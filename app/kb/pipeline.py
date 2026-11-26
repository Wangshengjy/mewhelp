"""End-to-end knowledge base flows shared by scripts and the /kb API.

Flows: document scanning/preview, kb_build (chunk→dedup→pending), kb_vectorize
(pending→embed→Milvus→done), kb_mine (history→LLM→staging), kb_dedup
(staging→kept/discarded) and staged-row approval (kept→knowledge_chunks).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.kb.chunker import Chunk, chunk_document, contains_key_clause, document_title, resolve_content_type
from app.kb.embeddings import embed_texts
from app.kb.milvus_store import MilvusStore
from app.kb import retrieval as kb_retrieval
from app.kb.retrieval import RetrievalResult, retrieve as dense_retrieve
from app.models import Conversation, KnowledgeChunk, Message
from app.repositories import knowledge as knowledge_repository
from app.repositories import qa_staging as staging_repository
from app.settings import Settings

logger = logging.getLogger(__name__)

MINED_CATEGORY = "历史对话挖掘"
MINED_CONTENT_TYPE = "mined"
DEFAULT_BATCH_SIZE = 20
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_TOP_K = 3

MINING_SYSTEM_PROMPT = (
    "你是客服知识挖掘助手。下面给你若干段历史客服对话，请从中抽取“用户问法 + 客服答案”问答对。\n"
    "要求：\n"
    "1. 只抽取客服给出明确、可复用答案的问答；寒暄、无法独立成答的内容不要输出。\n"
    "2. 问法保留用户的真实说法（可以合并同义追问）；答案保留客服给出的具体事实与数字。\n"
    "3. 严格输出 JSON，格式为两个并列数组：{\"questions\": [\"问法1\", \"问法2\"], \"answers\": [\"答案1\", \"答案2\"]}。\n"
    "4. questions 与 answers 一一对应，数组长度必须一致；不要输出其它字段，不要输出 markdown 代码块。"
)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


class StagingConflict(RuntimeError):
    """Raised when a staging row cannot be approved (wrong status or duplicate)."""


# ---------------------------------------------------------------------------
# 文档扫描与预览（kb_preview 脚本与 GET /api/kb/materials 共用）
# ---------------------------------------------------------------------------

def load_kb_documents(kb_dir: Path) -> list[dict]:
    """Read every data/kb/*.md and chunk it; chunks stay full-text for build."""
    documents: list[dict] = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        content_type = resolve_content_type(path.name)
        documents.append(
            {
                "file": path.name,
                "title": document_title(text) or path.stem,
                "chars": len(text),
                "content_type": content_type,
                "chunks": chunk_document(text, content_type),
            }
        )
    return documents


# ---------------------------------------------------------------------------
# 在线检索自测（POST /api/kb/search）：本章 dense 单路
# ---------------------------------------------------------------------------

def retrieve_for_api(
    session_factory: Callable[[], Session],
    settings: Settings,
    query: str,
    *,
    top_k: int = 5,
    mode: str = "dense",
) -> RetrievalResult:
    """API 检索入口：dense 走 ch03 密集单路（含降级），其余走 ch04 四策略管线。"""
    if mode == "dense":
        return dense_retrieve(session_factory, settings, query, top_k=top_k)
    if mode not in kb_retrieval.STRATEGIES:
        raise ValueError(f"暂不支持检索模式：{mode}")
    hits = kb_retrieval.search_knowledge(session_factory, settings, query, strategy=mode, top_k=top_k)
    return RetrievalResult(matches=hits)


def chunk_preview(chunk: Chunk, answer_limit: int = 120) -> dict:
    """JSON-safe single chunk preview (answer truncated)."""
    answer = chunk.answer.replace("\n", " ")
    if len(answer) > answer_limit:
        answer = answer[:answer_limit] + "…"
    return {
        "category": chunk.category,
        "questions": chunk.questions,
        "answer": answer,
        "answer_chars": len(chunk.answer),
        "section_path": chunk.section_path,
        "content_type": chunk.content_type,
        "is_key_clause": chunk.is_key_clause,
        "is_table_block": chunk.is_table_block,
    }


def document_previews(documents: Sequence[dict]) -> list[dict]:
    """Material list + per-chunk previews as returned by the preview endpoints."""
    previews: list[dict] = []
    for document in documents:
        chunks: list[Chunk] = document["chunks"]
        previews.append(
            {
                "file": document["file"],
                "title": document["title"],
                "content_type": document["content_type"],
                "chars": document["chars"],
                "chunk_count": len(chunks),
                "key_clause_count": sum(1 for chunk in chunks if chunk.is_key_clause),
                "table_block_count": sum(1 for chunk in chunks if chunk.is_table_block),
                "chunks": [chunk_preview(chunk) for chunk in chunks],
            }
        )
    return previews


# ---------------------------------------------------------------------------
# kb_build：全量切块 → 指纹查重 → 写 knowledge_chunks(pending)
# ---------------------------------------------------------------------------

def build_knowledge_base(
    session_factory: Callable[[], Session],
    kb_dir: Path,
    *,
    reset: bool = False,
    store: MilvusStore | None = None,
) -> dict:
    """Chunk every document and insert pending rows.

    kb_build itself is not idempotent across runs: when document-type chunks
    already exist it refuses and asks for ``--reset`` (clear then rebuild),
    matching the course reference implementation. ``reset`` removes document
    chunks (and their Milvus vectors when a store is reachable) first.
    """
    documents = load_kb_documents(kb_dir)
    with session_factory() as session:
        existing = knowledge_repository.count_doc_chunks(session)
        if existing and not reset:
            return {
                "ok": False,
                "refused": True,
                "existing_doc_chunks": existing,
                "message": (
                    f"knowledge_chunks 已有 {existing} 条文档类块；kb_build 无跨运行幂等，"
                    "请先清空重建：uv run python -m scripts.kb_build --reset"
                ),
            }
    if reset and existing:
        _clear_document_chunks(session_factory, store)

    inserted = 0
    skipped = 0
    per_document: list[dict] = []
    for document in documents:
        doc_inserted: list[int] = []
        with session_factory() as session:
            fingerprints = knowledge_repository.list_fingerprints(session)
            for chunk in document["chunks"]:
                fingerprint = knowledge_repository.chunk_fingerprint(chunk.category, chunk.questions, chunk.answer)
                if fingerprint in fingerprints:
                    skipped += 1
                    continue
                row = knowledge_repository.insert_chunk(
                    session,
                    category=chunk.category,
                    questions=chunk.questions,
                    answer=chunk.answer,
                    section_path=chunk.section_path,
                    content_type=chunk.content_type,
                    is_key_clause=chunk.is_key_clause,
                    is_table_block=chunk.is_table_block,
                )
                fingerprints.add(fingerprint)
                doc_inserted.append(row.id)
            inserted += len(doc_inserted)
            if doc_inserted:
                # 第二趟：全部插入拿到自增 id 后再回填同文档 prev/next 指针
                relinked = knowledge_repository.relink_document(session, document["title"], document["content_type"])
            session.commit()
        per_document.append(
            {"file": document["file"], "title": document["title"], "inserted": len(doc_inserted), "linked": relinked if doc_inserted else 0}
        )
    return {
        "ok": True,
        "inserted": inserted,
        "skipped": skipped,
        "documents": per_document,
        "message": f"建库完成：插入 {inserted} 块（pending），指纹跳过 {skipped} 块",
    }


def _clear_document_chunks(session_factory: Callable[[], Session], store: MilvusStore | None) -> dict:
    removed_ids: list[int] = []
    removed_vectors: list[int] = []
    with session_factory() as session:
        rows = knowledge_repository.delete_doc_chunks(session)
        for row in rows:
            removed_ids.append(row.id)
            if row.vector_id:
                try:
                    removed_vectors.append(int(row.vector_id))
                except ValueError:
                    continue
        session.commit()
    if store is not None and removed_vectors:
        try:
            store.delete(removed_vectors)
        except Exception as exc:
            logger.warning("reset could not clear Milvus vectors (%s)", type(exc).__name__)
    return {"removed": len(removed_ids), "vectors_removed": len(removed_vectors)}


# ---------------------------------------------------------------------------
# kb_vectorize：捞 pending → 批量嵌入 → upsert Milvus → 回填 → done
# ---------------------------------------------------------------------------

def vectorize_pending(
    session_factory: Callable[[], Session],
    settings: Settings,
    *,
    store: MilvusStore | None = None,
    embed_fn: Callable[[Settings, Sequence[str]], list[list[float]]] | None = None,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> dict:
    """Vectorize pending chunks batch by batch; each batch commits on its own.

    Interrupted runs simply leave the remaining batches pending, so re-running
    picks up exactly what is missing (per-primary-key upsert is idempotent).
    """
    embed = embed_fn or embed_texts
    store = store or MilvusStore(settings.milvus_uri)
    if embed_fn is None and not settings.embed_api_key:
        return {"ok": False, "message": "EMBED_API_KEY 未配置，无法向量化；请先配置后重试"}

    vectorized = 0
    batches = 0
    while True:
        with session_factory() as session:
            pending = knowledge_repository.list_pending(session)[:batch_size]
            if not pending:
                break
            texts = [f"{chunk.category}\n{chunk.questions}\n{chunk.answer}" for chunk in pending]
            vectors = embed(settings, texts)
            # ch04：新集合带 BM25 全文字段与标量列，upsert_rows 一次写入全部字段
            rows = [
                {
                    "id": chunk.id,
                    "dense": vector,
                    "text": f"{chunk.category}\n{chunk.questions}\n{chunk.answer}",
                    "question": chunk.questions,
                    "answer": chunk.answer,
                    "section_path": chunk.section_path or "",
                    "content_type": chunk.content_type or "",
                    "category": chunk.category,
                }
                for chunk, vector in zip(pending, vectors)
            ]
            store.upsert_rows(rows)
            for chunk, _vector in zip(pending, vectors):
                knowledge_repository.mark_done(session, chunk.id, str(chunk.id))
            session.commit()
            vectorized += len(pending)
            batches += 1
    stats = _kb_stats(session_factory, store)
    return {
        "ok": True,
        "vectorized": vectorized,
        "batches": batches,
        "stats": stats,
        "message": f"向量化完成：本轮处理 {vectorized} 块（{batches} 批）",
    }


def _kb_stats(session_factory: Callable[[], Session], store: MilvusStore | None) -> dict:
    with session_factory() as session:
        ledger = knowledge_repository.stats(session)
    milvus_count = None
    if store is not None:
        try:
            milvus_count = store.count()
        except Exception:
            milvus_count = None
    return {**ledger, "milvus": milvus_count}


# ---------------------------------------------------------------------------
# kb_repatch：文档改动 → 按小节路径 diff 出受影响块 → 更新/删除/新增 → 重嵌入
# ---------------------------------------------------------------------------

def repatch_knowledge_base(
    session_factory: Callable[[], Session],
    settings: Settings,
    kb_dir: Path,
    *,
    store: MilvusStore | None = None,
    embed_fn: Callable[[Settings, Sequence[str]], list[list[float]]] | None = None,
) -> dict:
    """Diff data/kb documents against the stored corpus and patch in place.

    Matching rule: within one document, chunks are keyed by ``section_path``.
    Same path + same fingerprint → untouched; same path + different content →
    update the row in place (id preserved, re-embedded); path gone → delete the
    chunk and its vector; new path → insert. Ids are preserved so citations that
    already point at a chunk stay valid after a content fix.
    """
    store = store or MilvusStore(settings.milvus_uri)
    embed = embed_fn or embed_texts
    documents = load_kb_documents(kb_dir)
    totals = {"updated": 0, "added": 0, "removed": 0, "unchanged": 0}
    per_document: list[dict] = []

    for document in documents:
        doc_updated: list[str] = []
        doc_added: list[str] = []
        doc_removed: list[str] = []
        doc_unchanged = 0
        with session_factory() as session:
            existing = knowledge_repository.list_doc_chunks(session, document["title"], document["content_type"])
            existing_by_path = {row.section_path or "": row for row in existing}
            new_paths = {chunk.section_path or "" for chunk in document["chunks"]}

            for chunk in document["chunks"]:
                path = chunk.section_path or ""
                fingerprint = knowledge_repository.chunk_fingerprint(chunk.category, chunk.questions, chunk.answer)
                row = existing_by_path.get(path)
                if row is not None:
                    row_fingerprint = knowledge_repository.chunk_fingerprint(row.category, row.questions, row.answer)
                    if row_fingerprint == fingerprint:
                        doc_unchanged += 1
                        continue
                    # 就地更新：保留原 id，重置为 pending 等待重嵌入
                    row.category = chunk.category
                    row.questions = chunk.questions
                    row.answer = chunk.answer
                    row.is_key_clause = chunk.is_key_clause
                    row.is_table_block = chunk.is_table_block
                    row.vectorize_status = knowledge_repository.STATUS_PENDING
                    row.updated_at = datetime.now()
                    doc_updated.append(path)
                else:
                    knowledge_repository.insert_chunk(
                        session,
                        category=chunk.category,
                        questions=chunk.questions,
                        answer=chunk.answer,
                        section_path=chunk.section_path,
                        content_type=chunk.content_type,
                        is_key_clause=chunk.is_key_clause,
                        is_table_block=chunk.is_table_block,
                    )
                    doc_added.append(path)

            removed_ids: list[int] = []
            removed_vectors: list[int] = []
            for path, row in existing_by_path.items():
                if path in new_paths:
                    continue
                removed_ids.append(row.id)
                if row.vector_id:
                    try:
                        removed_vectors.append(int(row.vector_id))
                    except ValueError:
                        continue
                session.delete(row)
                doc_removed.append(path)

            if doc_updated or doc_added or removed_ids:
                knowledge_repository.relink_document(session, document["title"], document["content_type"])
            session.commit()

        if removed_vectors and store is not None:
            try:
                store.delete(removed_vectors)
            except Exception as exc:
                logger.warning("kb_repatch could not clear Milvus vectors (%s)", type(exc).__name__)

        totals["updated"] += len(doc_updated)
        totals["added"] += len(doc_added)
        totals["removed"] += len(doc_removed)
        totals["unchanged"] += doc_unchanged
        per_document.append(
            {
                "file": document["file"],
                "updated": doc_updated,
                "added": doc_added,
                "removed": doc_removed,
                "unchanged": doc_unchanged,
            }
        )

    vectorized = 0
    try:
        result = vectorize_pending(session_factory, settings, store=store, embed_fn=embed_fn)
        vectorized = int(result.get("vectorized") or 0)
    except Exception as exc:
        logger.warning("kb_repatch re-vectorize failed (%s)", type(exc).__name__)

    return {
        "ok": True,
        "updated": totals["updated"],
        "added": totals["added"],
        "removed": totals["removed"],
        "unchanged": totals["unchanged"],
        "revectorized": vectorized,
        "documents": per_document,
        "message": (
            f"修块完成：更新 {totals['updated']} / 新增 {totals['added']} / 删除 {totals['removed']} /"
            f" 未变 {totals['unchanged']}，重嵌入 {vectorized} 块"
        ),
    }


# ---------------------------------------------------------------------------
# kb_mine：历史对话 → 分批喂 CHAT 模型 → qa_extraction_staging(extracted)
# ---------------------------------------------------------------------------

def load_mining_conversations(session_factory: Callable[[], Session]) -> list[dict]:
    """Conversations with complete user→assistant pairs (tool noise dropped)."""
    conversations: list[dict] = []
    with session_factory() as session:
        rows = session.scalars(select(Conversation).order_by(Conversation.created_at, Conversation.id)).all()
        for conversation in rows:
            messages = session.scalars(
                select(Message).where(Message.conversation_id == conversation.id).order_by(Message.id)
            ).all()
            pairs: list[tuple[str, str]] = []
            pending_question: str | None = None
            for message in messages:
                if message.role == "user":
                    pending_question = message.content
                elif message.role == "assistant" and message.content.strip() and pending_question:
                    pairs.append((pending_question, message.content))
                    pending_question = None
            if pairs:
                conversations.append({"id": conversation.id, "pairs": pairs})
    return conversations


def parse_parallel_qa(raw_text: str) -> tuple[list[str], list[str]]:
    """Parse the strict parallel-array JSON the mining model must return."""
    cleaned = _JSON_FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"挖掘模型输出不是合法 JSON：{type(exc).__name__}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("questions"), list) or not isinstance(
        payload.get("answers"), list
    ):
        raise RuntimeError("挖掘模型输出缺少并列数组 questions/answers")
    questions = [str(item) for item in payload["questions"] if str(item).strip()]
    answers = [str(item) for item in payload["answers"] if str(item).strip()]
    return questions, answers


def mine_conversations(
    session_factory: Callable[[], Session],
    settings: Settings,
    *,
    model: Callable[[list], object] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Mine knowledge from history conversations in batches of <=20 sessions.

    The model must answer with parallel scalar arrays ``questions[]`` /
    ``answers[]`` (nested object arrays make the upstream 502). When the two
    arrays disagree in length we truncate to the shorter one with a warning —
    never a silent misalignment.
    """
    conversations = load_mining_conversations(session_factory)
    if not conversations:
        return {"ok": True, "batch_no": None, "extracted": 0, "batches": 0, "message": "没有可挖掘的历史会话"}
    if model is None:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            base_url=settings.chat_base_url,
            api_key=settings.chat_api_key,
            model=settings.chat_model,
            temperature=0.1,
        )
        model = lambda messages: chat.invoke(messages)  # noqa: E731

    batch_no = f"mine-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    extracted = 0
    batches = 0
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start : start + batch_size]
        source_ref = ",".join(item["id"] for item in batch)[:255]
        prompt_text = "\n\n".join(
            f"[会话 {item['id']}]\n"
            + "\n".join(f"用户：{question}\n客服：{answer}" for question, answer in item["pairs"])
            for item in batch
        )
        response = model([SystemMessage(content=MINING_SYSTEM_PROMPT), HumanMessage(content=prompt_text)])
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        questions, answers = parse_parallel_qa(str(content))
        if len(questions) != len(answers):
            shorter = min(len(questions), len(answers))
            logger.warning(
                "mining batch %s: questions(%d) != answers(%d); truncating to %d",
                batch_no,
                len(questions),
                len(answers),
                shorter,
            )
            questions, answers = questions[:shorter], answers[:shorter]
        with session_factory() as session:
            for question, answer in zip(questions, answers):
                staging_repository.insert_extracted(
                    session, batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
                )
            session.commit()
        extracted += len(questions)
        batches += 1
    return {
        "ok": True,
        "batch_no": batch_no,
        "extracted": extracted,
        "batches": batches,
        "conversations": len(conversations),
        "message": f"挖掘完成：{len(conversations)} 个会话 / {batches} 批，抽出 {extracted} 条问答（暂存待去重审核）",
    }


# ---------------------------------------------------------------------------
# kb_dedup：按 batch_no 过滤 → 问法+正文指纹 → kept/discarded（不直接入库）
# ---------------------------------------------------------------------------

def dedup_batch(session_factory: Callable[[], Session], batch_no: str | None = None) -> dict:
    """Dedup one staging batch against the stored corpus and within itself.

    Only the requested batch (default: the newest) is processed, so stale
    ``extracted`` rows left by an interrupted earlier run are never swept in.
    Rows are only marked kept/discarded here; approval is a separate human step.
    """
    with session_factory() as session:
        batch = batch_no or staging_repository.latest_batch_no(session)
        if not batch:
            return {"ok": True, "batch_no": None, "kept": 0, "discarded": 0, "message": "暂存表为空，无需去重"}
        rows = staging_repository.list_by_batch(session, batch, status=staging_repository.STATUS_EXTRACTED)
        fingerprints = {
            knowledge_repository.qa_fingerprint(chunk.questions, chunk.answer)
            for chunk in session.scalars(select(KnowledgeChunk)).all()
        }
        seen: set[str] = set()
        kept = 0
        discarded = 0
        for row in rows:
            fingerprint = knowledge_repository.qa_fingerprint(row.question, row.answer)
            if fingerprint in fingerprints or fingerprint in seen:
                row.status = staging_repository.STATUS_DISCARDED
                discarded += 1
            else:
                row.status = staging_repository.STATUS_KEPT
                kept += 1
                seen.add(fingerprint)
                fingerprints.add(fingerprint)
        session.commit()
        return {
            "ok": True,
            "batch_no": batch,
            "kept": kept,
            "discarded": discarded,
            "total": len(rows),
            "message": f"批次 {batch} 去重完成：保留 {kept} 条待人工审核，丢弃重复 {discarded} 条",
        }


def approve_staging_row(session_factory: Callable[[], Session], row_id: int) -> dict:
    """采纳一条 kept 暂存行：写入 knowledge_chunks(pending) 并标 approved。

    Only ``kept`` rows are accepted; re-approving or submitting a
    rejected/discarded/approved row raises :class:`StagingConflict` (HTTP 409).
    """
    with session_factory() as session:
        row = staging_repository.get_staging_row(session, row_id)
        if row is None:
            raise LookupError(f"暂存行 {row_id} 不存在")
        if row.status != staging_repository.STATUS_KEPT:
            raise StagingConflict(f"仅 kept 状态可采纳（当前状态：{row.status}）")
        fingerprint = knowledge_repository.qa_fingerprint(row.question, row.answer)
        for chunk in session.scalars(select(KnowledgeChunk)).all():
            if knowledge_repository.qa_fingerprint(chunk.questions, chunk.answer) == fingerprint:
                raise StagingConflict("knowledge_chunks 已存在相同问法+正文，拒绝重复入库")
        chunk = knowledge_repository.insert_chunk(
            session,
            category=MINED_CATEGORY,
            questions=row.question,
            answer=row.answer,
            section_path=f"{MINED_CATEGORY}/{row.batch_no}",
            content_type=MINED_CONTENT_TYPE,
            is_key_clause=contains_key_clause(row.question, row.answer),
            is_table_block=False,
        )
        row.status = staging_repository.STATUS_APPROVED
        session.commit()
        return {
            "chunk_id": chunk.id,
            "staging_id": row.id,
            "status": staging_repository.STATUS_APPROVED,
            "vectorize_status": knowledge_repository.STATUS_PENDING,
            "message": f"已采纳为知识块 #{chunk.id}（pending，等待 kb_vectorize）",
        }


def reject_staging_row(session_factory: Callable[[], Session], row_id: int) -> dict:
    """人工弃用一条 kept 暂存行：标 rejected 留痕不删。"""
    with session_factory() as session:
        row = staging_repository.get_staging_row(session, row_id)
        if row is None:
            raise LookupError(f"暂存行 {row_id} 不存在")
        if row.status != staging_repository.STATUS_KEPT:
            raise StagingConflict(f"仅 kept 状态可弃用（当前状态：{row.status}）")
        row.status = staging_repository.STATUS_REJECTED
        session.commit()
        return {"staging_id": row.id, "status": staging_repository.STATUS_REJECTED, "message": "已弃用（留痕不删）"}


# ---------------------------------------------------------------------------
# 手工录入（POST /api/kb/entries 共用）
# ---------------------------------------------------------------------------

def add_manual_entry(
    session_factory: Callable[[], Session],
    settings: Settings,
    content: str,
    content_type: str,
    *,
    vectorize: bool = False,
    store: MilvusStore | None = None,
    embed_fn: Callable[[Settings, Sequence[str]], list[list[float]]] | None = None,
) -> dict:
    """Chunk pasted content, dedup by fingerprint, insert pending, optionally vectorize."""
    chunks = chunk_document(content, content_type)
    if not chunks:
        return {"ok": False, "message": "内容没有可入库的正文块（需要标题 + 正文）"}
    inserted_ids: list[int] = []
    skipped = 0
    with session_factory() as session:
        fingerprints = knowledge_repository.list_fingerprints(session)
        for chunk in chunks:
            fingerprint = knowledge_repository.chunk_fingerprint(chunk.category, chunk.questions, chunk.answer)
            if fingerprint in fingerprints:
                skipped += 1
                continue
            row = knowledge_repository.insert_chunk(
                session,
                category=chunk.category,
                questions=chunk.questions,
                answer=chunk.answer,
                section_path=chunk.section_path,
                content_type=content_type,
                is_key_clause=chunk.is_key_clause,
                is_table_block=chunk.is_table_block,
            )
            fingerprints.add(fingerprint)
            inserted_ids.append(row.id)
        session.commit()
    note: str | None = None
    vectorized = 0
    if vectorize and inserted_ids:
        if embed_fn is None and not settings.embed_api_key:
            note = "EMBED_API_KEY 未配置，仅入库为 pending，稍后用 kb_vectorize 补齐"
        else:
            result = vectorize_pending(session_factory, settings, store=store, embed_fn=embed_fn)
            vectorized = int(result.get("vectorized") or 0)
    return {
        "ok": True,
        "chunk_count": len(chunks),
        "inserted": len(inserted_ids),
        "skipped": skipped,
        "chunk_ids": inserted_ids,
        "vectorized": vectorized,
        "note": note,
        "message": f"录入完成：新增 {len(inserted_ids)} 块，指纹跳过 {skipped} 块"
        + (f"，已向量化 {vectorized} 块" if vectorized else ""),
    }
