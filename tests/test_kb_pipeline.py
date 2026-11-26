"""Tests for the ch03 pipeline flows: build, vectorize, mine, dedup, approval.

Milvus and embeddings are always faked here — no real network is touched.
"""

import json
from pathlib import Path

import pytest

from app.database import build_engine, build_session_factory
from app.kb import pipeline
from app.kb.chunker import Chunk
from app.kb.pipeline import StagingConflict
from app.models import Base, Conversation, KnowledgeChunk, Message
from app.repositories import qa_staging as staging_repository
from app.settings import Settings

SQLITE_URL = "sqlite+pysqlite:///:memory:"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_DIR = PROJECT_ROOT / "data" / "kb"

HERMETIC = Settings(
    chat_base_url="https://example.test/v1",
    chat_model="test-model",
    chat_api_key="test-key",
)


class FakeStore:
    """In-memory stand-in for MilvusStore keyed by primary chunk id."""

    def __init__(self):
        self.vectors: dict[int, list[float]] = []
        self.deleted: list[int] = []

    def upsert(self, ids, vectors):
        for chunk_id, vector in zip(ids, vectors):
            self.vectors.append((int(chunk_id), vector))

    def upsert_rows(self, rows):
        """ch04 schema upsert: rows carry id/dense plus the scalar columns."""
        for row in rows:
            self.vectors.append((int(row["id"]), row["dense"]))

    def delete(self, ids):
        self.deleted.extend(int(chunk_id) for chunk_id in ids)

    def count(self):
        return len(self.vectors)


def fake_embed(settings, texts):
    return [[float(len(text)), 0.5, 1024.0] for text in texts]


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


@pytest.fixture
def kb_documents():
    documents = pipeline.load_kb_documents(KB_DIR)
    assert len(documents) == 4
    return documents


def write_chunk(session_factory, chunk, content_type="policy"):
    with session_factory() as session:
        row = pipeline.knowledge_repository.insert_chunk(
            session,
            category=chunk.category,
            questions=chunk.questions,
            answer=chunk.answer,
            section_path=chunk.section_path,
            content_type=content_type,
            is_key_clause=chunk.is_key_clause,
            is_table_block=chunk.is_table_block,
        )
        session.commit()
        return row.id


# ---------------------------------------------------------------------------
# kb_build
# ---------------------------------------------------------------------------


def test_load_kb_documents_reads_the_four_corpus_files(kb_documents):
    by_file = {document["file"]: document for document in kb_documents}
    assert set(by_file) == {
        "returns-policy.md",
        "product-faq.md",
        "product-specs.md",
        "after-sales-manual.md",
    }
    returns = by_file["returns-policy.md"]
    assert returns["content_type"] == "policy"
    assert any("往返运费由买家承担" in chunk.answer for chunk in returns["chunks"])
    faq = by_file["product-faq.md"]
    assert faq["content_type"] == "faq"
    assert any("满 99 元包邮" in chunk.answer and "10 元运费" in chunk.answer for chunk in faq["chunks"])


def test_document_previews_are_json_safe_and_count_key_clauses(kb_documents):
    previews = pipeline.document_previews(kb_documents)
    returns = next(item for item in previews if item["file"] == "returns-policy.md")
    assert returns["chunk_count"] == len(returns["chunks"]) > 0
    assert returns["key_clause_count"] >= 1
    for chunk in returns["chunks"]:
        assert chunk["answer_chars"] == len(chunk["answer"].rstrip("…")) or chunk["answer"].endswith("…")


def test_build_knowledge_base_inserts_pending_and_links_pointers(session_factory, kb_documents):
    result = pipeline.build_knowledge_base(session_factory, KB_DIR)

    assert result["ok"] is True
    assert result["inserted"] > 0
    assert result["skipped"] == 0
    with session_factory() as session:
        rows = session.execute(
            KnowledgeChunk.__table__.select().order_by(KnowledgeChunk.id)
        ).mappings().all()
    assert all(row["vectorize_status"] == "pending" for row in rows)

    returns_rows = [row for row in rows if row["content_type"] == "policy" and (row["section_path"] or "").startswith("退货退款政策")]
    assert returns_rows, "returns-policy document must be persisted"
    first, second = returns_rows[0], returns_rows[1]
    assert first["prev_chunk_id"] is None and first["next_chunk_id"] == second["id"]
    assert second["prev_chunk_id"] == first["id"]


def test_build_knowledge_base_refuses_when_doc_chunks_exist(session_factory, kb_documents):
    chunk = Chunk(
        category="已有文档",
        questions="已有小节",
        answer="已有正文",
        section_path="已有文档 > 已有小节",
        content_type="policy",
        is_key_clause=False,
        is_table_block=False,
    )
    write_chunk(session_factory, chunk)

    result = pipeline.build_knowledge_base(session_factory, KB_DIR)

    assert result["ok"] is False and result["refused"] is True
    assert "--reset" in result["message"]
    with session_factory() as session:
        total = len(session.execute(KnowledgeChunk.__table__.select()).mappings().all())
    assert total == 1, "refused run must not insert anything"


def test_build_knowledge_base_reset_clears_doc_chunks_and_milvus_vectors(session_factory, kb_documents):
    chunk = Chunk(
        category="旧文档",
        questions="旧小节",
        answer="旧正文，运费条款。",
        section_path="旧文档 > 旧小节",
        content_type="policy",
        is_key_clause=True,
        is_table_block=False,
    )
    chunk_id = write_chunk(session_factory, chunk)
    with session_factory() as session:
        pipeline.knowledge_repository.mark_done(session, chunk_id, str(chunk_id))
        session.commit()
    store = FakeStore()

    result = pipeline.build_knowledge_base(session_factory, KB_DIR, reset=True, store=store)

    assert result["ok"] is True
    assert store.deleted == [chunk_id]
    with session_factory() as session:
        rows = session.execute(KnowledgeChunk.__table__.select()).mappings().all()
    assert all(row["content_type"] != "policy" or row["section_path"].startswith("退货退款政策") for row in rows)


def test_build_knowledge_base_skips_identical_chunks_within_one_run(session_factory, monkeypatch):
    chunks = [
        Chunk("重复文档", "运费说明", "正文内容完全一致。", "重复文档 > 小节", "policy", False, False),
        Chunk("重复文档", "运费说明", "正文内容完全一致。", "重复文档 > 另一节", "policy", False, False),
    ]
    monkeypatch.setattr(
        pipeline, "load_kb_documents", lambda kb_dir: [{"file": "重复文档.md", "title": "重复文档", "chars": 30, "content_type": "policy", "chunks": chunks}]
    )
    result = pipeline.build_knowledge_base(session_factory, Path("unused"))
    assert result["ok"] is True
    assert result["inserted"] == 1
    assert result["skipped"] == 1


# ---------------------------------------------------------------------------
# kb_vectorize
# ---------------------------------------------------------------------------


def test_vectorize_pending_embeds_upserts_and_marks_done(session_factory):
    chunk = Chunk("分类", "问法", "正文", "分类 > 小节", "policy", False, False)
    chunk_id = write_chunk(session_factory, chunk)
    store = FakeStore()

    result = pipeline.vectorize_pending(
        session_factory, HERMETIC, store=store, embed_fn=fake_embed
    )

    assert result["ok"] is True and result["vectorized"] == 1
    assert store.vectors and store.vectors[0][0] == chunk_id
    with session_factory() as session:
        row = session.get(KnowledgeChunk, chunk_id)
        assert row.vectorize_status == "done"
        assert row.vector_id == str(chunk_id)
        stats = pipeline.knowledge_repository.stats(session)
    assert stats["pending"] == 0 and stats["done"] == 1 and stats["total"] == 1


def test_vectorize_text_is_three_field_concatenation(session_factory):
    captured = {}

    def capture_embed(settings, texts):
        captured["texts"] = list(texts)
        return fake_embed(settings, texts)

    chunk = Chunk("货运政策", "运费计算", "满99包邮", "货运政策 > 运费计算", "policy", False, False)
    write_chunk(session_factory, chunk)
    pipeline.vectorize_pending(session_factory, HERMETIC, store=FakeStore(), embed_fn=capture_embed)

    assert captured["texts"] == ["货运政策\n运费计算\n满99包邮"]


def test_vectorize_only_processes_pending_so_reruns_are_cheap(session_factory):
    chunk = Chunk("分类", "问法", "正文", "分类 > 小节", "policy", False, False)
    chunk_id = write_chunk(session_factory, chunk)
    store = FakeStore()

    first = pipeline.vectorize_pending(session_factory, HERMETIC, store=store, embed_fn=fake_embed)
    second = pipeline.vectorize_pending(session_factory, HERMETIC, store=store, embed_fn=fake_embed)

    assert first["vectorized"] == 1
    assert second["vectorized"] == 0
    assert len(store.vectors) == 1, "done chunks must not be re-embedded or re-upserted"


def test_vectorize_interrupt_leaves_batch_pending_and_resume_completes_it(session_factory):
    for index in range(3):
        chunk = Chunk("分类", f"问法{index}", "正文", "分类 > 小节", "policy", False, False)
        write_chunk(session_factory, chunk)

    attempts = {"count": 0}

    def failing_embed(settings, texts):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("simulated interruption")
        return fake_embed(settings, texts)

    with pytest.raises(RuntimeError):
        pipeline.vectorize_pending(
            session_factory, HERMETIC, store=FakeStore(), embed_fn=failing_embed, batch_size=2
        )
    with session_factory() as session:
        assert pipeline.knowledge_repository.stats(session)["pending"] == 3

    resumed = pipeline.vectorize_pending(session_factory, HERMETIC, store=FakeStore(), embed_fn=fake_embed)
    assert resumed["vectorized"] == 3
    with session_factory() as session:
        assert pipeline.knowledge_repository.stats(session)["pending"] == 0


def test_vectorize_requires_embed_key(session_factory):
    chunk = Chunk("分类", "问法", "正文", "分类 > 小节", "policy", False, False)
    write_chunk(session_factory, chunk)
    keyless = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        embed_api_key="",
    )
    result = pipeline.vectorize_pending(session_factory, keyless, store=FakeStore())
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# kb_mine
# ---------------------------------------------------------------------------


class FakeModel:
    """Duck-typed ChatOpenAI returning canned parallel-array JSON."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls: list[list] = []

    def __call__(self, messages):
        self.calls.append(messages)
        payload = self.payloads.pop(0)

        class Response:
            content = json.dumps(payload, ensure_ascii=False)

        return Response()


def seed_history(session_factory):
    with session_factory() as session:
        for index in range(3):
            conversation = Conversation(id=f"conv-{index}", user_id="seed", status="closed")
            session.add(conversation)
            session.flush()
            Message(
                conversation_id=conversation.id, role="user", content=f"邮费怎么算{index}"
            )  # noqa: B015
            session.add(
                Message(conversation_id=conversation.id, role="user", content=f"邮费怎么算{index}")
            )
            session.add(
                Message(
                    conversation_id=conversation.id,
                    role="assistant",
                    content=f"满 99 元包邮，未满 99 元收取 10 元运费{index}",
                )
            )
            session.add(
                Message(conversation_id=conversation.id, role="tool", content="工具输出噪声")
            )
        session.commit()


def test_mine_conversations_writes_extracted_rows_with_batch_no(session_factory):
    seed_history(session_factory)
    model = FakeModel([
        {"questions": ["满99包邮吗", "运费多少"], "answers": ["满 99 包邮", "未满收 10 元"]}
    ] * 3)

    result = pipeline.mine_conversations(session_factory, HERMETIC, model=model, batch_size=2)

    assert result["ok"] is True
    assert result["batches"] == 2  # 3 conversations / batch_size=2 → 2 batches
    assert result["extracted"] == 4
    assert result["batch_no"].startswith("mine-")
    with session_factory() as session:
        rows = staging_repository.list_by_batch(session, result["batch_no"])
    assert len(rows) == 4
    assert all(row.status == "extracted" for row in rows)
    assert set(rows[0].source_ref.split(",")) <= {f"conv-{i}" for i in range(3)}
    # 并列标量数组 prompt 约束出现在系统消息里
    system_text = model.calls[0][0].content
    assert "questions" in system_text and "answers" in system_text


def test_mine_conversations_truncates_mismatched_arrays_with_warning(session_factory):
    seed_history(session_factory)
    model = FakeModel([
        {"questions": ["问1", "问2", "问3"], "answers": ["答1"]}
    ])

    result = pipeline.mine_conversations(session_factory, HERMETIC, model=model)

    assert result["extracted"] == 1, "length mismatch truncates to the shorter array"
    with session_factory() as session:
        rows = staging_repository.list_by_batch(session, result["batch_no"])
    assert [row.question for row in rows] == ["问1"]


def test_parse_parallel_qa_rejects_malformed_output():
    with pytest.raises(RuntimeError):
        pipeline.parse_parallel_qa("这不是 JSON")
    with pytest.raises(RuntimeError):
        pipeline.parse_parallel_qa(json.dumps([{"question": "q", "answer": "a"}]))
    questions, answers = pipeline.parse_parallel_qa(
        "```json\n{\"questions\": [\"q\"], \"answers\": [\"a\"]}\n```"
    )
    assert questions == ["q"] and answers == ["a"]


# ---------------------------------------------------------------------------
# kb_dedup + 审核
# ---------------------------------------------------------------------------


def seed_staging(session_factory, batch_no, items, status="extracted"):
    with session_factory() as session:
        for question, answer in items:
            staging_repository.insert_extracted(
                session, batch_no=batch_no, source_ref="conv-x", question=question, answer=answer
            )
        session.commit()


def test_dedup_batch_dedups_within_batch_and_against_corpus(session_factory):
    chunk = Chunk("分类", "邮费是多少", "满99包邮未满收10元", "分类 > 小节", "policy", False, False)
    write_chunk(session_factory, chunk)
    seed_staging(
        session_factory,
        "mine-1",
        [
            ("邮费是多少？", "满99包邮，未满收10元"),  # 与库内存量重复（归一化后）
            ("怎么退货运费谁出", "非质量问题10元一单"),
            ("怎么退货运费谁出！", "非质量问题10元一单"),  # 批内重复
        ],
    )

    result = pipeline.dedup_batch(session_factory, "mine-1")

    assert result["kept"] == 1 and result["discarded"] == 2
    with session_factory() as session:
        rows = staging_repository.list_by_batch(session, "mine-1")
    assert {row.status for row in rows} == {"kept", "discarded"}
    kept = [row for row in rows if row.status == "kept"]
    assert kept[0].question == "怎么退货运费谁出"


def test_dedup_batch_filters_by_batch_no_so_stale_rows_are_untouched(session_factory):
    seed_staging(session_factory, "mine-stale", [("旧批次问法", "旧批次答案")])
    seed_staging(session_factory, "mine-current", [("新批次问法", "新批次答案")])

    result = pipeline.dedup_batch(session_factory, "mine-current")

    assert result["batch_no"] == "mine-current" and result["kept"] == 1
    with session_factory() as session:
        stale = staging_repository.list_by_batch(session, "mine-stale")
    assert all(row.status == "extracted" for row in stale), "旧批次残留行不能被卷进本次去重"


def test_dedup_batch_defaults_to_the_latest_batch(session_factory):
    seed_staging(session_factory, "mine-20260101120000", [("老问法", "老答案")])
    seed_staging(session_factory, "mine-20260102120000", [("新问法", "新答案")])

    result = pipeline.dedup_batch(session_factory)

    assert result["batch_no"] == "mine-20260102120000"


def test_dedup_batch_on_empty_staging_is_a_noop(session_factory):
    result = pipeline.dedup_batch(session_factory)
    assert result["kept"] == 0 and result["batch_no"] is None


def test_approve_staging_row_inserts_mined_chunk_and_marks_approved(session_factory):
    seed_staging(session_factory, "mine-1", [("发货后多久能到", "江浙沪 1-2 天，其余 3-5 天")])
    with session_factory() as session:
        row = staging_repository.list_by_batch(session, "mine-1")[0]
        staging_repository.set_status(session, row.id, staging_repository.STATUS_KEPT)  # 去重后人工标记
        session.commit()
        row_id = row.id

    result = pipeline.approve_staging_row(session_factory, row_id)

    assert result["vectorize_status"] == "pending"
    with session_factory() as session:
        chunk = session.get(KnowledgeChunk, result["chunk_id"])
        row = staging_repository.get_staging_row(session, row_id)
    assert chunk.category == pipeline.MINED_CATEGORY
    assert chunk.content_type == "mined"
    assert chunk.questions == "发货后多久能到"
    assert chunk.vectorize_status == "pending"
    assert row.status == "approved"


def test_approve_staging_row_conflicts_on_wrong_status(session_factory):
    seed_staging(session_factory, "mine-1", [("问法", "答案")])
    with session_factory() as session:
        row_id = staging_repository.list_by_batch(session, "mine-1")[0].id

    with pytest.raises(StagingConflict):  # extracted 未去重不能直接采纳
        pipeline.approve_staging_row(session_factory, row_id)

    with session_factory() as session:
        staging_repository.set_status(session, row_id, staging_repository.STATUS_KEPT)
        session.commit()

    pipeline.approve_staging_row(session_factory, row_id)  # kept 可采纳
    with pytest.raises(StagingConflict):  # 重复提交（已 approved）
        pipeline.approve_staging_row(session_factory, row_id)


def test_approve_staging_row_conflicts_when_corpus_already_has_it(session_factory):
    chunk = Chunk("分类", "运费怎么算", "满99包邮", "分类 > 小节", "policy", False, False)
    write_chunk(session_factory, chunk)
    seed_staging(session_factory, "mine-1", [("运费怎么算！", "满99包邮")])  # 归一化后同文
    with session_factory() as session:
        row_id = staging_repository.list_by_batch(session, "mine-1")[0].id

    with pytest.raises(StagingConflict):
        pipeline.approve_staging_row(session_factory, row_id)


def test_approve_staging_row_on_unknown_row_raises_lookup(session_factory):
    with pytest.raises(LookupError):
        pipeline.approve_staging_row(session_factory, 4242)


def test_reject_staging_row_keeps_audit_trail(session_factory):
    seed_staging(session_factory, "mine-1", [("问法A", "答案A")])
    with session_factory() as session:
        row_id = staging_repository.list_by_batch(session, "mine-1")[0].id
        staging_repository.set_status(session, row_id, staging_repository.STATUS_KEPT)
        session.commit()

    result = pipeline.reject_staging_row(session_factory, row_id)

    assert result["status"] == "rejected"
    with session_factory() as session:
        row = staging_repository.get_staging_row(session, row_id)
    assert row.status == "rejected", "弃用留痕不删"
    with pytest.raises(StagingConflict):
        pipeline.reject_staging_row(session_factory, row_id)  # 再弃用返回冲突


def test_add_manual_entry_dedups_and_can_vectorize(session_factory):
    content_text = "# 手工文档\n\n## 运费\n\n满 99 元包邮，未满 99 元收取 10 元运费。"
    store = FakeStore()

    first = pipeline.add_manual_entry(
        session_factory, HERMETIC, content_text, "policy", vectorize=True, store=store, embed_fn=fake_embed
    )
    second = pipeline.add_manual_entry(
        session_factory, HERMETIC, content_text, "policy", vectorize=True, store=store, embed_fn=fake_embed
    )

    assert first["inserted"] == 1 and first["vectorized"] == 1
    assert second["inserted"] == 0 and second["skipped"] == 1
    assert len(store.vectors) == 1, "重复录入按指纹跳过，不重复向量化"
