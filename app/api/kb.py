"""The /kb knowledge-base console: materials, preview, entries, stats, search, review."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.orm import Session

from app.kb import pipeline as kb_pipeline
from app.kb.chunker import chunk_document
from app.kb.milvus_store import MilvusStore
from app.repositories import knowledge as knowledge_repository
from app.repositories import qa_staging as staging_repository
from app.schemas import KbEntryRequest, KbPreviewRequest, KbSearchRequest, KbStagingActionRequest
from app.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb")

KB_DIR = Path(__file__).resolve().parents[2] / "data" / "kb"


def _settings(request: Request) -> Settings:
    """Use the lifespan-owned settings so tests stay hermetic (no .env reads)."""
    return request.app.state.settings


@router.get("/materials")
def kb_materials(request: Request) -> dict:
    """材料清单与切块预览（与 kb_preview 脚本同一套逻辑）。"""
    documents = kb_pipeline.load_kb_documents(KB_DIR)
    previews = kb_pipeline.document_previews(documents)
    return {
        "documents": previews,
        "total_chunks": sum(item["chunk_count"] for item in previews),
        "total_key_clauses": sum(item["key_clause_count"] for item in previews),
    }


@router.post("/preview")
def kb_preview(payload: KbPreviewRequest) -> dict:
    """对粘贴的 Markdown 正文就地切块预览，只切不写。"""
    chunks = chunk_document(payload.content, payload.content_type)
    return {
        "content_type": payload.content_type,
        "chunk_count": len(chunks),
        "key_clause_count": sum(1 for chunk in chunks if chunk.is_key_clause),
        "table_block_count": sum(1 for chunk in chunks if chunk.is_table_block),
        "chunks": [kb_pipeline.chunk_preview(chunk) for chunk in chunks],
    }


@router.post("/entries")
def kb_entries(payload: KbEntryRequest, request: Request) -> dict:
    """手工录入：切块 + 指纹查重 + 入库(pending)，勾选后顺手向量化。"""
    result = kb_pipeline.add_manual_entry(
        request.app.state.session_factory,
        _settings(request),
        payload.content,
        payload.content_type,
        vectorize=payload.vectorize,
    )
    status_code = 200 if result.get("ok") else 400
    if status_code != 200:
        raise HTTPException(status_code=status_code, detail=result.get("message", "录入失败"))
    return result


@router.get("/stats")
def kb_stats(request: Request) -> dict:
    """三态账本：pending / done / Milvus 计数。"""
    session_factory: Callable[[], Session] = request.app.state.session_factory
    with session_factory() as session:
        ledger = knowledge_repository.stats(session)
    store = MilvusStore(_settings(request).milvus_uri)
    milvus_count = None
    note = None
    try:
        milvus_count = store.count()
    except Exception as exc:
        note = f"Milvus 不可达：{type(exc).__name__}"
    return {**ledger, "milvus": milvus_count, "note": note}


@router.post("/search")
def kb_search(payload: KbSearchRequest, request: Request) -> dict:
    """检索自测（本章 dense 单路）；降级时在 note 里说明。"""
    session_factory: Callable[[], Session] = request.app.state.session_factory
    result = kb_pipeline.retrieve_for_api(
        session_factory, _settings(request), payload.query, top_k=payload.top_k, mode=payload.mode
    )
    return {
        "query": payload.query,
        "mode": payload.mode,
        "top_k": payload.top_k,
        "degraded": result.degraded,
        "note": result.note,
        "results": result.matches,
    }


@router.post("/repatch")
def kb_repatch(request: Request) -> dict:
    """就地修块：diff data/kb 文档 → 更新/删除/新增 → 重嵌入（kb.html「就地修块」按钮）。"""
    session_factory: Callable[[], Session] = request.app.state.session_factory
    settings = _settings(request)
    kb_dir = Path(__file__).resolve().parents[2] / "data" / "kb"
    try:
        result = kb_pipeline.repatch_knowledge_base(session_factory, settings, kb_dir)
    except Exception as exc:
        logger.warning("kb repatch failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail=f"修块失败：{type(exc).__name__}") from exc
    return result


@router.get("/staging")
def kb_staging(request: Request, status: str | None = None) -> dict:
    """对话挖知识审核列表与计数。"""
    session_factory: Callable[[], Session] = request.app.state.session_factory
    with session_factory() as session:
        if status:
            rows = staging_repository.list_staging(session, statuses=[status])
        else:
            rows = staging_repository.list_staging(session)
        counts = staging_repository.staging_stats(session)
    return {
        "counts": counts,
        "rows": [
            {
                "id": row.id,
                "batch_no": row.batch_no,
                "source_ref": row.source_ref,
                "question": row.question,
                "answer": row.answer,
                "status": row.status,
            }
            for row in rows
        ],
    }


@router.post("/staging/approve")
def kb_staging_approve(payload: KbStagingActionRequest, request: Request) -> dict:
    """采纳一条 kept 暂存行：写 knowledge_chunks(pending) 并标 approved。

    仅 kept 状态可采纳；重复提交、弃用/丢弃行再提交一律 409。
    """
    session_factory: Callable[[], Session] = request.app.state.session_factory
    try:
        return kb_pipeline.approve_staging_row(session_factory, payload.id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except kb_pipeline.StagingConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/staging/reject")
def kb_staging_reject(payload: KbStagingActionRequest, request: Request) -> dict:
    """弃用一条 kept 暂存行：标 rejected 留痕不删；非 kept 状态返回 409。"""
    session_factory: Callable[[], Session] = request.app.state.session_factory
    try:
        return kb_pipeline.reject_staging_row(session_factory, payload.id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except kb_pipeline.StagingConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
