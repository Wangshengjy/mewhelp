"""ch09 · 审核后台 API:待审队列 / 详情(用户原话+当时召回快照) / 通过写回 / 拒绝留痕。

通过写回复用 ch03 落库流程(pipeline.add_manual_entry:切块+指纹查重+pending 入库,
embed 配置齐则顺手向量化)——落库后即可被检索/精排命中,飞轮整圈闭合。
"""

import logging
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

from app.core.flywheel import FlywheelUnavailable
from app.core import flywheel as flywheel_core
from app.settings import Settings

logger = logging.getLogger(__name__)

router = APIRouter()
_PAGE_PATH = Path(__file__).resolve().parents[2] / "static" / "review.html"
PAGE_PATH = _PAGE_PATH


class ReviewActionRequest(BaseModel):
    """审核动作载荷:approve 必带核准答案,reject 可带备注。"""

    approved_answer: str = Field(default="", max_length=8000, description="核准答案(approve 必填)")
    note: str = Field(default="", max_length=2000, description="驳回/备注原因")


def _get_settings(request) -> Settings:
    return request.app.state.settings


@router.get("/api/review/queue")
async def review_queue(request: Request, status: Literal["待审", "通过", "驳回"] | None = None) -> dict:
    """待审队列列表(按出现次数降序=优先级)。"""
    deps = flywheel_core.deps_from_app(request.app)
    items = await flywheel_core.list_queue(deps, status)
    return {"items": items}


@router.get("/api/review/{review_id}")
async def review_detail(request: Request, review_id: int) -> dict:
    """缺口详情:标准化问题 + 归并原话流水(带 source/召回快照),审核人判断真缺还是没检到。"""
    deps = flywheel_core.deps_from_app(request.app)
    try:
        detail = await flywheel_core.get_detail(deps, review_id)
    except FlywheelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if detail is None:
        raise HTTPException(status_code=404, detail="review item not found")
    item, raws = detail
    return {
        "id": item.id,
        "normalized_question": item.normalized_question,
        "ai_suggested_answer": item.ai_suggested_answer,
        "occurrence_count": item.occurrence_count,
        "review_status": item.review_status,
        "approved_answer": item.approved_answer,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "raw_questions": [
            {
                "id": raw.id,
                "raw_question": raw.raw_question,
                "source": raw.source,
                "reason": raw.reason,
                "retrieved_chunks": raw.retrieved_chunks or [],
                "created_at": raw.created_at.isoformat() if raw.created_at else None,
            }
            for raw in raws
        ],
    }


@router.post("/api/review/{review_id}/approve")
async def approve_review(request: Request, review_id: int, payload: ReviewActionRequest) -> dict:
    """通过 + 写回知识库(复用 ch03 落库+向量化);同答案二次写回靠指纹幂等。"""
    if not payload.approved_answer.strip():
        raise HTTPException(status_code=422, detail="approved_answer 不能为空")
    deps = flywheel_core.deps_from_app(request.app)
    try:
        result = await flywheel_core.approve_and_writeback(
            deps, review_id, payload.approved_answer.strip()
        )
    except FlywheelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result


@router.post("/api/review/{review_id}/reject")
async def reject_review(request: Request, review_id: int, payload: ReviewActionRequest) -> dict:
    """拒绝留痕:状态置驳回,approved_answer 存驳回备注(终审,查重命中只累次数)。"""
    deps = flywheel_core.deps_from_app(request.app)
    try:
        updated = await flywheel_core.reject(deps, review_id, payload.note or payload.approved_answer)
    except FlywheelUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not updated:
        raise HTTPException(status_code=409, detail="仅「待审」可流转,或条目不存在")
    return {"status": "rejected", "id": review_id}


@router.get("/review", response_class=HTMLResponse, include_in_schema=False)
async def review_page() -> HTMLResponse:
    """ch09 审核后台页(静态单文件)。"""
    return HTMLResponse(content=_PAGE_PATH.read_text(encoding="utf-8"))
