"""ch09 · 用户反馈入口:POST /api/feedback(👍/👎 落池,飞轮第三入口)。

- 会话/轮次校验:checkpoint 快照回捞(get_turn_snapshot)——404 会话不存在,
  400 轮次越界;
- 落池 source=user_feedback,reason 带 rating/turn/comment,当时的召回快照
  随条存 retrieved_chunks(审核页给人看);
- 同轮重复反馈幂等(created=false),不重复入池。
"""

import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.schemas import FeedbackRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest) -> dict:
    try:
        snapshot, created = await runtime.record_feedback(
            request.conversation_id, request.turn_index, request.rating, request.comment
        )
    except runtime.ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown conversation_id") from exc
    except runtime.TurnNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError:
        logger.exception("feedback persistence failed")
        raise HTTPException(status_code=503, detail="反馈保存失败，请稍后重试。") from None
    return {
        "status": "ok" if created else "duplicate",
        "rating": request.rating,
        "snapshot": {
            "query": snapshot["query"],
            "retrieved_snapshot": snapshot["retrieved_snapshot"],
            "citations": snapshot["citations"],
            "fallback_source": snapshot["fallback_source"],
        },
    }


@router.get("/api/feedback/snapshot")
async def feedback_snapshot(conversation_id: str, turn_index: int) -> dict:
    """调试用:直接回捞某轮快照(审核页「用户原话+当时召回」同源)。"""
    try:
        return await runtime.get_turn_snapshot(conversation_id, turn_index)
    except runtime.ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown conversation_id") from exc
    except runtime.TurnNotFound as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
