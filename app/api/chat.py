"""ch05/ch06 · POST /api/chat/stream:把 LangGraph runtime 事件流渲染成 SSE。

帧契约保留 ch04:conversation / tool / token / citations / done([DONE]) / error,
ch05 新增 actions(转人工/建工单提议),ch06 新增 interrupt(选单卡片,图暂停)。
FastAPI 0.142 的 EventSourceResponse 特判要求路由本身是 async generator,因此
404/DB 错误经 Depends(prepare_chat) 在开流前抛出,保持「错误路径不吐半个 SSE 帧」。
"""

import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# 帧契约常量(与 ch04 完全一致,前端零改动即可继续工作)
UPSTREAM_ERROR_CODE = "upstream_stream_error"
UPSTREAM_ERROR_MESSAGE = "客服服务暂时不可用，请稍后重试。"
PERSISTENCE_ERROR_CODE = "persistence_error"
PERSISTENCE_ERROR_MESSAGE = "会话记录保存失败，请稍后重试。"


async def prepare_chat(request: ChatRequest) -> tuple[str, bool]:
    """Validate the session and persist the user message before any SSE bytes are sent."""
    try:
        return await runtime.prepare_turn(request.conversation_id, request.message)
    except runtime.ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown conversation_id") from exc
    except SQLAlchemyError as exc:
        logger.error("chat prepare persistence failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail=PERSISTENCE_ERROR_MESSAGE) from exc


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def stream_chat(
    prepared: Annotated[tuple[str, bool], Depends(prepare_chat)],
    request: ChatRequest,
) -> AsyncIterator[ServerSentEvent]:
    """Stream one graph-driven conversation turn as SSE frames."""
    conversation_id, created = prepared
    if created:
        yield ServerSentEvent(
            event="conversation",
            data={"conversation_id": conversation_id},
        )
    try:
        async for event in runtime.stream_turn(conversation_id, request.message):
            yield _to_sse(event)
    except SQLAlchemyError:
        logger.exception("chat stream persistence failed")
        yield ServerSentEvent(
            event="error",
            data={"code": PERSISTENCE_ERROR_CODE, "message": PERSISTENCE_ERROR_MESSAGE},
        )
    except Exception as exc:
        # 上游挂了只给句式化提示:原始异常文本可能带 key/内网信息,绝不透传
        logger.error("chat stream failed (%s)", type(exc).__name__)
        yield ServerSentEvent(
            event="error",
            data={"code": UPSTREAM_ERROR_CODE, "message": UPSTREAM_ERROR_MESSAGE},
        )


def _to_sse(event: dict) -> ServerSentEvent:
    """Render one runtime event as an SSE frame(ch04 帧名/形状 + actions)。"""
    kind = event["type"]
    if kind == "delta":
        return ServerSentEvent(event="token", data={"delta": event["text"]})
    if kind == "tool":
        return ServerSentEvent(
            event="tool",
            data={
                "tool_call_id": event["tool_call_id"],
                "name": event["name"],
                "status": event["status"],
            },
        )
    if kind == "citations":
        return ServerSentEvent(
            event="citations",
            data={"conversation_id": event.get("conversation_id"), "items": event["items"]},
        )
    if kind == "actions":
        return ServerSentEvent(event="actions", data={"items": event["items"]})
    if kind == "interrupt":
        # ch06/ch08:图暂停等人工输入。本次流到此为止(后面不会再有帧,
        # 前端凭 payload.type 渲染:select_order 选单卡 / confirm_ticket 工单确认卡)
        payload = event.get("payload") or {}
        return ServerSentEvent(
            event="interrupt",
            data={
                "kind": payload.get("type"),
                "orders": payload.get("orders") or [],
                "preview": payload.get("preview") or {},
            },
        )
    if kind == "done":
        return ServerSentEvent(event="done", raw_data="[DONE]")
    raise ValueError(f"unknown runtime event type: {kind}")
