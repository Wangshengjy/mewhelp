"""ch05/ch06 · 前端动作端点:建工单 / 退款申请 / interrupt 续跑。

create_ticket / submit_refund 在图里被拦截为「提议」,真正写库发生在用户点
按钮确认后——本组端点就是那几次确认。复用 app/repositories/tickets.create_ticket
(同步函数,to_thread 包裹),工单类别收中文枚举(与前端下拉一致),落库原样
存字符串(tickets.ticket_type 是 varchar(64),加「退款」无需迁移)。
resume:图被 fetch_order 的 interrupt 暂停后,前端把选中订单号 POST 回来,
以 EventSourceResponse 续流(与 /api/chat/stream 同一帧契约)。
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Callable, Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.chat import _to_sse
from app.dependencies import get_session_factory
from app.graph import runtime
from app.repositories import tickets as tickets_repository

logger = logging.getLogger(__name__)

router = APIRouter()


class CreateTicketRequest(BaseModel):
    conversation_id: str
    description: str = Field(min_length=1, max_length=2000)
    ticket_type: Literal["售后", "投诉", "咨询"]


class CreateTicketResponse(BaseModel):
    ticket_no: str
    status: str = "已转人工"


class CreateRefundRequest(BaseModel):
    conversation_id: str
    order_id: str = Field(min_length=1, max_length=64)
    reason: Literal["七天无理由", "质量问题", "发错货", "不想要了", "其他"]


class CreateRefundResponse(BaseModel):
    ticket_no: str
    status: str = "退款申请已提交"


class ResumeRequest(BaseModel):
    """interrupt 续跑请求:选单卡回 order_id(ch06),工单确认卡回 confirmed(ch08)。

    二选一;都缺按 400 拒绝(前端按钮语义明确,不允许空续跑)。"""

    conversation_id: str
    order_id: str | None = Field(default=None, min_length=1, max_length=64)
    confirmed: bool | None = None

    def resume_value(self):
        if self.order_id:
            return self.order_id
        if self.confirmed is not None:
            return {"confirmed": self.confirmed}
        return None


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket(
    request: CreateTicketRequest,
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
) -> CreateTicketResponse:
    try:
        return await asyncio.to_thread(
            _create_ticket, session_factory, request
        )
    except SQLAlchemyError as exc:
        # 会话不存在(FK 违例)或库不可用都归为 503,不暴露内部细节
        logger.error("create ticket persistence failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="工单创建失败，请稍后重试。") from exc


def _create_ticket(session_factory, request: CreateTicketRequest) -> CreateTicketResponse:
    with session_factory() as session:
        ticket = tickets_repository.create_ticket(
            session,
            conversation_id=request.conversation_id,
            description=request.description,
            ticket_type=request.ticket_type,
        )
        session.commit()
        return CreateTicketResponse(ticket_no=ticket.ticket_no)


@router.post("/api/actions/create-refund", response_model=CreateRefundResponse)
async def create_refund(
    request: CreateRefundRequest,
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
) -> CreateRefundResponse:
    """退款表单提交(ch06):图里的 submit_refund 只是提议,这里才是真正落库。

    reason 是前端固定五类下拉(自由文本不落库);description 由后端拼装,
    订单号/原因可追溯。"""
    try:
        return await asyncio.to_thread(
            _create_refund, session_factory, request
        )
    except SQLAlchemyError as exc:
        logger.error("create refund persistence failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=503, detail="退款申请提交失败，请稍后重试。") from exc


def _create_refund(session_factory, request: CreateRefundRequest) -> CreateRefundResponse:
    with session_factory() as session:
        ticket = tickets_repository.create_ticket(
            session,
            conversation_id=request.conversation_id,
            description=f"退款申请 订单{request.order_id} 原因:{request.reason}",
            ticket_type="退款",
        )
        session.commit()
        return CreateRefundResponse(ticket_no=ticket.ticket_no)


# ── ch06 · interrupt 续跑 ────────────────────────────────────────────────────


async def prepare_resume(request: ResumeRequest) -> str:
    """开流前校验会话存在:404 必须在 SSE 开始前抛出,路由层才能回 404。

    (FastAPI 的 EventSourceResponse 特判要求路由本身是 async generator,
    生成器内部抛异常只会变成半途 error 帧,不是 HTTP 状态码。)"""
    try:
        await runtime.ensure_conversation(request.conversation_id)
    except runtime.ConversationNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown conversation_id") from exc
    return request.conversation_id


@router.post("/api/actions/resume", response_class=EventSourceResponse)
async def resume_action(
    prepared_conversation_id: Annotated[str, Depends(prepare_resume)],
    request: ResumeRequest,
) -> AsyncIterator[ServerSentEvent]:
    """interrupt 续跑(ch06 选单 / ch08 工单确认);帧契约与 /api/chat/stream 一致。

    resume 值按请求形状二选一:order_id(选中订单号)或 {"confirmed": bool}
    (工单确认卡按钮)。"""
    resume_value = request.resume_value()
    if resume_value is None:
        raise HTTPException(status_code=400, detail="order_id 或 confirmed 必须二选一。")
    try:
        async for event in runtime.stream_resume(prepared_conversation_id, resume_value):
            yield _to_sse(event)
    except Exception:
        logger.exception("resume stream failed (conv=%s)", prepared_conversation_id)
        yield ServerSentEvent(event="error", data={"message": "续跑失败，请稍后重试。"})
