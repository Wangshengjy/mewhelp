"""Idempotent, clearly fictional fixtures used by the local demo."""

from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Conversation, Faq, Message, Ticket

DEMO_CONVERSATION_ID = "00000000-0000-4000-8000-000000000001"
DEMO_TOOL_CALL_ID = "demo-query-faq-001"


def seed_demo_data(session_factory: Callable[[], Session]) -> None:
    """Insert fixed records using natural identifiers to support restarts."""
    with session_factory() as session:
        for question, answer, category in (
            ("演示商品可以退货吗？", "这是演示内容：商品签收后七天内可申请退货。", "退货"),
            ("如何联系人工客服？", "这是演示内容：请在工作日联系在线客服。", "服务"),
        ):
            existing_faq = session.scalar(
                select(Faq).where(Faq.question == question, Faq.category == category)
            )
            if existing_faq is None:
                session.add(Faq(question=question, answer=answer, category=category))

        conversation = session.get(Conversation, DEMO_CONVERSATION_ID)
        if conversation is None:
            conversation = Conversation(id=DEMO_CONVERSATION_ID, user_id="demo-user", status="open")
            session.add(conversation)
            session.flush()

        if not session.scalar(select(Message.id).where(Message.conversation_id == DEMO_CONVERSATION_ID)):
            session.add_all(
                [
                    Message(conversation_id=DEMO_CONVERSATION_ID, role="user", content="演示商品可以退货吗？"),
                    Message(
                        conversation_id=DEMO_CONVERSATION_ID,
                        role="assistant",
                        content="",
                        tool_calls=[{"id": DEMO_TOOL_CALL_ID, "name": "query_faq", "args": {"keyword": "退货"}}],
                        tool_call_id=DEMO_TOOL_CALL_ID,
                        tool_name="query_faq",
                    ),
                    Message(
                        conversation_id=DEMO_CONVERSATION_ID,
                        role="tool",
                        content="演示 FAQ：商品签收后七天内可申请退货。",
                        tool_call_id=DEMO_TOOL_CALL_ID,
                        tool_name="query_faq",
                    ),
                    Message(conversation_id=DEMO_CONVERSATION_ID, role="assistant", content="可以申请退货。"),
                ]
            )

        ticket = session.scalar(
            select(Ticket).where(
                Ticket.conversation_id == DEMO_CONVERSATION_ID,
                Ticket.source_tool_call_id == DEMO_TOOL_CALL_ID,
            )
        )
        if ticket is None:
            session.add(
                Ticket(
                    ticket_no="DEMO-0001",
                    conversation_id=DEMO_CONVERSATION_ID,
                    description="演示退货咨询工单",
                    ticket_type="return_request",
                    status="open",
                    source_tool_call_id=DEMO_TOOL_CALL_ID,
                )
            )
        session.commit()
