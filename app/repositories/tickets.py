"""Ticket persistence with per-tool-call idempotency."""

from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Ticket


def find_ticket(
    session: Session, conversation_id: str, source_tool_call_id: str | None
) -> Ticket | None:
    """Return the ticket already created for one conversation tool call."""
    if source_tool_call_id is None:
        return None
    return session.scalar(
        select(Ticket).where(
            Ticket.conversation_id == conversation_id,
            Ticket.source_tool_call_id == source_tool_call_id,
        )
    )


def _generate_ticket_no() -> str:
    return f"T{uuid4().hex[:12].upper()}"


def create_ticket(
    session: Session,
    *,
    conversation_id: str,
    description: str,
    ticket_type: str,
    source_tool_call_id: str | None = None,
) -> Ticket:
    """Create the ticket for a tool call, or return the one already created."""
    existing = find_ticket(session, conversation_id, source_tool_call_id)
    if existing is not None:
        return existing

    ticket = Ticket(
        ticket_no=_generate_ticket_no(),
        conversation_id=conversation_id,
        description=description,
        ticket_type=ticket_type,
        status="open",
        source_tool_call_id=source_tool_call_id,
    )
    try:
        with session.begin_nested():
            session.add(ticket)
            session.flush()
    except IntegrityError:
        session.expire_all()
        existing = find_ticket(session, conversation_id, source_tool_call_id)
        if existing is None:
            raise
        return existing
    return ticket
