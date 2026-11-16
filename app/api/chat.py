"""The POST /api/chat/stream Server-Sent Events route."""

from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.language_models import BaseChatModel
from sqlalchemy.orm import Session

from app.chat import PromptBudgetExceeded
from app.dependencies import get_chat_model, get_session_factory, get_settings
from app.schemas import ChatRequest
from app.services.chat import ChatEvent, PreparedTurn, UnknownConversation, prepare_turn, stream_turn
from app.settings import Settings

router = APIRouter()


def prepare_chat(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
) -> PreparedTurn:
    """Validate the session and prepare the turn before any SSE bytes are sent."""
    try:
        return prepare_turn(
            session_factory,
            conversation_id=request.conversation_id,
            message=request.message,
            token_budget=settings.chat_context_token_budget,
            safety_margin=settings.chat_token_safety_margin,
        )
    except UnknownConversation as exc:
        raise HTTPException(status_code=404, detail="Unknown conversation_id") from exc
    except PromptBudgetExceeded as exc:
        raise HTTPException(
            status_code=413,
            detail="The message exceeds the configured context budget",
        ) from exc


@router.post("/api/chat/stream", response_class=EventSourceResponse)
async def stream_chat(
    prepared: Annotated[PreparedTurn, Depends(prepare_chat)],
    model: Annotated[BaseChatModel, Depends(get_chat_model)],
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[ServerSentEvent]:
    """Stream one tool-aware conversation turn."""
    if prepared.created:
        yield ServerSentEvent(
            event="conversation",
            data={"conversation_id": prepared.conversation_id},
        )
    async for event in stream_turn(
        model=model,
        session_factory=session_factory,
        conversation_id=prepared.conversation_id,
        prompt_messages=prepared.prompt_messages,
        tool_timeout_seconds=settings.tool_timeout_seconds,
    ):
        yield _to_sse(event)


def _to_sse(event: ChatEvent) -> ServerSentEvent:
    """Render one domain event as an SSE frame."""
    if event.raw_data is not None:
        return ServerSentEvent(event=event.event, raw_data=event.raw_data)
    return ServerSentEvent(event=event.event, data=event.data)