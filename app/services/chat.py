"""One-turn customer-support orchestration: prepare, decide, execute, converge."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from app.chat import PromptBudgetExceeded, trim_history
from app.prompts import CHAT_SYSTEM_PROMPT, chat_prompt
from app.repositories.conversations import (
    append_message,
    create_conversation,
    get_conversation,
    load_history,
)
from app.services.tool_executor import ToolCallResult, execute_tool_call
from app.tools.registry import build_tool_registry

logger = logging.getLogger(__name__)

MULTIPLE_TOOL_CALLS_CODE = "unsupported_multiple_tool_calls"
MULTIPLE_TOOL_CALLS_MESSAGE = "本轮同时申请了多个工具，已安全拒绝，请一次只问一个问题。"
UPSTREAM_ERROR_CODE = "upstream_stream_error"
UPSTREAM_ERROR_MESSAGE = "客服服务暂时不可用，请稍后重试。"
PERSISTENCE_ERROR_CODE = "persistence_error"
PERSISTENCE_ERROR_MESSAGE = "会话记录保存失败，请稍后重试。"
TOOL_EXECUTION_ERROR_MESSAGE = "工具执行失败，请稍后重试。"


class UnknownConversation(LookupError):
    """Raised when the caller references a conversation that does not exist."""


@dataclass(frozen=True, slots=True)
class ChatEvent:
    """A domain event the API layer renders as one SSE frame."""

    event: str
    data: dict[str, Any] | None = None
    raw_data: str | None = None


@dataclass(frozen=True, slots=True)
class PreparedTurn:
    """The validated, trimmed turn ready for the model decision call."""

    conversation_id: str
    prompt_messages: list[BaseMessage]
    created: bool


def prepare_turn(
    session_factory: Callable[[], Session],
    *,
    conversation_id: str | None,
    message: str,
    token_budget: int,
    safety_margin: int,
) -> PreparedTurn:
    """Resolve the conversation, trim whole-turn history and persist the user message."""
    with session_factory() as session:
        if conversation_id is None:
            resolved_id = create_conversation(session).id
            created = True
        else:
            if get_conversation(session, conversation_id) is None:
                raise UnknownConversation(conversation_id)
            resolved_id = conversation_id
            created = False

        history = _complete_turns_only(load_history(session, resolved_id))
        trimmed_history = trim_history(
            system_text=CHAT_SYSTEM_PROMPT,
            history=history,
            current_text=message,
            token_budget=token_budget,
            safety_margin=safety_margin,
        )
        append_message(session, resolved_id, role="user", content=message)
        session.commit()

    prompt_messages = chat_prompt.format_messages(history=trimmed_history, message=message)
    return PreparedTurn(conversation_id=resolved_id, prompt_messages=prompt_messages, created=created)


def _complete_turns_only(history: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Keep only turns that start with a user message and reach a final answer.

    A failed decision, a rejected multi-tool decision or a failed final stream
    leaves the persisted tail of a turn unfinished (a lone user row, or a user
    row with the assistant tool request and tool result but no final answer).
    Task 3's ``trim_history`` is a strict validator and rejects such a tail, so
    the orchestrator drops it before trimming instead of failing the next turn
    with an unmapped error. The persisted rows stay in the database untouched.
    """
    turns: list[list[BaseMessage]] = []
    for message in history:
        if message.type == "human":
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    return [message for turn in turns if _is_complete_turn(turn) for message in turn]


def _is_complete_turn(turn: Sequence[BaseMessage]) -> bool:
    """Return whether a grouped turn ends with an answer and pairs every tool call."""
    if not turn or turn[0].type != "human" or turn[-1].type != "ai":
        return False
    requested: list[Any] = []
    for message in turn:
        if message.type == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                requested.append(call.get("id"))
        elif message.type == "tool" and getattr(message, "tool_call_id", None) in requested:
            requested.remove(message.tool_call_id)
    return not requested


def _with_usable_call_id(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Give the call a non-empty id so frames, persistence and ToolMessage agree."""
    call_id = tool_call.get("id")
    if isinstance(call_id, str) and call_id.strip():
        return tool_call
    return {**tool_call, "id": f"call_{uuid4().hex}"}


async def stream_turn(
    *,
    model: BaseChatModel,
    session_factory: Callable[[], Session],
    conversation_id: str,
    prompt_messages: list[BaseMessage],
    tool_timeout_seconds: float = 10,
) -> AsyncIterator[ChatEvent]:
    """Run exactly one bound decision, at most one tool, then one unbound stream."""
    registry = build_tool_registry(session_factory)

    try:
        bound_model = model.bind_tools(list(registry.values()))
        decision = await bound_model.ainvoke(prompt_messages)
    except Exception as exc:
        yield _upstream_error(exc, conversation_id)
        return

    tool_calls = list(getattr(decision, "tool_calls", None) or [])
    if len(tool_calls) > 1:
        yield ChatEvent(
            event="error",
            data={"code": MULTIPLE_TOOL_CALLS_CODE, "message": MULTIPLE_TOOL_CALLS_MESSAGE},
        )
        return

    final_messages = list(prompt_messages)

    if len(tool_calls) == 1:
        tool_call = _with_usable_call_id(tool_calls[0])
        yield ChatEvent(
            event="tool",
            data={
                "tool_call_id": tool_call.get("id"),
                "name": tool_call.get("name"),
                "status": "running",
            },
        )
        result = await _run_tool(
            tool_call,
            registry=registry,
            conversation_id=conversation_id,
            session_factory=session_factory,
            timeout_seconds=tool_timeout_seconds,
        )
        try:
            await asyncio.to_thread(
                _persist_tool_trace,
                session_factory,
                conversation_id=conversation_id,
                assistant_content=_text_delta(decision),
                tool_call=tool_call,
                result=result,
            )
        except Exception as exc:
            yield _persistence_error(exc, conversation_id)
            return
        yield ChatEvent(
            event="tool",
            data={
                "tool_call_id": result.tool_call_id,
                "name": result.tool_name,
                "status": result.status,
            },
        )
        final_messages.append(
            AIMessage(content=_text_delta(decision), tool_calls=[_tool_call_payload(tool_call)])
        )
        final_messages.append(
            ToolMessage(
                content=json.dumps(result.content, ensure_ascii=False),
                tool_call_id=result.tool_call_id,
                name=result.tool_name,
            )
        )

    answer_parts: list[str] = []
    try:
        async for chunk in model.astream(final_messages):
            text = _text_delta(chunk)
            if not text:
                continue
            answer_parts.append(text)
            yield ChatEvent(event="token", data={"delta": text})
    except Exception as exc:
        yield _upstream_error(exc, conversation_id)
        return

    try:
        await asyncio.to_thread(
            _persist_assistant,
            session_factory,
            conversation_id=conversation_id,
            content="".join(answer_parts),
        )
    except Exception as exc:
        yield _persistence_error(exc, conversation_id)
        return
    yield ChatEvent(event="done", raw_data="[DONE]")


async def _run_tool(
    tool_call: dict[str, Any],
    *,
    registry: dict[str, BaseTool],
    conversation_id: str,
    session_factory: Callable[[], Session],
    timeout_seconds: float,
) -> ToolCallResult:
    """Execute one tool on a worker thread; the executor never leaks raw errors."""
    try:
        return await asyncio.to_thread(
            execute_tool_call,
            tool_call,
            registry=registry,
            conversation_id=conversation_id,
            session_factory=session_factory,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        logger.warning(
            "Tool execution raised for conversation_id=%s error_type=%s",
            conversation_id,
            type(exc).__name__,
        )
        return ToolCallResult(
            str(tool_call.get("name") or ""),
            str(tool_call.get("id") or ""),
            "failed",
            {"error": "tool_error", "message": TOOL_EXECUTION_ERROR_MESSAGE},
        )


def _tool_call_payload(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Normalize a model tool call into a JSON-serializable message payload."""
    return {
        "name": tool_call.get("name"),
        "args": dict(tool_call.get("args") or {}),
        "id": tool_call.get("id"),
        "type": tool_call.get("type") or "tool_call",
    }


def _persist_tool_trace(
    session_factory: Callable[[], Session],
    *,
    conversation_id: str,
    assistant_content: str,
    tool_call: dict[str, Any],
    result: ToolCallResult,
) -> None:
    """Persist the assistant tool request and its tool result as one atomic write."""
    with session_factory() as session:
        append_message(
            session,
            conversation_id,
            role="assistant",
            content=assistant_content,
            tool_calls=[_tool_call_payload(tool_call)],
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
        )
        append_message(
            session,
            conversation_id,
            role="tool",
            content=json.dumps(result.content, ensure_ascii=False),
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
        )
        session.commit()


def _persist_assistant(
    session_factory: Callable[[], Session],
    *,
    conversation_id: str,
    content: str,
) -> None:
    """Persist the final assistant answer for the turn."""
    with session_factory() as session:
        append_message(session, conversation_id, role="assistant", content=content)
        session.commit()


def _upstream_error(exc: Exception, conversation_id: str) -> ChatEvent:
    """Return the stable, sanitized error frame for any upstream model failure."""
    logger.warning(
        "Chat upstream failed for conversation_id=%s error_type=%s status=%s",
        conversation_id,
        type(exc).__name__,
        getattr(exc, "status_code", None),
    )
    return ChatEvent(
        event="error",
        data={"code": UPSTREAM_ERROR_CODE, "message": UPSTREAM_ERROR_MESSAGE},
    )


def _persistence_error(exc: Exception, conversation_id: str) -> ChatEvent:
    """Return the stable error frame for a failed database write."""
    logger.error(
        "Chat persistence failed for conversation_id=%s error_type=%s",
        conversation_id,
        type(exc).__name__,
    )
    return ChatEvent(
        event="error",
        data={"code": PERSISTENCE_ERROR_CODE, "message": PERSISTENCE_ERROR_MESSAGE},
    )


def _text_delta(chunk: object) -> str:
    """Return textual content from a LangChain message or streaming chunk."""
    if not isinstance(chunk, AIMessageChunk):
        content = getattr(chunk, "content", "")
    else:
        content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


__all__ = [
    "ChatEvent",
    "PreparedTurn",
    "PromptBudgetExceeded",
    "UnknownConversation",
    "prepare_turn",
    "stream_turn",
]
