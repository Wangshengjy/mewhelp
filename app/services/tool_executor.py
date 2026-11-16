"""Bounded, single-tool execution with validation, timeout and one retry."""

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

MAX_ATTEMPTS = 2

FAILURE_MESSAGES = {
    "invalid_tool_call": "工具调用格式无效，未执行。",
    "unknown_tool": "请求的工具不在允许列表中，未执行。",
    "invalid_arguments": "工具参数校验失败，未执行。",
    "tool_timeout": "工具执行超时，请稍后重试。",
    "tool_error": "工具执行失败，请稍后重试。",
}


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Normalized outcome of executing, or rejecting, one tool call."""

    tool_name: str
    tool_call_id: str
    status: str
    content: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_call_id": self.tool_call_id,
            "status": self.status,
            "content": self.content,
        }


def execute_tool_call(
    tool_call: object,
    *,
    registry: dict[str, Any],
    conversation_id: str,
    session_factory: Callable[[], Session] | None,
    timeout_seconds: float = 10,
) -> ToolCallResult:
    """Validate one tool call, execute at most one allowlisted tool and normalize the result.

    The registry already binds the database session factory into each tool, so
    ``session_factory`` is accepted to keep the approved call signature and is
    not used directly here.
    """
    name, call_id, args = _extract_call(tool_call)
    if name is None or call_id is None or args is None:
        return _failure(name or "", call_id or "", "invalid_tool_call")

    business_tool = registry.get(name)
    if business_tool is None:
        return _failure(name, call_id, "unknown_tool")

    if not _arguments_are_valid(business_tool, args, conversation_id, call_id):
        return _failure(name, call_id, "invalid_arguments")

    call = {
        "name": name,
        "args": {**args, "conversation_id": conversation_id},
        "id": call_id,
        "type": "tool_call",
    }

    error_code = "tool_error"
    for _ in range(MAX_ATTEMPTS):
        try:
            message = _invoke_with_timeout(business_tool, call, timeout_seconds)
        except TimeoutError:
            error_code = "tool_timeout"
        except Exception:
            error_code = "tool_error"
        else:
            return _success(name, call_id, message)
    return _failure(name, call_id, error_code)


def _extract_call(tool_call: object) -> tuple[str | None, str | None, dict | None]:
    """Keep every valid field of the call so a rejection still echoes its identity."""
    if not isinstance(tool_call, dict):
        return None, None, None
    name = tool_call.get("name")
    call_id = tool_call.get("id")
    args = tool_call.get("args")
    valid_name = name if isinstance(name, str) and name.strip() else None
    valid_call_id = call_id if isinstance(call_id, str) and call_id.strip() else None
    valid_args = args if isinstance(args, dict) else None
    return valid_name, valid_call_id, valid_args


def _arguments_are_valid(
    business_tool: Any, args: dict, conversation_id: str, call_id: str
) -> bool:
    """Validate the model arguments before anything is executed.

    Only this pre-execution pydantic validation is non-retryable. Anything that
    raises from inside the tool body afterwards is a transient execution error.
    """
    args_schema = getattr(business_tool, "args_schema", None)
    if not isinstance(args_schema, type) or not issubclass(args_schema, BaseModel):
        return True
    fields = set(args_schema.model_fields)
    candidate = dict(args)
    if "conversation_id" in fields:
        candidate["conversation_id"] = conversation_id
    if "tool_call_id" in fields:
        candidate["tool_call_id"] = call_id
    try:
        args_schema.model_validate(candidate)
    except ValidationError:
        return False
    return True


def _invoke_with_timeout(business_tool: Any, call: dict, timeout_seconds: float) -> Any:
    """Run the tool on an abandoned-safe daemon thread and enforce the timeout."""
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["result"] = business_tool.invoke(call)
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError("tool execution exceeded the configured timeout")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")


def _success(name: str, call_id: str, message: Any) -> ToolCallResult:
    raw = getattr(message, "content", message)
    return ToolCallResult(name, call_id, "completed", _normalize_content(raw))


def _normalize_content(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"result": raw}
        return parsed if isinstance(parsed, dict) else {"result": parsed}
    return {"result": str(raw)}


def _failure(name: str, call_id: str, error_code: str) -> ToolCallResult:
    return ToolCallResult(
        name,
        call_id,
        "failed",
        {"error": error_code, "message": FAILURE_MESSAGES[error_code]},
    )
