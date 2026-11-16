"""Tests for the bounded, single-tool execution service."""

import json
import time

import pytest
from sqlalchemy import select

from app.database import build_engine, build_session_factory
from app.demo_seed import seed_demo_data
from app.models import Base, Ticket
from app.repositories import tickets as tickets_repository
from app.repositories.conversations import create_conversation
from app.repositories.tickets import create_ticket
from app.services.tool_executor import ToolCallResult, execute_tool_call
from app.tools.registry import build_tool_registry
from app.tools.schemas import QueryOrderArgs

SQLITE_URL = "sqlite+pysqlite:///:memory:"


class StubTool:
    """A deterministic tool double whose behavior is fully controlled by a callback."""

    def __init__(self, name, behavior):
        self.name = name
        self._behavior = behavior
        self.calls = 0

    def invoke(self, tool_call):
        self.calls += 1
        return self._behavior(tool_call)


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    seed_demo_data(factory)
    yield factory
    engine.dispose()


@pytest.fixture
def registry(session_factory):
    return build_tool_registry(session_factory)


@pytest.fixture
def conversation_id(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        session.commit()
        return conversation.id


def tool_call(name, args, call_id="call-1"):
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_executes_registered_tool_and_returns_normalized_result(registry, session_factory, conversation_id):
    result = execute_tool_call(
        tool_call("query_order", {"order_id": "1001"}),
        registry=registry,
        conversation_id=conversation_id,
        session_factory=session_factory,
    )

    assert isinstance(result, ToolCallResult)
    assert result.tool_name == "query_order"
    assert result.tool_call_id == "call-1"
    assert result.status == "completed"
    assert result.content["order_id"] == "1001"
    assert json.loads(json.dumps(result.content, ensure_ascii=False)) == result.content
    assert result.as_dict() == {
        "tool_name": "query_order",
        "tool_call_id": "call-1",
        "status": "completed",
        "content": result.content,
    }


def test_ticket_retry_with_same_call_returns_same_ticket(registry, session_factory, conversation_id):
    call = tool_call("create_ticket", {"description": "需要退货", "ticket_type": "refund"}, call_id="call-7")

    first = execute_tool_call(
        call, registry=registry, conversation_id=conversation_id, session_factory=session_factory
    )
    retry = execute_tool_call(
        call, registry=registry, conversation_id=conversation_id, session_factory=session_factory
    )

    assert first.status == "completed"
    assert retry.status == "completed"
    assert first.content["ticket_no"] == retry.content["ticket_no"]

    with session_factory() as session:
        persisted = session.scalars(select(Ticket).where(Ticket.conversation_id == conversation_id)).all()
    assert len(persisted) == 1


def test_unknown_tool_is_rejected_and_not_executed():
    tool = StubTool("known", lambda call: json.dumps({"ok": True}))
    result = execute_tool_call(
        tool_call("unknown_tool", {}),
        registry={"known": tool},
        conversation_id="conv",
        session_factory=None,
    )

    assert result.status == "failed"
    assert result.tool_name == "unknown_tool"
    assert result.content["error"] == "unknown_tool"
    assert tool.calls == 0


def test_malformed_tool_call_is_rejected(registry, session_factory, conversation_id):
    not_a_mapping = execute_tool_call(
        "not-a-dict",
        registry=registry,
        conversation_id=conversation_id,
        session_factory=session_factory,
    )
    bad_args = execute_tool_call(
        tool_call("query_order", "not-a-dict"),
        registry=registry,
        conversation_id=conversation_id,
        session_factory=session_factory,
    )

    for result in (not_a_mapping, bad_args):
        assert result.status == "failed"
        assert result.content["error"] == "invalid_tool_call"

    assert not_a_mapping.tool_name == ""
    assert not_a_mapping.tool_call_id == ""
    assert bad_args.tool_name == "query_order"
    assert bad_args.tool_call_id == "call-1"


def test_schema_invalid_arguments_fail_closed_without_executing(registry, session_factory, conversation_id):
    results = [
        execute_tool_call(
            tool_call(
                "create_ticket",
                {"description": "需要退货", "ticket_type": "refund", "extra": "boom"},
                call_id="call-extra",
            ),
            registry=registry,
            conversation_id=conversation_id,
            session_factory=session_factory,
        ),
        execute_tool_call(
            tool_call("query_order", {"order_id": "   "}, call_id="call-blank"),
            registry=registry,
            conversation_id=conversation_id,
            session_factory=session_factory,
        ),
        execute_tool_call(
            tool_call("query_order", {}, call_id="call-missing"),
            registry=registry,
            conversation_id=conversation_id,
            session_factory=session_factory,
        ),
        execute_tool_call(
            tool_call(
                "create_ticket",
                {"description": "需要退货", "ticket_type": "not_a_type"},
                call_id="call-bad-type",
            ),
            registry=registry,
            conversation_id=conversation_id,
            session_factory=session_factory,
        ),
    ]

    for result in results:
        assert result.status == "failed"
        assert result.content["error"] == "invalid_arguments"

    with session_factory() as session:
        persisted = session.scalars(select(Ticket).where(Ticket.conversation_id == conversation_id)).all()
    assert persisted == []


def test_transient_error_retries_once_then_succeeds():
    attempts = {"count": 0}

    def behavior(call):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary upstream failure")
        return json.dumps({"ok": True})

    tool = StubTool("flaky", behavior)
    result = execute_tool_call(
        tool_call("flaky", {}),
        registry={"flaky": tool},
        conversation_id="conv",
        session_factory=None,
        timeout_seconds=1,
    )

    assert result.status == "completed"
    assert result.content == {"ok": True}
    assert tool.calls == 2


def test_transient_failure_retries_once_then_fails_sanitized():
    def behavior(call):
        raise RuntimeError("SECRET token=abc123 SELECT * FROM tickets password=hunter2")

    tool = StubTool("broken", behavior)
    result = execute_tool_call(
        tool_call("broken", {"order_id": "1001"}),
        registry={"broken": tool},
        conversation_id="conv",
        session_factory=None,
        timeout_seconds=1,
    )

    assert result.status == "failed"
    assert result.content["error"] == "tool_error"
    assert tool.calls == 2

    serialized = json.dumps(result.content, ensure_ascii=False)
    for secret in ("SECRET", "abc123", "SELECT", "tickets", "password", "hunter2", "RuntimeError", "1001"):
        assert secret not in serialized


def test_timeout_retries_once_then_fails():
    def behavior(call):
        time.sleep(0.2)
        return json.dumps({"ok": True})

    tool = StubTool("slow", behavior)
    result = execute_tool_call(
        tool_call("slow", {}),
        registry={"slow": tool},
        conversation_id="conv",
        session_factory=None,
        timeout_seconds=0.02,
    )

    assert result.status == "failed"
    assert result.content["error"] == "tool_timeout"
    assert tool.calls == 2


def test_schema_validation_failure_does_not_execute_or_retry():
    tool = StubTool("guard", lambda call: pytest.fail("tool must not run with invalid arguments"))
    tool.args_schema = QueryOrderArgs

    result = execute_tool_call(
        tool_call("guard", {"order_id": "   "}),
        registry={"guard": tool},
        conversation_id="conv",
        session_factory=None,
        timeout_seconds=1,
    )

    assert result.status == "failed"
    assert result.content["error"] == "invalid_arguments"
    assert tool.calls == 0


def test_internal_error_is_treated_as_transient_and_retried():
    attempts = {"count": 0}

    def behavior(call):
        attempts["count"] += 1
        raise ValueError("genuine bug inside the tool")

    tool = StubTool("buggy", behavior)
    result = execute_tool_call(
        tool_call("buggy", {}),
        registry={"buggy": tool},
        conversation_id="conv",
        session_factory=None,
        timeout_seconds=1,
    )

    assert result.status == "failed"
    assert result.content["error"] == "tool_error"
    assert tool.calls == 2


def test_ticket_timeout_then_write_keeps_one_ticket(registry, session_factory, conversation_id):
    real_tool = registry["create_ticket"]
    state = {"first": True}

    def behavior(call):
        if state["first"]:
            state["first"] = False
            real_tool.invoke(call)
            raise TimeoutError("simulated abandoned worker timeout")
        return real_tool.invoke(call)

    tool = StubTool("create_ticket", behavior)
    call = tool_call(
        "create_ticket",
        {"description": "需要退货", "ticket_type": "refund"},
        call_id="call-timeout",
    )

    result = execute_tool_call(
        call,
        registry={"create_ticket": tool},
        conversation_id=conversation_id,
        session_factory=session_factory,
        timeout_seconds=1,
    )

    assert result.status == "completed"
    assert tool.calls == 2

    with session_factory() as session:
        persisted = session.scalars(
            select(Ticket).where(
                Ticket.conversation_id == conversation_id,
                Ticket.source_tool_call_id == "call-timeout",
            )
        ).all()

    assert len(persisted) == 1
    assert result.content["ticket_no"] == persisted[0].ticket_no


def test_ticket_creation_recovers_from_unique_constraint_race(session_factory, monkeypatch):
    with session_factory() as session:
        conversation = create_conversation(session)
        session.commit()
        conversation_id = conversation.id
        first = create_ticket(
            session,
            conversation_id=conversation_id,
            description="已存在的工单",
            ticket_type="refund",
            source_tool_call_id="call-race",
        )
        session.commit()
        first_no = first.ticket_no

    real_find = tickets_repository.find_ticket
    calls = {"count": 0}

    def flaky_find(session, conversation_id, source_tool_call_id):
        calls["count"] += 1
        if calls["count"] == 1:
            return None
        return real_find(session, conversation_id, source_tool_call_id)

    monkeypatch.setattr(tickets_repository, "find_ticket", flaky_find)

    with session_factory() as session:
        recovered = create_ticket(
            session,
            conversation_id=conversation_id,
            description="重复工单",
            ticket_type="refund",
            source_tool_call_id="call-race",
        )
        assert recovered.ticket_no == first_no
        assert calls["count"] == 2

        follow_up = create_ticket(
            session,
            conversation_id=conversation_id,
            description="另一个工单",
            ticket_type="other",
            source_tool_call_id="call-race-2",
        )
        session.commit()
        follow_up_no = follow_up.ticket_no

    assert follow_up_no != first_no

    with session_factory() as session:
        persisted = session.scalars(select(Ticket).where(Ticket.conversation_id == conversation_id)).all()

    assert len(persisted) == 2
    assert {ticket.ticket_no for ticket in persisted} == {first_no, follow_up_no}
