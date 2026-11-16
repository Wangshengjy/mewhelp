"""API and service tests for the single-tool decision SSE turn."""

import json
from collections import deque

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from sqlalchemy import select

from app.database import build_engine, build_session_factory, initialize_database
from app.dependencies import get_chat_model, get_session_factory, get_settings
from app.models import Message
from app.settings import Settings

TEST_SETTINGS = Settings(
    chat_base_url="https://example.test/v1",
    chat_model="test-model",
    chat_api_key="test-key",
    chat_context_token_budget=6000,
    chat_token_safety_margin=500,
)


class ScriptedBoundModel:
    """The tool-bound view returned by ``bind_tools``; only decisions are allowed."""

    def __init__(self, parent):
        self.parent = parent
        self.invoke_count = 0

    async def ainvoke(self, messages, **kwargs):
        self.parent.decision_calls.append(list(messages))
        self.invoke_count += 1
        item = self.parent.decisions.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedChatModel:
    """Deterministic model double with separate decision and final-stream scripts."""

    def __init__(self, decisions=(), streams=()):
        self.decisions = deque(decisions)
        self.streams = deque(streams)
        self.bind_tools_calls = []
        self.bound = ScriptedBoundModel(self)
        self.decision_calls = []
        self.final_calls = []

    def bind_tools(self, tools, **kwargs):
        self.bind_tools_calls.append(list(tools))
        return self.bound

    async def ainvoke(self, messages, **kwargs):
        raise AssertionError("the service must not call ainvoke on the unbound model")

    async def astream(self, messages):
        self.final_calls.append(list(messages))
        for part in self.streams.popleft():
            if isinstance(part, Exception):
                raise part
            yield AIMessageChunk(content=part)


def tool_decision(*calls):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
            for name, args, call_id in calls
        ],
    )


def no_tool_decision(text=""):
    return AIMessage(content=text, tool_calls=[])


@pytest.fixture
def tool_client(test_app):
    model = ScriptedChatModel()
    test_app.dependency_overrides[get_chat_model] = lambda: model
    test_app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    try:
        with TestClient(test_app) as client:
            yield client, model, test_app
    finally:
        test_app.dependency_overrides.clear()


def _spy_on_executor(monkeypatch):
    """Record every executor call while still running the real implementation."""
    from app.services import chat as chat_service

    executions = []
    real = chat_service.execute_tool_call

    def spy(tool_call, **kwargs):
        executions.append(tool_call)
        return real(tool_call, **kwargs)

    monkeypatch.setattr(chat_service, "execute_tool_call", spy)
    return executions


def _parse_sse_events(body):
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        lines = block.splitlines()
        event_line = next((line for line in lines if line.startswith("event: ")), None)
        data_lines = [line.removeprefix("data: ") for line in lines if line.startswith("data: ")]
        if event_line is None:
            continue
        raw_data = "\n".join(data_lines)
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            data = raw_data
        events.append({"event": event_line.removeprefix("event: "), "data": data})
    return events


def _conversation_id(body):
    for event in _parse_sse_events(body):
        if event["event"] == "conversation":
            return event["data"]["conversation_id"]
    raise AssertionError("conversation event missing")


def _rows(session_factory, conversation_id):
    with session_factory() as session:
        return list(
            session.scalars(
                select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
            ).all()
        )


def test_single_tool_turn_emits_tool_frames_and_streams_final_text(tool_client, monkeypatch):
    client, model, _ = tool_client
    executions = _spy_on_executor(monkeypatch)
    model.decisions.append(tool_decision(("query_logistics", {"order_id": "1001"}, "call-1")))
    model.streams.append(["订单 1001 已揽收", "，正在派送中。"])

    response = client.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == [
        "conversation",
        "tool",
        "tool",
        "token",
        "token",
        "done",
    ]
    assert events[0]["data"]["conversation_id"]
    assert events[1]["data"] == {
        "tool_call_id": "call-1",
        "name": "query_logistics",
        "status": "running",
    }
    assert events[2]["data"] == {
        "tool_call_id": "call-1",
        "name": "query_logistics",
        "status": "completed",
    }
    assert [event["data"] for event in events[3:5]] == [
        {"delta": "订单 1001 已揽收"},
        {"delta": "，正在派送中。"},
    ]
    assert events[-1]["data"] == "[DONE]"

    assert len(executions) == 1
    assert len(model.bind_tools_calls) == 1
    assert {tool.name for tool in model.bind_tools_calls[0]} == {
        "query_order",
        "query_product",
        "query_logistics",
        "query_faq",
        "create_ticket",
    }
    assert model.bound.invoke_count == 1
    assert len(model.final_calls) == 1
    tool_messages = [message for message in model.final_calls[0] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call-1"
    assert json.loads(tool_messages[0].content)["order_id"] == "1001"


def test_no_tool_decision_streams_output_from_the_unbound_model(tool_client):
    client, model, _ = tool_client
    model.decisions.append(no_tool_decision("我会直接回答"))
    model.streams.append(["你好，", "请问有什么可以帮您？"])

    response = client.post("/api/chat/stream", json={"message": "你好"})

    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["conversation", "token", "token", "done"]
    assert [event["data"] for event in events[1:3]] == [
        {"delta": "你好，"},
        {"delta": "请问有什么可以帮您？"},
    ]
    assert [message.type for message in model.final_calls[0]] == ["system", "human"]
    assert model.bound.invoke_count == 1
    assert len(model.bind_tools_calls) == 1
    rows = _rows(tool_client[2].state.session_factory, _conversation_id(response.text))
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[1].content == "你好，请问有什么可以帮您？"


def test_two_tool_calls_in_one_decision_execute_nothing(tool_client, monkeypatch):
    client, model, test_app = tool_client
    executions = _spy_on_executor(monkeypatch)
    model.decisions.append(
        tool_decision(
            ("query_order", {"order_id": "1001"}, "call-a"),
            ("query_logistics", {"order_id": "1001"}, "call-b"),
        )
    )

    response = client.post("/api/chat/stream", json={"message": "帮我查订单和物流"})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert executions == []
    assert model.final_calls == []
    assert "tool" not in [event["event"] for event in events]
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "unsupported_multiple_tool_calls"
    assert events[-1]["data"]["message"]
    rows = _rows(test_app.state.session_factory, _conversation_id(response.text))
    assert [row.role for row in rows] == ["user"]


def test_tool_trace_is_persisted_and_reloaded_for_the_next_turn(tool_client):
    client, model, test_app = tool_client
    model.decisions.append(tool_decision(("query_logistics", {"order_id": "1001"}, "call-1")))
    model.streams.append(["订单 1001 已揽收。"])

    first = client.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})
    conversation_id = _conversation_id(first.text)
    rows = _rows(test_app.state.session_factory, conversation_id)

    assert [row.role for row in rows] == ["user", "assistant", "tool", "assistant"]
    assert rows[0].content == "订单 1001 的物流到哪了"
    assert rows[1].tool_calls[0]["name"] == "query_logistics"
    assert rows[1].tool_calls[0]["args"] == {"order_id": "1001"}
    assert rows[1].tool_calls[0]["id"] == "call-1"
    assert rows[2].tool_call_id == "call-1"
    assert rows[2].tool_name == "query_logistics"
    assert json.loads(rows[2].content)["order_id"] == "1001"
    assert rows[3].content == "订单 1001 已揽收。"

    model.decisions.append(no_tool_decision())
    model.streams.append(["好的。"])
    second = client.post(
        "/api/chat/stream",
        json={"message": "还有别的要确认吗", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    second_prompt = model.decision_calls[1]
    assert [message.type for message in second_prompt] == ["system", "human", "ai", "tool", "ai", "human"]
    assert second_prompt[1].content == "订单 1001 的物流到哪了"
    assert second_prompt[2].tool_calls[0]["id"] == "call-1"
    assert second_prompt[3].tool_call_id == "call-1"
    assert second_prompt[4].content == "订单 1001 已揽收。"
    assert second_prompt[5].content == "还有别的要确认吗"


def test_decision_failure_emits_sanitized_error_and_keeps_the_user_row(tool_client):
    client, model, test_app = tool_client
    model.decisions.append(RuntimeError("upstream secret body"))

    response = client.post("/api/chat/stream", json={"message": "订单 1001"})

    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["conversation", "error"]
    assert events[1]["data"]["code"] == "upstream_stream_error"
    assert events[1]["data"]["message"]
    assert "upstream secret body" not in response.text
    rows = _rows(test_app.state.session_factory, _conversation_id(response.text))
    assert [row.role for row in rows] == ["user"]


def test_final_stream_failure_keeps_tool_trace_and_fabricates_no_answer(tool_client):
    client, model, test_app = tool_client
    model.decisions.append(tool_decision(("query_logistics", {"order_id": "1001"}, "call-1")))
    model.streams.append([RuntimeError("upstream secret body")])

    response = client.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})

    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["conversation", "tool", "tool", "error"]
    assert events[-1]["data"]["code"] == "upstream_stream_error"
    assert "upstream secret body" not in response.text
    rows = _rows(test_app.state.session_factory, _conversation_id(response.text))
    assert [row.role for row in rows] == ["user", "assistant", "tool"]


def test_session_factory_dependency_can_be_overridden(test_app):
    engine = build_engine("sqlite+pysqlite:///:memory:")
    factory = build_session_factory(engine)
    initialize_database(factory)
    model = ScriptedChatModel([no_tool_decision()], [["你好"]])
    test_app.dependency_overrides[get_chat_model] = lambda: model
    test_app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    test_app.dependency_overrides[get_session_factory] = lambda: factory
    try:
        with TestClient(test_app) as client:
            response = client.post("/api/chat/stream", json={"message": "你好"})
        conversation_id = _conversation_id(response.text)
        rows = _rows(factory, conversation_id)
    finally:
        test_app.dependency_overrides.clear()
        engine.dispose()

    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[0].content == "你好"


def test_next_turn_after_a_failed_decision_still_succeeds(tool_client):
    client, model, _ = tool_client
    model.decisions.append(RuntimeError("upstream secret body"))
    first = client.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})
    conversation_id = _conversation_id(first.text)
    assert _parse_sse_events(first.text)[-1]["event"] == "error"

    model.decisions.append(no_tool_decision())
    model.streams.append(["好的。"])
    second = client.post(
        "/api/chat/stream",
        json={"message": "那先不查了", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    assert _parse_sse_events(second.text)[-1]["event"] == "done"
    assert [message.type for message in model.decision_calls[1]] == ["system", "human"]


def test_unanswered_history_tail_is_dropped_before_trimming():
    from langchain_core.messages import HumanMessage

    from app.chat import trim_history
    from app.services import chat as chat_service

    history = [
        HumanMessage(content="已经答完的问题"),
        AIMessage(content="已经答完的答案"),
        HumanMessage(content="上游失败、没有答案的问题"),
    ]

    with pytest.raises(ValueError):
        trim_history("sys", history, "cur", token_budget=200, safety_margin=0)
    assert chat_service._complete_turns_only(history) == history[:2]


def test_next_turn_after_a_failed_final_stream_still_succeeds(tool_client):
    client, model, test_app = tool_client
    model.decisions.append(tool_decision(("query_logistics", {"order_id": "1001"}, "call-1")))
    model.streams.append([RuntimeError("upstream secret body")])
    first = client.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})
    conversation_id = _conversation_id(first.text)

    rows = _rows(test_app.state.session_factory, conversation_id)
    assert [row.role for row in rows] == ["user", "assistant", "tool"]

    model.decisions.append(no_tool_decision())
    model.streams.append(["好的。"])
    second = client.post(
        "/api/chat/stream",
        json={"message": "换一个问题", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    assert _parse_sse_events(second.text)[-1]["event"] == "done"
    assert [message.type for message in model.decision_calls[1]] == ["system", "human"]


def test_blank_tool_call_id_is_replaced_consistently(tool_client):
    client, model, test_app = tool_client
    model.decisions.append(tool_decision(("query_logistics", {"order_id": "1001"}, "")))
    model.streams.append(["订单 1001 已揽收。"])

    response = client.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})

    tool_events = [event["data"] for event in _parse_sse_events(response.text) if event["event"] == "tool"]
    assert len(tool_events) == 2
    call_id = tool_events[0]["tool_call_id"]
    assert call_id
    assert tool_events[1]["tool_call_id"] == call_id
    rows = _rows(test_app.state.session_factory, _conversation_id(response.text))
    assert rows[1].tool_calls[0]["id"] == call_id
    assert rows[2].tool_call_id == call_id
    tool_messages = [message for message in model.final_calls[0] if isinstance(message, ToolMessage)]
    assert tool_messages[0].tool_call_id == call_id


def test_persistence_failure_emits_an_error_frame_instead_of_truncating(tool_client, monkeypatch):
    from app.services import chat as chat_service

    client, model, _ = tool_client
    model.decisions.append(tool_decision(("query_logistics", {"order_id": "1001"}, "call-1")))
    model.streams.append(["订单 1001 已揽收。"])

    def boom(*args, **kwargs):
        raise RuntimeError("db down secret detail")

    monkeypatch.setattr(chat_service, "_persist_tool_trace", boom)

    response = client.post("/api/chat/stream", json={"message": "订单 1001 的物流到哪了"})

    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["conversation", "tool", "error"]
    assert events[-1]["data"]["code"] == "persistence_error"
    assert events[-1]["data"]["message"]
    assert "db down secret detail" not in response.text


def test_failed_tool_turn_emits_failed_frame_and_still_streams_final_text(tool_client):
    client, model, _ = tool_client
    model.decisions.append(tool_decision(("no_such_tool", {}, "call-failed")))
    model.streams.append(["工具不可用", "，我先如实说明。"])

    response = client.post("/api/chat/stream", json={"message": "帮我查下单号 1001 的物流"})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    event_names = [event["event"] for event in events]
    assert event_names[0] == "conversation"
    assert event_names[1] == "tool"
    assert event_names[2] == "tool"
    assert event_names[-1] == "done"
    assert len(event_names[3:-1]) >= 1
    assert all(name == "token" for name in event_names[3:-1])

    running = events[1]["data"]
    failed = events[2]["data"]
    assert running == {
        "tool_call_id": "call-failed",
        "name": "no_such_tool",
        "status": "running",
    }
    assert failed["status"] == "failed"
    assert failed["tool_call_id"] == running["tool_call_id"] == "call-failed"
    assert failed["name"] == running["name"] == "no_such_tool"
    assert events[-1]["data"] == "[DONE]"

    assert len(model.final_calls) == 1
    assert len(model.bind_tools_calls) == 1

    failed_frame_json = json.dumps(failed, ensure_ascii=False)
    assert "Traceback" not in failed_frame_json
    assert "KeyError" not in failed_frame_json


def test_continuation_turn_emits_no_conversation_frame(tool_client):
    client, model, _ = tool_client

    model.decisions.append(no_tool_decision("你好"))
    model.streams.append(["你好，请问有什么可以帮您？"])
    first = client.post("/api/chat/stream", json={"message": "你好"})
    assert first.status_code == 200
    conversation_id = _conversation_id(first.text)

    model.decisions.append(no_tool_decision("继续"))
    model.streams.append(["好的，请继续说。"])
    second = client.post(
        "/api/chat/stream",
        json={"message": "继续", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    events = _parse_sse_events(second.text)
    event_names = [event["event"] for event in events]
    assert "conversation" not in event_names
    assert event_names[0] == "token"
    assert event_names[-1] == "done"
    assert events[-1]["data"] == "[DONE]"


def test_final_assistant_persistence_failure_emits_error_without_done(tool_client, monkeypatch):
    from app.services import chat as chat_service

    client, model, _ = tool_client
    model.decisions.append(no_tool_decision("你好"))
    model.streams.append(["你好，", "请问有什么可以帮您？"])

    def boom(*args, **kwargs):
        raise RuntimeError("boom-secret")

    monkeypatch.setattr(chat_service, "_persist_assistant", boom)

    failed = client.post("/api/chat/stream", json={"message": "你好"})

    events = _parse_sse_events(failed.text)
    event_names = [event["event"] for event in events]
    assert event_names[0] == "conversation"
    assert event_names[-1] == "error"
    assert "done" not in event_names
    assert len(event_names[1:-1]) >= 1
    assert all(name == "token" for name in event_names[1:-1])
    assert events[-1]["data"]["code"] == "persistence_error"
    assert events[-1]["data"]["message"]
    assert "boom-secret" not in failed.text

    conversation_id = _conversation_id(failed.text)

    monkeypatch.undo()
    model.decisions.append(no_tool_decision("继续"))
    model.streams.append(["好的。"])
    second = client.post(
        "/api/chat/stream",
        json={"message": "继续", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    assert _parse_sse_events(second.text)[-1]["event"] == "done"
