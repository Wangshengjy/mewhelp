"""ch05 · SSE 聊天路由的会话处理与流式契约测试(测图)。

事件映射与真实图行为在 tests/graph/ 有专门覆盖;这里用 Fake 事件流钉死
/api/chat/stream 的 SSE 帧契约(ch04 全部帧保留 + actions 新增)与错误路径。
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select

from app.graph import runtime
from app.models import Message
from app.repositories.conversations import create_conversation
from starlette.requests import Request
from app import dependencies
from app.dependencies import get_chat_model


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


def _fake_prepare(created=True, error=None):
    async def _prepare(conversation_id, message):
        if error is not None:
            raise error
        return (conversation_id or "conv-fake-1"), created

    return _prepare


def _fake_stream(events, error=None):
    async def _stream(conversation_id, message):
        for event in events:
            yield event
        if error is not None:
            raise error
        yield {"type": "done", "conversation_id": conversation_id}

    return _stream


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        yield test_client, test_app


def test_sse_events_include_conversation_tokens_and_done(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(runtime, "prepare_turn", _fake_prepare(created=True))
    monkeypatch.setattr(
        runtime,
        "stream_turn",
        _fake_stream([{"type": "delta", "text": "你好"}, {"type": "delta", "text": "，欢迎咨询。"}]),
    )

    response = test_client.post("/api/chat/stream", json={"message": "你好"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["conversation", "token", "token", "done"]
    assert events[0]["data"]["conversation_id"] == "conv-fake-1"
    assert [event["data"] for event in events[1:3]] == [{"delta": "你好"}, {"delta": "，欢迎咨询。"}]
    assert events[-1]["data"] == "[DONE]"


def test_actions_tool_and_citations_frames_are_forwarded(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(runtime, "prepare_turn", _fake_prepare(created=False))
    monkeypatch.setattr(
        runtime,
        "stream_turn",
        _fake_stream(
            [
                {"type": "delta", "text": "好的"},
                {"type": "tool", "tool_call_id": "t1", "name": "query_order", "status": "running"},
                {"type": "tool", "tool_call_id": "t1", "name": "query_order", "status": "completed"},
                {"type": "citations", "conversation_id": "conv-fake-1", "items": [{"n": 1, "title": "t"}]},
                {"type": "actions", "items": [{"type": "transfer_human"}, {"type": "create_ticket", "draft": {}}]},
            ]
        ),
    )

    response = test_client.post("/api/chat/stream", json={"message": "问题"})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == [
        "token", "tool", "tool", "citations", "actions", "done",
    ]
    assert events[1]["data"] == {"tool_call_id": "t1", "name": "query_order", "status": "running"}
    assert events[3]["data"]["items"][0]["n"] == 1
    assert events[4]["data"]["items"][0]["type"] == "transfer_human"


def test_second_turn_receives_no_second_conversation_event(client, monkeypatch):
    """既有会话续聊:不再重发 conversation 帧(ch04 行为)。"""
    test_client, _ = client
    monkeypatch.setattr(runtime, "prepare_turn", _fake_prepare(created=False))
    monkeypatch.setattr(runtime, "stream_turn", _fake_stream([{"type": "delta", "text": "好的"}]))

    response = test_client.post(
        "/api/chat/stream", json={"message": "接着问", "conversation_id": "conv-existing"}
    )

    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["token", "done"]


def test_unknown_conversation_id_returns_404_before_sse(client):
    """真实 prepare_turn(测试应用的 SQLite)校验:未知会话 404 且无 SSE 帧。"""
    test_client, _ = client

    response = test_client.post(
        "/api/chat/stream", json={"message": "继续", "conversation_id": "not-a-session"}
    )

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "event:" not in response.text


def test_blank_message_is_rejected_before_sse(client):
    test_client, _ = client

    response = test_client.post("/api/chat/stream", json={"message": "  \t  "})

    assert response.status_code == 422
    assert "event:" not in response.text


def test_prepare_db_error_maps_503(client, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    test_client, _ = client
    monkeypatch.setattr(
        runtime, "prepare_turn", _fake_prepare(error=SQLAlchemyError("db down"))
    )

    response = test_client.post("/api/chat/stream", json={"message": "你好"})

    assert response.status_code == 503
    assert "event:" not in response.text


def test_stream_error_does_not_commit_partial_turn(client, monkeypatch):
    test_client, test_app = client
    with test_app.state.session_factory() as session:
        conversation_id = create_conversation(session).id
        session.commit()
    # prepare 用真实实现(会落 user 行);只桩掉图流,模拟上游在流中途炸掉
    monkeypatch.setattr(
        runtime,
        "stream_turn",
        _fake_stream(
            [{"type": "delta", "text": "partial"}],
            error=RuntimeError("upstream secret detail"),
        ),
    )

    response = test_client.post(
        "/api/chat/stream", json={"message": "问题", "conversation_id": conversation_id}
    )

    error = next(event for event in _parse_sse_events(response.text) if event["event"] == "error")
    assert error["data"]["code"] == "upstream_stream_error"
    assert error["data"]["message"] == "客服服务暂时不可用，请稍后重试。"
    assert "upstream secret detail" not in response.text
    assert "done" not in {event["event"] for event in _parse_sse_events(response.text)}
    with test_app.state.session_factory() as session:
        rows = session.scalars(
            select(Message).where(Message.conversation_id == conversation_id)
        ).all()
    assert [row.role for row in rows] == ["user"]


def test_stream_persistence_error_frame(client, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError

    test_client, _ = client
    monkeypatch.setattr(runtime, "prepare_turn", _fake_prepare(created=False))
    monkeypatch.setattr(
        runtime,
        "stream_turn",
        _fake_stream([], error=SQLAlchemyError("log insert failed")),
    )

    response = test_client.post(
        "/api/chat/stream", json={"message": "问题", "conversation_id": "conv-1"}
    )

    error = next(event for event in _parse_sse_events(response.text) if event["event"] == "error")
    assert error["data"]["code"] == "persistence_error"
    assert error["data"]["message"] == "会话记录保存失败，请稍后重试。"


def test_openapi_lists_new_action_route(client):
    _, test_app = client
    paths = test_app.openapi()["paths"]

    assert "post" in paths["/api/chat/stream"]
    assert "post" in paths["/api/actions/create-ticket"]
    assert "post" in paths["/api/after-sales/extract"]


def test_get_chat_model_reuses_one_lazily_created_client(monkeypatch, test_app):
    """结构化抽取端点仍走请求级模型缓存(ch04 行为保留)。"""
    created = []
    previous_model = getattr(test_app.state, "chat_model", None)
    test_app.state.chat_model = None

    def fake_chat_openai(**kwargs):
        client_stub = object()
        created.append((kwargs, client_stub))
        return client_stub

    monkeypatch.setattr(dependencies, "ChatOpenAI", fake_chat_openai)
    from app.settings import Settings

    settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        chat_temperature=0.1,
        checkpointer_db_path=":memory:",
    )
    try:
        request = Request({"type": "http", "app": test_app})
        first = dependencies.get_chat_model(request, settings)
        second = dependencies.get_chat_model(request, settings)
    finally:
        test_app.state.chat_model = previous_model

    assert first is second
    assert len(created) == 1
