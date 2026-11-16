"""Tests for the SSE chat route's session handling and streaming contract."""

from collections import deque
import json

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import select

from app import dependencies
from app.dependencies import get_chat_model, get_settings
from app.models import Message
from app.prompts import CHAT_SYSTEM_PROMPT
from app.repositories.conversations import create_conversation
from app.settings import Settings


class FakeBoundModel:
    """The tool-bound view of the fake model; the chat route only decides here."""

    def __init__(self, parent):
        self.parent = parent

    async def ainvoke(self, messages, **kwargs):
        self.parent.decision_calls.append(list(messages))
        item = self.parent.decisions.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class FakeStreamingModel:
    """Scripts tool decisions and final text streams for the chat route."""

    def __init__(self, decisions=(), streams=()):
        self.decisions = deque(decisions)
        self.streams = deque(streams)
        self.bound = FakeBoundModel(self)
        self.bind_tools_calls = []
        self.decision_calls = []
        self.final_calls = []

    def bind_tools(self, tools, **kwargs):
        self.bind_tools_calls.append(list(tools))
        return self.bound

    async def astream(self, messages):
        self.final_calls.append(list(messages))
        for part in self.streams.popleft():
            if isinstance(part, Exception):
                raise part
            yield AIMessageChunk(content=part)


def _answer(text):
    return AIMessage(content=text, tool_calls=[])


@pytest.fixture
def chat_client(test_app):
    model = FakeStreamingModel(
        decisions=[_answer("直接回答"), _answer("直接回答")],
        streams=[["你好", "，欢迎咨询。"], ["我记得", "上一轮问题。"]],
    )
    test_app.dependency_overrides[get_chat_model] = lambda: model
    test_app.dependency_overrides[get_settings] = lambda: Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        chat_context_token_budget=6000,
        chat_token_safety_margin=500,
    )
    try:
        with TestClient(test_app) as client:
            yield client, model
    finally:
        test_app.dependency_overrides.clear()


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


def test_sse_events_include_conversation_tokens_and_done(chat_client):
    client, _ = chat_client

    response = client.post("/api/chat/stream", json={"message": "你好"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse_events(response.text)
    assert [event["event"] for event in events] == ["conversation", "token", "token", "done"]
    assert events[0]["data"]["conversation_id"]
    assert [event["data"] for event in events[1:3]] == [{"delta": "你好"}, {"delta": "，欢迎咨询。"}]
    assert events[-1]["data"] == "[DONE]"


def test_second_turn_receives_first_turn_history(chat_client):
    client, model = chat_client
    first = client.post("/api/chat/stream", json={"message": "第一问"})
    conversation_id = _conversation_id(first.text)

    second = client.post(
        "/api/chat/stream",
        json={"message": "接着刚才的问题", "conversation_id": conversation_id},
    )

    assert second.status_code == 200
    assert [(message.type, message.content) for message in model.decision_calls[1]] == [
        ("system", CHAT_SYSTEM_PROMPT),
        ("human", "第一问"),
        ("ai", "你好，欢迎咨询。"),
        ("human", "接着刚才的问题"),
    ]


def test_unknown_conversation_id_returns_404_before_sse(chat_client):
    client, _ = chat_client

    response = client.post(
        "/api/chat/stream",
        json={"message": "继续", "conversation_id": "not-a-session"},
    )

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "event:" not in response.text


def test_blank_message_is_rejected_before_sse(chat_client):
    client, _ = chat_client

    response = client.post("/api/chat/stream", json={"message": "  \t  "})

    assert response.status_code == 422
    assert "event:" not in response.text


def test_stream_error_does_not_commit_partial_turn(chat_client, test_app):
    client, _ = chat_client
    with test_app.state.session_factory() as session:
        conversation_id = create_conversation(session).id
        session.commit()
    model = FakeStreamingModel(
        decisions=[_answer("直接回答")],
        streams=[["partial", RuntimeError("upstream secret detail")]],
    )
    test_app.dependency_overrides[get_chat_model] = lambda: model

    response = client.post(
        "/api/chat/stream",
        json={"message": "问题", "conversation_id": conversation_id},
    )

    error = next(event for event in _parse_sse_events(response.text) if event["event"] == "error")
    assert error["data"]["code"] == "upstream_stream_error"
    assert error["data"]["message"]
    assert "upstream secret detail" not in response.text
    with test_app.state.session_factory() as session:
        rows = session.scalars(
            select(Message).where(Message.conversation_id == conversation_id)
        ).all()
    assert [row.role for row in rows] == ["user"]


def test_fixed_prompt_over_budget_returns_413(chat_client, test_app):
    client, _ = chat_client
    test_app.dependency_overrides[get_settings] = lambda: Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        chat_context_token_budget=1,
        chat_token_safety_margin=0,
    )
    try:
        response = client.post("/api/chat/stream", json={"message": "hello"})
        assert response.status_code == 413
    finally:
        test_app.dependency_overrides.pop(get_settings, None)


def test_openapi_lists_both_post_routes(chat_client, test_app):
    paths = test_app.openapi()["paths"]

    assert "post" in paths["/api/chat/stream"]
    assert "post" in paths["/api/after-sales/extract"]


def test_get_chat_model_reuses_one_lazily_created_client(monkeypatch, test_app):
    created = []
    previous_model = getattr(test_app.state, "chat_model", None)
    test_app.state.chat_model = None

    def fake_chat_openai(**kwargs):
        client = object()
        created.append((kwargs, client))
        return client

    monkeypatch.setattr(dependencies, "ChatOpenAI", fake_chat_openai)
    settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        chat_temperature=0.1,
    )
    try:
        request = Request({"type": "http", "app": test_app})
        first = dependencies.get_chat_model(request, settings)
        second = dependencies.get_chat_model(request, settings)
    finally:
        test_app.state.chat_model = previous_model

    assert first is second
    assert len(created) == 1
    assert created[0][0] == {
        "base_url": "https://example.test/v1",
        "api_key": "test-key",
        "model": "test-model",
        "temperature": 0.1,
    }
