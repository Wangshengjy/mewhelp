"""Round-trip tests for the SQLAlchemy conversation/message repositories."""

import pytest
from sqlalchemy import select

from app.database import build_engine, build_session_factory
from app.models import Base, Message
from app.repositories.conversations import (
    append_message,
    create_conversation,
    get_conversation,
    load_history,
)

SQLITE_URL = "sqlite+pysqlite:///:memory:"


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def test_create_and_get_conversation_round_trip(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        assert conversation.id
        assert conversation.user_id == "demo-user"

        session.commit()
        found = get_conversation(session, conversation.id)

        assert found is not None
        assert found.id == conversation.id
        assert found.user_id == "demo-user"
        assert get_conversation(session, "missing-id") is None


def test_append_and_load_history_round_trip_preserves_tool_calls(session_factory):
    tool_calls = [
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call-1", "type": "tool_call"}
    ]

    with session_factory() as session:
        conversation = create_conversation(session)
        append_message(session, conversation.id, role="user", content="可以退货吗？")
        append_message(
            session,
            conversation.id,
            role="assistant",
            content="",
            tool_calls=tool_calls,
            tool_call_id="call-1",
            tool_name="query_faq",
        )
        append_message(
            session,
            conversation.id,
            role="tool",
            content="七天内可退货。",
            tool_call_id="call-1",
            tool_name="query_faq",
        )
        append_message(session, conversation.id, role="assistant", content="可以退货。")
        session.commit()

        history = load_history(session, conversation.id)

        assert [(message.type, message.content) for message in history] == [
            ("human", "可以退货吗？"),
            ("ai", ""),
            ("tool", "七天内可退货。"),
            ("ai", "可以退货。"),
        ]
        assert history[1].tool_calls == tool_calls
        assert history[2].tool_call_id == "call-1"
        assert history[2].name == "query_faq"


def test_appended_messages_persist_json_compatible_tool_call_ids(session_factory):
    tool_calls = [
        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call-9", "type": "tool_call"}
    ]

    with session_factory() as session:
        conversation = create_conversation(session)
        assistant = append_message(
            session,
            conversation.id,
            role="assistant",
            content="",
            tool_calls=tool_calls,
            tool_call_id="call-9",
            tool_name="query_faq",
        )
        tool = append_message(
            session,
            conversation.id,
            role="tool",
            content="工具结果",
            tool_call_id="call-9",
            tool_name="query_faq",
        )
        session.commit()

        assert assistant.id
        assert tool.id
        stored = session.scalars(select(Message).order_by(Message.id)).all()
        assert [message.role for message in stored] == ["assistant", "tool"]
        assert stored[0].tool_calls == tool_calls
        assert stored[0].tool_call_id == "call-9"
        assert stored[0].tool_name == "query_faq"
        assert stored[1].tool_call_id == "call-9"


def test_load_history_orders_by_append_order(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        for role, content in (
            ("user", "一"),
            ("assistant", "二"),
            ("user", "三"),
            ("assistant", "四"),
        ):
            append_message(session, conversation.id, role=role, content=content)
        session.commit()

        history = load_history(session, conversation.id)

        assert [message.content for message in history] == ["一", "二", "三", "四"]


def test_load_history_for_unknown_conversation_is_empty(session_factory):
    with session_factory() as session:
        assert load_history(session, "missing-id") == []
