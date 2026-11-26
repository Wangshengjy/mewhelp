"""Round-trip tests for the SQLAlchemy conversation/message repositories."""

import pytest
from sqlalchemy import select

from app.database import build_engine, build_session_factory
from app.models import Base, Message
from app.repositories.conversations import (
    append_message,
    count_messages_after,
    create_conversation,
    get_conversation,
    list_dialog_messages,
    list_summary_segments,
    load_history,
    update_conversation_summary,
    update_layer1_from,
    append_summary_segment,
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


# ── ch07 · 分层上下文:锚点字段 + 分段摘要 ────────────────────────────────────


def test_new_conversation_has_empty_layer_fields(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        session.commit()
        found = get_conversation(session, conversation.id)
        assert found.summary is None
        assert found.summary_upto_msg_id is None
        assert found.layer1_from_msg_id is None


def test_count_messages_after(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        first = append_message(session, conversation.id, role="user", content="一")
        append_message(session, conversation.id, role="assistant", content="二")
        session.commit()
        assert count_messages_after(session, conversation.id, 0) == 2
        assert count_messages_after(session, conversation.id, first.id) == 1
        assert count_messages_after(session, conversation.id, 10**9) == 0
        assert count_messages_after(session, conversation.id, None) == 2


def test_list_dialog_messages_filters_tool_rows(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        append_message(session, conversation.id, role="user", content="一")
        append_message(session, conversation.id, role="assistant", content="", tool_call_id="c1")
        append_message(session, conversation.id, role="tool", content="工具", tool_call_id="c1")
        append_message(session, conversation.id, role="assistant", content="二")
        session.commit()
        rows = list_dialog_messages(session, conversation.id)
        assert [(row.role, row.content) for row in rows] == [
            ("user", "一"),
            ("assistant", ""),
            ("assistant", "二"),
        ]


def test_update_layer1_from_and_summary_fields(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        append_message(session, conversation.id, role="user", content="一")
        session.commit()
        update_layer1_from(session, conversation.id, 7)
        update_conversation_summary(session, conversation.id, "梗概内容", 7)
        session.commit()
        found = get_conversation(session, conversation.id)
        assert found.layer1_from_msg_id == 7
        assert found.summary == "梗概内容"
        assert found.summary_upto_msg_id == 7


def test_append_summary_segment_appends_and_updates_projection(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        append_message(session, conversation.id, role="user", content="一")
        session.commit()
        seq1 = append_summary_segment(session, conversation.id, 1, 4, "第一段梗概")
        seq2 = append_summary_segment(session, conversation.id, 5, 9, "第二段梗概")
        session.commit()
        assert (seq1, seq2) == (1, 2)
        found = get_conversation(session, conversation.id)
        assert found.summary == "第一段梗概\n第二段梗概"
        assert found.summary_upto_msg_id == 9
        segments = list_summary_segments(session, conversation.id)
        assert segments == ["第一段梗概", "第二段梗概"]


def test_list_summary_segments_returns_latest_n_in_order(session_factory):
    with session_factory() as session:
        conversation = create_conversation(session)
        append_message(session, conversation.id, role="user", content="一")
        session.commit()
        for index in range(1, 5):
            append_summary_segment(session, conversation.id, index, index + 1, f"第{index}段")
        session.commit()
        assert list_summary_segments(session, conversation.id) == ["第1段", "第2段", "第3段", "第4段"]
        assert list_summary_segments(session, conversation.id, limit=2) == ["第3段", "第4段"]
