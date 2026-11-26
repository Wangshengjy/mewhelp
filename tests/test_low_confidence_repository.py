"""ch04 · low_confidence_questions 表 repository（确定性写入，TDD）。"""

import pytest

from app.repositories import low_confidence as low_confidence_repository


@pytest.fixture
def session(session_factory):
    with session_factory() as session:
        yield session


def test_insert_low_confidence_with_conversation(session):
    from app.models import Conversation

    conversation = Conversation()
    session.add(conversation)
    session.flush()

    row = low_confidence_repository.insert_low_confidence(
        session,
        conversation_id=conversation.id,
        raw_question="邮费到底多少啊",
        source="retrieval_low_conf",
        reason="检索证据不足(top=0.12)",
    )
    session.commit()

    assert row.id > 0
    assert row.raw_question == "邮费到底多少啊"
    assert row.source == "retrieval_low_conf"
    assert row.reason == "检索证据不足(top=0.12)"
    assert row.conversation_id == conversation.id
    assert row.created_at is not None


def test_insert_low_confidence_allows_null_conversation(session):
    row = low_confidence_repository.insert_low_confidence(
        session, conversation_id=None, raw_question="无会话问题", source="self_check", reason=None
    )
    session.commit()
    assert row.id > 0
    assert row.conversation_id is None


def test_insert_low_confidence_rejects_unknown_source(session):
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(ValueError):
        low_confidence_repository.insert_low_confidence(
            session, conversation_id=None, raw_question="x", source="bad_source", reason=None
        )


def test_recent_rows_orders_by_id_desc(session):
    low_confidence_repository.insert_low_confidence(session, conversation_id=None, raw_question="问题一", source="self_check", reason=None)
    low_confidence_repository.insert_low_confidence(session, conversation_id=None, raw_question="问题二", source="retrieval_low_conf", reason="低分")
    session.commit()

    rows = low_confidence_repository.recent_rows(session, limit=1)
    assert len(rows) == 1
    assert rows[0].raw_question == "问题二"

    counts = low_confidence_repository.counts_by_source(session)
    assert counts == {"self_check": 1, "retrieval_low_conf": 1}
