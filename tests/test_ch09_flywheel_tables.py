"""ch09 地基：review_queue / eval_runs ORM 与 repository 飞轮方法（TDD）。"""

import pytest

from app.models import LowConfidenceQuestion
from app.repositories import low_confidence as low_confidence_repository
from app.repositories import review_queue as review_queue_repository


@pytest.fixture
def session(session_factory):
    with session_factory() as session:
        yield session


def test_insert_low_confidence_with_snapshot(session):
    snap = [{"question": "退货运费谁出", "answer": "满99包邮", "rerank_score": 0.91, "section_path": "售后 / 退货"}]
    row = low_confidence_repository.insert_low_confidence(
        session, conversation_id=None, raw_question="猫窝能水洗吗",
        source="retrieval_low_conf", reason="top1=0.2", retrieved_chunks=snap,
    )
    session.commit()
    assert row.id > 0
    assert row.retrieved_chunks[0]["rerank_score"] == 0.91
    assert row.matched_review_id is None
    unmatched = review_queue_repository.fetch_unmatched(session, limit=10)
    assert [r.id for r in unmatched] == [row.id]


def test_insert_low_confidence_without_snapshot(session):
    row = low_confidence_repository.insert_low_confidence(
        session, conversation_id=None, raw_question="没走检索的问题", source="user_feedback", reason=None
    )
    session.commit()
    assert row.retrieved_chunks is None


def test_review_item_create_and_merge(session):
    item = review_queue_repository.insert_review_item(
        session, normalized_question="猫窝是否支持水洗", ai_suggested_answer="可以,答案备查"
    )
    session.commit()
    candidates = review_queue_repository.list_review_candidates(session)
    assert candidates == [{"id": item.id, "normalized_question": "猫窝是否支持水洗"}]
    review_queue_repository.increment_occurrence(session, item.id)
    review_queue_repository.increment_occurrence(session, item.id)
    session.commit()
    detail = review_queue_repository.get_review_detail(session, item.id)
    assert detail is not None
    got, raws = detail
    assert got.occurrence_count == 3
    assert got.review_status == "待审"
    assert raws == []


def test_set_matched_review_links_raws(session):
    item = review_queue_repository.insert_review_item(session, normalized_question="猫窝是否支持水洗", ai_suggested_answer=None)
    lcq = low_confidence_repository.insert_low_confidence(
        session, conversation_id=None, raw_question="猫窝可以洗吗", source="user_feedback", reason=None
    )
    review_queue_repository.set_matched_review(session, lcq.id, item.id)
    session.commit()
    _, raws = review_queue_repository.get_review_detail(session, item.id)
    assert [r.id for r in raws] == [lcq.id]
    assert review_queue_repository.fetch_unmatched(session, limit=10) == []  # 归并后不再是待处理


def test_update_review_status_only_from_pending(session):
    item = review_queue_repository.insert_review_item(session, normalized_question="猫窝是否支持水洗", ai_suggested_answer=None)
    session.commit()
    assert review_queue_repository.update_review_status(session, item.id, "通过", approved_answer="可以水洗") is True
    session.commit()
    got, _ = review_queue_repository.get_review_detail(session, item.id)
    assert got.review_status == "通过" and got.approved_answer == "可以水洗"
    # 已通过的不许再改（终审）
    assert review_queue_repository.update_review_status(session, item.id, "驳回") is False


def test_list_review_queue_orders_by_occurrence(session):
    a = review_queue_repository.insert_review_item(session, normalized_question="问题A", ai_suggested_answer=None)
    b = review_queue_repository.insert_review_item(session, normalized_question="问题B", ai_suggested_answer=None)
    review_queue_repository.increment_occurrence(session, b.id)
    session.commit()
    rows = review_queue_repository.list_review_queue(session, "待审")
    assert [r.id for r in rows] == [b.id, a.id]
    assert review_queue_repository.update_review_status(session, a.id, "驳回") is True
    session.commit()
    assert [r.id for r in review_queue_repository.list_review_queue(session, "待审")] == [b.id]
    assert len(review_queue_repository.list_review_queue(session, None)) == 2


def test_eval_runs_roundtrip(session):
    review_queue_repository.insert_eval_run(session, triggered_by="手动", dataset_size=40, metrics={"recall_at_10": 0.85, "mrr": 0.72})
    review_queue_repository.insert_eval_run(session, triggered_by="定时", dataset_size=40, metrics={"recall_at_10": 0.88, "mrr": 0.74})
    session.commit()
    runs = review_queue_repository.list_eval_runs(session)
    assert len(runs) == 2
    assert runs[0].metrics["recall_at_10"] == 0.88  # 最新在前
    assert runs[1].triggered_by == "手动"
