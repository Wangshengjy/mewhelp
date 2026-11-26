"""ch10 · 主题分布 API（/api/topics/distribution + /api/topics/questions）（TDD）。"""

import pytest

from app.models import LowConfidenceQuestion, ReviewQueue


@pytest.fixture
def client(test_app):
    from fastapi.testclient import TestClient

    with TestClient(test_app) as c:
        yield c


@pytest.fixture
def session_factory(test_app):
    return test_app.state.session_factory


def _seed(session_factory, raw="猫窝买大了想退", norm=None, source="retrieval_low_conf"):
    with session_factory() as session:
        review_id = None
        if norm:
            rq = ReviewQueue(normalized_question=norm)
            session.add(rq)
            session.flush()
            review_id = rq.id
        q = LowConfidenceQuestion(
            raw_question=raw, source=source, reason=None, matched_review_id=review_id)
        session.add(q)
        session.commit()
        return q.id


def test_distribution_empty_returns_all_17(client):
    resp = client.get("/api/topics/distribution")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert len(body["classes"]) == 17


def test_distribution_counts(client, session_factory):
    qid = _seed(session_factory, raw="猫窝买大了想退", norm="猫窝能否退货")
    from app.repositories.topic_classification import insert_topic_classifications

    with session_factory() as session:
        insert_topic_classifications(session, [{"question_id": qid, "labels": ["尺码", "退换货"]}])
        session.commit()
    resp = client.get("/api/topics/distribution")
    body = resp.json()
    by_label = {c["label"]: c for c in body["classes"]}
    assert body["total"] == 1
    assert by_label["尺码"]["count"] == 1 and by_label["退换货"]["count"] == 1
    assert by_label["物流"]["count"] == 0


def test_questions_paginates_and_returns_peers(client, session_factory):
    qid = _seed(session_factory, norm="猫窝买大了能不能退货")
    from app.repositories.topic_classification import insert_topic_classifications

    with session_factory() as session:
        insert_topic_classifications(session, [{"question_id": qid, "labels": ["尺码", "退换货"]}])
        session.commit()
    resp = client.get("/api/topics/questions", params={"label": "退换货"})
    body = resp.json()
    assert resp.status_code == 200
    assert body["label"] == "退换货" and body["total"] == 1
    item = body["items"][0]
    assert item["text"] == "猫窝买大了能不能退货" and item["normalized"] is True
    assert item["labels"] == ["尺码", "退换货"] and item["peer_labels"] == ["尺码"]


def test_questions_invalid_label_returns_400(client):
    resp = client.get("/api/topics/questions", params={"label": "不存在的类目"})
    assert resp.status_code == 400


def test_questions_page_bounds(client, session_factory):
    resp = client.get("/api/topics/questions", params={"label": "退换货", "page": 0})
    assert resp.status_code == 400
