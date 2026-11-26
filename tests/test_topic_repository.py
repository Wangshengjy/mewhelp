"""ch10 地基：topic_classifications ORM 与 repository 池捞取/写表/分布/类目列表（TDD）。"""

import pytest

from app.models import LowConfidenceQuestion, ReviewQueue
from app.repositories import topic_classification as topic_repository


@pytest.fixture
def session(session_factory):
    with session_factory() as session:
        yield session


def _seed_question(session, raw="猫窝买大了想退", norm=None, source="retrieval_low_conf"):
    review_id = None
    if norm:
        rq = ReviewQueue(normalized_question=norm)
        session.add(rq)
        session.flush()
        review_id = rq.id
    q = LowConfidenceQuestion(
        raw_question=raw, source=source, reason=None, matched_review_id=review_id
    )
    session.add(q)
    session.flush()
    return q.id


def test_insert_and_exclude_classified(session):
    qid = _seed_question(session, norm="猫窝买大了能不能退货")  # 已归并才进分类
    session.commit()
    rows = topic_repository.list_unclassified_questions(session)
    assert any(r["question_id"] == qid for r in rows)

    n = topic_repository.insert_topic_classifications(
        session, [{"question_id": qid, "labels": ["尺码", "退换货"]}]
    )
    session.commit()
    assert n == 1
    rows = topic_repository.list_unclassified_questions(session)
    assert not any(r["question_id"] == qid for r in rows)


def test_unmerged_excluded_from_classify(session):
    _seed_question(session, raw="猫抓板回收吗")  # 无 norm=未归并
    session.commit()
    rows = topic_repository.list_unclassified_questions(session)
    assert rows == []


def test_text_prefers_normalized(session):
    qid = _seed_question(
        session, raw="这个还能退吗", norm="智能猫砂盆能否七天无理由退货"
    )
    session.commit()
    rows = topic_repository.list_pool_texts(session)
    row = next(r for r in rows if r["question_id"] == qid)
    assert row["text"] == "智能猫砂盆能否七天无理由退货"


def test_topic_distribution_counts_all_17(session):
    qid = _seed_question(session)
    session.commit()
    topic_repository.insert_topic_classifications(
        session, [{"question_id": qid, "labels": ["尺码", "退换货"]}]
    )
    session.commit()
    dist = topic_repository.topic_distribution(session)
    assert len(dist["classes"]) == 17
    by_label = {c["label"]: c for c in dist["classes"]}
    assert by_label["尺码"]["count"] == 1 and by_label["退换货"]["count"] == 1
    assert by_label["物流"]["count"] == 0
    assert by_label["尺码"]["samples"]
    assert dist["total"] == 1 and dist["latest"] is not None


def test_distribution_samples_dedupe_by_text(session):
    """归并后同一句标准化问法对应池里好几行：三条样例不能都一样，计数照旧按条算。"""
    qid1 = _seed_question(session, raw="还能退吗", norm="智能猫砂盆能否退货")
    qid2 = _seed_question(session, raw="求退货", norm="智能猫砂盆能否退货")
    session.commit()
    topic_repository.insert_topic_classifications(
        session,
        [
            {"question_id": qid1, "labels": ["退换货"]},
            {"question_id": qid2, "labels": ["退换货"]},
        ],
    )
    session.commit()
    dist = topic_repository.topic_distribution(session, samples_per_class=3)
    by_label = {c["label"]: c for c in dist["classes"]}
    assert by_label["退换货"]["count"] == 2
    assert by_label["退换货"]["samples"] == ["智能猫砂盆能否退货"]


def test_topic_questions_pagination_and_peers(session):
    qid1 = _seed_question(session, norm="猫窝买大了能不能退货")
    qid2 = _seed_question(session, raw="发票和退货一起问", norm="能否开发票并退货")
    session.commit()
    topic_repository.insert_topic_classifications(
        session,
        [
            {"question_id": qid1, "labels": ["尺码", "退换货"]},
            {"question_id": qid2, "labels": ["发票", "退换货"]},
        ],
    )
    session.commit()
    out = topic_repository.topic_questions(session, label="退换货", page=1, size=1)
    assert out["label"] == "退换货"
    assert out["total"] == 2 and out["pages"] == 2 and out["size"] == 1
    item = out["items"][0]
    assert item["normalized"] is True
    assert "退换货" in item["labels"] and item["peer_labels"] == ["尺码"]
    assert item["review_status"] == "待审"
    assert item["occurrence_count"] == 1
    # 翻页能拿到另一条
    out2 = topic_repository.topic_questions(session, label="退换货", page=2, size=1)
    assert out2["items"][0]["question_id"] != item["question_id"]
    # 多标签问题在它命中的每个类目下都出现
    for label in ("尺码", "发票"):
        sub = topic_repository.topic_questions(session, label=label)
        assert sub["total"] >= 1


def test_topic_questions_unmerged_shows_raw(session):
    """分类只吃已归并行，但列表兜底能展示未归并的原话（normalized=false、审核状态为空）。"""
    qid = _seed_question(session, raw="猫抓板回收吗")  # 未归并
    session.commit()
    topic_repository.insert_topic_classifications(
        session, [{"question_id": qid, "labels": ["其他"]}]
    )
    session.commit()
    out = topic_repository.topic_questions(session, label="其他")
    assert out["total"] == 1
    item = out["items"][0]
    assert item["text"] == "猫抓板回收吗"
    assert item["normalized"] is False
    assert item["review_status"] is None
    assert item["occurrence_count"] == 1
