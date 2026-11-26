"""ch09 飞轮流水线单测:normalize_and_match 全 mock(真模型只跑样例验证),库操作真跑 SQLite。"""
import json

import pytest

from app.core import flywheel
from app.repositories import low_confidence as low_confidence_repository
from app.repositories import review_queue as review_queue_repository


@pytest.fixture
def seeded(session_factory):
    """三池条目:两条同义,一条新知识。"""
    raws = []
    with session_factory() as session:
        for q in ("猫窝可以机洗么", "猫窝能不能洗衣机洗", "猫粮多久发一次货"):
            raws.append(low_confidence_repository.insert_low_confidence(
                session, conversation_id=None, raw_question=q,
                source="user_feedback", reason=None))
        session.commit()
    return raws


def test_verdict_schema_fields():
    verdict = flywheel.FlywheelVerdict(normalized_question="猫窝是否支持机洗", matched_id=None, suggested_answer="")
    assert verdict.normalized_question == "猫窝是否支持机洗"
    assert verdict.matched_id is None


def test_run_pipeline_creates_and_merges(session_factory, seeded, monkeypatch):
    """首条建缺口,后两条同义归并累次数,一次输出不二次调用。"""
    calls = []

    def fake_normalize(settings, raw_question, candidates):
        calls.append({"raw": raw_question, "n_candidates": len(candidates)})
        if raw_question.startswith("猫窝"):
            # 后两条进来自动看到第一条建的缺口(候选随流水线推进)
            hit = next((c for c in candidates if "猫窝" in c["normalized_question"]), None)
            return flywheel.FlywheelVerdict(
                normalized_question="猫窝是否支持机洗",
                matched_id=hit["id"] if hit else None)
        return flywheel.FlywheelVerdict(normalized_question="猫粮补货周期是多久", matched_id=None)

    monkeypatch.setattr(flywheel, "normalize_and_match", fake_normalize)
    result = flywheel.run_pipeline(session_factory, object(), limit=10)
    assert result["processed"] == 3
    assert result["created"] == 2   # 猫窝缺口 + 猫粮缺口
    assert result["merged"] == 1    # 第二条猫窝归并
    assert result["failed"] == 0
    assert len(calls) == 3          # 每条原话恰好一次 LLM 调用

    with session_factory() as session:
        queue = review_queue_repository.list_review_queue(session)
        by_q = {row.normalized_question: row for row in queue}
        cat_bed = next((v for k, v in by_q.items() if "猫窝" in k), None)
        assert cat_bed is not None
        assert cat_bed.occurrence_count == 2          # 建槽 1 次 + 归并 1 次
        unmatched = review_queue_repository.fetch_unmatched(session)
        assert unmatched == []                        # 全部归并,游标清空


def test_run_pipeline_rejects_hallucinated_id(session_factory, seeded, monkeypatch):
    """模型返回候选里没有的 ID → 当新缺口处理,不指向不存在的行。"""

    def fake_normalize(settings, raw_question, candidates):
        return flywheel.FlywheelVerdict(normalized_question="随便一个问题", matched_id=99999)

    monkeypatch.setattr(flywheel, "normalize_and_match", fake_normalize)
    result = flywheel.run_pipeline(session_factory, object(), limit=10)
    assert result["failed"] == 0
    with session_factory() as session:
        queue = review_queue_repository.list_review_queue(session)
        assert all(row.id != 99999 for row in queue)
        assert len(queue) == 3  # 三条各建各的缺口
        for row in queue:
            assert row.occurrence_count == 1


def test_run_pipeline_failure_leaves_cursor(session_factory, seeded, monkeypatch):
    """单条 LLM 失败 → 该条游标不动留给下轮,其余正常推进。"""
    state = {"n": 0}

    def fake_normalize(settings, raw_question, candidates):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("上游超时")
        return flywheel.FlywheelVerdict(normalized_question=f"标准化问题{state['n']}", matched_id=None)

    monkeypatch.setattr(flywheel, "normalize_and_match", fake_normalize)
    result = flywheel.run_pipeline(session_factory, object(), limit=10)
    assert result["failed"] == 1 and result["created"] == 2
    assert result["failures"][0]["error"].startswith("RuntimeError")
    with session_factory() as session:
        unmatched = review_queue_repository.fetch_unmatched(session)
        assert len(unmatched) == 1 and unmatched[0].raw_question == "猫窝可以机洗么"


def test_run_pipeline_limit(session_factory, seeded, monkeypatch):
    monkeypatch.setattr(
        flywheel, "normalize_and_match",
        lambda s, q, c: flywheel.FlywheelVerdict(normalized_question=q[:20], matched_id=None))
    result = flywheel.run_pipeline(session_factory, object(), limit=1)
    assert result["processed"] == 1


def test_normalize_prompt_contract():
    """提示词契约:含候选列表与原话占位(真模型样例验证同源)。"""
    assert "{candidates}" in flywheel.NORMALIZE_PROMPT
    assert "{raw_question}" in flywheel.NORMALIZE_PROMPT
