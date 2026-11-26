"""ch04 · faith_cases 编造个案台账 repository（确定性方法，TDD）。

一题一行（uk_eval_id）：重复判出 → 更新答案/理由/证据快照、seen_count+1、
已处置个案状态退回「未解决」（复发），resolution 清空、resolved_at 保留标记复发。
"""

import pytest

from app.repositories import faith_cases as faith_cases_repository

CASE = {
    "eval_id": "A43",
    "bucket": "A_policy",
    "query": "退货之后钱什么时候到账",
    "strategy": "hybrid_rerank",
    "answer": "退款会在 24 小时内到账[1]。",
    "reason": "「24 小时内到账」证据未支撑,属编造承诺。",
    "citations": [{"n": 1, "chunk_id": 5, "section_path": "退货政策", "question": "退款", "answer": "原路退回"}],
    "judge_model": "judge-x",
}


@pytest.fixture
def session(session_factory):
    with session_factory() as session:
        yield session


def test_record_case_inserts_then_updates_on_repeat(session):
    created = faith_cases_repository.record_case(session, **CASE)
    session.commit()
    assert created.status == "未解决"
    assert created.seen_count == 1
    assert created.first_seen_at is not None
    assert created.last_seen_at is not None
    assert created.citations == CASE["citations"]

    updated = faith_cases_repository.record_case(session, **{**CASE, "answer": "新版答案", "seen": 2})
    session.commit()
    assert updated.id == created.id, "同题不再新增行"
    assert updated.seen_count == 2
    assert updated.answer == "新版答案"
    assert updated.last_seen_at >= created.last_seen_at


def test_resolve_and_resolve_without_fix(session):
    row = faith_cases_repository.record_case(session, **CASE)
    session.commit()

    resolved = faith_cases_repository.resolve_case(
        session, eval_id="A43", status="已解决", resolution="补充了退款时效知识块并重建"
    )
    session.commit()
    assert resolved.status == "已解决"
    assert resolved.resolved_at is not None

    # 复发：同题再次被判编造 → 退回未解决、resolution 清空、resolved_at 保留
    recurrence = faith_cases_repository.record_case(session, **{**CASE, "answer": "又编了", "seen": 3})
    session.commit()
    assert recurrence.id == row.id
    assert recurrence.status == "未解决"
    assert recurrence.resolution is None
    assert recurrence.resolved_at is not None, "resolved_at 保留用来标记复发"
    assert recurrence.seen_count == 3

    no_need = faith_cases_repository.resolve_case(
        session, eval_id="A43", status="无需解决", resolution="裁判误判,答案有据"
    )
    session.commit()
    assert no_need.status == "无需解决"


def test_resolve_case_rejects_bad_status(session):
    faith_cases_repository.record_case(session, **CASE)
    session.commit()
    with pytest.raises(ValueError):
        faith_cases_repository.resolve_case(session, eval_id="A43", status="随便", resolution="x")
    with pytest.raises(ValueError):
        faith_cases_repository.resolve_case(session, eval_id="不存在", status="已解决", resolution="x")


def test_list_cases_filter_by_status(session):
    faith_cases_repository.record_case(session, **CASE)
    faith_cases_repository.record_case(
        session,
        eval_id="B7",
        bucket="B_model",
        query="MH-LP100 多少钱",
        strategy="vector",
        answer="卖 999 元。",
        reason="价格编造。",
        citations=None,
        judge_model="judge-x",
    )
    session.commit()
    faith_cases_repository.resolve_case(session, eval_id="B7", status="无需解决", resolution="库里本就无价格")

    unresolved = faith_cases_repository.list_cases(session, status="未解决")
    assert [case.eval_id for case in unresolved] == ["A43"]
    all_cases = faith_cases_repository.list_cases(session)
    assert {case.eval_id for case in all_cases} == {"A43", "B7"}

    stats = faith_cases_repository.status_counts(session)
    assert stats == {"未解决": 1, "无需解决": 1, "已解决": 0}
