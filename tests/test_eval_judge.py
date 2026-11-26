"""ch04 · 裁判与生成提示词测试（judge）+ 台账一致性回归测试（judge_check）。

裁判输出扁平字段 + 标量数组；调用/解析失败返回 None 由调用方跳过。
judge_ledger 口径：已解决→期望 faithful=False、无需解决→期望 True、
未解决/缺快照跳过、调用失败不计入一致率分母。
"""

import json

import pytest

from app.database import build_engine, build_session_factory
from app.eval import judge as judge_module
from app.models import Base
from app.repositories import faith_cases as faith_cases_repository
from scripts import judge_check

SQLITE_URL = "sqlite+pysqlite:///:memory:"


class FakeResponse:
    def __init__(self, content):
        self.content = content


def make_model_fn(payload: dict | str | None):
    """payload=None 时模拟调用抛错。"""

    def model_fn(messages, temperature=0.0):
        if payload is None:
            raise RuntimeError("simulated outage")
        if isinstance(payload, str):
            return FakeResponse(payload)
        return FakeResponse(json.dumps(payload, ensure_ascii=False))

    return model_fn


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------

VERDICT = {
    "covered_points": ["满 99 元包邮"],
    "faithful": False,
    "fabricated_points": ["退款 24 小时内到账"],
    "reason": "凭空数值",
}


def test_judge_parses_flat_fields_from_model_json():
    verdict = judge_module.judge(
        make_model_fn(VERDICT), "退款多久到账", ["退款 1-3 个工作日原路退回"], ["[1] 退款 1-3 个工作日"], "退款 24 小时内到账[1]"
    )
    assert verdict == VERDICT


def test_judge_tolerates_markdown_fences():
    fenced = "```json\n" + json.dumps(VERDICT, ensure_ascii=False) + "\n```"
    verdict = judge_module.judge(make_model_fn(fenced), "q", ["p"], ["e"], "a")
    assert verdict["faithful"] is False


def test_judge_returns_none_on_model_failure():
    assert judge_module.judge(make_model_fn(None), "q", ["p"], ["e"], "a") is None


def test_judge_returns_none_on_non_json_output():
    assert judge_module.judge(make_model_fn("抱歉我不懂"), "q", ["p"], ["e"], "a") is None


def test_answer_coverage_exact_and_substring_matching():
    points = ["满 99 元包邮", "10 元运费"]
    assert judge_module.answer_coverage(points, ["满 99 元包邮", "未满收取 10 元运费"]) == 1.0
    assert judge_module.answer_coverage(points, ["满 99 元包邮"]) == 0.5
    assert judge_module.answer_coverage([], ["随便"]) is None


# ---------------------------------------------------------------------------
# judge_check · judge_ledger（台账重放）
# ---------------------------------------------------------------------------


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def seed_case(session_factory, eval_id, status, *, query, with_citations=True):
    citations = (
        [{"n": 1, "chunk_id": 1, "section_path": "退货政策", "question": "退款", "answer": "1-3 个工作日"}]
        if with_citations
        else None
    )
    with session_factory() as session:
        faith_cases_repository.record_case(
            session,
            eval_id=eval_id,
            bucket="A_policy",
            query=query,
            strategy="hybrid_rerank",
            answer="答案" + eval_id,
            reason="裁判理由",
            citations=citations,
            judge_model="judge-x",
        )
        if status != "未解决":
            faith_cases_repository.resolve_case(session, eval_id=eval_id, status=status, resolution="人工处置")
        session.commit()


def test_judge_ledger_consistency_and_skips(session_factory, monkeypatch):
    seed_case(session_factory, "A1", "已解决", query="q-A1")  # 期望 faithful=False
    seed_case(session_factory, "B1", "无需解决", query="q-B1")  # 期望 faithful=True
    seed_case(session_factory, "C1", "未解决", query="q-C1")  # 跳过
    seed_case(session_factory, "D1", "已解决", query="q-D1", with_citations=False)  # 缺快照跳过
    seed_case(session_factory, "E1", "已解决", query="q-E1")  # 调用失败不计入分母

    monkeypatch.setattr(judge_check, "session_factory_from", lambda settings: session_factory)
    monkeypatch.setattr(judge_check, "_chat_model", lambda settings: (lambda messages, temperature=0.0: None))

    def fake_judge(model_fn, query, points, evidence, answer):
        if query == "q-A1":
            return {"covered_points": [], "faithful": False, "fabricated_points": [], "reason": "判编造"}
        if query == "q-B1":
            return {"covered_points": [], "faithful": True, "fabricated_points": [], "reason": "判忠实"}
        return None  # q-E1 → 调用失败

    monkeypatch.setattr(judge_check.judge_module, "judge", fake_judge)

    result = judge_check.judge_ledger(SettingsStub(), status=None)

    assert result["total_ledger"] == 5
    assert result["judged"] == 2, "未解决/缺快照/调用失败都不计入一致率分母"
    assert result["consistent"] == 2
    assert result["consistency_rate"] == 1.0
    rows = {row["eval_id"]: row for row in result["rows"]}
    assert rows["C1"]["note"] == "未解决跳过"
    assert rows["D1"]["note"] == "缺证据快照跳过"
    assert rows["E1"]["consistent"] is None and "调用失败" in rows["E1"]["note"]


def test_judge_ledger_detects_inconsistency(session_factory, monkeypatch):
    seed_case(session_factory, "A2", "已解决", query="q-A2")  # 期望 False，裁判给 True → 不一致

    monkeypatch.setattr(judge_check, "session_factory_from", lambda settings: session_factory)
    monkeypatch.setattr(judge_check, "_chat_model", lambda settings: (lambda messages, temperature=0.0: None))
    monkeypatch.setattr(
        judge_check.judge_module,
        "judge",
        lambda model_fn, query, points, evidence, answer: {
            "covered_points": [], "faithful": True, "fabricated_points": [], "reason": "误判"
        },
    )

    result = judge_check.judge_ledger(SettingsStub(), status=None)

    assert result["judged"] == 1 and result["consistent"] == 0
    assert result["consistency_rate"] == 0.0


class SettingsStub:
    """judge_ledger 只把它透传给被 monkeypatch 的工厂与模型构造器。"""
