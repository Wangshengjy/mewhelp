"""ch09 闸升级：置信度低→retrieval_low_conf；置信度过但自评不过→self_check；
快照只要走了检索就写 state；forced_rag 两道闸落池都带召回快照。"""
import asyncio

import pytest
from langchain_core.messages import HumanMessage

from app.graph import nodes
from app.models import LowConfidenceQuestion
from app.repositories import conversations as conversations_repository
from tests.graph.conftest import cfg


def _conversation_id(graph_deps) -> str:
    """low_confidence_questions 对会话有外键：先造一条真实会话。"""
    with graph_deps.session_factory() as session:
        conversation = conversations_repository.create_conversation(session)
        session.commit()
        return conversation.id


def _mk_hits(*scores):
    return [{"id": i, "question": f"q{i} 退货", "answer": f"a{i} 运费", "rerank_score": s,
             "section_path": "s", "content_type": "faq", "category": "售后"}
            for i, s in enumerate(scores)]


@pytest.fixture
def stub_retrieval(monkeypatch):
    def set_hits(hits):
        monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: hits)
        monkeypatch.setattr(nodes.kb_retrieval, "arrange_head_tail", lambda hits_: list(hits_))
    return set_hits


@pytest.fixture
def stub_selfcheck(monkeypatch):
    def set_result(useful, reason=""):
        monkeypatch.setattr(
            nodes.selfcheck_module, "check_sufficient",
            lambda *a, **k: {"useful": useful, "reason": reason},
        )
    return set_result


def _threshold(graph_deps, value):
    """hermetic settings 是 frozen dataclass：换带新阈值的 GraphDeps 配置。"""
    from app.graph.deps import GraphDeps
    from dataclasses import replace
    return cfg(GraphDeps(session_factory=graph_deps.session_factory,
                         settings=replace(graph_deps.settings, evidence_confidence_threshold=value)))


def test_low_confidence_blocks_with_source_and_snapshot(graph_deps, stub_retrieval, stub_selfcheck):
    config = _threshold(graph_deps, 0.5)
    stub_retrieval(_mk_hits(0.15, 0.14))          # 弱证据 → 置信度闸拦下
    stub_selfcheck(True)
    out = asyncio.run(
        nodes.forced_rag({"messages": [HumanMessage("猫窝能水洗吗")]}, config)
    )
    assert out["evidence_strong"] is False
    assert out["trace"]["gate"] == "mechanical"
    assert out["retrieved_snapshot"]              # 走了检索就有快照
    assert out["evidence_confidence"] < 0.5
    assert "confidence_signals" in out["trace"]


def test_selfcheck_fail_labels_self_check(graph_deps, stub_retrieval, stub_selfcheck):
    config = _threshold(graph_deps, 0.3)
    stub_retrieval(_mk_hits(0.9, 0.4, 0.35))
    stub_selfcheck(False, "缺关键信息")
    out = asyncio.run(
        nodes.forced_rag({"messages": [HumanMessage("怎么注销账号")]}, config)
    )
    assert out["evidence_strong"] is False
    assert out["trace"]["gate"] == "self_check"
    assert out["retrieved_snapshot"]              # 自评不过同样有快照


def test_strong_evidence_keeps_snapshot_for_feedback(graph_deps, stub_retrieval, stub_selfcheck):
    config = _threshold(graph_deps, 0.3)
    stub_retrieval(_mk_hits(0.9, 0.4, 0.35))
    stub_selfcheck(True)
    out = asyncio.run(
        nodes.forced_rag({"messages": [HumanMessage("退货政策是什么")]}, config)
    )
    assert out["evidence_strong"] is True
    assert len(out["retrieved_snapshot"]) == 3    # 闸都过也留快照，供事后 👎 回捞
    assert out["citations"]


def test_both_gates_persist_snapshot_to_pool(graph_deps, stub_retrieval, stub_selfcheck):
    """两道闸落池都把快照随条写进 retrieved_chunks 列。"""
    conversation_id = _conversation_id(graph_deps)
    config = _threshold(graph_deps, 0.9)  # 高阈值：0.9 分的证据也被拦
    stub_retrieval(_mk_hits(0.6, 0.55))
    stub_selfcheck(True)
    asyncio.run(nodes.forced_rag(
        {"messages": [HumanMessage("怎么注销账号")], "conversation_id": conversation_id}, config))
    config_pass = _threshold(graph_deps, 0.1)
    stub_retrieval(_mk_hits(0.9, 0.85))
    stub_selfcheck(False, "证据只讲退货,没讲注销")
    asyncio.run(nodes.forced_rag(
        {"messages": [HumanMessage("怎么注销账号")], "conversation_id": conversation_id}, config_pass))
    with graph_deps.session_factory() as session:
        rows = session.query(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id).all()
    assert [r.source for r in rows] == ["retrieval_low_conf", "self_check"]
    for row in rows:
        assert row.retrieved_chunks and row.retrieved_chunks[0]["rerank_score"] > 0
        assert "evidence_confidence=" in row.reason
        assert "signals=" in row.reason


def test_self_check_reason_carries_verdict(graph_deps, stub_retrieval, stub_selfcheck):
    conversation_id = _conversation_id(graph_deps)
    config = _threshold(graph_deps, 0.1)
    stub_retrieval(_mk_hits(0.9, 0.85))
    stub_selfcheck(False, "证据只讲退货,没讲注销")
    asyncio.run(nodes.forced_rag(
        {"messages": [HumanMessage("怎么注销账号")], "conversation_id": conversation_id}, config))
    with graph_deps.session_factory() as session:
        row = session.query(LowConfidenceQuestion).one()
    assert row.source == "self_check"
    assert "注销" in row.reason


def test_no_hits_still_pools_without_snapshot(graph_deps, stub_retrieval):
    conversation_id = _conversation_id(graph_deps)
    stub_retrieval([])
    asyncio.run(nodes.forced_rag(
        {"messages": [HumanMessage("怎么注销账号")], "conversation_id": conversation_id},
        cfg(graph_deps)))
    with graph_deps.session_factory() as session:
        row = session.query(LowConfidenceQuestion).one()
    assert row.source == "retrieval_low_conf"
    assert row.retrieved_chunks is None           # 没走有效检索存 NULL
