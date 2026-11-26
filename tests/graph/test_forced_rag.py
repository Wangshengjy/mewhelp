"""ch05 · 知识路强制检索节点 forced_rag(mock 检索,测两道闸与证据组装)。"""
import asyncio

from langchain_core.messages import HumanMessage

from app.graph import nodes
from app.models import LowConfidenceQuestion
from app.repositories import conversations as conversations_repository
from tests.graph.conftest import cfg


def _conversation_id(graph_deps) -> str:
    """low_confidence_questions 对会话有外键:先造一条真实会话。"""
    with graph_deps.session_factory() as session:
        conversation = conversations_repository.create_conversation(session)
        session.commit()
        return conversation.id


def _aval(value):
    async def _f(*a, **k):
        return value
    return _f


def _hit(**overrides):
    hit = {
        "id": 1,
        "question": "退货政策",
        "answer": "7天无理由",
        "rerank_score": 0.9,
        "section_path": "政策/退货",
        "content_type": "policy",
        "category": "售后",
        "score": 0.5,
    }
    hit.update(overrides)
    return hit


def test_forced_rag_strong_builds_evidence(graph_deps, monkeypatch):
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: [_hit()])
    monkeypatch.setattr(nodes.kb_retrieval, "arrange_head_tail", lambda hits: list(hits))
    monkeypatch.setattr(
        nodes.selfcheck_module, "check_sufficient", lambda *a, **k: {"useful": True, "reason": ""}
    )
    out = asyncio.run(
        nodes.forced_rag({"messages": [HumanMessage("退货政策是什么")]}, cfg(graph_deps))
    )
    assert out["evidence_strong"] is True
    assert "[1]" in out["evidence"]
    assert out["citations"][0]["n"] == 1
    assert out["citations"][0]["chunk_id"] == 1
    assert out["trace"]["forced_rag"] is True
    assert out["trace"]["gate"] == "pass"
    assert out["trace"]["evidence_top"] == 0.9


def test_forced_rag_weak_when_no_hits_records_retrieval_low_conf(graph_deps, monkeypatch):
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: [])
    conversation_id = _conversation_id(graph_deps)
    out = asyncio.run(
        nodes.forced_rag(
            {"messages": [HumanMessage("怎么注销账号")], "conversation_id": conversation_id},
            cfg(graph_deps),
        )
    )
    assert out["evidence_strong"] is False
    assert out["trace"]["gate"] == "mechanical"
    assert out["trace"]["evidence_top"] == 0.0
    with graph_deps.session_factory() as session:
        row = session.query(LowConfidenceQuestion).one()
    assert row.source == "retrieval_low_conf"
    assert row.raw_question == "怎么注销账号"
    assert row.conversation_id == conversation_id


def test_forced_rag_weak_when_below_threshold(graph_deps, monkeypatch):
    """rerank 0.1 的弱证据:0.17 总分(<0.5)应被拦。ch09 校准阈值 0.138 下
    该分数段属「过闸但低」,故此用例显式用 0.5 严阈值钉住「低于阈值拦截」语义。"""
    from dataclasses import replace

    from app.graph.deps import GraphDeps

    strict = GraphDeps(
        session_factory=graph_deps.session_factory,
        settings=replace(graph_deps.settings, evidence_confidence_threshold=0.5),
    )
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: [_hit(rerank_score=0.1)])
    out = asyncio.run(
        nodes.forced_rag({"messages": [HumanMessage("怎么注销账号")]}, cfg(strict))
    )
    assert out["evidence_strong"] is False
    assert "低于阈值" in out["trace"]["gate_reason"]


def test_forced_rag_weak_when_selfcheck_rejects_records_self_check(graph_deps, monkeypatch):
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: [_hit()])
    monkeypatch.setattr(
        nodes.selfcheck_module,
        "check_sufficient",
        lambda *a, **k: {"useful": False, "reason": "证据只讲退货,没讲注销"},
    )
    conversation_id = _conversation_id(graph_deps)
    out = asyncio.run(
        nodes.forced_rag(
            {"messages": [HumanMessage("怎么注销账号")], "conversation_id": conversation_id},
            cfg(graph_deps),
        )
    )
    assert out["evidence_strong"] is False
    assert out["trace"]["gate"] == "self_check"
    assert out["citations"] == []
    with graph_deps.session_factory() as session:
        row = session.query(LowConfidenceQuestion).one()
    assert row.source == "self_check"
    assert "注销" in row.reason


def test_forced_rag_degrades_when_retrieval_unconfigured(graph_deps, monkeypatch):
    def raise_unconfigured(*a, **k):
        raise nodes.kb_retrieval.RetrievalNotConfiguredError("no milvus")

    degraded = nodes.kb_retrieval.RetrievalResult(
        matches=[{"id": 9, "question": "退货", "answer": "7天", "category": "售后"}],
        degraded=True,
        note="已降级",
    )
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", raise_unconfigured)
    monkeypatch.setattr(nodes.kb_retrieval, "retrieve", lambda *a, **k: degraded)
    out = asyncio.run(
        nodes.forced_rag({"messages": [HumanMessage("退货政策")]}, cfg(graph_deps))
    )
    assert out["evidence_strong"] is True
    assert out["trace"]["gate"] == "degraded"
    assert out["citations"][0]["chunk_id"] == 9
    with graph_deps.session_factory() as session:
        # ch04 行为:降级路径不落低置信池
        assert session.query(LowConfidenceQuestion).count() == 0


def test_confidence_check_traces_decision(graph_deps):
    config = cfg(graph_deps)
    assert asyncio.run(nodes.confidence_check({"evidence_strong": True}, config))["trace"]["confidence"] == "strong"
    assert asyncio.run(nodes.confidence_check({"evidence_strong": False}, config))["trace"]["confidence"] == "weak"
