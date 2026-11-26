"""ch05/ch06 · 确定性出口节点(complaint / fallback / script)。"""
import asyncio

from langchain_core.messages import HumanMessage

from app.graph import nodes
from app.models import LowConfidenceQuestion
from tests.graph.conftest import cfg


def test_complaint_reply_offers_two_actions():
    out = asyncio.run(
        nodes.complaint_reply({"messages": [HumanMessage("我要投诉你们")], "conversation_id": "c-7"})
    )
    assert out["answer"] == nodes.COMPLAINT_REPLY
    types = [action["type"] for action in out["suggested_actions"]]
    assert types == ["transfer_human", "create_ticket"]
    ticket = next(a for a in out["suggested_actions"] if a["type"] == "create_ticket")
    assert ticket["draft"]["ticket_type"] == "投诉"
    assert ticket["draft"]["description"] == "我要投诉你们"


def test_fallback_reply_fixed_and_does_not_double_record(graph_deps):
    """低置信落池归 forced_rag 两道闸(带 reason/evidence_top);fallback 只出话术,
    不再重复落池——同一问题一次对话只该在低置信池里留两道闸各自的痕。"""
    out = asyncio.run(
        nodes.fallback_reply(
            {"messages": [HumanMessage("怎么注销账号")], "trace": {"evidence_top": 0.1}},
            cfg(graph_deps),
        )
    )
    assert out["answer"] == nodes.FALLBACK_REPLY
    assert out["trace"]["route"] == "fallback"
    with graph_deps.session_factory() as session:
        assert session.query(LowConfidenceQuestion).count() == 0
