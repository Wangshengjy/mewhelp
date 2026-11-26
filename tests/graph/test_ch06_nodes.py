"""ch06 · 图节点单测:意图 v2 落态 / coref 改写 / fetch_order 中断 / retrieve_policy / script_reply。

monkeypatch 一律打在 nodes 模块引用上(与 ch05 test_forced_rag 同风格),
零上游模型调用。
"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes
from app.prompts import SCRIPT_REPLY_CHITCHAT
from tests.graph.conftest import cfg


# ── classify_intent ──────────────────────────────────────────────────────────


def test_classify_intent_writes_confidence_and_route(monkeypatch, graph_deps):
    async def fake_classify(query, settings, chat_model_factory=None, history=""):
        assert query == "蓝牙耳机还能申请退货吗"  # 吃 coref 改写后的 resolved_query
        assert "订单1001到哪了" in history  # 多轮历史注入
        return {"intent": "退款退货", "confidence": 0.83}

    monkeypatch.setattr(nodes, "classify_intent_label", fake_classify)
    state = {
        "messages": [
            HumanMessage("订单1001到哪了"),
            AIMessage("已发货"),
            HumanMessage("蓝牙耳机那个"),
        ],
        "resolved_query": "蓝牙耳机还能申请退货吗",
    }
    out = asyncio.run(nodes.classify_intent(state, cfg(graph_deps)))
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.83
    assert out["route"] == "refund_flow"
    assert out["trace"]["intent_confidence"] == 0.83


def test_history_text_keeps_recent_turns_only():
    state = {
        "messages": [
            HumanMessage("第1句"),
            AIMessage("答1"),
            HumanMessage("第2句"),
            AIMessage("答2"),
            HumanMessage("第3句"),
            AIMessage("答3"),
            HumanMessage("第4句"),
        ]
    }
    text = nodes._history_text(state)
    assert "第1句" not in text  # 只保留最近 6 条
    assert "答1" in text and "第2句" in text and "第4句" in text
    assert "用户:" in text and "客服:" in text


# ── coref ────────────────────────────────────────────────────────────────────


def test_coref_writes_resolved_query(monkeypatch, graph_deps):
    async def fake_resolve(query, history, settings, chat_model_factory=None):
        assert query == "这个能退吗"
        return "蓝牙耳机还能申请退货吗"

    monkeypatch.setattr(nodes, "coref_resolver", fake_resolve)
    state = {"messages": [HumanMessage("蓝牙耳机不错"), AIMessage("已下单"), HumanMessage("这个能退吗")]}
    out = asyncio.run(nodes.coref(state, cfg(graph_deps)))
    assert out["resolved_query"] == "蓝牙耳机还能申请退货吗"
    assert out["trace"]["coref"] == "rewrite"


def test_coref_passthrough_keeps_original_query(monkeypatch, graph_deps):
    async def fake_resolve(query, history, settings, chat_model_factory=None):
        return query  # 无指代,原样返回

    monkeypatch.setattr(nodes, "coref_resolver", fake_resolve)
    state = {"messages": [HumanMessage("发票怎么开")], "query": "发票怎么开"}
    out = asyncio.run(nodes.coref(state, cfg(graph_deps)))
    assert out["resolved_query"] == "发票怎么开"
    assert out["trace"]["coref"] == "passthrough"


# ── fetch_order ──────────────────────────────────────────────────────────────


class _Interrupted(Exception):
    """nodes.interrupt 被 monkeypatch 成抛这个,捕获载荷。"""

    def __init__(self, value):
        self.value = value


def test_extract_order_id_finds_four_digit_ids():
    assert nodes._extract_order_id("订单1001的物流") == "1001"
    assert nodes._extract_order_id("订单 2002 这个能退吗") == "2002"
    assert nodes._extract_order_id("尾号20260701那单") == "20260701"
    assert nodes._extract_order_id("帮我查个快递") is None
    assert nodes._extract_order_id("单号12345678") == "12345678"  # 4+ 位数字串都算锚点


def test_fetch_order_uses_query_order_id(graph_deps):
    state = {"resolved_query": "订单1001能退吗"}
    out = asyncio.run(nodes.fetch_order(state, cfg(graph_deps)))
    assert out["order_id"] == "1001"
    assert out["order_data"]["order_id"] == "1001"
    assert "fetch_order" in out["trace"]


def test_fetch_order_interrupts_with_user_orders(monkeypatch, graph_deps):
    captured: dict = {}

    def fake_interrupt(value):
        captured["value"] = value
        return "1001"  # resume 后被中断节点重跑,interrupt 返回所选订单号

    monkeypatch.setattr(nodes, "interrupt", fake_interrupt)
    state = {"resolved_query": "我要退款", "user_id": "user-ch06"}
    out = asyncio.run(nodes.fetch_order(state, cfg(graph_deps)))
    payload = captured["value"]
    assert payload["type"] == "select_order"
    orders = payload["orders"]
    assert 1 <= len(orders) <= 4
    assert {"order_id", "product", "status", "amount"} <= set(orders[0])
    # resume 值被选中,节点继续把快照写进 State
    assert out["order_id"] == "1001"
    assert out["order_data"]["order_id"] == "1001"


# ── retrieve_policy ──────────────────────────────────────────────────────────


def test_retrieve_policy_merges_expanded_queries(monkeypatch, graph_deps):
    async def fake_expand(query, settings=None, chat_model_factory=None):
        return ["退款政策", "七天无理由", "蓝牙耳机退货"]

    seen_queries: list[str] = []

    def fake_search(session_factory, settings, q, **kwargs):
        seen_queries.append(q)
        if q == "退款政策":
            return [{"id": "kb-1", "question": "退货政策", "answer": "七天", "section_path": "售后/退货", "rerank_score": 0.9}]
        if q == "七天无理由":
            return [{"id": "kb-1", "question": "退货政策", "answer": "七天", "section_path": "售后/退货", "rerank_score": 0.8},
                    {"id": "kb-2", "question": "无理由时限", "answer": "7天", "section_path": "售后/时限", "rerank_score": 0.85}]
        return [{"id": "kb-3", "question": "耳机退货", "answer": "可以", "section_path": "售后/耳机", "rerank_score": 0.7}]

    monkeypatch.setattr(nodes, "query_expander", fake_expand)
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", fake_search)
    state = {"resolved_query": "蓝牙耳机能退吗", "order_data": {"order_id": "1001", "status": "shipped"}}
    out = asyncio.run(nodes.retrieve_policy(state, cfg(graph_deps)))
    assert seen_queries == ["退款政策", "七天无理由", "蓝牙耳机退货"]  # 只搜扩展查询
    # kb-1 在两路命中,去重保留高分;首尾组装后编号连续
    assert out["citations"][0]["n"] == 1
    assert len({c["chunk_id"] for c in out["citations"]}) == len(out["citations"])
    assert "[1]" in out["evidence"]
    assert out["trace"]["retrieve_policy"]["queries"] == ["退款政策", "七天无理由", "蓝牙耳机退货"]


def test_retrieve_policy_survives_search_errors(monkeypatch, graph_deps):
    async def fake_expand(query, settings=None, chat_model_factory=None):
        return ["q1", "q2"]

    calls: list[str] = []

    def fake_search(session_factory, settings, q, **kwargs):
        calls.append(q)
        if q == "q1":
            raise RuntimeError("retrieval down")
        return [{"id": "kb-9", "question": "仍可用", "answer": "一路", "section_path": "售后", "rerank_score": 0.6}]

    monkeypatch.setattr(nodes, "query_expander", fake_expand)
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", fake_search)
    out = asyncio.run(nodes.retrieve_policy({"resolved_query": "能退吗"}, cfg(graph_deps)))
    assert len(calls) == 2
    assert out["evidence"]


# ── script_reply ─────────────────────────────────────────────────────────────


def test_script_reply_uses_intent_specific_script():
    out = asyncio.run(nodes.script_reply({"intent": "闲聊"}))
    assert out["answer"] == SCRIPT_REPLY_CHITCHAT


def test_script_reply_other_gets_generic_script():
    out = asyncio.run(nodes.script_reply({"intent": "其他"}))
    assert out["answer"] != SCRIPT_REPLY_CHITCHAT
    assert out["trace"]["route"] == "fallback_script"
