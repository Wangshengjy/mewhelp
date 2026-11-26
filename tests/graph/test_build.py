"""ch06 · 图组装:节点齐全 + 拓扑(五出口 + refund_flow 链)。"""
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.build import build_graph


def _edges(graph) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in graph.get_graph().edges}


def test_graph_has_all_nodes():
    graph = build_graph(checkpointer=InMemorySaver())
    nodes = set(graph.get_graph().nodes)
    assert {
        "coref", "classify_intent", "forced_rag", "confidence_check",
        "fetch_order", "retrieve_policy",
        "agent_llm", "agent_tools", "complaint_reply", "script_reply",
        "fallback_reply", "log",
    } <= nodes
    # ch06 退役:闲聊并入 script_reply
    assert "chitchat_reply" not in nodes


def test_topology_follows_course_spec():
    graph = build_graph(checkpointer=InMemorySaver())
    edges = _edges(graph)
    # 入口与意图分流(八类 → 五出口)
    assert ("__start__", "coref") in edges
    assert ("coref", "classify_intent") in edges
    assert ("classify_intent", "forced_rag") in edges
    assert ("classify_intent", "fetch_order") in edges  # 退款退货 → 人工接管前自动链
    assert ("classify_intent", "agent_llm") in edges
    assert ("classify_intent", "complaint_reply") in edges
    assert ("classify_intent", "script_reply") in edges
    # 退款链:抓订单 → 检索政策 → 带证据生成
    assert ("fetch_order", "retrieve_policy") in edges
    assert ("retrieve_policy", "agent_llm") in edges
    # 知识路:强制检索 → 证据闸 → strong→agent_llm / weak→fallback
    assert ("forced_rag", "confidence_check") in edges
    assert ("confidence_check", "agent_llm") in edges
    assert ("confidence_check", "fallback_reply") in edges
    # ReAct 环
    assert ("agent_llm", "agent_tools") in edges
    assert ("agent_tools", "agent_llm") in edges
    # 五出口全部 → log → END
    assert ("agent_llm", "log") in edges
    assert ("complaint_reply", "log") in edges
    assert ("script_reply", "log") in edges
    assert ("fallback_reply", "log") in edges
    assert ("log", "__end__") in edges


def test_build_graph_accepts_checkpointer():
    graph = build_graph(checkpointer=InMemorySaver())
    assert graph.checkpointer is not None
