"""ch05/ch06 · StateGraph 组装:五出口拓扑(ch06)。

    START → coref → classify_intent ─┬ knowledge     → forced_rag → confidence_check
                                     │                 ├ strong → agent_llm ↔ agent_tools
                                     │                 └ weak   → fallback_reply ──┐
                                     ├ refund_flow   → fetch_order → retrieve_policy
                                     │                 → agent_llm(带证据判定)      │
                                     ├ business      → agent_llm(ReAct 环收敛后)    │
                                     ├ complaint     → complaint_reply ────────────┤
                                     └ fallback_script → script_reply ─────────────┴→ log → END
"""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


def _builder() -> StateGraph:
    builder = StateGraph(ConversationState)
    builder.add_node("coref", nodes.coref)
    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("forced_rag", nodes.forced_rag)
    builder.add_node("confidence_check", nodes.confidence_check)
    builder.add_node("fetch_order", nodes.fetch_order)
    builder.add_node("retrieve_policy", nodes.retrieve_policy)
    builder.add_node("agent_llm", nodes.agent_llm)
    builder.add_node("agent_tools", nodes.agent_tools)
    builder.add_node("complaint_reply", nodes.complaint_reply)
    builder.add_node("script_reply", nodes.script_reply)
    builder.add_node("fallback_reply", nodes.fallback_reply)
    builder.add_node("log", nodes.log_node)

    # 入口与意图分流(条件边:八类意图 → 五出口,由 route_by_intent 决定)
    builder.add_edge(START, "coref")
    builder.add_edge("coref", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "knowledge": "forced_rag",
            "refund_flow": "fetch_order",
            "business": "agent_llm",
            "complaint": "complaint_reply",
            "fallback_script": "script_reply",
        },
    )

    # 退款链(ch06):抓订单(缺单号 interrupt 选单)→ 多路检索政策 → 带证据判定
    builder.add_edge("fetch_order", "retrieve_policy")
    builder.add_edge("retrieve_policy", "agent_llm")

    # 知识路:强制检索 → 生成前证据闸 → 强则带证据生成 / 弱则兜底
    builder.add_edge("forced_rag", "confidence_check")
    builder.add_conditional_edges(
        "confidence_check",
        confidence_gate,
        {"strong": "agent_llm", "weak": "fallback_reply"},
    )

    # 主力 Agent:ReAct 环(有 tool_calls 且未触停 → 执行工具 → 回到推理步;
    # 无 tool_calls 收敛 / 步数封顶 → 日志审计收尾)
    builder.add_conditional_edges(
        "agent_llm",
        should_continue,
        {"continue": "agent_tools", "stop": "log"},
    )
    builder.add_edge("agent_tools", "agent_llm")

    # 五出口全部 → log → END(单出口收尾,审计一处补齐)
    builder.add_edge("complaint_reply", "log")
    builder.add_edge("script_reply", "log")
    builder.add_edge("fallback_reply", "log")
    builder.add_edge("log", END)
    return builder


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    """编译图;checkpointer 由 runtime 提供(生产 AsyncSqliteSaver / 测试 InMemorySaver)。"""
    return _builder().compile(checkpointer=checkpointer)
