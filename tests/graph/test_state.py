"""ch05 · ConversationState + reducer 接线测试。"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.state import ConversationState, merge_dict


def test_merge_dict_merges_right_over_left():
    assert merge_dict({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}


def test_merge_dict_tolerates_none():
    assert merge_dict(None, {"a": 1}) == {"a": 1}
    assert merge_dict({"a": 1}, None) == {"a": 1}
    assert merge_dict(None, None) == {}


def test_state_reducers_wired_into_langgraph():
    """trace 走 merge_dict(两节点痕迹都保留)、messages 走 add_messages(追加)。"""

    def node_a(state: ConversationState) -> dict:
        return {"trace": {"a": 1}, "messages": [HumanMessage("hi")]}

    def node_b(state: ConversationState) -> dict:
        return {"trace": {"b": 2}, "messages": [AIMessage("hello")]}

    builder = StateGraph(ConversationState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    final = asyncio.run(graph.ainvoke({}, {"configurable": {"thread_id": "t1"}}))
    assert final["trace"] == {"a": 1, "b": 2}
    assert [m.type for m in final["messages"]] == ["human", "ai"]
    assert final["messages"][1].content == "hello"
