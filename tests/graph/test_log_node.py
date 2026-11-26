"""ch05 · 日志节点(resolve_answer + 消息审计)。"""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes
from app.models import Message
from app.repositories import conversations as conversations_repository
from tests.graph.conftest import cfg


def test_resolve_answer_prefers_explicit():
    assert nodes.resolve_answer({"answer": "固定话术", "messages": []}) == "固定话术"


def test_resolve_answer_falls_back_to_last_ai():
    state = {"messages": [HumanMessage("hi"), AIMessage("模型答复")]}
    assert nodes.resolve_answer(state) == "模型答复"


def test_resolve_answer_empty_state():
    assert nodes.resolve_answer({}) == ""


def test_log_node_persists_assistant(graph_deps):
    config = cfg(graph_deps)
    with graph_deps.session_factory() as session:
        conversation = conversations_repository.create_conversation(session)
        conversation_id = conversation.id
        session.commit()

    asyncio.run(
        nodes.log_node(
            {
                "conversation_id": conversation_id,
                "intent": "订单",
                "route": "business",
                "messages": [AIMessage("订单已发货")],
                "trace": {"forced_rag": False},
            },
            config,
        )
    )

    with graph_deps.session_factory() as session:
        rows = session.query(Message).filter_by(conversation_id=conversation_id).all()
    assert len(rows) == 1
    assert rows[0].role == "assistant"
    assert rows[0].content == "订单已发货"


def test_log_node_empty_answer_persists_empty_string(graph_deps):
    """内容列非空约束:空答复落空串,不落 NULL(与 ch04 审计一致)。"""
    with graph_deps.session_factory() as session:
        conversation = conversations_repository.create_conversation(session)
        conversation_id = conversation.id
        session.commit()

    asyncio.run(nodes.log_node({"conversation_id": conversation_id, "messages": []}, cfg(graph_deps)))

    with graph_deps.session_factory() as session:
        rows = session.query(Message).filter_by(conversation_id=conversation_id).all()
    assert len(rows) == 1 and rows[0].content == ""
