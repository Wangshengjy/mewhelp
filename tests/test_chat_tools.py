"""ch05 · 真图端到端(知识路/投诉/工具路):两道闸、编号证据、落池、审计与建议动作。

旧编排(查询理解→工具→流式)已被 LangGraph 图取代:这里把 ch04 的工具路测试
改造为「测图」——Fake 模型 + mock 检索 + 真实 SQLite,零上游。
SSE 帧契约在 tests/test_chat.py;事件映射在 tests/graph/test_runtime.py。
"""

import asyncio
import copy

import pytest
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes, runtime
from app.models import LowConfidenceQuestion, Message, Ticket
from app.prompts import COMPLAINT_REPLY_TEXT, FALLBACK_REPLY_TEXT
from app.settings import Settings

_HIT = {
    "id": 1,
    "question": "退货政策",
    "answer": "7天无理由",
    "rerank_score": 0.9,
    "section_path": "政策/退货",
    "content_type": "policy",
    "category": "售后",
    "score": 0.5,
}


class RecordingChatModel:
    """structured_output→固定标签;bind_tools→分步脚本 + 记录收到的消息。"""

    def __init__(self, intent_label: str, script):
        self._label = intent_label
        self._script = list(script)
        self._bound_calls = 0
        self.decision_calls = []

    def with_structured_output(self, schema, method=None):
        outer = self

        class _Structured:
            async def ainvoke(self, messages):
                return schema(**{"intent": outer._label})

        return _Structured()

    def bind_tools(self, tools, **kwargs):
        outer = self

        class _Bound:
            async def ainvoke(self, messages, config=None):
                outer.decision_calls.append(list(messages))
                step = min(outer._bound_calls, len(outer._script) - 1)
                outer._bound_calls += 1
                # 每次新对象:同 id 会被 add_messages 去重
                return copy.deepcopy(outer._script[step])

        return _Bound()


@pytest.fixture
def graph_runtime(session_factory):
    """启动图(InMemorySaver);chat_model_factory 由各测试注入。"""

    async def _init(chat_model_factory):
        await runtime.init_graph(
            session_factory=session_factory,
            settings=Settings.hermetic_defaults(),
            checkpointer=InMemorySaver(),
            chat_model_factory=chat_model_factory,
        )

    def _start(chat_model_factory):
        asyncio.run(_init(chat_model_factory))

    yield _start
    asyncio.run(runtime.close_graph())


def _make_model(label, script):
    model = RecordingChatModel(label, script)
    return model, (lambda settings: model)


def _run_turn(message, conversation_id=None):
    async def _scenario():
        new_id, _created = await runtime.prepare_turn(conversation_id, message)
        events = [event async for event in runtime.stream_turn(new_id, message)]
        return new_id, events

    return asyncio.run(_scenario())


def _audit_rows(session_factory, conversation_id):
    with session_factory() as session:
        return (
            session.query(Message)
            .filter_by(conversation_id=conversation_id)
            .order_by(Message.id)
            .all()
        )


def test_knowledge_route_injects_numbered_evidence(graph_runtime, session_factory, monkeypatch):
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: [dict(_HIT)])
    monkeypatch.setattr(nodes.kb_retrieval, "arrange_head_tail", lambda hits: list(hits))
    monkeypatch.setattr(
        nodes.selfcheck_module, "check_sufficient", lambda *a, **k: {"useful": True, "reason": ""}
    )
    model, factory = _make_model("商品咨询", [AIMessage("根据政策[1],支持7天无理由退货。")])
    graph_runtime(factory)

    conversation_id, _events = _run_turn("退货政策是什么")

    # ch07:编号证据进了用户侧 turn_ctx(「别再调 query_faq」提示随行),system 只剩人设
    first_decision = model.decision_calls[0]
    assert isinstance(first_decision[0], SystemMessage)
    turn_ctx = first_decision[-1].content
    assert "[1]" in turn_ctx
    assert "7天无理由" in turn_ctx
    assert "query_faq" in turn_ctx
    # 审计:user + assistant(最终答复)
    rows = _audit_rows(session_factory, conversation_id)
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[1].content == "根据政策[1],支持7天无理由退货。"
    # 双闸都过,低置信池无行
    with session_factory() as session:
        assert session.query(LowConfidenceQuestion).count() == 0


def test_knowledge_route_mechanical_weak_falls_back_and_records(graph_runtime, session_factory, monkeypatch):
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: [])
    model, factory = _make_model("商品咨询", [AIMessage("unused")])
    graph_runtime(factory)

    conversation_id, events = _run_turn("怎么注销账号")

    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == [FALLBACK_REPLY_TEXT]
    assert not [event for event in events if event["type"] == "citations"]
    with session_factory() as session:
        row = session.query(LowConfidenceQuestion).one()
    assert row.source == "retrieval_low_conf"
    assert row.raw_question == "怎么注销账号"
    rows = _audit_rows(session_factory, conversation_id)
    assert rows[1].content == FALLBACK_REPLY_TEXT
    # 模型从未被调起(证据闸就拦下了)
    assert model.decision_calls == []


def test_knowledge_route_selfcheck_weak_records_self_check(graph_runtime, session_factory, monkeypatch):
    monkeypatch.setattr(nodes.kb_retrieval, "search_knowledge", lambda *a, **k: [dict(_HIT)])
    monkeypatch.setattr(
        nodes.selfcheck_module,
        "check_sufficient",
        lambda *a, **k: {"useful": False, "reason": "证据只讲退货,没讲注销"},
    )
    model, factory = _make_model("商品咨询", [AIMessage("unused")])
    graph_runtime(factory)

    _conversation_id, _events = _run_turn("怎么注销账号")

    with session_factory() as session:
        row = session.query(LowConfidenceQuestion).one()
    assert row.source == "self_check"
    assert model.decision_calls == []


def test_complaint_route_offers_transfer_and_ticket(graph_runtime, session_factory):
    _model, factory = _make_model("投诉", [AIMessage("unused")])
    graph_runtime(factory)

    conversation_id, events = _run_turn("你们这服务太差了,我要投诉!")

    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == [COMPLAINT_REPLY_TEXT]
    actions = [event for event in events if event["type"] == "actions"][0]["items"]
    assert [action["type"] for action in actions] == ["transfer_human", "create_ticket"]
    ticket = actions[1]
    assert ticket["draft"]["ticket_type"] == "投诉"
    assert ticket["draft"]["description"] == "你们这服务太差了,我要投诉!"
    rows = _audit_rows(session_factory, conversation_id)
    assert [row.role for row in rows] == ["user", "assistant"]
    with session_factory() as session:
        assert session.query(Ticket).count() == 0  # 只是提议,没有写库


def test_unknown_tool_returns_failed_frame_and_model_recovers(graph_runtime, session_factory):
    script = [
        AIMessage("", tool_calls=[{"name": "no_such_tool", "args": {}, "id": "t1"}]),
        AIMessage("抱歉,这个我处理不了,已为您记录。"),
    ]
    _model, factory = _make_model("订单", script)
    graph_runtime(factory)

    _conversation_id, events = _run_turn("给我发个红包")

    tools = [event for event in events if event["type"] == "tool"]
    assert [(t["name"], t["status"]) for t in tools] == [
        ("no_such_tool", "running"),
        ("no_such_tool", "failed"),
    ]
