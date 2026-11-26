"""ch05/ch06 · runtime:生命周期 / prepare_turn / stream_turn 事件映射 / interrupt-resume / 真图端到端。"""
import asyncio
import copy
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import runtime
from app.graph.deps import GraphDeps
from app.models import Message
from app.prompts import SCRIPT_REPLY_CHITCHAT
from app.repositories import conversations as conversations_repository
from app.settings import Settings
from tests.graph.conftest import cfg


def _patch_runtime_graph(monkeypatch, chunks):
    """事件映射测试:同时桩掉图与依赖(绕过单例生命周期);分层字段不触库。"""
    fake_deps = GraphDeps(session_factory=None, settings=Settings.hermetic_defaults())
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph(chunks))
    monkeypatch.setattr(runtime, "get_deps", lambda: fake_deps)

    async def _fake_ensure(conversation_id):
        return "", 0, 0, 0

    monkeypatch.setattr(runtime, "_ensure_conversation", _fake_ensure)


# ── 生命周期 ─────────────────────────────────────────────────────────────────


def test_get_graph_raises_before_init():
    runtime._graph = None
    try:
        runtime.get_graph()
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_init_get_close_graph(session_factory, settings):
    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory, settings=settings, checkpointer=InMemorySaver()
        )
        try:
            assert runtime.get_graph() is not None
        finally:
            await runtime.close_graph()
        try:
            runtime.get_graph()
            raised = False
        except RuntimeError:
            raised = True
        assert raised

    asyncio.run(_scenario())


# ── prepare_turn ─────────────────────────────────────────────────────────────


def test_prepare_turn_creates_conversation_and_user_row(session_factory, settings):
    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory, settings=settings, checkpointer=InMemorySaver()
        )
        try:
            conversation_id, created = await runtime.prepare_turn(None, "你好")
            assert created is True
            assert isinstance(conversation_id, str) and conversation_id
            return conversation_id
        finally:
            await runtime.close_graph()

    conversation_id = asyncio.run(_scenario())
    with session_factory() as session:
        rows = session.query(Message).filter_by(conversation_id=conversation_id).all()
    assert [row.role for row in rows] == ["user"]
    assert rows[0].content == "你好"


def test_prepare_turn_raises_for_missing_conversation(session_factory, settings):
    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory, settings=settings, checkpointer=InMemorySaver()
        )
        raised = False
        try:
            await runtime.prepare_turn("no-such-conversation", "你好")
        except runtime.ConversationNotFound:
            raised = True
        finally:
            await runtime.close_graph()
        assert raised
        # 不存在时不落任何 user 行
        with session_factory() as session:
            assert session.query(Message).count() == 0

    asyncio.run(_scenario())


# ── ch06:每轮图输入清零多轮字段 ──────────────────────────────────────────────


def test_graph_input_zeroes_ch06_fields():
    """resolved_query/intent_confidence/order_id/order_data 不跨轮复用,
    防上一轮的改写与订单语境漏进这一轮。"""
    inp = runtime._graph_input("c-1", "订单 1001 到哪了")
    assert inp["resolved_query"] == ""
    assert inp["intent_confidence"] == 0.0
    assert inp["order_id"] == ""
    assert inp["order_data"] == {}
    assert inp["query"] == "订单 1001 到哪了"
    assert inp["steps"] == 0 and inp["tokens_used"] == 0


# ── stream_turn 事件映射(FakeGraph 钉死形状) ─────────────────────────────────


class FakeGraph:
    def __init__(self, chunks):
        self._chunks = chunks

    def astream(self, graph_input, config, stream_mode=None):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _collect(aiter):
    async def _run():
        return [event async for event in aiter]

    return asyncio.run(_run())


# ── ch06:interrupt / resume ──────────────────────────────────────────────────


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


def test_stream_turn_interrupt_frame_stops_stream(monkeypatch):
    """interrupt 帧:吐出待选订单列表后本次流直接结束(无 done 帧)。"""
    chunks = [
        ("updates", {"fetch_order": None}),
        (
            "updates",
            {
                "__interrupt__": (
                    _FakeInterrupt({"type": "select_order", "orders": [{"order_id": "1001"}]}),
                )
            },
        ),
    ]
    _patch_runtime_graph(monkeypatch, chunks)
    events = _collect(runtime.stream_turn("c-9", "我要退款"))
    assert events == [
        {"type": "interrupt", "payload": {"type": "select_order", "orders": [{"order_id": "1001"}]}}
    ]


class FakeResumeGraph:
    def __init__(self):
        self.captured = None

    async def ainvoke(self, graph_input, config):
        self.captured = graph_input
        return {"messages": [], "__interrupt__": ()}


def test_resume_turn_sends_command_and_reports_no_interrupt(monkeypatch):
    fake_graph = FakeResumeGraph()
    fake_deps = GraphDeps(session_factory=None, settings=Settings.hermetic_defaults())
    monkeypatch.setattr(runtime, "get_graph", lambda: fake_graph)
    monkeypatch.setattr(runtime, "get_deps", lambda: fake_deps)
    monkeypatch.setattr(runtime, "_conversation_exists", lambda factory, cid: True)

    result = asyncio.run(runtime.resume_turn("c-9", "1001"))
    assert isinstance(fake_graph.captured, Command)
    assert fake_graph.captured.resume == "1001"
    assert result["interrupt"] is None
    assert result["state"]["messages"] == []


def test_stream_resume_maps_events(monkeypatch):
    chunks = [("updates", {"script_reply": {"answer": "已继续退款流程"}})]
    _patch_runtime_graph(monkeypatch, chunks)
    monkeypatch.setattr(runtime, "_conversation_exists", lambda factory, cid: True)
    events = _collect(runtime.stream_resume("c-9", "1001"))
    assert [event["type"] for event in events] == ["delta", "done"]
    assert events[0]["text"] == "已继续退款流程"


def test_real_graph_refund_flow_interrupts_then_resumes(session_factory, settings):
    """真图端到端:缺单号 → interrupt 待选列表;resume 选中后走完退款链。"""
    script = [AIMessage("已进入退款确认环节,请在前端表单确认。")]

    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory,
            settings=settings,
            checkpointer=InMemorySaver(),
            chat_model_factory=_fake_factory("退款退货", script),
        )
        try:
            first = await runtime.run_turn(None, "我要退款")
            payload = first["interrupt"]
            assert payload and payload["type"] == "select_order", "缺单号应 interrupt 出待选订单列表"
            orders = payload["orders"]
            assert {"order_id", "product", "status", "amount"} <= set(orders[0])
            resumed = await runtime.resume_turn(first["conversation_id"], orders[0]["order_id"])
            return orders, resumed
        finally:
            await runtime.close_graph()

    orders, resumed = asyncio.run(_scenario())
    assert resumed["interrupt"] is None
    assert resumed["state"]["order_id"] == orders[0]["order_id"]
    assert resumed["state"]["order_data"]["order_id"] == orders[0]["order_id"]
    assert resumed["state"]["messages"][-1].content == "已进入退款确认环节,请在前端表单确认。"


def test_stream_turn_maps_events(monkeypatch):
    chunks = [
        # R1:意图分类器等内部 LLM 调用的 token 不进 delta
        ("messages", (AIMessage("查快递"), {"langgraph_node": "classify_intent"})),
        ("messages", (AIMessage("您的包裹"), {"langgraph_node": "agent_llm"})),
        (
            "updates",
            {
                "agent_llm": {
                    "messages": [
                        AIMessage(
                            "",
                            tool_calls=[
                                {"name": "create_ticket", "args": {}, "id": "t9"},  # 拦截,不发 running
                                {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "t1"},
                            ],
                        )
                    ],
                    "steps": 1,
                    "tokens_used": 9,
                }
            },
        ),
        (
            "updates",
            {
                "agent_tools": {
                    "messages": [ToolMessage(content="{}", tool_call_id="t1", name="query_logistics")],
                }
            },
        ),
        ("messages", (AIMessage("已发货[1]"), {"langgraph_node": "agent_llm"})),
        (
            "updates",
            {
                "forced_rag": {
                    "citations": [
                        {
                            "n": 1, "chunk_id": 3, "title": "政策/物流", "question": "q",
                            "answer": "a", "content_type": "faq", "category": "物流", "score": 0.9,
                        }
                    ]
                }
            },
        ),
        (
            "updates",
            {"complaint_reply": {"answer": "安抚话术", "suggested_actions": [{"type": "transfer_human"}]}},
        ),
    ]
    _patch_runtime_graph(monkeypatch, chunks)

    events = _collect(runtime.stream_turn("c-1", "hi"))
    types = [event["type"] for event in events]
    # delta:两个 agent_llm token + 一个确定性节点整块;内部分类 token 被过滤
    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == ["您的包裹", "已发货[1]", "安抚话术"]
    # tool:running(agent_llm 更新,create_ticket 被跳过)+ completed(agent_tools 更新)
    tools = [event for event in events if event["type"] == "tool"]
    assert tools == [
        {"type": "tool", "tool_call_id": "t1", "name": "query_logistics", "status": "running"},
        {"type": "tool", "tool_call_id": "t1", "name": "query_logistics", "status": "completed"},
    ]
    # citations:流结束后按角标 resolve,只留被引用的 [1]
    citations = [event for event in events if event["type"] == "citations"]
    assert len(citations) == 1 and citations[0]["items"][0]["n"] == 1
    # actions 帧 + done 收尾
    actions = [event for event in events if event["type"] == "actions"]
    assert actions == [{"type": "actions", "items": [{"type": "transfer_human"}]}]
    assert types[-1] == "done"


def test_stream_turn_no_citations_when_answer_cites_nothing(monkeypatch):
    chunks = [
        ("messages", (AIMessage("随便答答"), {"langgraph_node": "agent_llm"})),
        (
            "updates",
            {
                "forced_rag": {
                    "citations": [
                        {"n": 1, "chunk_id": 3, "title": "t", "question": "q", "answer": "a",
                         "content_type": "faq", "category": "物流", "score": 0.9}
                    ]
                }
            },
        ),
    ]
    _patch_runtime_graph(monkeypatch, chunks)
    events = _collect(runtime.stream_turn("c-1", "hi"))
    assert not [event for event in events if event["type"] == "citations"]
    assert [event["type"] for event in events] == ["delta", "done"]


def test_stream_turn_failed_tool_maps_to_failed(monkeypatch):
    chunks = [
        (
            "updates",
            {"agent_llm": {"messages": [AIMessage("", tool_calls=[{"name": "no_such_tool", "args": {}, "id": "t2"}])]}},
        ),
        (
            "updates",
            {"agent_tools": {"messages": [
                ToolMessage(content=json.dumps({"error": "无此工具"}), tool_call_id="t2",
                            name="no_such_tool", status="error")
            ]}},
        ),
    ]
    _patch_runtime_graph(monkeypatch, chunks)
    events = _collect(runtime.stream_turn("c-1", "hi"))
    tools = [event for event in events if event["type"] == "tool"]
    assert tools == [
        {"type": "tool", "tool_call_id": "t2", "name": "no_such_tool", "status": "running"},
        {"type": "tool", "tool_call_id": "t2", "name": "no_such_tool", "status": "failed"},
    ]


# ── 真图端到端(Fake 模型 + InMemorySaver,零上游) ────────────────────────────


class FakeIntentChatModel:
    """chat_model_factory 假体:structured_output → 固定标签;bind_tools → 分步脚本。

    结构化调用与 ReAct 调用分开计数:意图分类不算 ReAct 步数。
    """

    def __init__(self, intent_label: str, scripted_aimessages):
        self._intent_label = intent_label
        self._script = list(scripted_aimessages)
        self._bound_calls = 0

    def with_structured_output(self, schema, method=None):
        outer = self

        class _Structured:
            async def ainvoke(self, messages):
                return schema(**{"intent": outer._intent_label})

        return _Structured()

    def bind_tools(self, tools):
        outer = self

        class _Bound:
            async def ainvoke(self, messages, config=None):
                step = min(outer._bound_calls, len(outer._script) - 1)
                outer._bound_calls += 1
                # 每次返回新对象:同一 AIMessage 复用会被 add_messages 按 id 去重
                return copy.deepcopy(outer._script[step])

        return _Bound()


def _fake_factory(label, script):
    # 缓存单实例:agent_llm 每步都会调 factory,新实例会把脚本进度重置回第一步
    model = FakeIntentChatModel(label, script)
    return lambda settings: model


def test_real_graph_chitchat_turn_end_to_end(session_factory, settings):
    """ch06:闲聊走 script_reply 脚本出口(文案 = SCRIPT_REPLY_CHITCHAT)。"""
    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory,
            settings=settings,
            checkpointer=InMemorySaver(),
            chat_model_factory=_fake_factory("闲聊", [AIMessage("这不重要")]),
        )
        try:
            conversation_id, created = await runtime.prepare_turn(None, "你好")
            events = [event async for event in runtime.stream_turn(conversation_id, "你好")]
            return conversation_id, created, events
        finally:
            await runtime.close_graph()

    conversation_id, created, events = asyncio.run(_scenario())
    assert created is True
    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert "".join(deltas) == SCRIPT_REPLY_CHITCHAT
    assert [event["type"] for event in events][-1] == "done"
    # 审计:user(prepare)+ assistant(log 节点)
    with session_factory() as session:
        rows = session.query(Message).filter_by(conversation_id=conversation_id).all()
    assert [row.role for row in rows] == ["user", "assistant"]
    assert rows[1].content == SCRIPT_REPLY_CHITCHAT


def test_real_graph_tool_turn_persists_and_keeps_history(session_factory, settings):
    script = [
        AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "t1"}]),
        AIMessage("您的订单 1001 已发货。"),
    ]

    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory,
            settings=settings,
            checkpointer=InMemorySaver(),
            chat_model_factory=_fake_factory("订单", script),
        )
        try:
            conversation_id, _created = await runtime.prepare_turn(None, "订单 1001 到哪了")
            events = [
                event async for event in runtime.stream_turn(conversation_id, "订单 1001 到哪了")
            ]
            tools = [event for event in events if event["type"] == "tool"]
            # 第二轮:同 thread 再走一步(验证 checkpointer 跨轮历史;脚本走完后续轮复用末条)
            # ch07:每轮先 prepare 落库拿锚点 db-{msg_id},同 id 覆盖语义才不会误伤
            await runtime.prepare_turn(conversation_id, "再问一次")
            _ = [event async for event in runtime.stream_turn(conversation_id, "再问一次")]
            snapshot = runtime.get_graph().get_state({"configurable": {"thread_id": conversation_id}})
            return conversation_id, tools, snapshot.values.get("messages", [])
        finally:
            await runtime.close_graph()

    conversation_id, tools, messages = asyncio.run(_scenario())
    statuses = [(t["name"], t["status"]) for t in tools]
    assert ("query_order", "running") in statuses
    assert ("query_order", "completed") in statuses
    # 跨轮历史:两轮各有 user;首轮的 tool 消息与 AI 答复都在 checkpointer 里
    types = [m.type for m in messages]
    assert types.count("human") == 2
    assert types.count("tool") == 1
    assert types.count("ai") >= 2
    with session_factory() as session:
        rows = session.query(Message).filter_by(conversation_id=conversation_id).all()
    # 审计:两轮 user + 两轮 assistant(log 各补一条)
    assert [row.role for row in rows].count("assistant") == 2
