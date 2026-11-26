"""ch08 · 建工单确认流(真图端到端,零上游):Fake 模型 + 真实 SQLite。

- 信息不够:引擎校验拦下回灌,模型追问,不瞎编(无 interrupt、无写库);
- 参数齐:interrupt 推工单预览卡(类型+描述),等前端确认/取消;
- 确认(resume confirmed=True):引擎放行落 tickets,回复带工单号,审计「成功」;
- 取消(resume confirmed=False):走权限门拒绝,不写 tickets,审计「权限拒绝」。
复用 ch06 interrupt/resume 钉形状;ch05 投诉按钮建单路(/api/actions/create-ticket)不动。
"""

import asyncio
import copy

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph import nodes, runtime
from app.models import ToolAuditLog, Ticket
from app.settings import Settings
from tests.test_chat_tools import RecordingChatModel, _make_model  # 复用 ch05 Fake 模型装置


@pytest.fixture
def graph_runtime(session_factory):
    def _start(chat_model_factory):
        async def _init():
            await runtime.init_graph(
                session_factory=session_factory,
                settings=Settings.hermetic_defaults(),
                checkpointer=InMemorySaver(),
                chat_model_factory=chat_model_factory,
            )

        asyncio.run(_init())

    yield _start
    asyncio.run(runtime.close_graph())


def _run_turn(message, conversation_id=None):
    async def _scenario():
        new_id, _created = await runtime.prepare_turn(conversation_id, message)
        events = [event async for event in runtime.stream_turn(new_id, message)]
        return new_id, events

    return asyncio.run(_scenario())


def _resume_turn(conversation_id, resume_value):
    async def _scenario():
        events = [event async for event in runtime.stream_resume(conversation_id, resume_value)]
        return events

    return asyncio.run(_scenario())


def _tickets(session_factory):
    with session_factory() as session:
        return session.query(Ticket).all()


def _audit(session_factory, tool_name):
    with session_factory() as session:
        rows = (
            session.query(ToolAuditLog)
            .filter_by(tool_name=tool_name)
            .order_by(ToolAuditLog.id)
            .all()
        )
        return [(r.status, r.retry_count, r.tool_source) for r in rows]


def test_create_ticket_confirm_flow_creates_ticket(graph_runtime, session_factory):
    """验收 4:确认 → 落 tickets 带工单号;审计「成功」。"""
    script = [
        AIMessage(
            "",
            tool_calls=[
                {"name": "create_ticket", "args": {"description": "商品有质量问题", "ticket_type": "售后"}, "id": "t1"}
            ],
        ),
        AIMessage("工单已创建成功,人工客服会尽快跟进。"),
    ]
    model, factory = _make_model("售后", script)
    graph_runtime(factory)

    conversation_id, events = _run_turn("我要建个售后工单,商品有质量问题")

    # interrupt 预览卡:类型 + 描述;无 done 帧;未写任何工单
    interrupts = [event for event in events if event["type"] == "interrupt"]
    assert len(interrupts) == 1
    payload = interrupts[0]["payload"]
    assert payload["type"] == "confirm_ticket"
    assert payload["preview"] == {"ticket_type": "售后", "description": "商品有质量问题"}
    assert not [event for event in events if event["type"] == "done"]
    assert _tickets(session_factory) == []

    # 前端确认 → resume 放行
    resume_events = _resume_turn(conversation_id, {"confirmed": True})
    deltas = [event["text"] for event in resume_events if event["type"] == "delta"]
    assert deltas == ["工单已创建成功,人工客服会尽快跟进。"]
    tickets = _tickets(session_factory)
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.ticket_no.startswith("T") and len(ticket.ticket_no) == 13  # T+12位hex
    assert ticket.ticket_type == "售后"
    assert ticket.description == "商品有质量问题"
    assert ticket.status == "open"
    assert _audit(session_factory, "create_ticket") == [("成功", 0, "builtin")]


def test_create_ticket_cancel_denied_and_audited(graph_runtime, session_factory):
    """验收 5:取消 = 引擎权限拒绝,不写 tickets,审计「权限拒绝」。"""
    script = [
        AIMessage(
            "",
            tool_calls=[
                {"name": "create_ticket", "args": {"description": "不想买了想退", "ticket_type": "售后"}, "id": "t1"}
            ],
        ),
        AIMessage("已按您的要求取消,工单未创建。"),
    ]
    _model, factory = _make_model("售后", script)
    graph_runtime(factory)

    conversation_id, _events = _run_turn("帮我建个工单")

    resume_events = _resume_turn(conversation_id, {"confirmed": False})
    deltas = [event["text"] for event in resume_events if event["type"] == "delta"]
    assert deltas == ["已按您的要求取消,工单未创建。"]
    assert _tickets(session_factory) == []  # 取消不落 tickets
    assert _audit(session_factory, "create_ticket") == [("权限拒绝", 0, "builtin")]


def test_create_ticket_missing_info_asks_instead_of_fabricating(graph_runtime, session_factory):
    """信息不够先追问:校验拦下回灌,模型追问用户;无 interrupt、无写库、无审计成功行。"""
    script = [
        AIMessage(
            "",
            tool_calls=[{"name": "create_ticket", "args": {"description": "", "ticket_type": "售后"}, "id": "t1"}],
        ),
        AIMessage("请问具体遇到了什么问题呢?描述清楚后我再为您登记工单。"),
    ]
    _model, factory = _make_model("售后", script)
    graph_runtime(factory)

    _conversation_id, events = _run_turn("给我建个工单")

    # description 为空被引擎校验拦下(不是 interrupt),模型第二步追问
    assert not [event for event in events if event["type"] == "interrupt"]
    deltas = [event["text"] for event in events if event["type"] == "delta"]
    assert deltas == ["请问具体遇到了什么问题呢?描述清楚后我再为您登记工单。"]
    assert _tickets(session_factory) == []
    assert _audit(session_factory, "create_ticket") == []  # 未进引擎执行段(节点层追问)


def test_system_prompt_rule8_matches_confirm_flow():
    """prompt 样例验证:规则 8 与确认流一致,旧口径(ticket_type 传 other)已删。"""
    from app.prompts import CHAT_SYSTEM_PROMPT

    assert "先向用户追问" in CHAT_SYSTEM_PROMPT  # 信息不足先追问,不瞎编
    assert "售后/投诉/咨询" in CHAT_SYSTEM_PROMPT  # 类型三选一(中文化)
    assert "工单确认卡" in CHAT_SYSTEM_PROMPT  # 只有用户确认才真正创建
    assert "ticket_type 传 other" not in CHAT_SYSTEM_PROMPT  # ch05 旧口径已退役


def test_create_ticket_confirmation_audit_row_has_tool_call_id(graph_runtime, session_factory):
    """确认放行时审计带 tool_call_id(可对回 messages 流水)。"""
    script = [
        AIMessage(
            "",
            tool_calls=[
                {"name": "create_ticket", "args": {"description": "物流丢了", "ticket_type": "投诉"}, "id": "tc-42"}
            ],
        ),
        AIMessage("已登记。"),
    ]
    _model, factory = _make_model("售后", script)
    graph_runtime(factory)

    conversation_id, _events = _run_turn("你们太差了,给我建个投诉工单")
    _resume_turn(conversation_id, {"confirmed": True})

    with session_factory() as session:
        row = session.query(ToolAuditLog).filter_by(tool_name="create_ticket").one()
    assert row.tool_call_id == "tc-42"
    assert row.arguments["description"] == "物流丢了"
    assert row.duration_ms is not None
