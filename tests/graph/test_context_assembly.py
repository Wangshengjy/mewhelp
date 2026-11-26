"""ch07 · 上下文组装与轮末结算:锚点入图 / 拼装契约 / settle_layers 降级 / 日志留痕。"""

import asyncio
import dataclasses
import logging
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.core import budget as budget_mod
from app.core import memory
from app.graph import nodes, runtime
from app.repositories import conversations as repo
from app.settings import Settings


def _user(text: str, msg_id: int | None = None) -> HumanMessage:
    return HumanMessage(content=text, id=f"db-{msg_id}" if msg_id is not None else None)


def _healthy() -> Settings:
    return dataclasses.replace(
        Settings.hermetic_defaults(),
        model_context_window=32768,
        max_output_tokens=4000,
        tool_result_max_tokens=1200,
    )


# ── 入图锚点 ──────────────────────────────────────────────────────────────────


def test_graph_input_carries_db_anchor_and_layer_fields():
    payload = runtime._graph_input("c-1", "订单1001到哪了", summary="梗概", summary_upto=4, layer1_from=7, msg_id=9)
    (human,) = payload["messages"]
    assert human.id == "db-9"
    assert payload["summary"] == "梗概"
    assert payload["summary_upto_msg_id"] == 4
    assert payload["layer1_from_msg_id"] == 7
    assert payload["intent_confidence"] == 0.0  # ch06 字段照旧清零


# ── _agent_messages 拼装契约 ──────────────────────────────────────────────────


def test_agent_messages_summary_goes_user_side_after_last_human():
    state = {
        "summary": "用户咨询订单1001,客服答复已发货",
        "summary_upto_msg_id": 0,  # 摘要行注入与滑窗起点解耦:这里只验拼装
        "layer1_from_msg_id": 2,
        "route": "knowledge",
        "evidence": "[1] 退货: 7天",
        "messages": [
            _user("历史问题", 1),
            AIMessage(content="历史答复"),
            _user("本轮问题", 3),
        ],
    }
    msgs = nodes._agent_messages(state)
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == nodes.CHAT_SYSTEM_PROMPT
    # 顺序:窗内原文(层2 历史轮 + 层1 本轮) → turn_ctx(摘要+证据)挂在最后一条 human 之后
    assert msgs[1].content == "历史问题"
    assert msgs[2].content == "历史答复"
    assert msgs[3].content == "本轮问题"
    turn_ctx = msgs[4].content
    assert turn_ctx.startswith("(早前对话摘要:")
    assert turn_ctx.endswith("[1] 退货: 7天")


def test_agent_messages_refund_snapshot_in_turn_ctx():
    state = {
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已发货"},
        "evidence": "[1] 退款政策",
        "messages": [_user("我要退款", 5)],
    }
    msgs = nodes._agent_messages(state)
    turn_ctx = msgs[-1].content
    assert '"order_id": "1001"' in turn_ctx
    assert nodes.REFUND_JUDGE_HINT.strip() in turn_ctx
    assert "订单快照" in turn_ctx


def test_log_model_context_emits_preview(caplog):
    state = {"conversation_id": "c-42", "summary": "用户咨询订单1001"}
    messages = [
        SystemMessage(content="system 提示"),
        HumanMessage(content="订单1001到哪了"),
    ]
    caplog.set_level(logging.INFO, logger="app.graph.nodes")
    nodes._log_model_context(state, messages)
    assert "model_ctx" in caplog.text
    assert "c-42" in caplog.text
    assert "订单1001到哪了" in caplog.text


def test_history_text_prefixes_summary_and_uses_window():
    state = {
        "summary": "用户咨询订单1001,答复已发货",
        "summary_upto_msg_id": 2,
        "layer1_from_msg_id": 0,
        "messages": [_user("最开始那个订单怎么样", 3), AIMessage(content="已发货,预计三天内送达")],
    }
    with mock.patch.object(nodes.logger, "info"):
        text = nodes._history_text(state)
    assert text.startswith("(早前对话摘要:用户咨询订单1001")
    assert "最开始那个订单怎么样" in text


# ── settle_layers ─────────────────────────────────────────────────────────────


def _seed_long_history(session_factory):
    """7 轮带锚点长对话,返回 (cid, anchors=[(u_id, a_id)×7])。"""
    with session_factory() as session:
        conversation = repo.create_conversation(session)
        anchors = []
        for index in range(1, 8):
            user = repo.append_message(
                session, conversation.id, role="user", content=f"第{index}轮,订单100{index}。" + "细节" * 500
            )
            assistant = repo.append_message(
                session, conversation.id, role="assistant", content=f"第{index}轮答复。" + "信息" * 500
            )
            anchors.append((user.id, assistant.id))
        session.commit()
        return conversation.id, anchors


def _state_messages(anchors) -> list:
    messages = []
    for index, (u_id, _a_id) in enumerate(anchors, start=1):
        messages.append(_user(f"第{index}轮,订单100{index}。" + "细节" * 500, u_id))
        messages.append(AIMessage(content=f"第{index}轮答复。" + "信息" * 500))
    return messages


def test_settle_layers_degrades_boundary(session_factory):
    """7 轮长对话在健康预算(层1=9322)下必超 → 边界推进到某个中间锚点。"""

    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory, settings=_healthy(), checkpointer=InMemorySaver()
        )
        try:
            cid, anchors = await asyncio.to_thread(_seed_long_history, session_factory)
            messages = _state_messages(anchors)
            result = budget_mod.compute(_healthy())
            _, l1_tokens = memory.layer_tokens(messages, 0, 0, _healthy())
            assert l1_tokens > result.layer1_budget  # 前置:确实超了
            await runtime.settle_layers(cid, {"messages": messages, "summary_upto_msg_id": 0, "layer1_from_msg_id": 0})
            with session_factory() as session:
                found = repo.get_conversation(session, cid)
            assert found.layer1_from_msg_id is not None
            assert anchors[0][0] <= found.layer1_from_msg_id < anchors[-1][0]  # 边界落在中间锚点
        finally:
            await runtime.close_graph()

    asyncio.run(_scenario())


def test_settle_layers_keeps_boundary_when_layer1_fits(session_factory):
    """层 1 达标不写库:边界只进不退。"""

    async def _scenario():
        await runtime.init_graph(
            session_factory=session_factory, settings=_healthy(), checkpointer=InMemorySaver()
        )
        try:
            cid, anchors = await asyncio.to_thread(_seed_long_history, session_factory)
            boundary = anchors[3][0]  # 层 2 = 前四轮,层 1 只剩最后一轮
            with session_factory() as session:
                repo.update_layer1_from(session, cid, boundary)
                session.commit()
            short_state = {
                "messages": [_user("就一句", anchors[4][0]), AIMessage(content="就一答")],
                "summary_upto_msg_id": 0,
                "layer1_from_msg_id": boundary,
            }
            await runtime.settle_layers(cid, short_state)
            with session_factory() as session:
                found = repo.get_conversation(session, cid)
            assert found.layer1_from_msg_id == boundary  # 未被挪动
        finally:
            await runtime.close_graph()

    asyncio.run(_scenario())


def test_settle_layers_swallows_exceptions(monkeypatch, caplog):
    """分层结算里的任何异常吞掉只留痕,不影响当轮回复。"""
    from app.graph.deps import GraphDeps

    async def _scenario():
        fake_deps = GraphDeps(session_factory=None, settings=_healthy())
        monkeypatch.setattr(runtime, "get_deps", lambda: fake_deps)
        monkeypatch.setattr(
            runtime.memory,
            "layer_tokens",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        caplog.set_level(logging.WARNING, logger="app.graph.runtime")
        await runtime.settle_layers(
            "c-9", {"messages": [_user("hi", 1)], "summary_upto_msg_id": 0, "layer1_from_msg_id": 0}
        )
        assert "settle_layers 失败" in caplog.text

    asyncio.run(_scenario())
