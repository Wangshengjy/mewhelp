"""ch05/ch08 · 主力 Agent 节点(agent_llm / agent_tools)+ 统一执行引擎接线。

ch08:agent_tools 的普通工具走 engine.execute_tool_call(六段管道);
create_ticket / submit_refund 仍在节点层拦截为『提议』(ch05/ch06 语义)。
单元测试 monkeypatch 注册中心的 get_all_specs,避免真连 MCP Server。"""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.graph import nodes
from app.graph.deps import GraphDeps
from app.tools.registry import ToolSpec
from tests.graph.conftest import cfg


def _fake_spec(name, fn, *, injected=(), **kw):
    tool = type("T", (), {"name": name, "ainvoke": staticmethod(fn)})()
    return ToolSpec(
        name=name, description="测试工具",
        json_schema={"type": "object", "properties": {}},
        tool=tool, permission=kw.get("permission", "read"),
        source=kw.get("source", "builtin"), injected_args=injected,
    )


def _patch_specs(monkeypatch, specs):
    async def fake_get_all_specs(settings=None):
        return {spec.name: spec for spec in specs}

    monkeypatch.setattr(nodes.tool_registry, "get_all_specs", fake_get_all_specs)


def test_agent_messages_injects_evidence_on_knowledge():
    # ch07:system 只剩人设红线;摘要/证据/快照等可变内容走用户侧 turn_ctx
    msgs = nodes._agent_messages(
        {
            "route": "knowledge",
            "evidence": "[1] 退货: 7天",
            "messages": [HumanMessage("能退吗")],
        }
    )
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == nodes.CHAT_SYSTEM_PROMPT
    assert len([m for m in msgs if isinstance(m, SystemMessage)]) == 1  # 整条只一条 system
    assert isinstance(msgs[-1], HumanMessage)
    assert msgs[-1].content.endswith("[1] 退货: 7天")  # 证据在 turn_ctx 尾部
    assert "query_faq" in msgs[-1].content  # 指示别再检索
    assert msgs[1].content == "能退吗"  # 窗内用户原话在 turn_ctx 之前


def test_agent_messages_no_evidence_on_business():
    msgs = nodes._agent_messages({"route": "business", "messages": [HumanMessage("订单1001")]})
    assert isinstance(msgs[0], SystemMessage)
    assert "已检索到的知识证据" not in msgs[0].content
    assert len(msgs) == 2  # system + 窗内原文;空材料不产 turn_ctx


def test_agent_llm_accumulates_steps_and_tokens(graph_deps, monkeypatch):
    ai = AIMessage("好的", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    bound = {}

    class FakeModel:
        def bind_tools(self, tools):
            bound["tools"] = list(tools)
            return self

        async def ainvoke(self, msgs, config=None):
            return ai

    _patch_specs(monkeypatch, [])
    deps = GraphDeps(
        session_factory=graph_deps.session_factory,
        settings=graph_deps.settings,
        chat_model_factory=lambda s: FakeModel(),
    )
    out = asyncio.run(
        nodes.agent_llm({"messages": [HumanMessage("hi")], "steps": 1, "tokens_used": 100}, cfg(deps))
    )
    assert out["steps"] == 2
    assert out["tokens_used"] == 115
    assert out["messages"][0] is ai
    assert bound["tools"] == []  # 清单为空时也能 bind(结构不动)


def test_agent_tools_executes_normal_tool_via_engine(graph_deps, monkeypatch):
    captured = {}

    async def query_order(args):
        captured["conversation_id"] = args.get("conversation_id")
        captured["order_id"] = args.get("order_id")
        return {"status": "已发货"}

    _patch_specs(monkeypatch, [_fake_spec("query_order", query_order, injected=("conversation_id",))])
    ai = AIMessage(
        "",
        tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "t1"}],
    )
    out = asyncio.run(
        nodes.agent_tools({"messages": [ai], "conversation_id": "c-5"}, cfg(graph_deps))
    )
    message = out["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.name == "query_order"
    assert message.tool_call_id == "t1"
    assert message.status == "success"
    payload = json.loads(message.content)
    assert payload == {"status": "已发货"} and "已发货" in message.content  # 中文不转义
    assert captured["conversation_id"] == "c-5"  # 引擎注入 conversation_id
    assert captured["order_id"] == "1001"
    assert not out.get("suggested_actions")


def test_agent_tools_marks_unknown_tool_error(graph_deps, monkeypatch):
    async def must_not_run(*args, **kwargs):
        raise AssertionError("未知工具不应被调用")

    _patch_specs(monkeypatch, [_fake_spec("query_order", must_not_run)])
    ai = AIMessage("", tool_calls=[{"name": "no_such_tool", "args": {}, "id": "t2"}])
    out = asyncio.run(
        nodes.agent_tools({"messages": [ai], "conversation_id": "c-5"}, cfg(graph_deps))
    )
    message = out["messages"][0]
    assert message.status == "error"
    assert "未知工具" in message.content  # 引擎分诊:坏消息如实回灌


def test_agent_tools_generates_call_id_when_missing(graph_deps, monkeypatch):
    async def ok(args):
        return {"order_id": "1001"}

    _patch_specs(monkeypatch, [_fake_spec("query_order", ok)])
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": ""}])
    out = asyncio.run(
        nodes.agent_tools({"messages": [ai], "conversation_id": "c-5"}, cfg(graph_deps))
    )
    assert out["messages"][0].tool_call_id.startswith("call_")
