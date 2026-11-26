"""ch08 · 统一执行引擎六段管道:校验拦下/权限拒绝/超时重试/写不重试/分诊/格式化/审计齐备。

审计经 monkeypatch engine._write_audit_row 捕获(不触真库);工具用 _tool 假体(ainvoke 可控)。
"""

import asyncio

import pytest

from app.tools import engine
from app.tools.registry import ToolSpec

SCHEMA = {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}
_SF = object()  # session_factory 哨兵:有值才会走审计写入路径


def _spec(name="query_order", *, permission="read", source="builtin", tool=None,
          schema=SCHEMA, timeout=None, max_retries=None, injected=(), fmt=None):
    return ToolSpec(
        name=name, description="测试工具", json_schema=schema, tool=tool,
        permission=permission, source=source, timeout=timeout,
        max_retries=max_retries, injected_args=injected, format_result=fmt,
    )


def _tool(fn, name="query_order"):
    return type("T", (), {"name": name, "ainvoke": staticmethod(fn)})()


@pytest.fixture()
def audits(monkeypatch):
    rows = []

    def fake_write(session_factory, record):
        rows.append(record)

    monkeypatch.setattr(engine, "_write_audit_row", fake_write)
    return rows


async def test_unknown_tool(audits):
    run = await engine.execute_tool_call({"name": "nope", "args": {}, "id": "c1"}, "conv-1", {}, session_factory=_SF)
    assert run.ok is False and "未知工具" in run.tool_message.content
    assert audits[-1]["status"] == "失败"


async def test_validation_blocks_and_feeds_back(audits):
    async def boom(_):
        raise AssertionError("校验不过不许执行")

    spec = _spec(tool=_tool(boom))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {}, "id": "c1"}, "conv-1", {"query_order": spec}, session_factory=_SF
    )
    assert run.ok is False and run.status == "校验拦下"
    assert "参数校验未通过" in run.tool_message.content and run.tool_message.status == "error"
    assert audits[-1]["status"] == "校验拦下" and audits[-1]["retry_count"] == 0


async def test_write_without_confirmation_denied(audits):
    async def boom(_):
        raise AssertionError("未确认不许执行")

    schema = {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}
    spec = _spec("create_ticket", permission="write", tool=_tool(boom, "create_ticket"), schema=schema)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        "conv-1", {"create_ticket": spec}, session_factory=_SF,
    )
    assert run.ok is False and run.status == "权限拒绝"
    assert audits[-1]["status"] == "权限拒绝"


async def test_write_confirmed_executes_and_not_retried(audits):
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        raise RuntimeError("boom")

    schema = {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}
    spec = _spec("create_ticket", permission="write", tool=_tool(flaky, "create_ticket"), schema=schema)
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        "conv-1", {"create_ticket": spec}, session_factory=_SF, confirmed=True,
    )
    assert run.ok is False and calls["n"] == 1 and run.retry_count == 0  # 写操作恒不重试
    assert audits[-1]["retry_count"] == 0


async def test_transient_timeout_retries_then_gives_up(audits):
    async def slow(_):
        await asyncio.sleep(1)

    spec = _spec(tool=_tool(slow), timeout=0.05)
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}, "conv-1",
        {"query_order": spec}, session_factory=_SF,
    )
    assert run.ok is False and run.status == "超时"
    assert run.retry_count == 2  # settings.tool_max_retries 默认 2(settings 未注入走引擎默认,同值)
    assert audits[-1]["status"] == "超时" and audits[-1]["retry_count"] == 2
    assert audits[-1]["duration_ms"] is not None and audits[-1]["duration_ms"] > 0


async def test_business_error_not_retried(audits):
    calls = {"n": 0}

    async def fail(_):
        calls["n"] += 1
        raise ValueError("业务错")  # 非暂时性 → 不重试

    spec = _spec(tool=_tool(fail))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}, "conv-1",
        {"query_order": spec}, session_factory=_SF,
    )
    assert run.ok is False and calls["n"] == 1 and "工具暂时不可用" in run.tool_message.content
    assert audits[-1]["status"] == "失败" and "ValueError" in audits[-1]["error_message"]


async def test_retry_succeeds_second_attempt(audits):
    calls = {"n": 0}

    async def flaky(_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("网络抖动")
        return {"order_id": "1", "status": "已发货"}

    spec = _spec(tool=_tool(flaky))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}, "conv-1",
        {"query_order": spec}, session_factory=_SF,
    )
    assert run.ok is True and calls["n"] == 2 and run.retry_count == 1
    assert "已发货" in run.tool_message.content  # ensure_ascii=False,中文不转义
    assert audits[-1]["status"] == "成功" and audits[-1]["retry_count"] == 1


async def test_format_result_hook_translates_enum(audits):
    async def ok(_):
        return {"tracking_no": "SF1", "status_code": "IN_TRANSIT", "internal_ref": "x9"}

    def fmt(d):
        return {
            "tracking_no": d["tracking_no"],
            "status": {"IN_TRANSIT": "运输中"}.get(d.get("status_code"), d.get("status_code")),
        }

    spec = _spec(
        "query_logistics", source="mcp", tool=_tool(ok, "query_logistics"),
        schema={"type": "object", "properties": {"tracking_no": {"type": "string"}}, "required": ["tracking_no"]},
        fmt=fmt,
    )
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF1"}, "id": "c1"}, "conv-1",
        {"query_logistics": spec}, session_factory=_SF,
    )
    assert "运输中" in run.tool_message.content and "internal_ref" not in run.tool_message.content
    assert audits[-1]["tool_source"] == "mcp" and audits[-1]["mcp_server"] is None


async def test_mcp_content_block_list_is_unwrapped(audits):
    """adapters 0.3.x 的工具返回 [content-block] 列表:引擎解包 text 块解回 dict 再走格式化。"""

    async def mcp_shape(_):
        import json as _json

        return [{"type": "text", "text": _json.dumps({"status_code": "IN_TRANSIT"}, ensure_ascii=False)}]

    spec = _spec(
        "query_logistics", source="mcp", tool=_tool(mcp_shape, "query_logistics"),
        schema={"type": "object", "properties": {"tracking_no": {"type": "string"}}, "required": ["tracking_no"]},
    )
    run = await engine.execute_tool_call(
        {"name": "query_logistics", "args": {"tracking_no": "SF9"}, "id": "c1"}, "conv-1",
        {"query_logistics": spec}, session_factory=_SF,
    )
    assert run.ok is True and "IN_TRANSIT" in run.tool_message.content
    assert "'type'" not in run.tool_message.content  # 内容块外壳不进回灌


async def test_audit_failure_never_blocks_execution(monkeypatch):
    def audit_boom(session_factory, record):
        raise RuntimeError("审计库挂了")

    monkeypatch.setattr(engine, "_write_audit_row", audit_boom)

    async def ok(_):
        return {"order_id": "1"}

    spec = _spec(tool=_tool(ok))
    run = await engine.execute_tool_call(
        {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}, "conv-1",
        {"query_order": spec}, session_factory=_SF,
    )
    assert run.ok is True  # 审计失败不反拦


async def test_inject_runtime_args_after_validation(audits):
    seen = {}

    async def ok(args):
        seen.update(args)
        return {"ticket_no": "T1"}

    schema = {
        "type": "object",
        "properties": {"description": {"type": "string"}},
        "required": ["description"],
        "additionalProperties": False,
    }
    spec = _spec(
        "create_ticket", permission="write", tool=_tool(ok, "create_ticket"),
        schema=schema, injected=("conversation_id", "session_factory"),
    )
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": {"description": "x"}, "id": "c1"},
        "conv-7", {"create_ticket": spec}, session_factory=_SF, confirmed=True,
    )
    assert run.ok is True
    # 校验后注入:schema 不含注入参数,注入值来自调用上下文,模型填的不算数
    assert seen.get("conversation_id") == "conv-7" and seen.get("session_factory") is _SF
    assert audits[-1]["status"] == "成功"


async def test_validate_args_reports_missing_required():
    spec = _spec(schema=SCHEMA)
    err = engine.validate_args(spec, {})
    assert err is not None and "order_id" in err
    assert engine.validate_args(spec, {"order_id": "1001"}) is None
