"""ch08 · ToolSpec 注册中心:三样必齐(名字/描述/JSON Schema)、注册即定义、权限只认我们侧。

内置 query_logistics 下线——物流查询由 MCP 接管,不留重名。
"""

import logging

from langchain_core.tools import tool as langchain_tool
from pydantic import BaseModel, Field

from app.tools import registry


def test_builtin_scan_registers_five_tools_without_query_logistics():
    names = {s.name for s in registry.builtin_specs()}
    # ch08:物流查询由 MCP 接管,内置 query_logistics 下线,不留重名
    assert names == {"query_order", "query_product", "query_faq", "create_ticket", "submit_refund"}


def test_every_spec_has_three_essentials():
    for s in registry.builtin_specs():
        assert s.name and s.description  # 工具名 + 用途描述
        assert isinstance(s.json_schema, dict) and s.json_schema.get("properties") is not None


def test_permissions_only_trust_our_side():
    by = {s.name: s for s in registry.builtin_specs()}
    assert by["create_ticket"].permission == "write"
    assert all(s.permission == "read" for n, s in by.items() if n != "create_ticket")
    # 未登记(含所有 MCP)工具一律按只读放行——权限只认我们侧 WRITE_TOOLS
    assert registry.permission_for("mcp_random_tool") == "read"


def test_create_ticket_schema_hides_injected_runtime_args():
    spec = registry.get_builtin_spec("create_ticket")
    assert set(spec.injected_args) == {"conversation_id", "tool_call_id", "session_factory", "settings"}
    props = spec.json_schema.get("properties", {})
    # 模型可见口径不含任何注入参数
    assert "conversation_id" not in props and "session_factory" not in props
    assert "settings" not in props and "tool_call_id" not in props


def test_faq_needs_runtime_deps_injected():
    spec = registry.get_builtin_spec("query_faq")
    assert {"session_factory", "settings", "tool_call_id", "conversation_id"} <= set(spec.injected_args)
    # query_faq 走 RAG 管线,多次上游往返:不重试,超时放宽到 30s
    assert spec.timeout == 30.0 and spec.max_retries == 0


def test_duplicate_register_is_dropped_with_warning(caplog):
    spec = registry.builtin_specs()[0]
    dup = registry.ToolSpec(
        name=spec.name, description="dup",
        json_schema={"type": "object", "properties": {}},
        tool=spec.tool, permission="read", source="builtin",
    )
    with caplog.at_level(logging.WARNING):
        registry.register(dup)
    assert registry.get_builtin_spec(spec.name).description != "dup"  # 先到者保留
    assert any("重名" in r.message for r in caplog.records)


def test_spec_from_langchain_tool_extracts_pydantic_schema():
    class X(BaseModel):
        x: str = Field(description="参数x")

    @langchain_tool(args_schema=X)
    def x_tool(x: str) -> dict:
        """测试工具。"""
        return {}

    spec = registry.spec_from_langchain_tool(x_tool, source="mcp", mcp_server="demo")
    assert spec.json_schema["properties"]["x"]["type"] == "string"
    assert spec.name == "x_tool" and spec.description == "测试工具。"
    assert spec.permission == "read" and spec.source == "mcp" and spec.mcp_server == "demo"
    assert spec.injected_args == ()  # dict/MCP 工具无注入参数


def test_spec_from_dict_schema_tool():
    """MCP adapters 回来的工具 args_schema 可能就是 dict:原样收作 JSON Schema。"""

    class DictTool:
        name = "dict_tool"
        description = "字典 schema 工具"
        args_schema = {"type": "object", "properties": {"y": {"type": "integer"}}}

    spec = registry.spec_from_langchain_tool(DictTool(), source="mcp", mcp_server="s")
    assert spec.json_schema == {"type": "object", "properties": {"y": {"type": "integer"}}}
    assert spec.injected_args == ()


# 模型可见口径(不含注入参数)与必填/可选切分
EXPECTED_VISIBLE_SCHEMAS = {
    "query_order": {"order_id"},
    "query_product": {"product_id"},
    "query_faq": {"keyword", "strategy"},  # ch04 新增可选检索策略入参
    "create_ticket": {"description", "ticket_type"},
    "submit_refund": {"order_id", "reason"},  # ch06 拦截型退款工具
}
OPTIONAL_VISIBLE_FIELDS = {"strategy", "reason"}
INJECTED_ARGUMENTS = {"conversation_id", "tool_call_id", "session_factory", "settings"}


def test_builtin_visible_schemas_are_exact_and_hide_runtime_arguments():
    for name, expected in EXPECTED_VISIBLE_SCHEMAS.items():
        schema = registry.get_builtin_spec(name).json_schema
        assert set(schema["properties"]) == expected
        assert expected - OPTIONAL_VISIBLE_FIELDS == set(schema.get("required", []))
        # 注入参数一律不在模型可见口径里
        assert set(schema["properties"]).isdisjoint(INJECTED_ARGUMENTS)
        # 输入口径:有界非空字符串
        for field_name in expected - {"ticket_type"} - OPTIONAL_VISIBLE_FIELDS:
            prop = schema["properties"][field_name]
            assert prop["type"] == "string" and prop["minLength"] == 1 and prop["maxLength"] > 1


def test_faq_strategy_is_optional_with_pattern_constraint():
    schema = registry.get_builtin_spec("query_faq").json_schema
    strategy = schema["properties"]["strategy"]
    assert "strategy" not in schema.get("required", [])
    choices = strategy.get("anyOf", [strategy])
    assert "^(vector|bm25|hybrid|hybrid_rerank)$" in {choice.get("pattern") for choice in choices}


def test_create_ticket_ticket_type_is_chinese_enum():
    schema = registry.get_builtin_spec("create_ticket").json_schema
    assert set(schema["properties"]["ticket_type"]["enum"]) == {"售后", "投诉", "咨询"}
