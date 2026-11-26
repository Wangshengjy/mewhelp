"""ch08 MCP Server 集成测试:真起子进程 → adapters client 列工具(三要素齐)+ 真调拿 mock 数据。

两台 Server 是本章交付物,连不上直接 fail(不 skip)。端口用 18101/18102 避开默认端口,
与手跑的 :8101/:8102 互不干扰。"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

ROOT = Path(__file__).resolve().parents[2]
LOGISTICS_PORT = 18101
AFTERSALES_PORT = 18102


def _wait_port(port: int, timeout: float = 20.0) -> None:
    """轮询探活:必须等到真实来源应答(本机对无监听端口会回 502 Bad Gateway,
    502/503/504 一律不算就绪,避免抢在 uvicorn 完成绑定前开测)。"""
    deadline = time.time() + timeout
    last_seen = "not tried"
    while time.time() < deadline:
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/mcp", timeout=1.0)
            last_seen = f"HTTP {response.status_code}"
            if response.status_code not in (502, 503, 504):
                return  # 真实 uvicorn 应答(406/200/307 都算)
        except httpx.TransportError as exc:
            last_seen = type(exc).__name__
        time.sleep(0.25)
    raise RuntimeError(f"MCP Server :{port} {timeout}s 未就绪(最后状态:{last_seen})")


@pytest.fixture(scope="module")
def mcp_procs():
    procs = []
    for name, script, port in (
        ("logistics", ROOT / "mcp_servers" / "logistics_server.py", LOGISTICS_PORT),
        ("aftersales", ROOT / "mcp_servers" / "aftersales_server.py", AFTERSALES_PORT),
    ):
        env = {**os.environ, "PORT": str(port), "PYTHONIOENCODING": "utf-8"}
        procs.append(subprocess.Popen([sys.executable, str(script)], cwd=str(ROOT), env=env))
    try:
        _wait_port(LOGISTICS_PORT)
        _wait_port(AFTERSALES_PORT)
        yield
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "logistics": {"transport": "streamable_http", "url": f"http://127.0.0.1:{LOGISTICS_PORT}/mcp"},
            "aftersales": {"transport": "streamable_http", "url": f"http://127.0.0.1:{AFTERSALES_PORT}/mcp"},
        },
        handle_tool_errors=False,
    )


async def test_list_tools_three_essentials(mcp_procs):
    tools = await _client().get_tools()
    by = {t.name: t for t in tools}
    assert set(by) == {"query_logistics", "query_warranty", "query_return_status"}
    for t in by.values():
        assert t.description  # 用途描述
        schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
        assert schema.get("properties")  # JSON Schema 参数定义


async def test_invoke_logistics_stable_mock(mcp_procs):
    import json

    tools = {t.name: t for t in await _client().get_tools(server_name="logistics")}
    r1 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    r2 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    # 种子固定 → 同单号稳定(比对解包后的 JSON 载荷;内容块外壳 id 每次生成,本就不同)
    assert json.loads(r1[0]["text"]) == json.loads(r2[0]["text"])
    # adapters 返回 [content-block] 列表:text 块解回是含内部枚举码的 dict
    payload = json.loads(r1[0]["text"])
    assert payload["status_code"] in {"PICKED_UP", "IN_TRANSIT", "DELIVERING", "DELIVERED"}
    assert "carrier_code" in payload  # 内部编码在 Server 侧不删,client 侧格式化时剔除


async def test_invoke_aftersales_warranty_and_return(mcp_procs):
    import json

    tools = {t.name: t for t in await _client().get_tools(server_name="aftersales")}
    warranty = json.loads((await tools["query_warranty"].ainvoke({"order_id": "1001"}))[0]["text"])
    assert warranty["order_id"] == "1001" and warranty["warranty_code"] in {"IN_WARRANTY", "EXPIRED"}
    ret = json.loads((await tools["query_return_status"].ainvoke({"order_id": "1001"}))[0]["text"])
    assert ret["return_code"] in {"AUDITING", "RETURNING", "REFUNDED", "NONE"}
