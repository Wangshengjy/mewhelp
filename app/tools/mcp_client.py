"""ch08 · MCP 客户端:MultiServerMCPClient 多 Server、现问现拿动态发现。

- 两台自建 Server(logistics/aftersales)地址来自 settings(可换端口/可关);
- 现问现拿:每轮现连现列工具,Server 侧加了工具,客服侧重启即用,不动代码;
- 与内置清单合成一份(specs 清单层合并),主力 Agent 一视同仁;
- 结果格式化我们侧登记:挑字段 + 枚举码翻人话(内部编码不进模型上下文);
  未登记的工具透传;不可达 Server 只告警跳过,内置清单照常可用。
"""

import asyncio
import logging
from collections.abc import Callable

from langchain_mcp_adapters.client import MultiServerMCPClient

from app.tools.registry import ToolSpec, spec_from_langchain_tool

logger = logging.getLogger(__name__)

# 我们侧格式化登记表:Server 内部枚举码在这翻人话、内部编码字段在这里剔除。
# 新 Server 的工具要么进来登记,要么透传原文(仍可用,只是模型读得费劲)。
_LOGISTICS_STATUS = {"PICKED_UP": "已揽收", "IN_TRANSIT": "运输中", "DELIVERING": "派送中", "DELIVERED": "已签收"}
_RETURN_STATUS = {"AUDITING": "审核中", "RETURNING": "退货中", "REFUNDED": "已退款", "NONE": "无退货记录"}


def _fmt_logistics(data: dict) -> dict:
    return {
        "tracking_no": data.get("tracking_no"),
        "status": _LOGISTICS_STATUS.get(data.get("status_code"), data.get("status_code")),
        "current_city": data.get("current_city"),
        "trace": data.get("trace"),
    }


def _fmt_warranty(data: dict) -> dict:
    return {
        "order_id": data.get("order_id"),
        "warranty": {"IN_WARRANTY": "在保"}.get(data.get("warranty_code"), "已过保"),
        "warranty_until": data.get("warranty_until"),
    }


def _fmt_return_status(data: dict) -> dict:
    return {
        "order_id": data.get("order_id"),
        "status": _RETURN_STATUS.get(data.get("return_code"), data.get("return_code")),
        "updated_at": data.get("updated_at"),
    }


FORMATTERS: dict[str, Callable[[dict], dict]] = {
    "query_logistics": _fmt_logistics,
    "query_warranty": _fmt_warranty,
    "query_return_status": _fmt_return_status,
}


def build_client(settings) -> MultiServerMCPClient:
    """按 settings 组 MultiServerMCPClient;handle_tool_errors=False 让异常原样抛出,
    由统一执行引擎分诊(重试/回灌),不在 client 层吞。"""
    return MultiServerMCPClient(
        {
            "logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url},
            "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url},
        },
        handle_tool_errors=False,
    )


async def fetch_mcp_specs(settings) -> list[ToolSpec]:
    """现问现拿:逐 Server 列工具并登记成 ToolSpec(server 归属记进 spec.mcp_server)。
    Server 不可达 → 告警跳过(不炸清单合并;只读放行的权限模型天然安全)。"""
    if settings is None:  # eval/演示直调:按环境配置自建
        from app.settings import Settings

        settings = Settings.load()
    specs: list[ToolSpec] = []
    client = build_client(settings)
    for server_name in ("logistics", "aftersales"):
        tools = None
        for attempt in (1, 2):  # 清单拉取偶发网络抖动:快速重试一次(工具调用层另有引擎重试)
            try:
                tools = await client.get_tools(server_name=server_name)
                break
            except Exception as exc:  # noqa: BLE001 连不上就是连不上,别拖垮整条链
                if attempt == 1:
                    logger.warning("MCP Server %s 拉取清单失败(%s),重试一次", server_name, type(exc).__name__)
                    await asyncio.sleep(0.5)
                    continue
                logger.warning("MCP Server %s 不可达(%s),跳过其工具", server_name, type(exc).__name__)
        if not tools:
            continue
        specs.extend(
            spec_from_langchain_tool(tool, source="mcp", mcp_server=server_name, format_result=FORMATTERS.get(tool.name))
            for tool in tools
        )
    return specs
