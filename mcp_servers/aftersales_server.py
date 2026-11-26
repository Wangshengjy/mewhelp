"""售后 MCP Server(ch08 自建,mock 数据,不接真实系统、不建表)。
独立进程:.\\.venv\\Scripts\\python.exe mcp_servers/aftersales_server.py
查在保(query_warranty)、查退货进度(query_return_status)。
MOCK_DELAY_SECONDS 与 PORT 环境变量用法同 logistics_server.py。"""

import asyncio
import os
import random
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(
    "aftersales",
    host="127.0.0.1",
    port=int(os.environ.get("PORT", "8102")),
)

_DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0"))


@mcp.tool()
async def query_warranty(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单商品是否在保修期内(在保状态、到期日)。用于用户问保修/在保时。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"warranty:{order_id}")  # 种子固定 → 同单号稳定
    code = rng.choice(["IN_WARRANTY", "EXPIRED"])
    return {
        "order_id": order_id,
        "warranty_code": code,  # 内部枚举码,client 侧翻人话
        "warranty_until": f"2026-{rng.randint(8, 12):02d}-{rng.randint(1, 28):02d}",
        "policy_ref": "AS-POLICY-07",  # 内部政策编号,回答用不上,client 侧应剔除
    }


@mcp.tool()
async def query_return_status(
    order_id: Annotated[str, Field(description="订单号,例如 1001")],
) -> dict:
    """查询某订单的退货进度(审核中/退货中/已退款/无退货记录)。用于用户问退货到哪一步了。"""
    if _DELAY > 0:
        await asyncio.sleep(_DELAY)
    rng = random.Random(f"return:{order_id}")
    return {
        "order_id": order_id,
        "return_code": rng.choice(["AUDITING", "RETURNING", "REFUNDED", "NONE"]),
        "updated_at": f"2026-07-{rng.randint(1, 16):02d} 10:00",
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
