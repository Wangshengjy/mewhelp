"""ch08 · 订单/商品查询工具(ch02 mock 数据原样复用 mock_data)。

query_order 的描述保留「先查订单拿 tracking_no 再查物流」的引导语义:
物流轨迹自本章起由物流 MCP Server 的 query_logistics 接管(内置下线,不留重名)。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import mock_data, registry


class OrderInput(BaseModel):
    order_id: str = Field(min_length=1, max_length=64, description="订单号,例如 1001")


class ProductInput(BaseModel):
    product_id: str = Field(min_length=1, max_length=64, description="商品编号,例如 P-1001")


@tool(args_schema=OrderInput)
def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号(tracking_no)。用于用户询问某个订单情况时。
    要查物流轨迹,需先用本工具拿到订单的 tracking_no,再把它传给 query_logistics。"""
    return mock_data.build_order(order_id)


@tool(args_schema=ProductInput)
def query_product(product_id: str) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    return mock_data.build_product(product_id)


registry.register(registry.spec_from_langchain_tool(query_order, source="builtin"))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
