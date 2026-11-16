"""LangChain tools for demo order, product, logistics, FAQ and ticket queries."""

import json
from collections.abc import Callable
from typing import Annotated

from langchain_core.tools import BaseTool, InjectedToolArg, InjectedToolCallId, tool
from sqlalchemy.orm import Session

from app.repositories import faq as faq_repository
from app.repositories import tickets as tickets_repository
from app.tools import mock_data
from app.tools.schemas import (
    CreateTicketArgs,
    QueryFaqArgs,
    QueryLogisticsArgs,
    QueryOrderArgs,
    QueryProductArgs,
)


def build_business_tools(session_factory: Callable[[], Session]) -> list[BaseTool]:
    """Build the five approved tools bound to a session factory."""

    @tool(args_schema=QueryOrderArgs)
    def query_order(
        order_id: str,
        conversation_id: Annotated[str, InjectedToolArg],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """查询订单基础信息（状态、金额、下单时间）。订单事实只能来自本工具结果。"""
        return json.dumps(mock_data.build_order(order_id), ensure_ascii=False)

    @tool(args_schema=QueryProductArgs)
    def query_product(
        product_id: str,
        conversation_id: Annotated[str, InjectedToolArg],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """查询商品基础信息（名称、分类、价格、库存）。商品事实只能来自本工具结果。"""
        return json.dumps(mock_data.build_product(product_id), ensure_ascii=False)

    @tool(args_schema=QueryLogisticsArgs)
    def query_logistics(
        order_id: str,
        conversation_id: Annotated[str, InjectedToolArg],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """查询订单物流状态和轨迹。物流事实只能来自本工具结果。"""
        return json.dumps(mock_data.build_logistics(order_id), ensure_ascii=False)

    @tool(args_schema=QueryFaqArgs)
    def query_faq(
        keyword: str,
        conversation_id: Annotated[str, InjectedToolArg],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """用关键词检索演示 FAQ。无匹配时返回空结果，不得编造政策。"""
        with session_factory() as session:
            rows = faq_repository.search_faq(session, keyword)
        matches = [
            {"question": row.question, "answer": row.answer, "category": row.category} for row in rows
        ]
        payload: dict = {"keyword": keyword, "found": bool(matches), "matches": matches}
        if not matches:
            payload["message"] = "未找到匹配的演示 FAQ，请勿编造政策。"
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CreateTicketArgs)
    def create_ticket(
        description: str,
        ticket_type: str,
        conversation_id: Annotated[str, InjectedToolArg],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """创建人工售后工单。仅在用户明确要求人工或开单时调用。"""
        with session_factory() as session:
            ticket = tickets_repository.create_ticket(
                session,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
                source_tool_call_id=tool_call_id,
            )
            payload = {
                "ticket_no": ticket.ticket_no,
                "status": ticket.status,
                "ticket_type": ticket.ticket_type,
                "description": ticket.description,
            }
            session.commit()
        return json.dumps(payload, ensure_ascii=False)

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
