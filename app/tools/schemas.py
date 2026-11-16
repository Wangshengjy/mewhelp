"""Pydantic argument schemas for the customer-support LangChain tools."""

from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, InjectedToolCallId
from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_IDENTIFIER_LENGTH = 64
MAX_KEYWORD_LENGTH = 128
MAX_DESCRIPTION_LENGTH = 2000

TicketType = Literal["refund", "exchange", "logistics_query", "quality_issue", "other"]
TICKET_TYPES: tuple[str, ...] = ("refund", "exchange", "logistics_query", "quality_issue", "other")


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("visible tool arguments must not be blank")
    return value.strip()


class InjectedToolArgs(BaseModel):
    """Base schema that forbids unknown arguments and carries runtime-only inputs."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: Annotated[str, InjectedToolArg] = Field(description="当前会话 ID，由运行时注入，模型不可见")
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(description="模型工具调用 ID，由运行时注入，模型不可见")


class QueryOrderArgs(InjectedToolArgs):
    order_id: Annotated[str, Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH, description="订单号")]

    @field_validator("order_id")
    @classmethod
    def order_id_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value)


class QueryProductArgs(InjectedToolArgs):
    product_id: Annotated[str, Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH, description="商品 ID")]

    @field_validator("product_id")
    @classmethod
    def product_id_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value)


class QueryLogisticsArgs(InjectedToolArgs):
    order_id: Annotated[str, Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH, description="订单号")]

    @field_validator("order_id")
    @classmethod
    def order_id_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value)


class QueryFaqArgs(InjectedToolArgs):
    keyword: Annotated[str, Field(min_length=1, max_length=MAX_KEYWORD_LENGTH, description="FAQ 检索关键词")]

    @field_validator("keyword")
    @classmethod
    def keyword_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value)


class CreateTicketArgs(InjectedToolArgs):
    description: Annotated[
        str, Field(min_length=1, max_length=MAX_DESCRIPTION_LENGTH, description="用户描述的售后问题")
    ]
    ticket_type: Annotated[TicketType, Field(description="工单类型")]

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value)
