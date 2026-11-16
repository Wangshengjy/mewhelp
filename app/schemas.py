"""Validated API request and structured extraction schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, description="本轮用户消息")
    conversation_id: str | None = Field(default=None, description="已有会话 ID；省略时创建新会话")

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, description="用户描述的售后问题")

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank")
        return value.strip()


class AfterSalesExtraction(BaseModel):
    """Three fixed fields extracted from a customer after-sales description."""

    model_config = ConfigDict(extra="forbid")

    order_number: str | None = Field(description="订单号原文；描述未提供时为 null")
    request_type: Literal["refund", "exchange", "logistics_query", "quality_issue"] | None = Field(
        description=(
            "用户希望客服办理的售后事项，不是商品故障原因；明确退款时填 refund，"
            "明确换新时填 exchange，查询发货或物流时填 logistics_query；只描述商品问题时填 quality_issue；无法判断时为 null"
        )
    )
    expected_solution: str | None = Field(
        description=(
            "用户明确提出的处理方式，简短规范地概括且不重复诉求类型；例如“换一副新的”概括为“换新”，"
            "“最好退回原支付方式”概括为“退回原支付方式”；未提出时为 null"
        )
    )
