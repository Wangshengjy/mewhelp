"""ch08 · submit_refund(拦截型,ch06 语义原样保留)。

agent_tools 在执行前截获本工具转成前端退款表单提议,函数体永不运行、不经引擎
权限门变化(ch05/ch06 的拦截语义不动)。仅保留注册动作让清单三样齐。
"""

from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from app.tools import registry

MAX_IDENTIFIER_LENGTH = 64


class RefundArgs(BaseModel):
    """ch06 · 模型判定可退后发起,前端表单确认后落库。"""

    order_id: Annotated[str, Field(min_length=1, max_length=MAX_IDENTIFIER_LENGTH, description="要退款的订单号")]
    reason: Annotated[
        str | None,
        Field(default=None, description="退款原因(用户原话概括;最终类目以前端固定下拉为准)"),
    ]

    @field_validator("order_id")
    @classmethod
    def order_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("visible tool arguments must not be blank")
        return value.strip()


@tool(args_schema=RefundArgs)
def submit_refund(order_id: str, reason: str | None = None) -> dict:
    """判定订单符合退款政策后调用,发起退款申请。实际提交以用户在前端退款表单确认为准;本工具仅表示「该单可退,已把确认入口交给用户」,不写数据库。"""
    # 拦截型工具:agent_tools 在执行前截获,本函数体永不运行(同 create_ticket 语义)
    return {"order_id": order_id, "reason": reason or "", "status": "待用户确认"}


registry.register(registry.spec_from_langchain_tool(submit_refund, source="builtin"))
