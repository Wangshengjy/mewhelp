"""ch08 · create_ticket(全系统唯一写操作)。

执行由前端确认流把门:agent_tools 先 interrupt 推工单预览,用户确认后引擎才带
confirmed=True 放行到本函数;没确认的调用引擎直接拒绝(权限门,模型绕不过)。
demo_ticket_delay_seconds>0 时人为变慢(验收 6 写超时演示钩子)。
"""

import time
from typing import Annotated, Any, Literal

from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tools import registry

MAX_DESCRIPTION_LENGTH = 2000


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("visible tool arguments must not be blank")
    return value.strip()


class CreateTicketArgs(BaseModel):
    """模型可见口径只有 description/ticket_type;其余全是运行期注入参数。"""

    model_config = ConfigDict(extra="forbid")

    description: Annotated[
        str, Field(min_length=1, max_length=MAX_DESCRIPTION_LENGTH, description="用户描述的售后问题")
    ]
    ticket_type: Literal["售后", "投诉", "咨询"]
    conversation_id: Annotated[str | None, InjectedToolArg] = Field(default=None, description="当前会话 ID,由运行时注入,模型不可见")
    tool_call_id: Annotated[str | None, InjectedToolCallId] = Field(default=None, description="模型工具调用 ID,由运行时注入,模型不可见")
    session_factory: Annotated[Any, InjectedToolArg] = Field(default=None, description="DB 会话工厂,由运行时注入,模型不可见")
    settings: Annotated[Any, InjectedToolArg] = Field(default=None, description="应用配置,由运行时注入,模型不可见")

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value)


@tool(args_schema=CreateTicketArgs)
def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: str | None = None,
    tool_call_id: str | None = None,
    session_factory=None,
    settings=None,
) -> dict:
    """创建人工工单。仅当用户明确要求建工单/要求人工跟进时才调用;调用前必须确认 description
    (问题描述)已从用户处问清,信息不足时先向用户追问,严禁编造或用占位文本。
    ticket_type 从 售后/投诉/咨询 中选;工单关联的会话号由系统注入,你不要传。"""
    demo_delay = float(getattr(settings, "demo_ticket_delay_seconds", 0.0) or 0.0)
    if demo_delay > 0:  # 验收 6:环境变量注入写操作超时演示
        time.sleep(demo_delay)
    from app.repositories import tickets as tickets_repository

    with session_factory() as session:
        ticket = tickets_repository.create_ticket(
            session,
            conversation_id=conversation_id,
            description=description,
            ticket_type=ticket_type,
            source_tool_call_id=tool_call_id,
        )
        session.commit()
        return {
            "ticket_no": ticket.ticket_no,
            "status": "已转人工",
            "ticket_type": ticket.ticket_type,
            "description": ticket.description,
        }


registry.register(registry.spec_from_langchain_tool(create_ticket, source="builtin"))

