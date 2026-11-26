"""ch08 · 工具调用审计留痕(tool_audit_logs)仓储。

表无外键(审计不能被引用约束拦);审计写失败不许反拦工具执行——engine 侧
自行 try/except,本模块只管往 session 里 add 一行,commit 由调用方负责。
"""

from sqlalchemy.orm import Session

from app.models import ToolAuditLog


def insert_tool_audit(
    session: Session,
    *,
    conversation_id: str | None,
    tool_call_id: str | None,
    tool_name: str,
    tool_source: str,
    mcp_server: str | None = None,
    arguments: dict | None = None,
    result_summary: str | None = None,
    status: str,
    error_message: str | None = None,
    retry_count: int = 0,
    duration_ms: int | None = None,
) -> None:
    """工具调用审计落一条(含被权限拒、被校验拦的调用)。"""
    session.add(
        ToolAuditLog(
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_source=tool_source,
            mcp_server=mcp_server,
            arguments=arguments,
            result_summary=result_summary,
            status=status,
            error_message=error_message,
            retry_count=retry_count,
            duration_ms=duration_ms,
        )
    )
