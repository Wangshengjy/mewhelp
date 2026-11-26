"""ch08 · tool_audit_logs 审计仓储:五种状态可落、无会话上下文可空、来源/Server 字段齐。

表无外键(审计不能被引用约束拦),审计写失败不反拦工具执行由 engine 侧负责。
"""

from sqlalchemy import select

from app.models import ToolAuditLog
from app.repositories import tool_audit as tool_audit_repository


def test_insert_tool_audit_minimal(session_factory):
    with session_factory() as session:
        tool_audit_repository.insert_tool_audit(
            session,
            conversation_id=None,
            tool_call_id=None,
            tool_name="query_order",
            tool_source="builtin",
            mcp_server=None,
            arguments={"order_id": "1001"},
            result_summary="{}",
            status="成功",
            error_message=None,
            retry_count=0,
            duration_ms=12,
        )
        session.commit()

    with session_factory() as session:
        row = session.scalars(select(ToolAuditLog)).one()

    assert row.tool_name == "query_order" and row.status == "成功"
    assert row.conversation_id is None  # 无会话上下文可空
    assert row.arguments == {"order_id": "1001"}
    assert row.tool_source == "builtin" and row.mcp_server is None
    assert row.retry_count == 0 and row.duration_ms == 12


def test_insert_tool_audit_all_statuses(session_factory):
    with session_factory() as session:
        for st in ["成功", "失败", "超时", "校验拦下", "权限拒绝"]:
            tool_audit_repository.insert_tool_audit(
                session,
                conversation_id="conv-1",
                tool_call_id="tc-1",
                tool_name="create_ticket",
                tool_source="mcp",
                mcp_server="logistics",
                arguments=None,
                result_summary=None,
                status=st,
                error_message="原因",
                retry_count=2,
                duration_ms=None,
            )
        session.commit()

    with session_factory() as session:
        rows = session.scalars(select(ToolAuditLog)).all()

    assert {r.status for r in rows} == {"成功", "失败", "超时", "校验拦下", "权限拒绝"}
    assert all(r.mcp_server == "logistics" for r in rows)
    assert all(r.retry_count == 2 for r in rows)
