"""ch07 · GET /api/conversations 与 /api/conversations/{id}/messages 端点测试(真实 SQLite 仓库)。"""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.repositories.conversations import append_message, create_conversation


def _make_conversation(session_factory, *, created_at, summary="", messages=()):
    """建一个会话并按 (role, content) 顺序插消息,返回 (conversation_id, 首问内容)。"""
    with session_factory() as session:
        conversation = create_conversation(session)
        conversation.created_at = created_at
        if summary:
            conversation.summary = summary
        first_question = ""
        for index, (role, content) in enumerate(messages):
            append_message(session, conversation.id, role=role, content=content)
            if index == 0:
                first_question = content
        session.commit()
        return conversation.id, first_question


def test_conversations_list_orders_newest_first_with_preview_and_summary_badge(test_app):
    """列表新会话在前;每项带首问预览与 has_summary 徽标位。"""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    with TestClient(test_app) as client:
        factory = test_app.state.session_factory
        old_id, _ = _make_conversation(
            factory,
            created_at=base,
            messages=[("user", "订单 1001 的物流到哪了?"), ("assistant", "已发货")],
        )
        mid_id, _ = _make_conversation(
            factory,
            created_at=base + timedelta(minutes=5),
            summary="用户咨询订单 1001 物流,已告知发货状态。",
            messages=[("user", "订单 2002 能退吗?"), ("assistant", "支持七天无理由")],
        )
        new_id, _ = _make_conversation(
            factory,
            created_at=base + timedelta(minutes=10),
            messages=[("user", "我要转人工客服。"), ("assistant", "好的")],
        )

        response = client.get("/api/conversations")

    assert response.status_code == 200
    items = response.json()["items"]
    # 库里可能还有 seed 演示会话,这里只验证三个测试会话"新在前"的相对顺序
    order = [item["id"] for item in items]
    assert order.index(new_id) < order.index(mid_id) < order.index(old_id)
    by_id = {item["id"]: item for item in items}
    assert by_id[new_id]["preview"] == "我要转人工客服。"
    assert by_id[new_id]["has_summary"] is False
    assert by_id[mid_id]["preview"] == "订单 2002 能退吗?"
    assert by_id[mid_id]["has_summary"] is True
    assert by_id[old_id]["preview"] == "订单 1001 的物流到哪了?"
    assert by_id[old_id]["has_summary"] is False


def test_conversation_messages_returns_user_assistant_rows_in_order(test_app):
    """历史消息按 id 升序返回对话行;tool 行不进历史。"""
    with TestClient(test_app) as client:
        factory = test_app.state.session_factory
        with factory() as session:
            conversation = create_conversation(session)
            append_message(session, conversation.id, role="user", content="订单 1001 到哪了?")
            append_message(
                session,
                conversation.id,
                role="assistant",
                content="",
                tool_calls={"name": "query_logistics"},
                tool_call_id="call-1",
                tool_name="query_logistics",
            )
            append_message(session, conversation.id, role="tool", content="已发货", tool_call_id="call-1", tool_name="query_logistics")
            append_message(session, conversation.id, role="assistant", content="您的订单已发货。")
            session.commit()
            conversation_id = conversation.id

        response = client.get(f"/api/conversations/{conversation_id}/messages")

    assert response.status_code == 200
    items = response.json()["items"]
    assert items == [
        {"role": "user", "content": "订单 1001 到哪了?"},
        {"role": "assistant", "content": "您的订单已发货。"},
    ]
    assert all(set(item) == {"role", "content"} for item in items)


def test_conversation_messages_unknown_id_maps_404(test_app):
    """会话不存在 → 404,不抛 500。"""
    with TestClient(test_app) as client:
        response = client.get("/api/conversations/no-such-conv/messages")

    assert response.status_code == 404
