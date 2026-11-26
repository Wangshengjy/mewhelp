"""ch06 · POST /api/actions/create-refund 与 /api/actions/resume 端点测试。

create-refund:真实 SQLite 仓库建退款工单(ticket_type=「退款」,列是 varchar(64)
无需迁移);reason 是固定五类下拉,自由文本不落库。
resume:monkeypatch runtime.stream_resume,断言 SSE 帧契约(text/refund_form/[DONE])。
"""

import pytest
from fastapi.testclient import TestClient

from app.models import Ticket
from app.repositories.conversations import create_conversation


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        yield test_client, test_app


def _new_conversation(test_app) -> str:
    with test_app.state.session_factory() as session:
        conversation_id = create_conversation(session).id
        session.commit()
    return conversation_id


# ── create-refund ────────────────────────────────────────────────────────────


def test_create_refund_persists_row(client):
    test_client, test_app = client
    conversation_id = _new_conversation(test_app)

    response = test_client.post(
        "/api/actions/create-refund",
        json={
            "conversation_id": conversation_id,
            "order_id": "1001",
            "reason": "七天无理由",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_no"]
    assert body["status"] == "退款申请已提交"
    with test_app.state.session_factory() as session:
        rows = session.query(Ticket).filter_by(conversation_id=conversation_id).all()
    assert len(rows) == 1
    assert rows[0].ticket_type == "退款"
    assert "1001" in rows[0].description
    assert "七天无理由" in rows[0].description


def test_create_refund_rejects_unknown_reason(client):
    test_client, _ = client
    response = test_client.post(
        "/api/actions/create-refund",
        json={"conversation_id": "conv-1", "order_id": "1001", "reason": "随便编的"},
    )
    assert response.status_code == 422


def test_create_refund_rejects_blank_order_id(client):
    test_client, _ = client
    response = test_client.post(
        "/api/actions/create-refund",
        json={"conversation_id": "conv-1", "order_id": "", "reason": "其他"},
    )
    assert response.status_code == 422


def test_create_refund_unknown_conversation_maps_503(client):
    """会话不存在 → 外键违例 → 503(与 create-ticket 同语义,不暴露内部细节)。"""
    test_client, _ = client
    response = test_client.post(
        "/api/actions/create-refund",
        json={"conversation_id": "no-such-conv", "order_id": "1001", "reason": "质量问题"},
    )
    assert response.status_code == 503


# ── resume(SSE) ──────────────────────────────────────────────────────────────


def test_resume_streams_sse_frames(client, monkeypatch):
    """resume 端点:EventSourceResponse 帧契约与 /api/chat/stream 一致。"""
    test_client, test_app = client
    conversation_id = _new_conversation(test_app)

    async def fake_stream_resume(cid, resume_value):
        assert cid == conversation_id
        assert resume_value == "1001"
        yield {"type": "delta", "text": "该订单符合七天无理由退款条件。"}
        yield {"type": "actions", "items": [{"type": "refund_form", "order_id": "1001"}]}
        yield {"type": "done", "conversation_id": cid}

    from app.api import actions

    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream_resume)

    with test_client.stream(
        "POST", "/api/actions/resume", json={"conversation_id": conversation_id, "order_id": "1001"}
    ) as response:
        body = "".join(response.iter_text())

    # ServerSentEvent 对 dict data 是 ensure_ascii JSON:按帧解析后断言语义
    import json as jsonlib

    frames = []
    for block in body.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    frames.append("[DONE]")
                else:
                    frames.append(jsonlib.loads(raw))

    deltas = "".join(frame["delta"] for frame in frames if "delta" in frame)
    assert "该订单符合七天无理由退款条件。" in deltas
    assert any(
        item["type"] == "refund_form" for frame in frames for item in frame.get("items", [])
    )
    assert frames[-1] == "[DONE]"


def test_resume_unknown_conversation_maps_404(client):
    test_client, test_app = client
    with test_client.stream(
        "POST", "/api/actions/resume", json={"conversation_id": "no-such-conv", "order_id": "1001"}
    ) as response:
        assert response.status_code == 404
