"""ch05 · POST /api/actions/create-ticket 端点测试(真实 SQLite 仓库)。"""

import pytest
from fastapi.testclient import TestClient

from app.models import Ticket
from app.repositories.conversations import create_conversation


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        yield test_client, test_app


def test_create_ticket_persists_row(client):
    test_client, test_app = client
    with test_app.state.session_factory() as session:
        conversation_id = create_conversation(session).id
        session.commit()

    response = test_client.post(
        "/api/actions/create-ticket",
        json={
            "conversation_id": conversation_id,
            "description": "商品屏幕碎了,要求处理",
            "ticket_type": "售后",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_no"]
    assert body["status"] == "已转人工"
    with test_app.state.session_factory() as session:
        rows = session.query(Ticket).filter_by(conversation_id=conversation_id).all()
    assert len(rows) == 1
    assert rows[0].description == "商品屏幕碎了,要求处理"
    assert rows[0].ticket_type == "售后"


def test_create_ticket_rejects_unknown_type(client):
    test_client, _ = client

    response = test_client.post(
        "/api/actions/create-ticket",
        json={"conversation_id": "conv-1", "description": "随便", "ticket_type": "火星类别"},
    )

    assert response.status_code == 422


def test_create_ticket_rejects_blank_description(client):
    test_client, _ = client

    response = test_client.post(
        "/api/actions/create-ticket",
        json={"conversation_id": "conv-1", "description": "", "ticket_type": "投诉"},
    )

    assert response.status_code == 422


def test_create_ticket_unknown_conversation_maps_503(client):
    """会话不存在 → 外键违例 → 503(不暴露内部细节)。"""
    test_client, _ = client

    response = test_client.post(
        "/api/actions/create-ticket",
        json={"conversation_id": "no-such-conv", "description": "描述", "ticket_type": "咨询"},
    )

    assert response.status_code == 503
