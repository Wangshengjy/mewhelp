"""ch09 反馈入口 + 审核 API:POST /api/feedback 落池回捞、review 队列/详情/通过写回/拒绝。"""
import pytest
from fastapi.testclient import TestClient

from app.core import flywheel
from app.graph import runtime
from app.models import KnowledgeChunk, LowConfidenceQuestion


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        yield test_client, test_app


def _seed_conversation(app) -> str:
    from app.repositories.conversations import create_conversation
    with app.state.session_factory() as session:
        row = create_conversation(session)
        session.commit()
        return row.id


def _seed_low_confidence(app, conversation_id: str, raw: str, source: str, chunks=None):
    with app.state.session_factory() as session:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw, source=source,
            reason="evidence_confidence=0.123", retrieved_chunks=chunks)
        session.add(row)
        session.commit()
        return row.id


_SNAPSHOT = {
    "query": "猫窝能水洗吗",
    "retrieved_snapshot": [{"question": "q", "answer": "a", "rerank_score": 0.7, "section_path": "s"}],
    "citations": [{"n": 1, "section_path": "s"}],
    "fallback_source": "",
    "evidence_confidence": 0.8,
    "answer": "可以水洗。",
}


def test_feedback_down_pools_with_snapshot(client, monkeypatch):
    test_client, app = client
    conversation_id = _seed_conversation(app)

    async def fake_snapshot(cid, turn):
        return dict(_SNAPSHOT)

    monkeypatch.setattr(runtime, "get_turn_snapshot", fake_snapshot)

    resp = test_client.post("/api/feedback", json={
        "conversation_id": conversation_id, "turn_index": 1, "rating": "down", "comment": "答非所问"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["snapshot"]["query"] == "猫窝能水洗吗"
    assert body["snapshot"]["retrieved_snapshot"][0]["rerank_score"] == 0.7

    with app.state.session_factory() as session:
        rows = session.query(LowConfidenceQuestion).all()
    assert len(rows) == 1
    assert rows[0].source == "user_feedback"
    assert rows[0].retrieved_chunks[0]["rerank_score"] == 0.7
    assert "rating=down" in rows[0].reason and "comment=答非所问" in rows[0].reason

    # 幂等:同轮再点 👎 → duplicate,不重复入池
    resp2 = test_client.post("/api/feedback", json={
        "conversation_id": conversation_id, "turn_index": 1, "rating": "down"})
    assert resp2.json()["status"] == "duplicate"
    with app.state.session_factory() as session:
        assert session.query(LowConfidenceQuestion).count() == 1


def test_feedback_404_and_400(client, monkeypatch):
    test_client, _ = client

    async def not_found(cid, turn):
        raise runtime.ConversationNotFound(cid)

    async def turn_oob(cid, turn):
        raise runtime.TurnNotFound("越界")

    monkeypatch.setattr(runtime, "get_turn_snapshot", not_found)
    assert test_client.post("/api/feedback", json={
        "conversation_id": "missing", "turn_index": 1, "rating": "down"}).status_code == 404
    monkeypatch.setattr(runtime, "get_turn_snapshot", turn_oob)
    assert test_client.post("/api/feedback", json={
        "conversation_id": "conv", "turn_index": 99, "rating": "down"}).status_code == 400


def test_review_queue_and_detail_flow(client, monkeypatch):
    test_client, app = client
    conversation_id = _seed_conversation(app)
    lcq_a = _seed_low_confidence(app, conversation_id, "猫窝可以洗吗", "retrieval_low_conf",
                                 chunks=[{"question": "q1", "answer": "a1", "rerank_score": 0.2, "section_path": "s"}])
    lcq_b = _seed_low_confidence(app, conversation_id, "cat bed 能洗不", "user_feedback")

    # 队列空 → 跑飞轮(mock 标准化) → 两缺口
    def fake_normalize(settings, raw_question, candidates):
        hit = next((c for c in candidates if "猫窝" in c["normalized_question"]), None)
        return flywheel.FlywheelVerdict(
            normalized_question="猫窝是否支持水洗",
            matched_id=hit["id"] if hit else None,
            suggested_answer="可以,需手洗")
    monkeypatch.setattr(flywheel, "normalize_and_match", fake_normalize)
    result = flywheel.run_pipeline(app.state.session_factory, app.state.settings, limit=10)
    assert result["created"] == 1 and result["merged"] == 1

    items = test_client.get("/api/review/queue").json()["items"]
    assert len(items) == 1
    item = items[0]
    assert item["normalized_question"] == "猫窝是否支持水洗"
    assert item["occurrence_count"] == 2 and item["review_status"] == "待审"

    detail = test_client.get(f"/api/review/{item['id']}").json()
    assert {raw["raw_question"] for raw in detail["raw_questions"]} == {"猫窝可以洗吗", "cat bed 能洗不"}
    raw_a = next(r for r in detail["raw_questions"] if r["raw_question"] == "猫窝可以洗吗")
    assert raw_a["retrieved_chunks"][0]["rerank_score"] == 0.2
    assert detail["ai_suggested_answer"] == "可以,需手洗"


def test_review_approve_writes_back_to_kb(client, monkeypatch):
    test_client, app = client
    conversation_id = _seed_conversation(app)
    _seed_low_confidence(app, conversation_id, "猫窝可以洗吗", "user_feedback")
    monkeypatch.setattr(flywheel, "normalize_and_match",
                        lambda s, q, c: flywheel.FlywheelVerdict(
                            normalized_question="猫窝是否支持水洗", matched_id=None, suggested_answer=""))
    flywheel.run_pipeline(app.state.session_factory, app.state.settings, limit=10)
    item = test_client.get("/api/review/queue").json()["items"][0]

    resp = test_client.post(f"/api/review/{item['id']}/approve", json={
        "approved_answer": "可以水洗,阴干即可,不可烘干。"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "approved"
    assert body["kb"]["inserted"] == 1 and body["kb"]["vectorized"] == 0  # hermetic 无 embed key → pending

    with app.state.session_factory() as session:
        chunks = session.query(KnowledgeChunk).all()
    assert len(chunks) == 1
    assert "猫窝是否支持水洗" in chunks[0].questions
    assert chunks[0].vector_id == "" or chunks[0].vector_id is None  # pending 待向量化

    # 队列里该条已不是待审
    assert test_client.get("/api/review/queue?status=待审").json()["items"] == []
    # 重复流转被拒(终态)
    assert test_client.post(f"/api/review/{item['id']}/approve", json={
        "approved_answer": "再改一次"}).status_code == 503


def test_review_reject_keeps_record(client, monkeypatch):
    test_client, app = client
    conversation_id = _seed_conversation(app)
    _seed_low_confidence(app, conversation_id, "随机问题", "user_feedback")
    monkeypatch.setattr(flywheel, "normalize_and_match",
                        lambda s, q, c: flywheel.FlywheelVerdict(
                            normalized_question="随机问题标准化", matched_id=None, suggested_answer=""))
    flywheel.run_pipeline(app.state.session_factory, app.state.settings, limit=10)
    item = test_client.get("/api/review/queue").json()["items"][0]

    assert test_client.post(f"/api/review/{item['id']}/reject", json={
        "note": "查过库,已有同类条目"}).status_code == 200
    detail = test_client.get(f"/api/review/{item['id']}").json()
    assert detail["review_status"] == "驳回"
    assert detail["approved_answer"] == "查过库,已有同类条目"  # 驳回备注留痕
    # 驳回是终审:再试流转 → 409/503
    assert test_client.post(f"/api/review/{item['id']}/approve", json={
        "approved_answer": "x"}).status_code == 503


def test_review_404(client):
    test_client, _ = client
    assert test_client.get("/api/review/9999").status_code == 404
    assert test_client.post("/api/review/9999/reject", json={}).status_code in (409, 503)
