"""ch09 观测与成本 API + 白名单作业运行器(单测不起真子进程,聚合逻辑用 SQLite 真跑)。"""
import json

import pytest
from fastapi.testclient import TestClient

from app.core import jobs as jobs_core


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        yield test_client, test_app


def test_overview_three_blocks(client):
    test_client, app = client
    resp = test_client.get("/api/observability/overview")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"gap_top", "refusal_stats", "cost_report", "langfuse_enabled"}
    assert body["gap_top"]["items"] == []
    assert body["refusal_stats"]["total"] == 0


def test_gap_report_orders_by_occurrence(client):
    test_client, app = client
    from app.repositories import review_queue as repo
    with app.state.session_factory() as session:
        a = repo.insert_review_item(session, normalized_question="缺口A", ai_suggested_answer=None)
        b = repo.insert_review_item(session, normalized_question="缺口B", ai_suggested_answer=None)
        repo.increment_occurrence(session, b.id)
        repo.increment_occurrence(session, b.id)
        session.commit()
    body = test_client.get("/api/observability/gaps").json()
    items = body["items"]
    assert [it["normalized_question"] for it in items] == ["缺口B", "缺口A"]
    assert items[0]["occurrence_count"] == 3


def test_refusal_report_counts_by_source(client):
    test_client, app = client
    from app.models import LowConfidenceQuestion
    from app.repositories.conversations import create_conversation
    with app.state.session_factory() as session:
        conv = create_conversation(session)
        session.add_all([
            LowConfidenceQuestion(conversation_id=conv.id, raw_question="q1", source="retrieval_low_conf"),
            LowConfidenceQuestion(conversation_id=conv.id, raw_question="q2", source="user_feedback"),
            LowConfidenceQuestion(conversation_id=conv.id, raw_question="q3", source="user_feedback"),
        ])
        session.commit()
    body = test_client.get("/api/observability/overview").json()
    assert body["refusal_stats"]["counts"]["user_feedback"] == 2
    assert body["refusal_stats"]["counts"]["retrieval_low_conf"] == 1
    assert body["refusal_stats"]["total"] == 3


def test_jobs_whitelist_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_core, "LOG_DIR", tmp_path)
    monkeypatch.setattr(jobs_core, "REPORT_DIR", tmp_path)
    with pytest.raises(KeyError):
        jobs_core.start_job("rm -rf /")  # 非白名单拒绝
    status = jobs_core.job_status("flywheel_pipeline")
    assert status["running"] is False and status["log_tail"] == ""
    # 产物读取:白名单外拒绝
    assert jobs_core.read_artifact("eval_trend") is None
    (tmp_path / "eval_trend.json").write_text(json.dumps({"runs": [1]}), encoding="utf-8")
    assert jobs_core.read_artifact("eval_trend") == {"runs": [1]}
    assert jobs_core.read_artifact("../../etc/passwd") is None


def test_cost_artifact_endpoint_404_then_ok(client, monkeypatch, tmp_path):
    test_client, _ = client
    # 产物目录指到空 tmp:与真跑产物(data/ch09/reports)隔离,断言才稳定
    monkeypatch.setattr(jobs_core, "REPORT_DIR", tmp_path)
    assert test_client.get("/api/observability/cost").status_code == 404
    (tmp_path / "cost_by_intent.json").write_text(json.dumps({"grand_total": {"total_tokens": 10}}), encoding="utf-8")
    body = test_client.get("/api/observability/cost").json()
    assert body["grand_total"]["total_tokens"] == 10
    assert test_client.get("/api/observability/jobs-data/eval_trend").status_code == 404
    assert test_client.get("/api/observability/jobs-data/../secret").status_code in (404, 400)
