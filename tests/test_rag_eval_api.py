"""ch04 · /api/rag-eval 评估台与 /api/kb 新端点契约测试（SQLite + 全 mock）。"""

import json

import pytest
from fastapi.testclient import TestClient

from app.models import FaithCase
from app.repositories import faith_cases as faith_cases_repository
from app.settings import Settings


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        yield test_client


def seed_case(test_app, eval_id, *, bucket="A_policy", status="未解决"):
    with test_app.state.session_factory() as session:
        case = faith_cases_repository.record_case(
            session,
            eval_id=eval_id,
            bucket=bucket,
            query=f"问题 {eval_id}",
            strategy="hybrid_rerank",
            answer="回答 " + eval_id,
            reason="裁判理由 " + eval_id,
            citations=[{"n": 1, "chunk_id": 1, "section_path": "退货政策", "question": "退款", "answer": "1-3 个工作日"}],
            judge_model="judge-x",
        )
        if status != "未解决":
            faith_cases_repository.resolve_case(session, eval_id=eval_id, status=status, resolution="处置说明")
        session.commit()
        return case.id


# ---------------------------------------------------------------------------
# 报告只读
# ---------------------------------------------------------------------------


def test_report_returns_404_before_first_run(client, tmp_path, monkeypatch):
    monkeypatch.setattr("app.api.rag_eval.REPORT_PATH", tmp_path / "missing.json")
    response = client.get("/api/rag-eval/report")
    assert response.status_code == 404


def test_report_reads_the_latest_json_report(client, tmp_path, monkeypatch):
    report_path = tmp_path / "rag_eval.json"
    report_path.write_text(json.dumps({"meta": {"questions": 300}, "strategies": []}), encoding="utf-8")
    monkeypatch.setattr("app.api.rag_eval.REPORT_PATH", report_path)

    response = client.get("/api/rag-eval/report")

    assert response.status_code == 200
    assert response.json()["meta"]["questions"] == 300


def test_rag_eval_page_is_served(client):
    response = client.get("/rag-eval")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


# ---------------------------------------------------------------------------
# 台账列表与处置
# ---------------------------------------------------------------------------


def test_faith_cases_listing_supports_pagination_and_status_filter(test_app, client):
    for index in range(1, 4):
        seed_case(test_app, f"A{index}")
    seed_case(test_app, "B1", bucket="B_model", status="已解决")

    data = client.get("/api/rag-eval/faith-cases", params={"page": 1, "page_size": 2}).json()
    assert data["total"] == 4
    assert data["page"] == 1 and data["page_size"] == 2
    assert len(data["items"]) == 2
    assert set(data["status_counts"]) == {"未解决", "已解决", "无需解决"}
    assert data["status_counts"]["未解决"] == 3

    filtered = client.get("/api/rag-eval/faith-cases", params={"status": "已解决"}).json()
    assert [item["eval_id"] for item in filtered["items"]] == ["B1"]

    item = data["items"][0]
    assert set(item) == {
        "id", "eval_id", "bucket", "query", "strategy", "answer", "reason",
        "citations", "judge_model", "status", "seen_count",
        "first_seen_at", "last_seen_at", "resolution", "resolved_at",
    }


def test_resolve_faith_case_persists_resolution(test_app, client):
    seed_case(test_app, "A9")

    response = client.post(
        "/api/rag-eval/faith-cases/A9/resolve",
        json={"status": "已解决", "resolution": "补充了运费知识块并重建索引"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "已解决"
    with test_app.state.session_factory() as session:
        row = session.query(FaithCase).filter(FaithCase.eval_id == "A9").one()
    assert row.resolution == "补充了运费知识块并重建索引"
    assert row.resolved_at is not None


def test_resolve_faith_case_rejects_bad_payload_and_unknown_id(client):
    # status 不在 pattern 内 / 说明缺失 → 422
    assert client.post("/api/rag-eval/faith-cases/A1/resolve", json={"status": "未解决", "resolution": "x"}).status_code == 422
    assert client.post("/api/rag-eval/faith-cases/A1/resolve", json={"status": "已解决"}).status_code == 422
    # 个案不存在 → 400
    assert client.post("/api/rag-eval/faith-cases/A1/resolve", json={"status": "已解决", "resolution": "x"}).status_code == 400


# ---------------------------------------------------------------------------
# 就地重跑
# ---------------------------------------------------------------------------


def test_rerun_conflicts_while_a_run_is_active(client, monkeypatch):
    monkeypatch.setattr("app.api.rag_eval._RERUN_STATE", {"running": True, "lines": [], "exit_code": None, "started_at": None})
    response = client.post("/api/rag-eval/rerun")
    assert response.status_code == 409


def test_rerun_status_reflects_global_state(client, monkeypatch):
    monkeypatch.setattr(
        "app.api.rag_eval._RERUN_STATE",
        {"running": False, "lines": ["line-1", "line-2"], "exit_code": 0, "started_at": "2026-01-01T00:00:00"},
    )
    data = client.get("/api/rag-eval/rerun/status").json()
    assert data["running"] is False
    assert data["exit_code"] == 0
    assert data["lines"] == ["line-1", "line-2"]


# ---------------------------------------------------------------------------
# /api/kb/repatch 就地修块
# ---------------------------------------------------------------------------


def test_repatch_returns_diff_statistics(client, monkeypatch):
    fake_result = {
        "ok": True, "updated": 2, "added": 1, "removed": 0, "unchanged": 41,
        "revectorized": 3, "documents": [], "message": "修块完成",
    }
    monkeypatch.setattr("app.api.kb.kb_pipeline.repatch_knowledge_base", lambda *args, **kwargs: fake_result)

    response = client.post("/api/kb/repatch")

    assert response.status_code == 200
    assert response.json()["updated"] == 2
    assert response.json()["revectorized"] == 3


def test_repatch_maps_failure_to_503(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("milvus down")

    monkeypatch.setattr("app.api.kb.kb_pipeline.repatch_knowledge_base", boom)
    response = client.post("/api/kb/repatch")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# /api/kb/search 四策略模式
# ---------------------------------------------------------------------------


def with_dense_settings(test_app):
    test_app.state.settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        embed_api_key="test-embed-key",
        milvus_uri="http://milvus.test:19530",
    )


HITS = [
    {
        "id": 7, "score": 0.8, "question": "退货政策", "answer": "7 天无理由退货。",
        "section_path": "退货退款政策 > 无理由退货", "content_type": "policy", "category": "退货退款政策",
    },
    {
        "id": 8, "score": 0.7, "question": "换货政策", "answer": "7 天内可换货。",
        "section_path": "退货退款政策 > 换货政策", "content_type": "policy", "category": "退货退款政策",
    },
]


class FakeSearchStore:
    def __init__(self, uri):
        self.uri = uri

    def dense_search(self, query_vector, top_k, *, category=None, content_type=None):
        return HITS[:top_k]

    def bm25_search(self, query_text, top_k, *, category=None, content_type=None):
        return HITS[:top_k]

    def hybrid_search(self, query_vector, query_text, top_k, **kwargs):
        return HITS[:top_k]


def install_search_mocks(test_app, monkeypatch, *, ranked=None):
    with_dense_settings(test_app)
    monkeypatch.setattr("app.kb.retrieval.MilvusStore", FakeSearchStore)
    monkeypatch.setattr("app.kb.retrieval.embed_texts", lambda settings, texts: [[0.1] * 1024 for _ in texts])
    monkeypatch.setattr(
        "app.kb.retrieval._understand_query",
        lambda settings, query, understand_fn=None: {"standard": query, "expanded": []},
    )
    if ranked is not None:
        monkeypatch.setattr("app.kb.rerank.rerank", lambda settings, query, docs, top_n=None: ranked)


def test_search_vector_mode_returns_dense_hits(test_app, client, monkeypatch):
    install_search_mocks(test_app, monkeypatch)

    response = client.post("/api/kb/search", json={"query": "退货", "mode": "vector", "top_k": 2})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "vector" and data["degraded"] is False
    assert [item["id"] for item in data["results"]] == [7, 8]


def test_search_bm25_mode_returns_keyword_hits(test_app, client, monkeypatch):
    install_search_mocks(test_app, monkeypatch)

    data = client.post("/api/kb/search", json={"query": "MH-LP100 续航", "mode": "bm25"}).json()

    assert data["mode"] == "bm25" and data["degraded"] is False
    assert data["results"][0]["question"] == "退货政策"


def test_search_hybrid_rerank_mode_orders_by_rerank_score(test_app, client, monkeypatch):
    install_search_mocks(test_app, monkeypatch, ranked=[(1, 0.95), (0, 0.4)])

    data = client.post("/api/kb/search", json={"query": "换货", "mode": "hybrid_rerank"}).json()

    assert data["mode"] == "hybrid_rerank" and data["degraded"] is False
    assert [item["id"] for item in data["results"]] == [8, 7], "重排分决定顺序"
    assert [item["rerank_score"] for item in data["results"]] == [0.95, 0.4]
