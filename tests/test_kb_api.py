"""ch03 · /kb 控制台 API 契约测试（SQLite + 全 mock，不触真实网络）。"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.kb.milvus_store import MilvusUnavailableError
from app.kb.retrieval import DEGRADE_UNCONFIGURED_MESSAGE
from app.models import KnowledgeChunk
from app.repositories import knowledge as knowledge_repository
from app.repositories import qa_staging as staging_repository
from app.settings import Settings


@pytest.fixture
def client(test_app):
    with TestClient(test_app) as test_client:
        yield test_client


def with_dense_settings(test_app):
    """在 lifespan 之后替换 settings，让该用例具备嵌入配置（仍不触网，靠 mock）。"""
    test_app.state.settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        embed_api_key="test-embed-key",
        milvus_uri="http://milvus.test:19530",
    )


def seed_chunk(test_app, **overrides):
    row = {
        "category": "货运与运费 > 运费计算",
        "questions": "运费计算",
        "answer": "满 99 元包邮，未满 99 元收取 10 元运费。",
        "section_path": "货运与运费 > 运费计算",
        "content_type": "policy",
        "is_key_clause": True,
        **overrides,
    }
    with test_app.state.session_factory() as session:
        chunk = knowledge_repository.insert_chunk(session, **row)
        session.commit()
        return chunk.id


def seed_staging(test_app, question, answer, *, status=None, batch_no="mine-test"):
    with test_app.state.session_factory() as session:
        row = staging_repository.insert_extracted(
            session, batch_no=batch_no, source_ref="conv-1", question=question, answer=answer
        )
        if status is not None:
            staging_repository.set_status(session, row.id, status)
        session.commit()
        return row.id


def chunk_ids(test_app):
    with test_app.state.session_factory() as session:
        return list(session.scalars(select(KnowledgeChunk.id)).all())


# ---------------------------------------------------------------------------
# 页面与材料清单
# ---------------------------------------------------------------------------


def test_kb_console_page_is_served(client):
    response = client.get("/kb")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "/api/kb" in response.text


def test_materials_lists_the_four_seed_documents(client):
    response = client.get("/api/kb/materials")

    assert response.status_code == 200
    data = response.json()
    assert {document["file"] for document in data["documents"]} == {
        "returns-policy.md",
        "product-faq.md",
        "product-specs.md",
        "after-sales-manual.md",
    }
    assert data["total_chunks"] == sum(document["chunk_count"] for document in data["documents"])
    assert data["total_key_clauses"] >= 1

    returns = next(document for document in data["documents"] if document["file"] == "returns-policy.md")
    assert returns["content_type"] == "policy"
    assert returns["key_clause_count"] >= 1
    assert returns["chunks"], "退货政策文档必须切出正文块"
    preview_keys = {
        "category",
        "questions",
        "answer",
        "answer_chars",
        "section_path",
        "content_type",
        "is_key_clause",
        "is_table_block",
    }
    assert all(set(chunk) == preview_keys for chunk in returns["chunks"])


# ---------------------------------------------------------------------------
# 预览与手工录入
# ---------------------------------------------------------------------------


def test_preview_chunks_without_persisting(client, test_app):
    response = client.post(
        "/api/kb/preview",
        json={"content": "# 测试政策\n\n## 运费计算\n\n满 99 元包邮，未满 99 元收取 10 元运费。", "content_type": "policy"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["content_type"] == "policy"
    assert data["chunk_count"] == 1
    assert data["key_clause_count"] == 1
    assert data["chunks"][0]["questions"] == "运费计算"
    assert chunk_ids(test_app) == [], "预览只切不写"


def test_preview_rejects_unknown_content_type(client):
    response = client.post(
        "/api/kb/preview",
        json={"content": "# 标题\n\n## 小节\n\n正文", "content_type": "movie"},
    )

    assert response.status_code == 422


def test_entries_persist_pending_chunks_and_dedup_on_repeat(client, test_app):
    body = {"content": "# 手工录入政策\n\n## 退款时效\n\n自签收起 7 天内可退货。", "content_type": "manual"}

    first = client.post("/api/kb/entries", json=body)
    assert first.status_code == 200
    data = first.json()
    assert data["ok"] is True
    assert data["inserted"] == 1
    assert data["skipped"] == 0
    assert len(data["chunk_ids"]) == 1

    repeat = client.post("/api/kb/entries", json=body)
    assert repeat.status_code == 200
    assert repeat.json()["inserted"] == 0
    assert repeat.json()["skipped"] == 1, "指纹相同的重复录入必须跳过"

    with test_app.state.session_factory() as session:
        chunk = session.get(KnowledgeChunk, data["chunk_ids"][0])
    assert chunk.content_type == "manual"
    assert chunk.vectorize_status == "pending"
    assert chunk.vector_id is None


def test_entries_without_embed_key_records_pending_note(client, test_app):
    response = client.post(
        "/api/kb/entries",
        json={"content": "# 手工录入政策\n\n## 退款时效\n\n自签收起 7 天内可退货。", "vectorize": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["vectorized"] == 0
    assert "EMBED_API_KEY" in (data["note"] or "")
    with test_app.state.session_factory() as session:
        chunk = session.get(KnowledgeChunk, data["chunk_ids"][0])
    assert chunk.vectorize_status == "pending"


def test_entries_vectorize_marks_rows_done(test_app, client, monkeypatch):
    class FakeStore:
        instances = []

        def __init__(self, uri):
            self.uri = uri
            self.upserted = []
            FakeStore.instances.append(self)

        def upsert(self, ids, vectors):
            self.upserted.extend(zip(ids, vectors))

        def upsert_rows(self, rows):
            self.upserted.extend((row["id"], row["dense"]) for row in rows)

        def count(self):
            return len(self.upserted)

    def fake_embed(settings, texts):
        assert settings.embed_api_key == "test-embed-key"
        return [[0.2] * 1024 for _ in texts]

    with_dense_settings(test_app)
    monkeypatch.setattr("app.kb.pipeline.embed_texts", fake_embed)
    monkeypatch.setattr("app.kb.pipeline.MilvusStore", FakeStore)

    response = client.post(
        "/api/kb/entries",
        json={"content": "# 手工录入政策\n\n## 退款时效\n\n自签收起 7 天内可退货。", "content_type": "manual", "vectorize": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["vectorized"] >= 1
    assert data["note"] is None

    with test_app.state.session_factory() as session:
        chunk = session.get(KnowledgeChunk, data["chunk_ids"][0])
    assert chunk.vectorize_status == "done"
    assert chunk.vector_id == str(chunk.id)
    assert FakeStore.instances, "必须通过 MilvusStore 写入向量"
    assert [chunk_id for chunk_id, _ in FakeStore.instances[0].upserted] == [chunk.id]


# ---------------------------------------------------------------------------
# 账本统计
# ---------------------------------------------------------------------------


class FakeCountingStore:
    def __init__(self, *, count=0, error=None):
        self._count = count
        self._error = error

    def count(self):
        if self._error is not None:
            raise self._error
        return self._count


def test_stats_reports_ledger_with_milvus_count(test_app, client, monkeypatch):
    seed_chunk(test_app)
    seed_chunk(test_app, questions="包邮门槛", answer="满 99 元包邮。")
    monkeypatch.setattr("app.api.kb.MilvusStore", lambda uri: FakeCountingStore(count=7))

    response = client.get("/api/kb/stats")

    assert response.status_code == 200
    assert response.json() == {"pending": 2, "done": 0, "total": 2, "milvus": 7, "note": None}


def test_stats_reports_milvus_outage_as_note(test_app, client, monkeypatch):
    monkeypatch.setattr(
        "app.api.kb.MilvusStore", lambda uri: FakeCountingStore(error=MilvusUnavailableError("连接失败"))
    )

    response = client.get("/api/kb/stats")

    assert response.status_code == 200
    data = response.json()
    assert data["milvus"] is None
    assert "Milvus 不可达" in data["note"]


# ---------------------------------------------------------------------------
# 检索自测
# ---------------------------------------------------------------------------


def test_search_degrades_to_keyword_without_embedding_config(client):
    response = client.post("/api/kb/search", json={"query": "退货"})

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"query", "mode", "top_k", "degraded", "note", "results"}
    assert data["query"] == "退货"
    assert data["mode"] == "dense"
    assert data["degraded"] is True
    assert data["note"] == DEGRADE_UNCONFIGURED_MESSAGE
    assert data["results"], "降级后命中演示 FAQ"
    assert all(set(result) == {"id", "question", "answer", "category", "score"} for result in data["results"])
    assert all(result["score"] is None for result in data["results"])


def test_search_dense_path_returns_scored_chunk_results(test_app, client, monkeypatch):
    chunk_id = seed_chunk(test_app)

    def fake_embed(settings, texts):
        return [[0.1] * 1024 for _ in texts]

    class FakeStore:
        def __init__(self, uri):
            self.uri = uri

        def search(self, query_vector, top_k):
            assert top_k == 3
            return [(chunk_id, 0.93)]

    with_dense_settings(test_app)
    monkeypatch.setattr("app.kb.retrieval.embed_texts", fake_embed)
    monkeypatch.setattr("app.kb.retrieval.MilvusStore", FakeStore)

    response = client.post("/api/kb/search", json={"query": "运费是多少", "top_k": 3})

    assert response.status_code == 200
    data = response.json()
    assert data["degraded"] is False
    assert data["note"] is None
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == chunk_id
    assert data["results"][0]["score"] == 0.93
    assert "满 99 元包邮" in data["results"][0]["answer"]


def test_search_rejects_unknown_mode_and_blank_query(client):
    assert client.post("/api/kb/search", json={"query": "退货", "mode": "keyword"}).status_code == 422
    assert client.post("/api/kb/search", json={"query": "   "}).status_code == 422
    assert client.post("/api/kb/search", json={"query": "退货", "top_k": 0}).status_code == 422


# ---------------------------------------------------------------------------
# 对话挖知识审核
# ---------------------------------------------------------------------------


def test_staging_listing_groups_rows_and_counts(test_app, client):
    kept_id = seed_staging(test_app, "发票可以开吗", "可以，支持电子发票。", status=staging_repository.STATUS_KEPT)
    seed_staging(test_app, "什么时候发货", "48 小时内发货。")

    response = client.get("/api/kb/staging")

    assert response.status_code == 200
    data = response.json()
    assert set(data["counts"]) == set(staging_repository.ALL_STATUSES)
    assert data["counts"]["extracted"] == 1
    assert data["counts"]["kept"] == 1
    rows = {row["id"]: row for row in data["rows"]}
    assert rows[kept_id]["status"] == "kept"
    assert all(
        set(row) == {"id", "batch_no", "source_ref", "question", "answer", "status"} for row in data["rows"]
    )

    filtered = client.get("/api/kb/staging", params={"status": "kept"}).json()
    assert [row["id"] for row in filtered["rows"]] == [kept_id]


def test_staging_approve_persists_mined_chunk_and_marks_approved(test_app, client):
    row_id = seed_staging(test_app, "发票可以开吗", "可以，支持电子发票。", status=staging_repository.STATUS_KEPT)

    response = client.post("/api/kb/staging/approve", json={"id": row_id})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "approved"
    assert data["vectorize_status"] == "pending"
    assert data["staging_id"] == row_id
    with test_app.state.session_factory() as session:
        chunk = session.get(KnowledgeChunk, data["chunk_id"])
        row = staging_repository.get_staging_row(session, row_id)
    assert chunk.content_type == "mined"
    assert chunk.category == "历史对话挖掘"
    assert chunk.questions == "发票可以开吗"
    assert row.status == "approved"


def test_staging_approve_conflicts_are_explicit(test_app, client):
    kept_id = seed_staging(test_app, "发票可以开吗", "可以，支持电子发票。", status=staging_repository.STATUS_KEPT)
    extracted_id = seed_staging(test_app, "什么时候发货", "48 小时内发货。")
    discarded_id = seed_staging(test_app, "能便宜点吗", "不讲价。", status=staging_repository.STATUS_DISCARDED)

    first = client.post("/api/kb/staging/approve", json={"id": kept_id})
    assert first.status_code == 200

    repeated = client.post("/api/kb/staging/approve", json={"id": kept_id})
    assert repeated.status_code == 409, "重复提交必须 409"

    extracted = client.post("/api/kb/staging/approve", json={"id": extracted_id})
    assert extracted.status_code == 409, "未去重的 extracted 行不能直接采纳"

    discarded = client.post("/api/kb/staging/approve", json={"id": discarded_id})
    assert discarded.status_code == 409, "弃用行再提交必须 409"

    missing = client.post("/api/kb/staging/approve", json={"id": 9999})
    assert missing.status_code == 404


def test_staging_reject_marks_rejected_and_keeps_trace(test_app, client):
    kept_id = seed_staging(test_app, "发票可以开吗", "可以，支持电子发票。", status=staging_repository.STATUS_KEPT)

    first = client.post("/api/kb/staging/reject", json={"id": kept_id})
    assert first.status_code == 200
    assert first.json()["status"] == "rejected"

    repeated = client.post("/api/kb/staging/reject", json={"id": kept_id})
    assert repeated.status_code == 409, "已弃用行再弃用必须 409"

    approve_after_reject = client.post("/api/kb/staging/approve", json={"id": kept_id})
    assert approve_after_reject.status_code == 409, "弃用行不得再采纳"

    with test_app.state.session_factory() as session:
        row = staging_repository.get_staging_row(session, kept_id)
    assert row.status == "rejected", "留痕不删"


def test_approve_rejects_duplicate_corpus_entry(test_app, client):
    first_id = seed_staging(test_app, "发票可以开吗", "可以，支持电子发票。", status=staging_repository.STATUS_KEPT)
    approved = client.post("/api/kb/staging/approve", json={"id": first_id})
    assert approved.status_code == 200

    # 相同问法+正文（归一化后）的又一条 kept 行：去重环节漏掉的兜底防线
    duplicate_id = seed_staging(
        test_app, "  发票  可以开吗？ ", "可以，支持电子发票。", status=staging_repository.STATUS_KEPT, batch_no="mine-next"
    )
    duplicate = client.post("/api/kb/staging/approve", json={"id": duplicate_id})
    assert duplicate.status_code == 409
