"""ch03 · 在线检索链路测试：dense 命中组装与降级路径（全 mock，不打真实网络）。"""

import json
from dataclasses import replace

import pytest

from app.database import build_engine, build_session_factory
from app.demo_seed import seed_demo_data
from app.kb.embeddings import EmbeddingError
from app.kb.milvus_store import MilvusUnavailableError
from app.kb.retrieval import (
    DEGRADE_EMPTY_MESSAGE,
    DEGRADE_UNCONFIGURED_MESSAGE,
    DEGRADE_UNAVAILABLE_MESSAGE,
    retrieve,
)
from app.models import Base
from app.repositories import knowledge as knowledge_repository
from app.settings import Settings
from app.tools.builtin.faq import query_faq

SQLITE_URL = "sqlite+pysqlite:///:memory:"

DENSE_SETTINGS = Settings(
    chat_base_url="https://example.test/v1",
    chat_model="test-model",
    chat_api_key="test-key",
    embed_api_key="test-embed-key",
    milvus_uri="http://localhost:19530",
)

QUERY_VECTOR = [0.1] * 1024

CHUNK_ROWS = (
    {
        "category": "售后政策 > 退货",
        "questions": "退货政策",
        "answer": "自签收起 7 天内可退货，30 天内可换货。",
        "section_path": "售后政策 > 退货",
        "content_type": "policy",
    },
    {
        "category": "货运与运费 > 运费计算",
        "questions": "运费计算",
        "answer": "满 99 元包邮，未满 99 元收取 10 元运费。",
        "section_path": "货运与运费 > 运费计算",
        "content_type": "policy",
        "is_key_clause": True,
    },
)


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    seed_demo_data(factory)
    yield factory
    engine.dispose()


def seed_chunks(session_factory):
    with session_factory() as session:
        ids = []
        for row in CHUNK_ROWS:
            chunk = knowledge_repository.insert_chunk(session, **row)
            ids.append(chunk.id)
        session.commit()
    return ids


def install_fake_embed(monkeypatch, *, error=None):
    captured = {}

    def fake_embed(settings, texts):
        captured["texts"] = list(texts)
        if error is not None:
            raise error
        return [[0.1] * 1024 for _ in texts]

    monkeypatch.setattr("app.kb.retrieval.embed_texts", fake_embed)
    return captured


def install_fake_store(monkeypatch, *, hits=None, rows_by_id=None, error=None):
    captured = {}
    rows_by_id = rows_by_id or {}

    class FakeStore:
        def __init__(self, uri):
            captured["uri"] = uri

        def search(self, query_vector, top_k):
            """ch03 dense-only entry returning (chunk_id, score) tuples."""
            captured["vector"] = query_vector
            captured["top_k"] = top_k
            if error is not None:
                raise error
            return list(hits or [])

        @staticmethod
        def _hit(chunk_id, score):
            row = rows_by_id.get(chunk_id, {})
            return {
                "id": chunk_id,
                "score": score,
                "question": row.get("questions", ""),
                "answer": row.get("answer", ""),
                "section_path": row.get("section_path", ""),
                "content_type": row.get("content_type", ""),
                "category": row.get("category", ""),
            }

        def dense_search(self, query_vector, top_k, *, category=None, content_type=None):
            if error is not None:
                raise error
            return [self._hit(chunk_id, score) for chunk_id, score in (hits or [])][:top_k]

        def bm25_search(self, query_text, top_k, *, category=None, content_type=None):
            captured["bm25_text"] = query_text
            if error is not None:
                raise error
            return [self._hit(chunk_id, score) for chunk_id, score in (hits or [])][:top_k]

        def hybrid_search(self, query_vector, query_text, top_k, **kwargs):
            captured["hybrid_bm25_text"] = query_text
            if error is not None:
                raise error
            return [self._hit(chunk_id, score) for chunk_id, score in (hits or [])][:top_k]

    monkeypatch.setattr("app.kb.retrieval.MilvusStore", FakeStore)
    return captured


def invoke(tool, args, *, call_id="call-1", conversation_id="conv-1", session_factory=None, settings=None):
    merged = {**args, "conversation_id": conversation_id, "session_factory": session_factory, "settings": settings}
    return tool.invoke({"name": tool.name, "args": merged, "id": call_id, "type": "tool_call"})


# ---------------------------------------------------------------------------
# dense 正常路径
# ---------------------------------------------------------------------------


def test_dense_retrieval_assembles_matches_from_mysql_in_score_order(session_factory, monkeypatch):
    ids = seed_chunks(session_factory)
    embed = install_fake_embed(monkeypatch)
    store = install_fake_store(monkeypatch, hits=[(ids[1], 0.93), (ids[0], 0.81), (9999, 0.5)])

    result = retrieve(session_factory, DENSE_SETTINGS, "邮费是多少", top_k=3)

    assert result.degraded is False
    assert result.note is None
    assert [match["id"] for match in result.matches] == [ids[1], ids[0]], "按分数序且丢弃 Milvus 里的陈旧 id"
    assert [match["score"] for match in result.matches] == [0.93, 0.81]
    assert result.matches[0]["question"] == "运费计算"
    assert "满 99 元包邮" in result.matches[0]["answer"]
    assert set(result.matches[0]) == {"id", "question", "answer", "category", "score"}
    assert store["uri"] == DENSE_SETTINGS.milvus_uri
    assert store["vector"] == QUERY_VECTOR
    assert store["top_k"] == 3
    assert embed["texts"] == ["邮费是多少"]


# ---------------------------------------------------------------------------
# 未配置即降级（不触任何网络）
# ---------------------------------------------------------------------------


def test_empty_embed_key_degrades_to_keyword_lookup(session_factory):
    result = retrieve(session_factory, Settings.hermetic_defaults(), "退货", top_k=3)

    assert result.degraded is True
    assert result.note == DEGRADE_UNCONFIGURED_MESSAGE
    assert result.matches, "关键词降级应命中演示 FAQ"
    assert {match["question"] for match in result.matches} >= {"演示商品可以退货吗？"}
    assert all(set(match) == {"id", "question", "answer", "category", "score"} for match in result.matches)
    assert all(match["score"] is None for match in result.matches)


def test_missing_milvus_uri_degrades_before_any_network_call(session_factory, monkeypatch):
    embed = install_fake_embed(monkeypatch)
    store = install_fake_store(monkeypatch, hits=[(1, 0.9)])
    settings = replace(DENSE_SETTINGS, milvus_uri="")

    result = retrieve(session_factory, settings, "退货", top_k=3)

    assert result.degraded is True
    assert result.note == DEGRADE_UNCONFIGURED_MESSAGE
    assert "texts" not in embed, "未配置时不应发起嵌入调用"
    assert "vector" not in store, "未配置时不应访问 Milvus"
    assert result.matches


# ---------------------------------------------------------------------------
# 异常降级
# ---------------------------------------------------------------------------


def test_milvus_outage_degrades_to_keyword_lookup(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, error=MilvusUnavailableError("连接失败"))

    result = retrieve(session_factory, DENSE_SETTINGS, "退货", top_k=3)

    assert result.degraded is True
    assert result.note == DEGRADE_UNAVAILABLE_MESSAGE
    assert any("七天" in match["answer"] for match in result.matches)


def test_embedding_failure_degrades_without_touching_milvus(session_factory, monkeypatch):
    install_fake_embed(monkeypatch, error=EmbeddingError("上游 500"))
    store = install_fake_store(monkeypatch, hits=[(1, 0.9)])

    result = retrieve(session_factory, DENSE_SETTINGS, "退货", top_k=3)

    assert result.degraded is True
    assert result.note == DEGRADE_UNAVAILABLE_MESSAGE
    assert "vector" not in store, "嵌入失败后不应继续访问 Milvus"
    assert result.matches


def test_zero_recall_degrades_to_keyword_lookup(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, hits=[])

    result = retrieve(session_factory, DENSE_SETTINGS, "退货", top_k=3)

    assert result.degraded is True
    assert result.note == DEGRADE_EMPTY_MESSAGE
    assert result.matches


def test_degraded_lookup_keeps_explicit_no_match(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, error=MilvusUnavailableError("x"))

    result = retrieve(session_factory, DENSE_SETTINGS, "邮费", top_k=3)

    assert result.degraded is True
    assert result.matches == [], "演示语料不含“邮费”，降级后仍需显式空结果"


# ---------------------------------------------------------------------------
# query_faq 工具契约（ch02 出参结构一字不变）
# ---------------------------------------------------------------------------


def test_query_faq_keeps_exact_contract_on_retrieval_hits(session_factory, monkeypatch):
    """ch04 起默认走 hybrid_rerank 管线；出参契约与 ch02/ch03 完全一致。

    rerank 未配置 → 保持混合序；self-check 打桩为足够，避免触网。
    """
    ids = seed_chunks(session_factory)
    install_fake_embed(monkeypatch)
    install_fake_store(
        monkeypatch,
        hits=[(ids[1], 0.93)],
        rows_by_id={ids[0]: CHUNK_ROWS[0], ids[1]: CHUNK_ROWS[1]},
    )
    monkeypatch.setattr(
        "app.kb.retrieval._understand_query",
        lambda settings, query, understand_fn=None: {"standard": query, "expanded": []},
    )
    monkeypatch.setattr(
        "app.kb.selfcheck.check_sufficient", lambda *args, **kwargs: {"useful": True, "reason": ""}
    )
    payload = json.loads(
        invoke(query_faq, {"keyword": "邮费"}, session_factory=session_factory, settings=DENSE_SETTINGS).content
    )

    assert set(payload) == {"keyword", "found", "matches"}
    assert payload["keyword"] == "邮费"
    assert payload["found"] is True
    assert set(payload["matches"][0]) == {"question", "answer", "category"}, "matches 恒为三键，不得泄漏 id/score"
    assert payload["matches"][0]["question"] == "运费计算"
    assert "满 99 元包邮" in payload["matches"][0]["answer"]


def test_query_faq_degraded_note_is_explicit_for_the_model(session_factory):
    payload = json.loads(
        invoke(
            query_faq, {"keyword": "邮费"},
            session_factory=session_factory, settings=Settings.hermetic_defaults(),
        ).content
    )

    assert set(payload) == {"keyword", "found", "matches", "message"}
    assert payload["found"] is False
    assert payload["matches"] == []
    assert DEGRADE_UNCONFIGURED_MESSAGE in payload["message"]
    assert "未找到匹配的演示 FAQ" in payload["message"]


def test_query_faq_degraded_note_prepended_when_hits_found(session_factory):
    payload = json.loads(
        invoke(
            query_faq, {"keyword": "  退货  "},
            session_factory=session_factory, settings=None,  # 未传 settings → hermetic 降级
        ).content
    )

    assert payload["keyword"] == "退货"
    assert payload["found"] is True
    assert payload["message"].startswith(DEGRADE_UNCONFIGURED_MESSAGE)
    assert any(match["question"] == "演示商品可以退货吗？" for match in payload["matches"])
