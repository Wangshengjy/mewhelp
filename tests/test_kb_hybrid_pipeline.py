"""ch04 · 混合检索管线与两道闸测试（全 mock，不打真实网络）。

覆盖：四策略路由、Query 理解只进 hybrid、重排与降级、首尾组装、
query_faq 机械闸/语义闸落池、引用角标暂存与解析。
"""

import json

import pytest

from app.database import build_engine, build_session_factory
from app.demo_seed import DEMO_CONVERSATION_ID, seed_demo_data
from app.kb import citations as citations_store
from app.kb import retrieval as kb_retrieval
from app.kb.milvus_store import MilvusUnavailableError
from app.kb.retrieval import RetrievalNotConfiguredError, arrange_head_tail, search_knowledge
from app.models import Base
from app.repositories import low_confidence as low_confidence_repository
from app.settings import Settings
from app.tools.builtin.faq import query_faq

SQLITE_URL = "sqlite+pysqlite:///:memory:"

SETTINGS = Settings(
    chat_base_url="https://example.test/v1",
    chat_model="test-model",
    chat_api_key="test-key",
    embed_api_key="test-embed-key",
    milvus_uri="http://localhost:19530",
)

HITS = [
    {
        "id": 11,
        "score": 0.72,
        "question": "退货政策",
        "answer": "自签收起 7 天内可退货。",
        "section_path": "退货退款政策 > 无理由退货",
        "content_type": "policy",
        "category": "退货退款政策 > 无理由退货",
    },
    {
        "id": 12,
        "score": 0.65,
        "question": "换货运费",
        "answer": "非质量问题换货运费由买家承担。",
        "section_path": "退货退款政策 > 换货政策 > 换货运费",
        "content_type": "policy",
        "category": "退货退款政策 > 换货政策",
    },
    {
        "id": 13,
        "score": 0.58,
        "question": "运费计算",
        "answer": "满 99 元包邮，未满收取 10 元运费。",
        "section_path": "商品与购物 FAQ > 运费怎么算",
        "content_type": "faq",
        "category": "商品与购物 FAQ > 运费怎么算",
    },
]


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    seed_demo_data(factory)
    yield factory
    engine.dispose()


@pytest.fixture(autouse=True)
def clear_understand_cache():
    kb_retrieval._UNDERSTAND_CACHE.clear()
    yield
    kb_retrieval._UNDERSTAND_CACHE.clear()


def install_fake_embed(monkeypatch):
    captured = {}

    def fake_embed(settings, texts):
        captured["texts"] = list(texts)
        return [[0.1] * 1024 for _ in texts]

    monkeypatch.setattr("app.kb.retrieval.embed_texts", fake_embed)
    return captured


def install_fake_store(monkeypatch, *, hits=None, error=None):
    captured = {}

    class FakeStore:
        def __init__(self, uri):
            captured["uri"] = uri

        def search(self, query_vector, top_k):
            """ch03 dense-only entry（降级路径会调它）。"""
            if error is not None:
                raise error
            return [(hit["id"], hit["score"]) for hit in (hits or [])][:top_k]

        def dense_search(self, query_vector, top_k, *, category=None, content_type=None):
            captured["dense_top_k"] = top_k
            captured["dense_filter"] = (category, content_type)
            if error is not None:
                raise error
            return list(hits or [])[:top_k]

        def bm25_search(self, query_text, top_k, *, category=None, content_type=None):
            captured["bm25_text"] = query_text
            if error is not None:
                raise error
            return list(hits or [])[:top_k]

        def hybrid_search(self, query_vector, query_text, top_k, *, recall=50, rank_constant=60,
                          category=None, content_type=None):
            captured["hybrid_bm25_text"] = query_text
            captured["hybrid_recall"] = recall
            captured["hybrid_rank_constant"] = rank_constant
            if error is not None:
                raise error
            return list(hits or [])[:top_k]

    monkeypatch.setattr("app.kb.retrieval.MilvusStore", FakeStore)
    return captured


def install_fake_rerank(monkeypatch, ranked=None, error=None):
    captured = {}

    def fake_rerank(settings, query, documents, top_n=None):
        captured["query"] = query
        captured["documents"] = list(documents)
        captured["top_n"] = top_n
        if error is not None:
            raise error
        return list(ranked or [])

    monkeypatch.setattr("app.kb.rerank.rerank", fake_rerank)
    return captured


def install_selfcheck(monkeypatch, useful=True, reason=""):
    captured = {}

    def fake_check(model_fn, query, evidence_texts, temperature=0.0):
        captured["query"] = query
        captured["evidence"] = list(evidence_texts)
        return {"useful": useful, "reason": reason}

    monkeypatch.setattr("app.kb.selfcheck.check_sufficient", fake_check)
    return captured


def invoke(tool, args, *, call_id="call-1", conversation_id=DEMO_CONVERSATION_ID, session_factory=None, settings=None):
    merged = {**args, "conversation_id": conversation_id, "session_factory": session_factory, "settings": settings}
    return tool.invoke({"name": tool.name, "args": merged, "id": call_id, "type": "tool_call"})


def pool_rows(session_factory):
    with session_factory() as session:
        rows = low_confidence_repository.recent_rows(session, limit=10)
    return rows


# ---------------------------------------------------------------------------
# 四策略路由
# ---------------------------------------------------------------------------


def test_vector_strategy_hits_dense_route_without_query_understanding(session_factory, monkeypatch):
    embed = install_fake_embed(monkeypatch)
    store = install_fake_store(monkeypatch, hits=HITS)

    def fail_understand(query):
        raise AssertionError("vector 策略不得运行 Query 理解")

    hits = search_knowledge(session_factory, SETTINGS, "退换货政策", strategy="vector", top_k=2, understand_fn=fail_understand)

    assert [hit["id"] for hit in hits] == [11, 12], "top_k 截断生效"
    assert embed["texts"] == ["退换货政策"], "vector 用原始问句嵌入"
    assert "hybrid_bm25_text" not in store and "bm25_text" not in store
    assert set(hits[0]) == {"id", "score", "question", "answer", "section_path", "content_type", "category"}


def test_bm25_strategy_uses_raw_query_text(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    store = install_fake_store(monkeypatch, hits=HITS)

    hits = search_knowledge(session_factory, SETTINGS, "运费怎么算", strategy="bm25", top_k=3)

    assert store["bm25_text"] == "运费怎么算", "bm25 直接用原问句，不走理解与扩展"
    assert [hit["id"] for hit in hits] == [11, 12, 13]


def test_hybrid_strategy_expands_bm25_text_only(session_factory, monkeypatch):
    embed = install_fake_embed(monkeypatch)
    store = install_fake_store(monkeypatch, hits=HITS)

    def understand(query):
        return {"standard": "退换货政策是怎样的", "expanded": ["退货", "换货"]}

    hits = search_knowledge(session_factory, SETTINGS, "退货咋整", strategy="hybrid", top_k=3, understand_fn=understand)

    assert store["hybrid_bm25_text"] == "退换货政策是怎样的 退货 换货", "扩展词只拼进 BM25 检索文本"
    assert embed["texts"] == ["退换货政策是怎样的"], "dense 用 standard 改写"
    assert store["hybrid_recall"] == SETTINGS.recall_top_k
    assert store["hybrid_rank_constant"] == SETTINGS.rrf_rank_constant
    assert [hit["id"] for hit in hits] == [11, 12, 13]


def test_hybrid_rerank_reorders_by_rerank_score(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, hits=HITS)
    rerank = install_fake_rerank(monkeypatch, ranked=[(2, 0.9), (0, 0.5)])

    def understand(query):
        return {"standard": "标准问法", "expanded": []}

    hits = search_knowledge(
        session_factory, SETTINGS, "退货咋整", strategy="hybrid_rerank", top_k=3, understand_fn=understand
    )

    assert rerank["documents"] == [f"{hit['question']} {hit['answer']}" for hit in HITS]
    assert [hit["id"] for hit in hits] == [13, 11], "按重排分降序，未被重排点名的 12 被挤出"
    assert [hit["rerank_score"] for hit in hits] == [0.9, 0.5]
    assert len(hits) == 2


def test_hybrid_rerank_degrades_to_hybrid_order_when_rerank_fails(session_factory, monkeypatch):
    from app.kb.rerank import RerankUnavailableError

    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, hits=HITS)
    install_fake_rerank(monkeypatch, error=RerankUnavailableError("HTTP 500"))

    hits = search_knowledge(
        session_factory, SETTINGS, "退货咋整", strategy="hybrid_rerank", top_k=3,
        understand_fn=lambda query: {"standard": query, "expanded": []},
    )

    assert [hit["id"] for hit in hits] == [11, 12, 13], "重排不可用 → 保持混合序"
    assert all("rerank_score" not in hit for hit in hits)


def test_unknown_strategy_raises_value_error(session_factory):
    with pytest.raises(ValueError):
        search_knowledge(session_factory, SETTINGS, "q", strategy="graph")


def test_unconfigured_settings_raise_retrieval_not_configured(session_factory):
    with pytest.raises(RetrievalNotConfiguredError):
        search_knowledge(session_factory, Settings.hermetic_defaults(), "q", strategy="vector")


def test_arrange_head_tail_puts_best_first_and_second_last():
    assert arrange_head_tail([]) == []
    assert arrange_head_tail(["a"]) == ["a"]
    assert arrange_head_tail(["a", "b"]) == ["a", "b"]
    assert arrange_head_tail(["s1", "s2", "s3", "s4"]) == ["s1", "s3", "s4", "s2"]


# ---------------------------------------------------------------------------
# query_faq 两道闸与引用暂存
# ---------------------------------------------------------------------------


def test_query_faq_refusal_on_zero_recall_falls_into_low_confidence_pool(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, hits=[])
    monkeypatch.setattr(
        "app.kb.retrieval._understand_query",
        lambda settings, query, understand_fn=None: {"standard": query, "expanded": []},
    )
    payload = json.loads(
        invoke(query_faq, {"keyword": "货到付款"}, session_factory=session_factory, settings=SETTINGS).content
    )

    assert payload["found"] is False and payload["matches"] == []
    assert "知识库证据不足" in payload["message"]
    rows = pool_rows(session_factory)
    assert len(rows) == 1
    assert rows[0].source == "retrieval_low_conf"
    assert rows[0].raw_question == "货到付款"
    assert "检索无召回" in rows[0].reason
    assert rows[0].conversation_id == DEMO_CONVERSATION_ID
    stash = citations_store.pop("call-1")
    assert stash == {"citations": [], "refused": True}


def test_query_faq_mechanical_gate_uses_rerank_score(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, hits=HITS[:1])
    install_fake_rerank(monkeypatch, ranked=[(0, 0.12)])  # 低于 rerank_min_score=0.3
    monkeypatch.setattr(
        "app.kb.retrieval._understand_query",
        lambda settings, query, understand_fn=None: {"standard": query, "expanded": []},
    )
    payload = json.loads(
        invoke(query_faq, {"keyword": "太阳系有几颗行星"}, session_factory=session_factory, settings=SETTINGS).content
    )

    assert payload["found"] is False
    rows = pool_rows(session_factory)
    assert rows[0].source == "retrieval_low_conf"
    assert "低于阈值" in rows[0].reason and "0.12" in rows[0].reason


def test_query_faq_self_check_gate_falls_into_pool(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, hits=HITS[:1])
    install_fake_rerank(monkeypatch, ranked=[(0, 0.9)])  # 机械闸放行
    install_selfcheck(monkeypatch, useful=False, reason="证据只讲退货，没讲赔付金额")
    monkeypatch.setattr(
        "app.kb.retrieval._understand_query",
        lambda settings, query, understand_fn=None: {"standard": query, "expanded": []},
    )
    payload = json.loads(
        invoke(query_faq, {"keyword": "退货赔付多少"}, session_factory=session_factory, settings=SETTINGS).content
    )

    assert payload["found"] is False
    assert "self_check" in payload["message"]
    rows = pool_rows(session_factory)
    assert rows[0].source == "self_check"
    assert "退货" in rows[0].reason


def test_query_faq_success_stashes_citation_map_in_head_tail_order(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, hits=HITS)
    install_fake_rerank(monkeypatch, ranked=[(0, 0.9), (1, 0.8), (2, 0.7)])
    install_selfcheck(monkeypatch, useful=True)
    monkeypatch.setattr(
        "app.kb.retrieval._understand_query",
        lambda settings, query, understand_fn=None: {"standard": query, "expanded": []},
    )
    payload = json.loads(
        invoke(query_faq, {"keyword": "退货"}, session_factory=session_factory, settings=SETTINGS).content
    )

    assert payload["found"] is True
    assert set(payload["matches"][0]) == {"question", "answer", "category"}
    stash = citations_store.pop("call-1")
    assert stash is not None and stash["refused"] is False
    citations = stash["citations"]
    assert [entry["n"] for entry in citations] == [1, 2, 3]
    assert [entry["chunk_id"] for entry in citations] == [11, 13, 12], "首尾组装：第 2 名垫底"
    assert citations[0]["title"] == "退货退款政策 > 无理由退货"
    assert citations[0]["score"] == 0.9, "有重排分时快照记重排分"


def test_query_faq_store_outage_falls_back_to_keyword_retrieval(session_factory, monkeypatch):
    install_fake_embed(monkeypatch)
    install_fake_store(monkeypatch, error=MilvusUnavailableError("连接失败"))
    monkeypatch.setattr(
        "app.kb.retrieval._understand_query",
        lambda settings, query, understand_fn=None: {"standard": query, "expanded": []},
    )
    payload = json.loads(
        invoke(query_faq, {"keyword": "退货"}, session_factory=session_factory, settings=SETTINGS).content
    )

    assert set(payload) == {"keyword", "found", "matches", "message"}
    assert payload["found"] is True
    assert any(match["question"] == "演示商品可以退货吗？" for match in payload["matches"])
    assert pool_rows(session_factory) == [], "降级路径不落池"


# ---------------------------------------------------------------------------
# 引用角标解析
# ---------------------------------------------------------------------------


def test_parse_markers_extracts_distinct_numbers_in_first_seen_order():
    assert citations_store.parse_markers("依据[1]和[3]，另有[1]重复。") == [1, 3]
    assert citations_store.parse_markers("没有引用") == []
    assert citations_store.parse_markers("编号[12]也支持") == [12]


def test_resolve_citations_filters_unknown_numbers():
    citations = [
        {"n": 1, "chunk_id": 11, "title": "退货", "question": "q1", "answer": "a1"},
        {"n": 2, "chunk_id": 12, "title": "换货", "question": "q2", "answer": "a2"},
    ]
    resolved = citations_store.resolve_citations("结论[2]，另见[9]。", citations)
    assert [entry["chunk_id"] for entry in resolved] == [12], "模型幻觉出的 [9] 被静默丢弃"
    assert citations_store.resolve_citations("无引用", citations) == []
