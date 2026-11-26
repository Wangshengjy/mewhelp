"""Knowledge retrieval for the online chat (ch03 dense + ch04 hybrid pipeline).

Two entry points share this module:

- ``retrieve`` — ch03 dense-only lookup with keyword fallback (kept verbatim for
  the /api/kb/search console and the ch03 tests);
- ``search_knowledge`` — ch04 four-strategy pipeline
  (``vector`` / ``bm25`` / ``hybrid`` / ``hybrid_rerank``). Query understanding
  (rewrite + synonym expansion) feeds the hybrid strategies only, and expansion
  is concatenated into the BM25 search text exclusively — the dense query stays
  the standard rewrite. Rerank (bge-reranker-v2-m3) applies to ``hybrid_rerank``;
  when the rerank upstream is unconfigured/failing the hybrid order is kept.

Both online tools and the evaluation script call ``search_knowledge``, so the
evaluated retrieval is exactly the served one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from app.kb import rerank as rerank_module
from app.kb.embeddings import EmbeddingError, embed_texts
from app.kb.milvus_store import MilvusStore
from app.repositories import faq as faq_repository
from app.repositories import knowledge as knowledge_repository
from app.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 3
STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")

DEGRADE_UNCONFIGURED_MESSAGE = "向量检索未配置（EMBED_API_KEY 或 MILVUS_URI 缺失），已降级为关键词检索。"
DEGRADE_UNAVAILABLE_MESSAGE = "Milvus 向量检索不可用，已降级为关键词检索。"
DEGRADE_EMPTY_MESSAGE = "向量检索无召回结果，已降级为关键词检索。"


class RetrievalNotConfiguredError(RuntimeError):
    """Raised when hybrid retrieval is attempted without EMBED_API_KEY / MILVUS_URI."""


@dataclass(slots=True)
class RetrievalResult:
    """Rich retrieval outcome; the query_faq tool maps this onto its contract."""

    matches: list[dict] = field(default_factory=list)
    degraded: bool = False
    note: str | None = None


def retrieve(
    session_factory,
    settings: Settings,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> RetrievalResult:
    """Run dense retrieval; any failure degrades to the keyword lookup."""
    if not settings.embed_api_key or not settings.milvus_uri:
        return _keyword_fallback(session_factory, query, DEGRADE_UNCONFIGURED_MESSAGE)
    try:
        vector = embed_texts(settings, [query])[0]
        hits = MilvusStore(settings.milvus_uri).search(vector, top_k)
    except (EmbeddingError, Exception) as exc:
        logger.warning("dense retrieval failed (%s); falling back to keyword lookup", type(exc).__name__)
        return _keyword_fallback(session_factory, query, DEGRADE_UNAVAILABLE_MESSAGE)
    if not hits:
        return _keyword_fallback(session_factory, query, DEGRADE_EMPTY_MESSAGE)

    with session_factory() as session:
        rows = knowledge_repository.get_chunks_by_ids(session, [chunk_id for chunk_id, _ in hits])
    rows_by_id = {row.id: row for row in rows}
    matches: list[dict] = []
    for chunk_id, score in hits:
        row = rows_by_id.get(chunk_id)
        if row is None:
            continue
        matches.append(
            {
                "id": row.id,
                "question": row.questions,
                "answer": row.answer,
                "category": row.category,
                "score": score,
            }
        )
    return RetrievalResult(matches=matches)


def _keyword_fallback(session_factory, query: str, note: str) -> RetrievalResult:
    """Original ch02 behaviour: parameterized LIKE lookup over the faq table."""
    with session_factory() as session:
        rows = faq_repository.search_faq(session, query)
    matches = [
        {
            "id": row.id,
            "question": row.question,
            "answer": row.answer,
            "category": row.category,
            "score": None,
        }
        for row in rows
    ]
    return RetrievalResult(matches=matches, degraded=True, note=note)


# ---------------------------------------------------------------------------
# ch04 · 四策略混合检索管线
# ---------------------------------------------------------------------------

_UNDERSTAND_CACHE: dict[tuple[str, str], dict] = {}
_UNDERSTAND_CACHE_LIMIT = 512


def arrange_head_tail(items: list) -> list:
    """Most relevant first, second-most relevant last, the rest in the middle.

    Mitigates lost-in-the-middle when the evidence block is assembled into the
    generation prompt. Input is expected in final relevance order.
    """
    if len(items) <= 2:
        return list(items)
    return [items[0], *items[2:], items[1]]


def _understand_query(settings: Settings, query: str, understand_fn: Callable | None) -> dict:
    """Colloquial → standard rewrite + synonym expansion (cached, fail-open)."""
    cache_key = (settings.chat_model, query)
    if cache_key in _UNDERSTAND_CACHE:
        return _UNDERSTAND_CACHE[cache_key]
    if understand_fn is None:
        from app.kb.query_understanding import default_model_fn, understand

        model_fn = default_model_fn(settings)
        result = understand(model_fn, query)
    else:
        result = understand_fn(query)
    if len(_UNDERSTAND_CACHE) >= _UNDERSTAND_CACHE_LIMIT:
        _UNDERSTAND_CACHE.clear()
    _UNDERSTAND_CACHE[cache_key] = result
    return result


def _rerank_hits(settings: Settings, query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Rerank recalled hits; degrade to the pre-rerank order when unavailable."""
    documents = [f"{hit['question']} {hit['answer']}" for hit in hits]
    try:
        ranked = rerank_module.rerank(settings, query, documents, top_n=settings.rerank_top_n)
    except rerank_module.RerankUnavailableError as exc:
        logger.warning("rerank unavailable (%s); keeping hybrid order", exc)
        return hits[:top_k]
    ranked = ranked[: max(1, top_k)]
    reranked: list[dict] = []
    for index, score in ranked:
        if index >= len(hits):
            continue
        hit = dict(hits[index])
        hit["rerank_score"] = score
        reranked.append(hit)
    return reranked


def search_knowledge(
    session_factory,
    settings: Settings,
    query: str,
    *,
    strategy: str | None = None,
    top_k: int | None = None,
    category: str | None = None,
    content_type: str | None = None,
    understand_fn: Callable | None = None,
) -> list[dict]:
    """Four-strategy retrieval pipeline; online default comes from settings.

    Returns hits ordered by final relevance; each hit carries id / score /
    question / answer / section_path / content_type / category, plus
    ``rerank_score`` for the ``hybrid_rerank`` strategy when rerank succeeded.
    Raises EmbeddingError / MilvusUnavailableError on infrastructure failure;
    configuration checks raise ValueError for unknown strategies.
    """
    effective_strategy = strategy or settings.retrieval_strategy
    if effective_strategy not in STRATEGIES:
        raise ValueError(f"未知检索策略：{effective_strategy}（可选 {'/'.join(STRATEGIES)}）")
    effective_top_k = top_k if top_k is not None else settings.rerank_top_n
    if not settings.embed_api_key or not settings.milvus_uri:
        raise RetrievalNotConfiguredError("EMBED_API_KEY 或 MILVUS_URI 未配置，无法执行混合检索")

    store = MilvusStore(settings.milvus_uri)

    if effective_strategy == "vector":
        vector = embed_texts(settings, [query])[0]
        return store.dense_search(vector, effective_top_k, category=category, content_type=content_type)

    if effective_strategy == "bm25":
        return store.bm25_search(query, effective_top_k, category=category, content_type=content_type)

    # hybrid / hybrid_rerank：Query 理解（改写 + 同义词扩展，扩展只进 BM25 检索文本）
    understood = _understand_query(settings, query, understand_fn)
    standard = understood["standard"]
    bm25_text = standard + (" " + " ".join(understood["expanded"]) if understood["expanded"] else "")
    vector = embed_texts(settings, [standard])[0]
    hits = store.hybrid_search(
        vector,
        bm25_text,
        effective_top_k,
        recall=settings.recall_top_k,
        rank_constant=settings.rrf_rank_constant,
        category=category,
        content_type=content_type,
    )
    if effective_strategy == "hybrid":
        return hits
    return _rerank_hits(settings, standard or query, hits, effective_top_k)
