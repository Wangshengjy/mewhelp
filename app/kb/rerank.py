"""SiliconFlow ``/rerank`` client for BAAI/bge-reranker-v2-m3 (ch04).

The rerank upstream speaks the Jina/Cohere shape (``results`` + ``relevance_score``),
not the OpenAI protocol, so the request is hand-rolled with httpx. Never logs or
echoes the API key. Unconfigured or failing rerank degrades gracefully: callers
keep the pre-rerank order instead of breaking the online path.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import httpx

from app.settings import Settings

logger = logging.getLogger(__name__)

RERANK_TIMEOUT_SECONDS = 30
MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 1.0


class RerankUnavailableError(RuntimeError):
    """Raised when the rerank upstream is unreachable or answers badly."""


def rerank_available(settings: Settings) -> bool:
    """Rerank is optional: without a key the pipeline keeps the hybrid order."""
    return bool(settings.rerank_api_key)


def rerank(settings: Settings, query: str, documents: Sequence[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """Rerank ``documents`` against ``query``; returns (index, relevance_score).

    Ordered by relevance, best first, truncated to ``top_n`` (settings default).
    """
    if not documents:
        return []
    if not settings.rerank_api_key:
        raise RerankUnavailableError("RERANK_API_KEY 未配置，无法调用重排服务")
    effective_top_n = top_n if top_n is not None else settings.rerank_top_n
    payload = {
        "model": settings.rerank_model,
        "query": query,
        "documents": list(documents),
        "top_n": max(1, int(effective_top_n)),
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {settings.rerank_api_key}"}
    url = f"{settings.rerank_base_url}/rerank"
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=RERANK_TIMEOUT_SECONDS)
            if response.status_code != 200:
                raise RerankUnavailableError(f"重排服务返回 HTTP {response.status_code}")
            results = response.json().get("results") or []
            ranked: list[tuple[int, float]] = []
            for item in results:
                index = item.get("index")
                score = item.get("relevance_score")
                if index is None or score is None:
                    raise RerankUnavailableError("重排服务返回缺少 index/relevance_score")
                ranked.append((int(index), float(score)))
            ranked.sort(key=lambda pair: pair[1], reverse=True)
            return ranked
        except (httpx.HTTPError, RerankUnavailableError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                logger.warning("rerank call failed (%s), retrying once", type(exc).__name__)
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise RerankUnavailableError(f"重排调用失败（已重试 {MAX_RETRIES} 次）：{type(last_error).__name__}: {last_error}")
