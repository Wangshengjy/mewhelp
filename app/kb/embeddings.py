"""OpenAI-compatible embedding client for BAAI/bge-m3 (SiliconFlow by default).

Never logs or echoes the API key. Batches requests (<=32 inputs per call) and
retries once on transport or protocol failures with a clear error type.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import httpx

from app.settings import Settings

logger = logging.getLogger(__name__)

BATCH_SIZE = 32
EMBED_DIMENSIONS = 1024
EMBED_TIMEOUT_SECONDS = 30
MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 1.0


class EmbeddingError(RuntimeError):
    """Raised when the embedding upstream is unreachable or answers badly."""


def embed_texts(settings: Settings, texts: Sequence[str]) -> list[list[float]]:
    """Embed a list of texts, batching internally; order of input is preserved."""
    if not texts:
        return []
    if not settings.embed_api_key:
        raise EmbeddingError("EMBED_API_KEY 未配置，无法调用嵌入服务")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = list(texts[start : start + BATCH_SIZE])
        vectors.extend(_embed_batch(settings, batch))
    return vectors


def _embed_batch(settings: Settings, batch: list[str]) -> list[list[float]]:
    url = f"{settings.embed_base_url}/embeddings"
    headers = {"Authorization": f"Bearer {settings.embed_api_key}"}
    payload = {"model": settings.embed_model, "input": batch}
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=EMBED_TIMEOUT_SECONDS)
            if response.status_code != 200:
                # never include the request body (it carries the key) in errors
                raise EmbeddingError(f"嵌入服务返回 HTTP {response.status_code}")
            data: list[Any] = response.json().get("data") or []
            if len(data) != len(batch):
                raise EmbeddingError(f"嵌入服务返回条数不符：期望 {len(batch)}，实际 {len(data)}")
            vectors: list[list[float]] = []
            for item in data:
                vector = item.get("embedding")
                if not isinstance(vector, list) or len(vector) != EMBED_DIMENSIONS:
                    raise EmbeddingError(
                        f"嵌入向量维度异常：期望 {EMBED_DIMENSIONS}，实际 {len(vector) if isinstance(vector, list) else type(vector).__name__}"
                    )
                vectors.append(vector)
            return vectors
        except (httpx.HTTPError, EmbeddingError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                logger.warning("embedding batch failed (%s), retrying once", type(exc).__name__)
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise EmbeddingError(f"嵌入调用失败（已重试 {MAX_RETRIES} 次）：{type(last_error).__name__}: {last_error}")
