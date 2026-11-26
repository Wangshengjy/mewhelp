"""Milvus collection wrapper for the ch04 hybrid retrieval (pymilvus MilvusClient).

Collection ``knowledge`` (rebuilt in ch04):
- ``id`` INT64 primary key, auto_id=False, aligned with the MySQL chunk id;
- ``dense`` FLOAT_VECTOR(1024, COSINE) — bge-m3 embeddings;
- ``text`` VARCHAR with the built-in ``chinese`` analyzer + BM25 Function whose
  output lands in the ``sparse`` SPARSE_FLOAT_VECTOR field (Milvus 2.5+);
- scalar fields ``question`` / ``answer`` / ``section_path`` / ``content_type``
  / ``category`` fetched back in one search call (ch03 "Milvus 里取就行");

Every upsert is idempotent by primary key. BM25 full-text search and
``hybrid_search`` (RRF fusion) require Milvus Standalone >= 2.5.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from pymilvus import AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge"
PRIMARY_FIELD = "id"
VECTOR_FIELD = "dense"
SPARSE_FIELD = "sparse"
TEXT_FIELD = "text"
VECTOR_DIMENSION = 1024
METRIC_TYPE = "COSINE"
BM25_METRIC_TYPE = "BM25"
SEARCH_OUTPUT_FIELDS = ("question", "answer", "section_path", "content_type", "category")

# 入库文本 = category + questions + answer 拼接（与 dense 向量同源，两路召回同一批 chunk）
ANALYZER_PARAMS = {"type": "chinese"}

# Milvus VARCHAR max_length 按字符计；切块上限 800 字 + 标题/品类富余即可
_MAX_TEXT = 8192
_MAX_QUESTION = 1024
_MAX_ANSWER = 8192
_MAX_CATEGORY = 512
_MAX_SECTION = 512
_MAX_TYPE = 64


class MilvusUnavailableError(RuntimeError):
    """Raised when Milvus cannot be reached or refuses an operation."""


class MilvusStore:
    """Thin, test-friendly facade over ``MilvusClient`` for one collection."""

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._client: MilvusClient | None = None

    # -- connection --------------------------------------------------------
    def _ensure_client(self) -> MilvusClient:
        if self._client is None:
            try:
                self._client = MilvusClient(uri=self.uri)
            except Exception as exc:  # connection refused, DNS, auth...
                raise MilvusUnavailableError(f"Milvus 连接失败（{self.uri}）：{type(exc).__name__}") from exc
        return self._client

    def is_available(self) -> bool:
        try:
            self._ensure_client().list_collections()
            return True
        except Exception:
            return False

    # -- collection lifecycle ---------------------------------------------
    def ensure_collection(self) -> None:
        """Create the collection with the ch04 schema when missing (idempotent).

        An existing collection keeps whatever schema it was built with; a
        schema migration is an explicit drop + recreate (see drop_collection).
        """
        client = self._ensure_client()
        try:
            if client.has_collection(COLLECTION_NAME):
                return
            schema = self._build_schema()
            index_params = client.prepare_index_params()
            index_params.add_index(field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type=METRIC_TYPE)
            index_params.add_index(field_name=SPARSE_FIELD, index_type="SPARSE_INVERTED_INDEX", metric_type=BM25_METRIC_TYPE)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params=index_params,
            )
            logger.info("created Milvus collection %s (dense + BM25)", COLLECTION_NAME)
        except MilvusUnavailableError:
            raise
        except Exception as exc:
            raise MilvusUnavailableError(f"Milvus 建集合失败：{type(exc).__name__}: {exc}") from exc

    @staticmethod
    def _build_schema():
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(PRIMARY_FIELD, DataType.INT64, is_primary=True)
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        schema.add_field(TEXT_FIELD, DataType.VARCHAR, max_length=_MAX_TEXT, enable_analyzer=True, analyzer_params=ANALYZER_PARAMS)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("question", DataType.VARCHAR, max_length=_MAX_QUESTION)
        schema.add_field("answer", DataType.VARCHAR, max_length=_MAX_ANSWER)
        schema.add_field("section_path", DataType.VARCHAR, max_length=_MAX_SECTION)
        schema.add_field("content_type", DataType.VARCHAR, max_length=_MAX_TYPE)
        schema.add_field("category", DataType.VARCHAR, max_length=_MAX_CATEGORY)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=[TEXT_FIELD],
                output_field_names=[SPARSE_FIELD],
            )
        )
        return schema

    def drop_collection(self) -> None:
        """Drop the whole collection (schema migration / full rebuild only)."""
        client = self._ensure_client()
        try:
            if client.has_collection(COLLECTION_NAME):
                client.drop_collection(COLLECTION_NAME)
                logger.info("dropped Milvus collection %s", COLLECTION_NAME)
        except Exception as exc:
            raise MilvusUnavailableError(f"Milvus drop 集合失败：{type(exc).__name__}") from exc

    # -- data operations ---------------------------------------------------
    def upsert_rows(self, rows: Sequence[dict]) -> None:
        """Upsert ch04 rows: id / dense / text / question / answer / scalars.

        The ``sparse`` field is produced server-side by the BM25 Function from
        ``text``; it must not be included in the payload.
        """
        if not rows:
            return
        client = self._ensure_client()
        self.ensure_collection()
        data = []
        for row in rows:
            data.append(
                {
                    PRIMARY_FIELD: int(row["id"]),
                    VECTOR_FIELD: [float(x) for x in row["dense"]],
                    TEXT_FIELD: str(row["text"])[:_MAX_TEXT],
                    "question": str(row["question"])[:_MAX_QUESTION],
                    "answer": str(row["answer"])[:_MAX_ANSWER],
                    "section_path": str(row["section_path"])[:_MAX_SECTION],
                    "content_type": str(row["content_type"])[:_MAX_TYPE],
                    "category": str(row["category"])[:_MAX_CATEGORY],
                }
            )
        try:
            client.upsert(collection_name=COLLECTION_NAME, data=data)
        except Exception as exc:
            raise MilvusUnavailableError(f"Milvus upsert 失败：{type(exc).__name__}: {exc}") from exc

    def _filter_expr(self, category: str | None, content_type: str | None) -> str | None:
        """Milvus filter expression over the scalar fields (metadata pre-filter)."""
        clauses: list[str] = []
        if category:
            clauses.append(f'category like "{category}"')
        if content_type:
            clauses.append(f'content_type == "{content_type}"')
        return " and ".join(clauses) if clauses else None

    def _entities_to_hits(self, results) -> list[dict]:
        """Normalize client.search/hybrid_search output into hit dicts."""
        hits_raw = results[0] if results else []
        hits: list[dict] = []
        for hit in hits_raw:
            entity = hit.get("entity") or {}
            hits.append(
                {
                    "id": int(hit["id"]),
                    "score": float(hit.get("distance") or 0.0),
                    "question": entity.get("question") or "",
                    "answer": entity.get("answer") or "",
                    "section_path": entity.get("section_path") or "",
                    "content_type": entity.get("content_type") or "",
                    "category": entity.get("category") or "",
                }
            )
        return hits

    def _search(self, requests: list[AnnSearchRequest], top_k: int, *, ranker=None):
        client = self._ensure_client()
        try:
            if not client.has_collection(COLLECTION_NAME):
                return []
            if ranker is None:
                request = requests[0]
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    data=request.data,
                    anns_field=request.anns_field,
                    search_params=request.param,
                    limit=max(1, top_k),
                    filter=request.expr or "",
                    output_fields=list(SEARCH_OUTPUT_FIELDS),
                )
            else:
                results = client.hybrid_search(
                    collection_name=COLLECTION_NAME,
                    reqs=requests,
                    ranker=ranker,
                    limit=max(1, top_k),
                    output_fields=list(SEARCH_OUTPUT_FIELDS),
                )
        except Exception as exc:
            raise MilvusUnavailableError(f"Milvus search 失败：{type(exc).__name__}: {exc}") from exc
        return self._entities_to_hits(results)

    def _requests(self, anns_field: str, data: list, limit: int, expr: str | None, param: dict) -> AnnSearchRequest:
        """One search leg; metadata filtering rides on ``expr`` (per-request)."""
        return AnnSearchRequest(data=data, anns_field=anns_field, param=param, limit=max(1, limit), expr=expr)

    def dense_search(
        self, query_vector: Sequence[float], top_k: int, *, category: str | None = None, content_type: str | None = None
    ) -> list[dict]:
        """Pure dense route (evaluation strategy ``vector``)."""
        request = self._requests(
            VECTOR_FIELD,
            [[float(x) for x in query_vector]],
            top_k,
            self._filter_expr(category, content_type),
            {"metric_type": METRIC_TYPE},
        )
        return self._search([request], top_k)

    def bm25_search(
        self, query_text: str, top_k: int, *, category: str | None = None, content_type: str | None = None
    ) -> list[dict]:
        """Pure BM25 full-text route (evaluation strategy ``bm25``)."""
        request = self._requests(
            SPARSE_FIELD,
            [query_text],
            top_k,
            self._filter_expr(category, content_type),
            {"metric_type": BM25_METRIC_TYPE},
        )
        return self._search([request], top_k)

    def hybrid_search(
        self,
        query_vector: Sequence[float],
        query_text: str,
        top_k: int,
        *,
        recall: int = 50,
        rank_constant: int = 60,
        category: str | None = None,
        content_type: str | None = None,
    ) -> list[dict]:
        """dense + BM25 dual recall fused with RRF (evaluation strategy ``hybrid``)."""
        expr = self._filter_expr(category, content_type)
        requests = [
            self._requests(
                VECTOR_FIELD, [[float(x) for x in query_vector]], recall, expr, {"metric_type": METRIC_TYPE}
            ),
            self._requests(SPARSE_FIELD, [query_text], recall, expr, {"metric_type": BM25_METRIC_TYPE}),
        ]
        return self._search(requests, top_k, ranker=RRFRanker(rank_constant))

    def delete(self, ids: Sequence[int]) -> None:
        """Delete rows by primary key (used when resetting the doc corpus)."""
        client = self._ensure_client()
        try:
            if not client.has_collection(COLLECTION_NAME) or not ids:
                return
            client.delete(collection_name=COLLECTION_NAME, ids=[int(chunk_id) for chunk_id in ids])
        except Exception as exc:
            raise MilvusUnavailableError(f"Milvus delete 失败：{type(exc).__name__}") from exc

    def count(self) -> int:
        """Return the number of vectors currently stored in the collection."""
        client = self._ensure_client()
        try:
            if not client.has_collection(COLLECTION_NAME):
                return 0
            rows = client.query(
                collection_name=COLLECTION_NAME,
                filter=f"{PRIMARY_FIELD} >= 0",
                output_fields=["count(*)"],
            )
        except Exception as exc:
            raise MilvusUnavailableError(f"Milvus count 失败：{type(exc).__name__}") from exc
        if not rows:
            return 0
        return int(rows[0].get("count(*)") or 0)

    # -- ch03 兼容入口 -------------------------------------------------------
    def search(self, query_vector: Sequence[float], top_k: int) -> list[tuple[int, float]]:
        """ch03 dense-only lookup returning (chunk_id, score) tuples."""
        hits = self.dense_search(query_vector, top_k)
        return [(hit["id"], hit["score"]) for hit in hits]

    def upsert(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        """ch03-compatible upsert; scalar fields filled with placeholders.

        New code should call :meth:`upsert_rows` so the BM25 ``text`` field is
        populated (a placeholder text weakens BM25 recall).
        """
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        rows: list[dict] = []
        for chunk_id, vector in zip(ids, vectors):
            rows.append(
                {
                    "id": int(chunk_id),
                    "dense": vector,
                    "text": "",
                    "question": "",
                    "answer": "",
                    "section_path": "",
                    "content_type": "",
                    "category": "",
                }
            )
        self.upsert_rows(rows)

    def describe(self) -> dict[str, Any]:
        """Collection meta for diagnostics (schema/index presence)."""
        client = self._ensure_client()
        if not client.has_collection(COLLECTION_NAME):
            return {"exists": False}
        desc = client.describe_collection(COLLECTION_NAME)
        fields = [field.get("name") for field in desc.get("fields", [])]
        return {"exists": True, "fields": fields}
