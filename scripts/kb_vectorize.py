"""uv run python -m scripts.kb_vectorize —— 捞 pending → 批量嵌入 → upsert Milvus → done。

逐批（每批 ≤32 块）提交；中断后重跑只会处理剩余 pending，天然支持断点续跑。
"""

from __future__ import annotations

from scripts._common import embed_ready, load_settings, print_header, session_factory_from
from app.kb.embeddings import EmbeddingError
from app.kb.milvus_store import MilvusStore
from app.kb.pipeline import vectorize_pending


def main() -> int:
    print_header("向量化 pending → Milvus")
    settings = load_settings()
    if not embed_ready(settings):
        return 2
    store = MilvusStore(settings.milvus_uri)
    if not store.is_available():
        print(f"[error] Milvus 不可达（{settings.milvus_uri}），请先 docker compose up -d 并等待 healthy")
        return 1
    session_factory = session_factory_from(settings)
    try:
        result = vectorize_pending(session_factory, settings, store=store)
    except EmbeddingError as exc:
        print(f"[error] {exc}")
        print("已完成批次已提交；修复后重跑本命令即可续跑剩余 pending")
        return 1
    if not result.get("ok"):
        print(f"[error] {result.get('message')}")
        return 1
    stats = result.get("stats") or {}
    print(
        f"pending={stats.get('pending')} done={stats.get('done')} total={stats.get('total')} "
        f"milvus={stats.get('milvus', 'N/A（Milvus 不可达）')}"
    )
    print(result["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
