"""uv run python -m scripts.smoke_embed —— 开工冒烟：验证嵌入连通性与 1024 维。

Milvus 集合维度建好后不可改，因此建库前先用真实上游确认维度是 1024。
"""

from __future__ import annotations

from scripts._common import embed_ready, load_settings, print_header
from app.kb.embeddings import EMBED_DIMENSIONS, EmbeddingError, embed_texts

SMOKE_TEXTS = ("邮费是多少", "七天无理由退货怎么操作")


def main() -> int:
    print_header("嵌入服务冒烟测试 smoke_embed")
    settings = load_settings()
    if not embed_ready(settings):
        return 2
    print(f"base_url={settings.embed_base_url} model={settings.embed_model}")
    try:
        vectors = embed_texts(settings, list(SMOKE_TEXTS))
    except EmbeddingError as exc:
        print(f"[error] {exc}")
        return 1
    if len(vectors) != len(SMOKE_TEXTS):
        print(f"[error] 返回向量条数不符：期望 {len(SMOKE_TEXTS)}，实际 {len(vectors)}")
        return 1
    for index, vector in enumerate(vectors):
        if len(vector) != EMBED_DIMENSIONS:
            print(f"[error] 向量 {index} 维度 {len(vector)} != {EMBED_DIMENSIONS}（集合维度建好后不可改，先修配置再建库）")
            return 1
    print(f"[ok] {len(vectors)} 条文本全部返回 {EMBED_DIMENSIONS} 维向量，维度校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
