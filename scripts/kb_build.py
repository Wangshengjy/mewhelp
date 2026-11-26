"""uv run python -m scripts.kb_build [--reset] —— 全量切块 → 指纹查重 → 写 pending。

kb_build 无跨运行幂等：knowledge_chunks 里已有文档类块时拒绝执行，提示先清空
重建（--reset 会删掉文档类块及其 Milvus 向量后重建）。
"""

from __future__ import annotations

import argparse

from scripts._common import PROJECT_ROOT, load_settings, print_header, session_factory_from
from app.kb.pipeline import build_knowledge_base


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库建库：切块 + 查重 + 入库(pending)")
    parser.add_argument("--reset", action="store_true", help="先清空已有文档类块（含 Milvus 向量）再重建")
    args = parser.parse_args()

    print_header("知识库建库 kb_build" + ("（--reset 清空重建）" if args.reset else ""))
    session_factory = session_factory_from(load_settings())
    result = build_knowledge_base(session_factory, PROJECT_ROOT / "data" / "kb", reset=args.reset)
    if not result.get("ok"):
        print(f"[refused] {result.get('message')}")
        return 1
    for document in result.get("documents", []):
        print(f"  [{document['file']}] 插入 {document['inserted']} 块，指针回填 {document['linked']} 块")
    print(result["message"])
    print("下一步：uv run python -m scripts.kb_vectorize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
