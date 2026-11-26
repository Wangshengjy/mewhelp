"""scripts.kb_repatch — diff data/kb documents against the corpus and patch in place.

- 重切块并与库内指纹比对：同小节未变 → 不动；同小节内容变了 → 保留原 id 更新并重嵌入；
- 小节消失 → 删块并清向量；新小节 → 插入；
- 最后跑一轮 vectorize_pending（只处理 pending），输出改动统计。

用法：uv run python -m scripts.kb_repatch
"""

from __future__ import annotations

from scripts._common import load_settings, print_header, session_factory_from
from app.kb.pipeline import repatch_knowledge_base


def main() -> None:
    from pathlib import Path

    print_header("知识库就地修块 kb_repatch（diff → 更新/删除/新增 → 重嵌入）")
    settings = load_settings()
    factory = session_factory_from(settings)
    kb_dir = Path(__file__).resolve().parents[1] / "data" / "kb"
    result = repatch_knowledge_base(factory, settings, kb_dir)
    for doc in result["documents"]:
        changed = doc["updated"] or doc["added"] or doc["removed"]
        print(
            f"  [{doc['file']}] 更新 {len(doc['updated'])} / 新增 {len(doc['added'])} /"
            f" 删除 {len(doc['removed'])} / 未变 {doc['unchanged']}"
            + ("" if not changed else "")
        )
        for path in doc["updated"]:
            print(f"    ~ 更新: {path}")
        for path in doc["added"]:
            print(f"    + 新增: {path}")
        for path in doc["removed"]:
            print(f"    - 删除: {path}")
    print(f"  {result['message']}")


if __name__ == "__main__":
    main()
