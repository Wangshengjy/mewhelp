"""uv run python -m scripts.kb_preview —— 扫描 data/kb/*.md，打印切块预览。"""

from __future__ import annotations

from scripts._common import PROJECT_ROOT, print_header
from app.kb.pipeline import document_previews, load_kb_documents


def main() -> int:
    kb_dir = PROJECT_ROOT / "data" / "kb"
    print_header(f"知识库材料扫描 {kb_dir}")
    documents = load_kb_documents(kb_dir)
    if not documents:
        print("data/kb 下没有 Markdown 文档")
        return 1
    previews = document_previews(documents)
    for preview in previews:
        print(
            f"\n[{preview['file']}] 字符数={preview['chars']} 切块数={preview['chunk_count']} "
            f"关键条款={preview['key_clause_count']} 表格块={preview['table_block_count']} 类型={preview['content_type']}"
        )
        for index, chunk in enumerate(preview["chunks"], start=1):
            flags = []
            if chunk["is_key_clause"]:
                flags.append("关键条款")
            if chunk["is_table_block"]:
                flags.append("表格块")
            flag_text = f" ({', '.join(flags)})" if flags else ""
            print(
                f"  #{index} [{chunk['category']}] 问法/标题：{chunk['questions']} "
                f"（{chunk['answer_chars']} 字）{flag_text}"
            )
            print(f"      正文：{chunk['answer']}")
    print(f"\n共 {len(previews)} 份文档 / {sum(item['chunk_count'] for item in previews)} 块")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
