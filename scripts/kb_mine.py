"""uv run python -m scripts.kb_mine —— 历史客服对话分批喂 CHAT 模型抽问答对。

每批 ≤20 个会话拼进一个 prompt；模型输出并列标量数组 questions[]/answers[]
（嵌套对象数组会使上游 502），长度不一致按较短截断并记 warning。
结果写 qa_extraction_staging(status=extracted)，等待 kb_dedup 去重 + 人工审核。
"""

from __future__ import annotations

from scripts._common import load_settings, print_header, session_factory_from
from app.kb.pipeline import mine_conversations


def main() -> int:
    print_header("历史对话挖知识 kb_mine")
    settings = load_settings()
    session_factory = session_factory_from(settings)
    result = mine_conversations(session_factory, settings)
    if not result.get("ok"):
        print(f"[error] {result.get('message')}")
        return 1
    print(f"批次号 batch_no={result.get('batch_no')}（去重时可用 --batch 指定）")
    print(result["message"])
    print("下一步：uv run python -m scripts.kb_dedup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
