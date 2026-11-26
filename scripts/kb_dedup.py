"""uv run python -m scripts.kb_dedup [--batch BATCH_NO] —— 暂存表去重 + 人工审核流转。

只处理指定批次（默认最新一批）的 extracted 行：与库内存量、批内做"问法+正文"
指纹双重查重（问法归一化：去空白/标点、小写，中文保留）。kept 行不自动入库，
需在 /kb 页面"对话挖知识"卡片逐条采纳（approve）/弃用（reject）。
"""

from __future__ import annotations

import argparse

from scripts._common import load_settings, print_header, session_factory_from
from app.kb.pipeline import dedup_batch


def main() -> int:
    parser = argparse.ArgumentParser(description="暂存表 QA 去重（按批次，kept 不自动入库）")
    parser.add_argument("--batch", dest="batch_no", default=None, help="只处理该批次（默认最新一批）")
    args = parser.parse_args()

    print_header("暂存表去重 kb_dedup")
    session_factory = session_factory_from(load_settings())
    result = dedup_batch(session_factory, args.batch_no)
    print(result["message"])
    if result.get("kept"):
        print("下一步：打开 /kb 页面“对话挖知识”卡片逐条采纳或弃用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
