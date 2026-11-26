"""ch09 · 飞轮批处理:知识缺口流水线(问题池 → 标准化查重 → 待审队列)。

用法:python -m scripts.flywheel_pipeline [--limit 50]
    PYTHONPATH=.(Windows 用 $env:PYTHONPATH="." 或本仓库约定 python -m)
真模型调用(normalize_and_match),单测 mock;样例验证见 validate_flywheel_samples.py。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import flywheel  # noqa: E402
from scripts._common import load_settings, session_factory_from  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="ch09 飞轮批处理")
    parser.add_argument("--limit", type=int, default=50, help="本轮最多处理多少条未归并原话")
    args = parser.parse_args()

    settings = load_settings()
    session_factory = session_factory_from(settings)
    result = flywheel.run_pipeline(session_factory, settings, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    out = ROOT / "data" / "ch09" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "flywheel_pipeline_last.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
