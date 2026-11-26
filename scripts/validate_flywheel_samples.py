"""ch09 · 飞轮样例验证:10 条代表性原话真模型过 normalize_and_match,
校验标准化质量与查重归并正确性(mock 单测覆盖不了的真实语义路径)。

用法:python -m scripts.validate_flywheel_samples
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import flywheel  # noqa: E402
from app.repositories import review_queue as review_queue_repository  # noqa: E402
from scripts._common import load_settings, session_factory_from  # noqa: E402

SAMPLES = [
    "猫窝能放洗衣机里洗吗",
    "猫咪疫苗每年都要打不",
    "你们家猫粮啥时候发货啊",
    "运费险这个东西到底咋用",
    "cat litter box 可以水洗吗",
    "退货运费到底谁出啊",
    "怎样才算不影响二次销售呢",
    "发货后多久能到货",
    "pet bed 洗了会不会缩水",
    "疫苗第二针和第一针间隔多久",
]


def main() -> int:
    settings = load_settings()
    session_factory = session_factory_from(settings)
    results = []
    print(f"样例验证:{len(SAMPLES)} 条原话真模型过标准化+查重\n")
    for raw in SAMPLES:
        with session_factory() as session:
            candidates = review_queue_repository.list_review_candidates(session)
            verdict = flywheel.normalize_and_match(settings, raw, candidates)
            results.append({
                "raw": raw,
                "normalized": verdict.normalized_question,
                "matched_id": verdict.matched_id,
                "suggested": verdict.suggested_answer,
                "candidates_seen": len(candidates),
            })
            print(f"  原话: {raw}")
            print(f"  → 标准化: {verdict.normalized_question} | 归并: {verdict.matched_id} | 示例答: {verdict.suggested_answer}\n")
    out = ROOT / "data" / "ch09" / "reports" / "flywheel_samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物:{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
