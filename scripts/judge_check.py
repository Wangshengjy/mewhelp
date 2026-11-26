"""scripts.judge_check — 裁判一致性回归（课程验收 6/6 形式）。

读 faith_cases 台账里已处置且带 citations 快照的个案，用快照原样重放
（不重检索、不重生成，只换裁判），人工处置翻译成标准答案：
- 已解决 → 该判编造（faithful=false）
- 无需解决 → 该判忠实（faithful=true，裁判误判）
- 未解决 → 跳过
输出逐条对照与一致率；存在不一致时退出码 1。上游回空 completion 重试一次，
仍失败记「调用失败」且不计入一致率分母。

用法：uv run python -m scripts.judge_check [--status 已解决|无需解决|全部]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval import judge as judge_module  # noqa: E402
from app.kb.citations import parse_markers  # noqa: E402
from app.repositories import faith_cases as faith_cases_repository  # noqa: E402
from scripts._common import load_settings, session_factory_from  # noqa: E402


def _citations_snapshot_to_texts(citations: list[dict]) -> list[str]:
    """台账里的证据快照 → 裁判输入文本（[n] 编号保持入库时的顺序）。"""
    texts = []
    for index, entry in enumerate(citations or []):
        question = entry.get("question") or ""
        answer = entry.get("answer") or ""
        section = entry.get("section_path") or ""
        texts.append(f"{section}：{question} {answer}".strip())
    return texts


def judge_ledger(settings, status: str | None = None) -> dict:
    """Re-judge resolved/unneeded cases with the current criteria; consistency table."""
    factory = session_factory_from(settings)
    with factory() as session:
        cases = faith_cases_repository.list_cases(session, status=status)

    model = _chat_model(settings)
    rows: list[dict] = []
    for case in cases:
        if case.status == "未解决":
            rows.append(
                {
                    "eval_id": case.eval_id,
                    "bucket": case.bucket,
                    "strategy": case.strategy,
                    "status": case.status,
                    "expected": None,
                    "actual": None,
                    "consistent": None,
                    "note": "未解决跳过",
                }
            )
            continue
        if not case.citations:
            rows.append(
                {
                    "eval_id": case.eval_id,
                    "bucket": case.bucket,
                    "strategy": case.strategy,
                    "status": case.status,
                    "expected": None,
                    "actual": None,
                    "consistent": None,
                    "note": "缺证据快照跳过",
                }
            )
            continue
        expected = False if case.status == "已解决" else True
        evidence = _citations_snapshot_to_texts(case.citations)
        verdict = judge_module.judge(
            model,
            case.query,
            [],  # 一致性回归只看忠实度
            evidence,
            case.answer,
        )
        if verdict is None:
            rows.append(
                {
                    "eval_id": case.eval_id,
                    "bucket": case.bucket,
                    "strategy": case.strategy,
                    "status": case.status,
                    "expected": expected,
                    "actual": None,
                    "consistent": None,
                    "note": "调用失败（重试后仍失败，不计入一致率）",
                }
            )
            continue
        actual = verdict["faithful"]
        rows.append(
            {
                "eval_id": case.eval_id,
                "bucket": case.bucket,
                "strategy": case.strategy,
                "status": case.status,
                "expected": expected,
                "actual": actual,
                "consistent": actual == expected,
                "note": verdict["reason"],
                "cited": parse_markers(case.answer),
            }
        )

    judged = [row for row in rows if row["consistent"] is not None]
    consistent = [row for row in judged if row["consistent"]]
    return {
        "total_ledger": len(rows),
        "judged": len(judged),
        "consistent": len(consistent),
        "consistency_rate": (len(consistent) / len(judged)) if judged else None,
        "rows": rows,
    }


def _chat_model(settings):
    from langchain_openai import ChatOpenAI

    return lambda messages, temperature=0.0: ChatOpenAI(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        model=settings.judge_model,
        temperature=0.0,
        timeout=60,
    ).invoke(messages)


def main() -> None:
    parser = argparse.ArgumentParser(description="裁判一致性回归（台账重放）")
    parser.add_argument("--status", choices=["已解决", "无需解决", "全部"], default="全部")
    args = parser.parse_args()
    status = None if args.status == "全部" else args.status

    settings = load_settings()
    result = judge_ledger(settings, status=status)
    print("judge_check 裁判一致性回归（当前判据重放台账个案）")
    for row in result["rows"]:
        if row["consistent"] is None:
            print(f"  {row['eval_id']:<10} {row['status']:<6} {row['note']}")
        else:
            mark = "一致" if row["consistent"] else "不一致"
            print(
                f"  {row['eval_id']:<10} {row['status']:<6} 人工期望={row['expected']!s:<5}"
                f" 裁判={row['actual']!s:<5} [{mark}] {row['note']}"
            )
    rate = result["consistency_rate"]
    print(
        f"  一致率：{result['consistent']}/{result['judged']}"
        + (f" = {rate:.2%}" if rate is not None else "")
    )
    if rate is not None and rate < 1.0:
        print("  存在不一致，退出码 1")
        raise SystemExit(1)
    print("  全部一致")


if __name__ == "__main__":
    main()
