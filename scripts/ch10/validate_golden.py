"""ch10 预标质量闸：黄金样例上标签集合完全一致率 ≥ 80% 才放行批量预标。

运行：python -m scripts.ch10.validate_golden（需上游可用）。不过线就改 prompt，不改黄金样例来凑分。
报告一式两份落 data/ch10/reports/golden_report.{json,md}。
"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime, timezone

from scripts.ch10.prelabel import prelabel_batch

GOLDEN = pathlib.Path(__file__).parent / "golden_samples.jsonl"
REPORTS = pathlib.Path("data/ch10/reports")
PASS_RATE = 0.8


async def main() -> int:
    samples = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    predicted = await prelabel_batch([s["text"] for s in samples])
    hits, failures = 0, []
    for s, pred in zip(samples, predicted):
        ok = set(pred) == set(s["labels"])
        hits += ok
        if not ok:
            failures.append({"text": s["text"], "gold": s["labels"], "pred": pred})
            print(f"✗ {s['text']}\n    标准: {s['labels']}  预标: {pred}")
    rate = hits / len(samples) if samples else 0.0
    print(f"\n黄金样例 {len(samples)} 条，集合全对 {hits} 条，通过率 {rate:.0%}（闸线 {PASS_RATE:.0%}）")

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "total": len(samples),
        "hits": hits,
        "rate": round(rate, 4),
        "pass_line": PASS_RATE,
        "passed": rate >= PASS_RATE,
        "failures": failures,
    }
    (REPORTS / "golden_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# ch10 黄金样例验证报告", "",
        f"- 运行时间：{report['ran_at']}",
        f"- 黄金样例 {report['total']} 条，集合全对 {report['hits']} 条，通过率 **{rate:.0%}**（闸线 {PASS_RATE:.0%}）",
        f"- 结论：{'✅ 通过，放行批量预标' if report['passed'] else '🔴 不过线，改预标 prompt 后重跑'}", "",
        "## 错例", "",
    ]
    lines += [f"- {f['text']}：标准 {f['gold']} / 预标 {f['pred']}" for f in failures] or ["（无）"]
    (REPORTS / "golden_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已落 {REPORTS}/golden_report.{{json,md}}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
