"""ch09 · 置信度阈值校准：在 ch04 评估集(300 题,正负样本标注)上实跑检索,
对每题算 evidence_confidence 四信号总分,按正负样本分布上探校准阈值。

课程 §6.1:评估集=正负样本标注样例验证型——should_refuse=false(A/B/C/E 桶)是正样本
(知识库有答案,该过闸),should_refuse=true(D_absent)是负样本(库里没有,该被拦)。
阈值上探:拦下 ≥80% 负样本的分数点,同时尽量不误拦正样本(正样本 P10)。

用法:python -m scripts.calibrate_confidence [--limit N] [--out data/ch09/reports]
产物:confidence_calibration.txt(人读) + confidence_calibration.json(报表读)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.confidence import compute_evidence_confidence  # noqa: E402
from app.graph import deps as deps_module  # noqa: E402
from app.kb import retrieval as kb_retrieval  # noqa: E402
from scripts._common import load_settings, session_factory_from  # noqa: E402

CONCURRENCY = 8


def _quantile(sorted_values: list[float], q: float) -> float:
    """线性插值分位数(数据已升序)。空集返回 0。"""
    if not sorted_values:
        return 0.0
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _scores_from_txt(path: Path) -> list[dict]:
    """从已有报告 txt 的逐题明细行(``A1  A_policy  expect=answer  score=0.8616``)重建分数,
    支持改完推荐逻辑后零检索重算(上游 rerank 计费,300 题不重跑)。"""
    import re
    pattern = re.compile(r"^\s+(A\d+|B\d+|C\d+|D\d+|E\d+)\s+(\S+)\s+expect=(\w+)\s+score=([\d.]+)\s*$")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            rows.append({"id": m.group(1), "bucket": m.group(2),
                         "should_refuse": m.group(3) == "REFUSE", "score": float(m.group(4)),
                         "signals": {}, "error": None})
    return rows


async def score_questions(questions: list[dict]) -> list[dict]:
    settings = load_settings()
    session_factory = session_factory_from(settings)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def one(item: dict) -> dict:
        async with sem:
            try:
                hits = await asyncio.to_thread(
                    kb_retrieval.search_knowledge,
                    session_factory, settings, item["query"],
                    strategy=settings.retrieval_strategy, top_k=5,
                )
                conf = compute_evidence_confidence(hits)
                return {"id": item["id"], "bucket": item["bucket"],
                        "should_refuse": bool(item.get("should_refuse")),
                        "score": conf.score, "signals": conf.signals, "error": None}
            except Exception as exc:  # 单题失败不拖垮整轮
                return {"id": item["id"], "bucket": item["bucket"],
                        "should_refuse": bool(item.get("should_refuse")),
                        "score": 0.0, "signals": {}, "error": f"{type(exc).__name__}: {exc}"}

    results = []
    done = 0
    tasks = [asyncio.ensure_future(one(item)) for item in questions]
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)
        done += 1
        if done % 20 == 0 or done == len(tasks):
            print(f"  进度 {done}/{len(tasks)}")
    return results


def calibrate(rows: list[dict]) -> dict:
    pos = sorted(r["score"] for r in rows if not r["should_refuse"])
    neg = sorted(r["score"] for r in rows if r["should_refuse"])

    cand_block = _quantile(neg, 0.80)   # 拦下 ≥80% 负样本的分数点(上探)
    cand_keep = _quantile(pos, 0.10)    # 正样本 P10:至少 90% 可答题放行
    # 上探语义:在满足「拦下≥80%负样本」约束下尽量高(多拦漏网负样本);
    # cand_keep≥cand_block 时取 cand_keep(负样本约束仍满足),重叠严重时退到 cand_block
    recommended = round(max(cand_block, cand_keep), 3)

    def confusion(threshold: float) -> dict:
        tp = sum(1 for s in neg if s < threshold)   # 负样本被拦=对
        fn = len(neg) - tp
        tn = sum(1 for s in pos if s >= threshold)  # 正样本过闸=对
        fp = len(pos) - tn
        return {"blocked_neg": tp, "missed_neg": fn, "pass_pos": tn, "blocked_pos": fp,
                "block_rate_neg": round(tp / len(neg), 3) if neg else 0.0,
                "keep_rate_pos": round(tn / len(pos), 3) if pos else 0.0}

    buckets = {}
    for bucket in sorted({r["bucket"] for r in rows}):
        scores = [r["score"] for r in rows if r["bucket"] == bucket]
        buckets[bucket] = {
            "n": len(scores),
            "p05": round(_quantile(sorted(scores), 0.05), 3),
            "median": round(_quantile(sorted(scores), 0.50), 3),
            "p95": round(_quantile(sorted(scores), 0.95), 3),
        }
    return {
        "n_total": len(rows), "n_pos": len(pos), "n_neg": len(neg),
        "candidate_block_80pct_neg": round(cand_block, 3),
        "candidate_keep_90pct_pos": round(cand_keep, 3),
        "recommended_threshold": recommended,
        "confusion_at_recommended": confusion(recommended),
        "confusion_at_0_5": confusion(0.5),
        "bucket_stats": buckets,
        "errors": [r for r in rows if r["error"]],
        "scores": [{"id": r["id"], "bucket": r["bucket"],
                    "should_refuse": r["should_refuse"], "score": r["score"]} for r in rows],
    }


def render(report: dict, rows: list[dict]) -> str:
    lines = [
        "ch09 · evidence_confidence 阈值校准报告",
        f"评估集:data/ch04/eval_questions.json(正负样本标注,实跑 hybrid_rerank 检索)",
        f"样本:总 {report['n_total']} | 正样本(可答,应过闸) {report['n_pos']} | 负样本(D_absent,应被拦) {report['n_neg']}",
        "",
        "候选阈值:",
        f"  拦下≥80%负样本点(负样本P80)  = {report['candidate_block_80pct_neg']}",
        f"  放行≥90%正样本点(正样本P10)  = {report['candidate_keep_90pct_pos']}",
        f"  推荐阈值(保守取小)           = {report['recommended_threshold']}",
        "",
        f"推荐阈值下的混淆矩阵: {json.dumps(report['confusion_at_recommended'], ensure_ascii=False)}",
        f"占位阈值 0.5 下的混淆矩阵: {json.dumps(report['confusion_at_0_5'], ensure_ascii=False)}",
        "",
        "分组分布(score):",
    ]
    for bucket, st in report["bucket_stats"].items():
        lines.append(f"  {bucket:<14} n={st['n']:<4} P05={st['p05']:<7} 中位={st['median']:<7} P95={st['p95']}")
    if report["errors"]:
        lines.append(f"\n异常题({len(report['errors'])} 条,计 0 分):")
        for r in report["errors"][:10]:
            lines.append(f"  {r['id']}: {r['error']}")
    lines.append("\n逐题明细(score, 按桶):")
    for r in sorted(rows, key=lambda x: (x["bucket"], x["id"])):
        mark = "REFUSE" if r["should_refuse"] else "answer"
        lines.append(f"  {r['id']:<8} {r['bucket']:<12} expect={mark:<7} score={r['score']:.4f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ch09 置信度阈值校准")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题(调试用)")
    parser.add_argument("--out", default="data/ch09/reports", help="产物目录")
    parser.add_argument("--from-txt", default=None,
                        help="从已有报告 txt 的逐题明细重算(不重新检索,省上游费用)")
    args = parser.parse_args()

    if args.from_txt:
        rows = _scores_from_txt(ROOT / args.from_txt)
        print(f"从 {args.from_txt} 重建 {len(rows)} 条逐题分数,重算阈值(零检索)")
    else:
        questions = json.loads((ROOT / "data/ch04/eval_questions.json").read_text(encoding="utf-8"))
        if args.limit:
            questions = questions[: args.limit]
        print(f"校准开始:{len(questions)} 题,并发 {CONCURRENCY},strategy={load_settings().retrieval_strategy}")
        rows = asyncio.run(score_questions(questions))

    report = calibrate(rows)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render(report, rows)
    (out_dir / "confidence_calibration.txt").write_text(text, encoding="utf-8")

    print(text.split("逐题明细")[0])
    print(f"\n产物:{out_dir / 'confidence_calibration.txt'} / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
