"""ch09 · 自动化评估流水线:eval_runs 表 + 两轮趋势对比。

两轮来源(课程定稿:复用既有评估产物 + 新跑一轮):
  --import-existing data/ch04/reports/rag_eval.json
      把 ch04 评估产物里的 hybrid_rerank 指标(Recall@5/MRR@5/Faithfulness/拒答率)
      作为第一轮落 eval_runs,不重跑、不花钱;
  --run
      新跑一轮:240 可答题算 Recall@5/MRR@5(实跑检索,期望段落匹配口径与 ch04 一致),
      60 道 D_absent 题**用 ch09 新闸判拒答**(evidence_confidence < 校准阈值),
      这是对置信度闸的直接评估。生成段 faithfulness 需真生成+裁判(计费大),
      新轮暂不跑,趋势表标注沿用上轮(见 dev-notes 偏离记录)。

用法:
  python -m scripts.eval_flywheel --import-existing data/ch04/reports/rag_eval.json --run
产物:data/ch09/reports/eval_trend.txt / .json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.confidence import compute_evidence_confidence  # noqa: E402
from app.eval.metrics import mrr_at_k, recall_at_k  # noqa: E402
from app.repositories import review_queue as review_queue_repository  # noqa: E402
from scripts._common import load_settings, session_factory_from  # noqa: E402

K = 5
WORKERS = 8
GRADED_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")


def import_existing(path: Path) -> dict:
    """从 ch04 rag_eval.json 提取 hybrid_rerank 指标作第一轮(不重跑)。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    hr = data["retrieval"]["hybrid_rerank"]
    faith = (data.get("generation", {}).get("faithfulness", {}) or {}).get("hybrid_rerank", {})
    refusal = ((data.get("generation", {}).get("refusal", {}) or {}).get("D_absent", {}) or {}).get("hybrid_rerank", {})
    fabricated_rate = faith.get("fabricated_rate")
    metrics = {
        "recall_at_5": round(float(hr["recall"]), 4),
        "mrr_at_5": round(float(hr["mrr"]), 4),
        "faithfulness": round(1.0 - fabricated_rate, 4) if fabricated_rate is not None else None,
        "refusal_rate": round(float(refusal.get("rate", 0.0)), 4) or None,
        "note": "ch04 基线(生成段裁判口径),导入自 rag_eval.json",
    }
    return {"triggered_by": "手动", "dataset_size": int(data["meta"]["questions"]), "metrics": metrics}


def run_one_round(questions: list[dict], settings, session_factory) -> dict:
    """新跑一轮:检索段(Recall/MRR,同 ch04 口径) + D 桶置信度闸拒答率。"""
    import threading

    results: dict[str, dict] = {}
    lock = threading.Lock()

    def _one(item: dict):
        from app.kb import retrieval as kb_retrieval
        try:
            hits = kb_retrieval.search_knowledge(
                session_factory, settings, item["query"],
                strategy=settings.retrieval_strategy, top_k=K)
            conf = compute_evidence_confidence(hits)
            return item, [h.get("section_path", "") for h in hits], conf.score, None
        except Exception as exc:
            return item, [], 0.0, f"{item['id']}: {type(exc).__name__}"

    errors: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_one, q) for q in questions]
        for future in as_completed(futures):
            item, paths, conf_score, error = future.result()
            with lock:
                done += 1
                if done % 30 == 0:
                    print(f"  进度 {done}/{len(questions)}")
                if error:
                    errors.append(error)
                results[item["id"]] = {"bucket": item["bucket"],
                                       "expect": item.get("expect_section") or [],
                                       "paths": paths, "conf": conf_score,
                                       "should_refuse": bool(item.get("should_refuse"))}

    graded = [r for r in results.values() if not r["should_refuse"]]
    neg = [r for r in results.values() if r["should_refuse"]]
    recall_scores = [recall_at_k(r["paths"], [[alt for alt in group] for group in r["expect"]], K)
                     for r in graded]
    mrr_scores = [mrr_at_k(r["paths"], [[alt for alt in group] for group in r["expect"]], K)
                  for r in graded]
    threshold = settings.evidence_confidence_threshold
    refused = sum(1 for r in neg if r["conf"] < threshold)
    metrics = {
        "recall_at_5": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else None,
        "mrr_at_5": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else None,
        "faithfulness": None,  # 生成段需真生成+裁判(计费),本轮不跑,趋势表沿用上轮
        "refusal_rate": round(refused / len(neg), 4) if neg else None,
        "refusal_threshold": threshold,
        "note": f"ch09 新跑:检索段 {settings.retrieval_strategy},拒答=evidence_confidence<{threshold}(置信度闸)",
    }
    return {"triggered_by": "手动", "dataset_size": len(questions), "metrics": metrics,
            "errors": errors}


def trend(runs: list[dict]) -> dict:
    """两轮逐指标对比(最新在后)。"""
    if len(runs) < 2:
        return {"comparable": False, "message": f"仅 {len(runs)} 轮,无法对比"}
    first, second = runs[0], runs[1]
    deltas = {}
    for key in ("recall_at_5", "mrr_at_5", "faithfulness", "refusal_rate"):
        a, b = first["metrics"].get(key), second["metrics"].get(key)
        deltas[key] = {"first": a, "second": b,
                       "delta": round(b - a, 4) if (a is not None and b is not None) else None}
    return {"comparable": True, "deltas": deltas}


def render(rows: list[dict], trend_report: dict) -> str:
    lines = ["ch09 · 自动化评估流水线 · 两轮趋势", ""]
    for run in rows:
        lines.append(f"轮 {run['id']} | {run['triggered_by']} | n={run['dataset_size']} | {run['created_at']}")
        for key, value in run["metrics"].items():
            lines.append(f"    {key:<16} {value}")
        lines.append("")
    if trend_report.get("comparable"):
        lines.append("趋势对比(轮 N-1 → 轮 N):")
        for key, d in trend_report["deltas"].items():
            arrow = "→"
            delta = d["delta"]
            lines.append(f"    {key:<16} {d['first']} {arrow} {d['second']}  (Δ {delta})")
    else:
        lines.append(trend_report.get("message", ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ch09 评估流水线")
    parser.add_argument("--import-existing", default=None, help="ch04 rag_eval.json(第一轮)")
    parser.add_argument("--run", action="store_true", help="新跑一轮(检索段+置信度闸拒答)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="data/ch09/reports")
    args = parser.parse_args()

    settings = load_settings()
    session_factory = session_factory_from(settings)
    runs = []

    if args.import_existing:
        run = import_existing(ROOT / args.import_existing)
        with session_factory() as session:
            row = review_queue_repository.insert_eval_run(session, **run)
            session.commit()
            run_row = {"id": row.id, **run, "created_at": row.created_at.isoformat() if row.created_at else ""}
        runs.append(run_row)
        print(f"已导入既有产物为第 {row.id} 轮: {run['metrics']}")

    if args.run:
        questions = json.loads((ROOT / "data/ch04/eval_questions.json").read_text(encoding="utf-8"))
        if args.limit:
            questions = questions[: args.limit]
        print(f"新跑一轮:{len(questions)} 题,阈值={settings.evidence_confidence_threshold}")
        run = run_one_round(questions, settings, session_factory)
        errors = run.pop("errors", None)
        with session_factory() as session:
            row = review_queue_repository.insert_eval_run(session, **run)
            session.commit()
            run_row = {"id": row.id, **run, "created_at": row.created_at.isoformat() if row.created_at else ""}
        runs.append(run_row)
        print(f"新轮已落表(第 {row.id} 轮): {run['metrics']}")
        if errors:
            print(f"异常 {len(errors)} 条(计 0 分): {errors[:5]}")

    # 趋势:无论本轮动作,读全表(也许本轮只补跑了一段)
    with session_factory() as session:
        all_runs = review_queue_repository.list_eval_runs(session, limit=20)
        rows = [{"id": r.id, "triggered_by": r.triggered_by, "dataset_size": r.dataset_size,
                 "metrics": r.metrics,
                 "created_at": r.created_at.isoformat() if r.created_at else ""}
                for r in sorted(all_runs, key=lambda x: x.id)]

    trend_report = trend(rows)
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_trend.json").write_text(
        json.dumps({"runs": rows, "trend": trend_report}, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render(rows, trend_report)
    (out_dir / "eval_trend.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\n产物:{out_dir / 'eval_trend.txt'} / .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
