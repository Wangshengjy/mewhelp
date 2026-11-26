"""scripts.eval_rag — 四策略分桶评估（检索段 + 证据覆盖度 + 生成段）。

三段口径与课程一致：
- 段1 检索（确定性）：四策略 × 分桶 MRR@K / Recall@K（expect_section 匹配 section_path）；
- 段2 证据覆盖度（确定性）：expect_points 去空白子串匹配 Top-K 证据；
- 段3 生成（真实 LLM）：四策略各自用本轮 Top-K 证据生成答案 → 裁判
  （covered_points 答案覆盖度 + faithful 忠实度），D 桶只算拒答正确率；
  判出的编造个案入 faith_cases 台账（eval_id = 题号|策略缩写，跨轮累计）。

产物：data/ch04/reports/rag_eval.txt（人读日志）+ rag_eval.json（/rag-eval 页读）。
生成段失败不拖垮整轮：generation=null 照样落盘，补跑补全。

用法：uv run python -m scripts.eval_rag [--no-gen] [--limit N] [--strategies vector,bm25]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.eval import judge as judge_module  # noqa: E402
from app.eval import metrics  # noqa: E402
from app.kb.citations import parse_markers  # noqa: E402
from app.repositories import faith_cases as faith_cases_repository  # noqa: E402
from scripts._common import load_settings, session_factory_from  # noqa: E402

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid_rerank")
GRADED_BUCKETS = ("A_policy", "B_model", "C_colloquial", "E_multi")
STRATEGY_TAGS = {"vector": "vec", "bm25": "bm25", "hybrid": "hy", "hybrid_rerank": "hyrr"}
EVAL_DATA = ROOT / "data" / "ch04" / "eval_questions.json"
REPORT_DIR = ROOT / "data" / "ch04" / "reports"

RETRIEVAL_WORKERS = 8
GENERATION_WORKERS = 4
CALL_TIMEOUT_SECONDS = 60

_LOG_LINES: list[str] = []
_LOG_LOCK = threading.Lock()


def _log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    with _LOG_LOCK:
        _LOG_LINES.append(line)
    print(line, flush=True)


def load_questions(limit: int | None = None) -> list[dict]:
    questions = json.loads(EVAL_DATA.read_text(encoding="utf-8"))
    if limit:
        questions = questions[:limit]
    return questions


def _chat_model(settings, model_name: str):
    from langchain_openai import ChatOpenAI

    # 评估生成段与裁判共用专用提供商（JUDGE_*），未配置时回退 CHAT 提供商。
    base_url = getattr(settings, "judge_base_url", "") or settings.chat_base_url
    api_key = getattr(settings, "judge_api_key", "") or settings.chat_api_key
    return ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
        temperature=0.0,
        timeout=CALL_TIMEOUT_SECONDS,
    )


def _messages(model_fn):
    """Wrap a ChatOpenAI instance into the model_fn(messages, temperature) protocol.

    评估统一 temperature=0（裁判去抖）；构造时已固定，这里忽略传入值。
    """

    def _invoke(messages, temperature: float = 0.0):
        return model_fn.invoke(messages)

    return _invoke


# ---------------------------------------------------------------------------
# 段1+2：确定性检索评估
# ---------------------------------------------------------------------------

def run_retrieval_eval(settings, questions: list[dict], strategies: list[str], k: int) -> tuple[dict, dict, dict, list[str]]:
    """Returns (retrieval_report, coverage_report, HITS[(qid, strategy)] -> hits)."""
    factory = session_factory_from(settings)
    hits_map: dict[tuple[str, str], list[dict]] = {}
    errors: list[str] = []
    lock = threading.Lock()

    def _search_one(question: dict, strategy: str):
        from app.kb import retrieval as kb_retrieval

        try:
            hits = kb_retrieval.search_knowledge(
                factory, settings, question["query"], strategy=strategy, top_k=k
            )
            return (question["id"], strategy), hits, None
        except Exception as exc:  # 单题失败不拖垮整轮
            return (question["id"], strategy), [], f"{question['id']}/{strategy}: {type(exc).__name__}"

    total = len(questions) * len(strategies)
    done = 0
    with ThreadPoolExecutor(max_workers=RETRIEVAL_WORKERS) as pool:
        futures = [pool.submit(_search_one, q, s) for q in questions for s in strategies]
        for future in as_completed(futures):
            key, hits, error = future.result()
            hits_map[key] = hits
            if error:
                errors.append(error)
            done += 1
            if done % 50 == 0 or done == total:
                _log(f"  检索进度 {done}/{total}")

    retrieval_records: dict[str, list[dict]] = {strategy: [] for strategy in strategies}
    coverage_records: dict[str, list[dict]] = {strategy: [] for strategy in strategies}
    for question in questions:
        groups = metrics.split_expect_groups(question.get("expect_section"))
        points = question.get("expect_points") or []
        for strategy in strategies:
            hits = hits_map.get((question["id"], strategy), [])
            paths = [hit.get("section_path") or "" for hit in hits]
            answers = [hit.get("answer") or "" for hit in hits]
            if question["bucket"] != "D_absent" and groups:
                retrieval_records[strategy].append(
                    {
                        "id": question["id"],
                        "bucket": question["bucket"],
                        "mrr": metrics.mrr_at_k(paths, groups, k),
                        "recall": metrics.recall_at_k(paths, groups, k),
                    }
                )
            coverage_records[strategy].append(
                {
                    "id": question["id"],
                    "bucket": question["bucket"],
                    "coverage": metrics.evidence_coverage(points, answers),
                }
            )

    retrieval_report = {
        strategy: _summarize(retrieval_records[strategy], ("mrr", "recall")) for strategy in strategies
    }
    coverage_report = {
        strategy: _summarize(coverage_records[strategy], ("coverage",)) for strategy in strategies
    }
    return retrieval_report, coverage_report, hits_map, errors


def _summarize(records: list[dict], metrics_keys: tuple[str, ...]) -> dict:
    out: dict = {"count": len(records), "errors": 0, "buckets": {}}
    by_bucket: dict[str, list[dict]] = {}
    for record in records:
        by_bucket.setdefault(record["bucket"], []).append(record)
    for bucket, rows in by_bucket.items():
        out["buckets"][bucket] = {key: metrics.aggregate(row.get(key) for row in rows) for key in metrics_keys}
        out["buckets"][bucket]["count"] = len(rows)
    for key in metrics_keys:
        out[key] = metrics.aggregate(row.get(key) for row in records)
    return out


# ---------------------------------------------------------------------------
# 段3：生成 + 裁判
# ---------------------------------------------------------------------------

def run_generation_eval(settings, questions: list[dict], strategies: list[str], hits_map: dict) -> dict:
    # 生成段与裁判同走 JUDGE_* 通道（评估专用提供商），模型用 judge_model。
    answerer = _messages(_chat_model(settings, settings.judge_model))
    judge_fn = _messages(_chat_model(settings, settings.judge_model))
    factory = session_factory_from(settings)
    lock = threading.Lock()
    per_strategy: dict[str, dict] = {
        strategy: {"answered": 0, "failed": 0, "judged": 0, "fabricated": 0, "refusals": 0, "d_total": 0}
        for strategy in strategies
    }
    coverage_rows: dict[str, list[dict]] = {strategy: [] for strategy in strategies}
    cases: list[dict] = []
    errors: list[str] = []

    def _evidence_texts(hits: list[dict]) -> list[str]:
        return [f"{hit.get('section_path')}：{hit.get('question')} {hit.get('answer')}".strip() for hit in hits]

    def _gen_one(question: dict, strategy: str):
        nonlocal_errors = None
        try:
            hits = hits_map.get((question["id"], strategy), [])
            if question["bucket"] == "D_absent":
                evidence = _evidence_texts(hits)
                answer = judge_module._answer_with(
                    answerer, judge_module.RAG_ANSWER_SYSTEM, question["query"], evidence
                )
                refused = metrics.is_refusal(answer)
                with lock:
                    per_strategy[strategy]["answered"] += 1
                    per_strategy[strategy]["d_total"] += 1
                    per_strategy[strategy]["refusals"] += 1 if refused else 0
                return
            evidence = _evidence_texts(hits)
            answer = judge_module._answer_with(
                answerer, judge_module.RAG_ANSWER_SYSTEM, question["query"], evidence
            )
            verdict = judge_module.judge(
                judge_fn, question["query"], question.get("expect_points") or [], evidence, answer
            )
            with lock:
                per_strategy[strategy]["answered"] += 1
                if verdict is None:
                    per_strategy[strategy]["failed"] += 1
                    return
                per_strategy[strategy]["judged"] += 1
                coverage_value = judge_module.answer_coverage(question.get("expect_points") or [], verdict["covered_points"])
                coverage_rows[strategy].append(
                    {"id": question["id"], "bucket": question["bucket"], "answer_coverage": coverage_value}
                )
                if not verdict["faithful"]:
                    per_strategy[strategy]["fabricated"] += 1
                    snapshot = [
                        {
                            "n": index + 1,
                            "section_path": hit.get("section_path"),
                            "question": hit.get("question"),
                            "answer": hit.get("answer"),
                            "cited": (index + 1) in parse_markers(answer),
                        }
                        for index, hit in enumerate(hits)
                    ]
                    case = {
                        "eval_id": f"{question['id']}|{STRATEGY_TAGS[strategy]}",
                        "bucket": question["bucket"],
                        "query": question["query"],
                        "strategy": strategy,
                        "answer": answer,
                        "reason": verdict["reason"] or "；".join(verdict["fabricated_points"]),
                        "citations": snapshot,
                        "judge_model": settings.judge_model,
                    }
                    cases.append(case)
        except Exception as exc:
            with lock:
                per_strategy[strategy]["answered"] += 1
                per_strategy[strategy]["failed"] += 1
            nonlocal_errors = f"{question['id']}/{strategy}: {type(exc).__name__}"
        if nonlocal_errors:
            with lock:
                errors.append(nonlocal_errors)

    graded = [q for q in questions if q["bucket"] != "D_absent"]
    absent = [q for q in questions if q["bucket"] == "D_absent"]
    total = len(graded) * len(strategies) + len(absent) * len(strategies)
    done = 0
    with ThreadPoolExecutor(max_workers=GENERATION_WORKERS) as pool:
        futures = [pool.submit(_gen_one, q, s) for q in graded for s in strategies]
        futures += [pool.submit(_gen_one, q, s) for q in absent for s in strategies]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 25 == 0 or done == total:
                _log(f"  生成进度 {done}/{total}")

    answer_coverage_report = {
        strategy: _summarize(coverage_rows[strategy], ("answer_coverage",)) for strategy in strategies
    }
    faithfulness_report = {}
    for strategy in strategies:
        stats = per_strategy[strategy]
        judged = stats["judged"]
        faithfulness_report[strategy] = {
            "judged": judged,
            "fabricated": stats["fabricated"],
            "call_failed": stats["failed"],
            "fabricated_rate": (stats["fabricated"] / judged) if judged else None,
        }
    refusal_report = {}
    for strategy in strategies:
        stats = per_strategy[strategy]
        refusal_report[strategy] = {
            "total": stats["d_total"],
            "refused": stats["refusals"],
            "rate": (stats["refusals"] / stats["d_total"]) if stats["d_total"] else None,
        }

    return {
        "answer_coverage": answer_coverage_report,
        "faithfulness": faithfulness_report,
        "refusal": {"D_absent": refusal_report},
        "cases": cases,
        "errors": errors,
    }


def persist_cases(settings, cases: list[dict]) -> None:
    """编造个案入台账（eval_id = 题号|策略缩写；重复判出 seen_count+1、复发退回未解决）。"""
    factory = session_factory_from(settings)
    with factory() as session:
        for case in cases:
            faith_cases_repository.record_case(
                session,
                eval_id=case["eval_id"],
                bucket=case["bucket"],
                query=case["query"],
                strategy=case["strategy"],
                answer=case["answer"],
                reason=case["reason"],
                citations=case["citations"],
                judge_model=case["judge_model"],
            )
        session.commit()


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def write_report(meta: dict, retrieval: dict, coverage: dict, generation: dict | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"meta": meta, "retrieval": retrieval, "evidence_coverage": coverage, "generation": generation}
    report_path = REPORT_DIR / "rag_eval.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    txt_path = REPORT_DIR / "rag_eval.txt"
    txt_path.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")
    _log(f"  报告已写入 {report_path}")


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def print_summary(retrieval: dict, coverage: dict, generation: dict | None, *, gen_skipped: bool = False) -> None:
    _log("段1 检索（MRR@K / Recall@K）")
    for strategy, stats in retrieval.items():
        _log(f"  {strategy:<14} MRR={_fmt(stats.get('mrr'))} Recall={_fmt(stats.get('recall'))}")
    _log("段2 证据覆盖度")
    for strategy, stats in coverage.items():
        _log(f"  {strategy:<14} coverage={_fmt(stats.get('coverage'))}")
    if generation is None:
        _log("段3 生成：" + ("跳过（--no-gen）" if gen_skipped else "进行中，完成后在下方汇总"))
        return
    _log("段3 生成")
    for strategy, stats in generation["answer_coverage"].items():
        faith = generation["faithfulness"].get(strategy, {})
        refusal = generation["refusal"]["D_absent"].get(strategy, {})
        _log(
            f"  {strategy:<14} 答案覆盖={_fmt(stats.get('answer_coverage'))}"
            f" 编造率={_fmt(faith.get('fabricated_rate'))}（判 {faith.get('fabricated')}/{faith.get('judged')}）"
            f" D桶拒答率={_fmt(refusal.get('rate'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="ch04 RAG 四策略评估")
    parser.add_argument("--no-gen", action="store_true", help="只跑检索段与证据覆盖度")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 题（调试用）")
    parser.add_argument("--strategies", type=str, default=",".join(STRATEGIES), help="逗号分隔的策略子集")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    for strategy in strategies:
        if strategy not in STRATEGIES:
            raise SystemExit(f"未知策略：{strategy}")

    started = time.time()
    settings = load_settings()
    questions = load_questions(args.limit)
    k = settings.eval_top_k
    _log(f"评估开始：{len(questions)} 题 × {strategies}，K={k}")

    retrieval, coverage, hits_map, retrieval_errors = run_retrieval_eval(settings, questions, strategies, k)
    if retrieval_errors:
        _log(f"  检索失败 {len(retrieval_errors)} 项：{retrieval_errors[:5]}")
    print_summary(retrieval, coverage, None, gen_skipped=args.no_gen)

    generation = None
    if not args.no_gen:
        _log("生成段开始（生成 + 裁判 + 拒答）")
        generation = run_generation_eval(settings, questions, strategies, hits_map)
        if generation["cases"]:
            persist_cases(settings, generation["cases"])
            _log(f"  台账入库 {len(generation['cases'])} 条编造个案")
        print_summary(retrieval, coverage, generation)

    meta = {
        "questions": len(questions),
        "strategies": strategies,
        "k": k,
        "embed_model": settings.embed_model,
        "rerank_model": settings.rerank_model,
        "chat_model": settings.chat_model,
        "judge_model": settings.judge_model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.time() - started, 1),
    }
    write_report(meta, retrieval, coverage, generation)
    _log(f"评估完成，用时 {meta['duration_seconds']}s")


if __name__ == "__main__":
    main()
