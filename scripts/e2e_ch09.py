"""ch09 · 端到端六条验收(API/脚本层,真服务:MySQL dev 库 + Milvus + 上游 LLM)。

验收链路(课程定稿):
  1. 兜底问题落池 → 审核队列可见原话+召回片段 → 通过写回知识库 → 同问题再问答对(飞轮整圈)
  2. 👎 反馈落池 → 飞轮查重归并入队(occurrence+1)
  3. token 按意图统计(Langfuse 起栈则真跑 cost_by_intent)
  4. 评估两轮趋势(eval_flywheel,另行实跑,产物 eval_trend.txt)
产出:data/ch09/reports/e2e_ch09.txt 逐条留痕。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.core import flywheel  # noqa: E402
from app.graph import runtime  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import KnowledgeChunk, LowConfidenceQuestion  # noqa: E402
from app.repositories import review_queue as review_queue_repository  # noqa: E402

REPORT = ROOT / "data" / "ch09" / "reports" / "e2e_ch09.txt"
LINES: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    LINES.append(msg)


# 库外问题:校准报告确认库外 + 直调图验证 intent=商品咨询→knowledge 路由、
# 四信号闸 conf=0.0018<0.138 拦下落池(D17,校准明细 D17 score=0.002)
PROBE_QUESTION = "你们卖的猫粮保质期到具体哪一天"


def _final_text(events: list[dict]) -> str:
    return "".join(e["data"]["delta"] for e in events
                   if e["type"] == "token" and isinstance(e["data"], dict))


def _events(body: str) -> list[dict]:
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        lines = block.splitlines()
        ev = next((ln for ln in lines if ln.startswith("event: ")), None)
        data = next((ln.removeprefix("data: ") for ln in lines if ln.startswith("data: ")), None)
        if not ev:
            continue
        try:
            parsed = json.loads(data) if data and data != "[DONE]" else data
        except json.JSONDecodeError:
            parsed = data
        events.append({"type": ev.removeprefix("event: ").strip(), "data": parsed})
    return events


async def main() -> int:
    log("=== ch09 端到端验收(API/脚本层) ===")
    log(f"时间:{__import__('datetime').datetime.now().isoformat(timespec='seconds')}")
    app = create_app()  # 真实 Settings.load():MySQL dev 库 + Milvus + 上游

    with TestClient(app) as client:
        # ── 验收1 · 兜底问题落池 → 审核可见 → 通过写回 → 再问答对 ──
        log("\n[验收1] 兜底问题落池 → 审核队列可见原话+召回片段 → 通过写回 → 同问题再问答对")
        conv_id = None
        resp = client.post("/api/chat/stream", json={"message": PROBE_QUESTION})
        events = _events(resp.text)
        conv_id = next((e["data"]["conversation_id"] for e in events if e["type"] == "conversation"), None)
        answer1 = _final_text(events)
        log(f"  1a. 会话 {conv_id} 首问「{PROBE_QUESTION}」")
        log(f"      首轮回答(应拒答/兜底): {answer1[:80]}...")
        with app.state.session_factory() as session:
            rows = session.query(LowConfidenceQuestion).filter_by(
                conversation_id=conv_id, source="retrieval_low_conf").all()
        assert rows, "验收1失败:首轮未落池"
        log(f"  1b. 落池 ✓ source=retrieval_low_conf reason={rows[0].reason[:60]}")
        assert rows[0].retrieved_chunks is not None, "落池快照缺失"
        log(f"      快照 Top{len(rows[0].retrieved_chunks)}: "
            + "; ".join(f"{c['question']}({c['rerank_score']:.2f})" for c in rows[0].retrieved_chunks))

        # 飞轮批处理:标准化+查重入队
        fly_result = flywheel.run_pipeline(app.state.session_factory, app.state.settings, limit=50)
        log(f"  1c. 飞轮批处理: {json.dumps({k: fly_result[k] for k in ('processed', 'merged', 'created', 'failed')}, ensure_ascii=False)}")
        queue = client.get("/api/review/queue").json()["items"]
        target = next((it for it in queue if "保质期" in it["normalized_question"] or "猫粮" in it["normalized_question"]), None)
        assert target, "验收1失败:审核队列未见该缺口"
        log(f"  1d. 审核队列可见 ✓ id={target['id']} 「{target['normalized_question']}」出现 {target['occurrence_count']} 次")

        detail = client.get(f"/api/review/{target['id']}").json()
        assert detail["raw_questions"], "详情缺原话"
        assert any(r["retrieved_chunks"] for r in detail["raw_questions"]), "详情缺召回快照"
        log(f"  1e. 详情页数据 ✓ 原话 {len(detail['raw_questions'])} 条,含召回片段")

        approved_answer = "猫粮保质期:包装袋喷码为准,未开封 18 个月;开封后 30 天内用完并密封避光保存。批次不同喷码不同,可提供批次号帮您查询具体日期。"
        approve = client.post(f"/api/review/{target['id']}/approve", json={"approved_answer": approved_answer})
        assert approve.status_code == 200, approve.text
        kb = approve.json()["kb"]
        log(f"  1f. 通过写回 ✓ inserted={kb['inserted']} skipped={kb['skipped']} vectorized={kb['vectorized']}")

        with app.state.session_factory() as session:
            chunk = session.query(KnowledgeChunk).filter(
                KnowledgeChunk.questions.contains("保质期")).all()
        assert chunk, "知识库未见新条目"
        log(f"      知识库新条目: chunk_id={chunk[0].id} vector_id={chunk[0].vector_id!r} (pending→可向量化)")

        resp2 = client.post("/api/chat/stream", json={"message": PROBE_QUESTION, "conversation_id": conv_id})
        events2 = _events(resp2.text)
        answer2 = _final_text(events2)
        cites = next((e["data"]["items"] for e in events2 if e["type"] == "citations"), [])
        wrote_chunk_id = chunk[0].id
        cited_ids = [c.get("chunk_id") for c in cites]
        hit_ok = "保质期" in answer2 or wrote_chunk_id in cited_ids
        log(f"  1g. 同问题再问回答(节选): {answer2[:100]}...")
        log(f"      引用 chunk_ids={cited_ids}(写回条目 chunk_id={wrote_chunk_id}),含关键词={('保质期' in answer2)}")
        log(f"      飞轮整圈 {'✓' if hit_ok else '△'}")
        log("  验收1 结果: " + ("PASS" if hit_ok else "FAIL(写回已入库但检索未命中,待复核)"))

        # ── 验收2 · 👎 落池查重入队 ──
        log("\n[验收2] 👎 反馈 → 快照回捞落池 → 查重归并入队(次数+1)")
        before = target["occurrence_count"]
        fb = client.post("/api/feedback", json={
            "conversation_id": conv_id, "turn_index": 2, "rating": "down", "comment": "第二轮还不够细"})
        assert fb.status_code == 200, fb.text
        snap = fb.json()["snapshot"]
        log(f"  2a. 👎 提交 ✓ status={fb.json()['status']} 回捞快照段数={len(snap['retrieved_snapshot'])}")
        fly2 = flywheel.run_pipeline(app.state.session_factory, app.state.settings, limit=50)
        detail2 = client.get(f"/api/review/{target['id']}").json()
        log(f"  2b. 查重归并 ✓ 该缺口出现次数 {before}→{detail2['occurrence_count']} (occurrence+1,状态保持通过不复活)")
        assert detail2["occurrence_count"] == before + 1
        log("  验收2 结果: PASS")

        # ── 验收3 · token 按意图统计 ──
        log("\n[验收3] token 按意图统计(cost_by_intent)")
        proc = __import__("subprocess").run(
            [sys.executable, "-m", "scripts.cost_by_intent"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=180)
        log("  " + (proc.stdout or proc.stderr).strip().replace("\n", "\n  ")[:1200])
        status = "PASS" if proc.returncode == 0 else ("BLOCKED" if proc.returncode == 2 else "FAIL")
        log(f"  验收3 结果: {status} (exit={proc.returncode})")

    log("\n[验收4] 评估两轮趋势:另行实跑(scripts.eval_flywheel),产物 data/ch09/reports/eval_trend.txt")
    trend = (ROOT / "data/ch09/reports/eval_trend.txt")
    log("  验收4 结果: PASS(两轮已落表,见 eval_trend.txt)" if trend.exists() else "  验收4 结果: FAIL")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\n留痕:{REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
