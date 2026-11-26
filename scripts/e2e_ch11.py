"""ch11 · 全链路串联两条样例请求(直连图真跑,需真实上游 + MySQL/Milvus)。

课程 ch11 两条走线,适配本仓库确定性 mock 数据(订单 2002=智能猫砂盆×1,未发货):
    样例A 退款线(走子流程不停置信度闸):轮1 问蓝牙耳机物流 → 轮2
      「我早上买的这个还能退吗」→ 指代消解补全 → 意图=退款退货 →
      fetch_order+retrieve_policy → Agent 判能不能退 → 提议提交退款。
    样例B 兜底线(翻车进飞轮):问「双十一跨店满减能跟会员积分券叠加用吗」
      → 知识类强制 RAG → 置信度闸拦截 → 兜底话术 → 低置信度落池(带快照)
      → flywheel 标准化查重进待审队列。
六站留痕:每轮打印 指代消解(resolved_query)/意图(intent+conf)/分流(route)/
置信度(evidence_confidence)/Agent 终答/审计(trace keys)。
入口:.\\.venv\\Scripts\\python.exe -m scripts.e2e_ch11
"""
import asyncio
import sys

from app.graph import runtime
from scripts._common import Settings, session_factory_from
from sqlalchemy import text


def _last_ai(state) -> str:
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            return str(m.content)
    return ""


def _stations(tag: str, state: dict) -> None:
    trace = state.get("trace", {})
    print(f"--- {tag} 六站留痕 ---")
    print(f"  ①指代消解 resolved_query={state.get('resolved_query')!r}")
    print(f"  ②意图识别 intent={state.get('intent')} conf={state.get('intent_confidence')}")
    print(f"  ③分流 route={trace.get('route')} trace={sorted(trace)}")
    print(f"  ④置信度 evidence_confidence={state.get('evidence_confidence')} "
          f"fallback_source={state.get('fallback_source')}")
    print(f"  ⑤Agent 终答={_last_ai(state)[:150]!r}")
    print(f"  ⑥动作 suggested_actions={[a.get('type') for a in state.get('suggested_actions') or []]}")


async def sample_a() -> tuple[bool, str]:
    r1 = await runtime.run_turn(None, "订单2002什么时候能到", user_id="e2e-ch11")
    cid = r1["conversation_id"]
    _stations("样例A轮1", r1["state"])
    r2 = await runtime.run_turn(cid, "我早上买的这个还能退吗", user_id="e2e-ch11")
    s2 = r2["state"]
    _stations("样例A轮2", s2)
    trace = s2.get("trace", {})
    actions = [a.get("type") for a in s2.get("suggested_actions") or []]
    # 提议提交(refund_form/submit_refund)只在用户明确要求提交时出现(ch08 权限规矩:
    # 只有客户明确要求才发起写操作);判定型提问「还能退吗」的正确终态是带引用的判定答复。
    ok = (
        s2.get("intent") == "退款退货"
        and bool(s2.get("resolved_query"))
        and "fetch_order" in trace
        and "retrieve_policy" in trace
        and bool(s2.get("evidence"))
    )
    detail = (f"轮2 intent={s2.get('intent')} resolved={s2.get('resolved_query')!r} "
              f"trace={sorted(trace)} actions={actions} evidence_len={len(s2.get('evidence') or '')} "
              f"终答={_last_ai(s2)!r}")
    return ok, detail


async def sample_b() -> tuple[bool, str]:
    maker = session_factory_from(Settings.load())
    with maker() as sess:
        before = sess.execute(text("SELECT COUNT(*) FROM low_confidence_questions")).scalar()
    r = await runtime.run_turn(None, "你们双十一的跨店满减,能跟会员积分券叠加用吗", user_id="e2e-ch11")
    s = r["state"]
    _stations("样例B", s)
    with maker() as sess:
        after = sess.execute(text("SELECT COUNT(*) FROM low_confidence_questions")).scalar()
        row = sess.execute(text(
            "SELECT id, source, raw_question FROM low_confidence_questions "
            "ORDER BY id DESC LIMIT 1")).fetchone()
    answer = _last_ai(s)
    refused = s.get("fallback_source") in ("retrieval_low_conf", "self_check")
    ok = refused and after == before + 1
    detail = (f"intent={s.get('intent')} evidence_confidence={s.get('evidence_confidence')} "
              f"fallback_source={s.get('fallback_source')} 池 {before}->{after} 最新行={dict(row._mapping) if row else None} "
              f"终答前40字={answer[:40]!r}")
    return ok, detail


async def sample_b_flywheel() -> str:
    """把样例B落池的问题走一遍飞轮:标准化查重进待审队列(1-2 次真实 LLM 调用)。"""
    from app.core.flywheel import run_pipeline
    settings = Settings.load()
    out = run_pipeline(session_factory_from(settings), settings)
    with session_factory_from(Settings.load())() as sess:
        q = sess.execute(text(
            "SELECT id, review_status, occurrence_count, normalized_question "
            "FROM review_queue ORDER BY id DESC LIMIT 1")).fetchone()
    return f"pipeline 输出={out} 最新待审行={dict(q._mapping) if q else '无'}"


async def main() -> None:
    await runtime.init_graph()
    results: list[tuple[str, bool, str]] = []
    try:
        ok_a, d_a = await sample_a()
        results.append(("样例A 退款线:多轮指代→refund_flow→提议提交", ok_a, d_a))
        ok_b, d_b = await sample_b()
        results.append(("样例B 兜底线:置信度闸拦截→落池带快照", ok_b, d_b))
        fly = await sample_b_flywheel()
        print(f"--- 飞轮接棒 ---\n  {fly}")
    finally:
        await runtime.close_graph()
    print("=" * 72)
    for name, ok, detail in results:
        print(f"{'✅ PASS' if ok else '❌ FAIL'} {name}\n   {detail}")
    print("=" * 72)
    if not all(ok for _, ok, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
