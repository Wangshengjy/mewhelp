"""ch06 · 四条验收(直连图真跑,需真实上游 + MySQL/Milvus 可用)。

课程 Task 16 的四条验收,适配本仓库:无 /api/agent,eval 直接 runtime.run_turn /
resume_turn 直连图。interrupt 语义:run_turn 返回 ``interrupt``(待选订单列表)。
入口:
    .\\.venv\\Scripts\\python.exe -m scripts.eval_ch06
验收口径:
    1) 多轮意图漂移:物流→退款→物流,同一会话三轮意图各归其位;
    2) 意图 JSON 稳定:八类标注判对 + 怪问题落「其他」(详见 scripts/eval_intent);
    3) 带单号退款诉求:走 refund_flow,trace 有 fetch_order/retrieve_policy,证据非空;
    4) 缺单号退款诉求:run_turn interrupt 出待选订单,resume 选中后走完链路。
"""
import asyncio
import sys

from app.graph import runtime


async def acceptance_1_multi_turn_drift() -> tuple[bool, str]:
    r1 = await runtime.run_turn(None, "订单1001到哪了", user_id="eval-ch06")
    cid = r1["conversation_id"]
    s1, s2, s3 = r1["state"], None, None
    r2 = await runtime.run_turn(cid, "那我想把它退了", user_id="eval-ch06")
    s2 = r2["state"]
    r3 = await runtime.run_turn(cid, "算了,它现在到哪了", user_id="eval-ch06")
    s3 = r3["state"]
    ok = s1.get("intent") == "物流" and s2.get("intent") == "退款退货" and s3.get("intent") == "物流"
    detail = (
        f"轮1 intent={s1.get('intent')} 轮2 intent={s2.get('intent')}"
        f" resolved={s2.get('resolved_query')!r} 轮3 intent={s3.get('intent')}"
        f" resolved={s3.get('resolved_query')!r}"
    )
    return ok, detail


async def acceptance_3_refund_flow_with_order_id() -> tuple[bool, str]:
    result = await runtime.run_turn(None, "订单2002这个能退吗", user_id="eval-ch06")
    state = result["state"]
    trace = state.get("trace", {})
    has_chain = "fetch_order" in trace and "retrieve_policy" in trace
    has_evidence = bool(state.get("evidence"))
    actions = state.get("suggested_actions") or []
    detail = (
        f"intent={state.get('intent')} trace_keys={list(trace)} evidence_len={len(state.get('evidence') or '')}"
        f" actions={[a.get('type') for a in actions]} answer={bool(state.get('messages'))}"
    )
    return has_chain and has_evidence, detail


async def acceptance_4_interrupt_and_resume() -> tuple[bool, str]:
    first = await runtime.run_turn(None, "我要退款", user_id="eval-ch06")
    payload = first.get("interrupt")
    orders = (payload or {}).get("orders") if isinstance(payload, dict) else None
    if not orders:
        return False, f"缺单号未触发 interrupt,turn state keys={list(first['state'])}"
    orders_desc = [f"{o.get('order_id')}/{o.get('status')}" for o in orders]
    resumed = await runtime.resume_turn(first["conversation_id"], orders[0]["order_id"])
    state = resumed["state"]
    # 判定语义防退化:refund_flow 终态回答必须是判定,不得追问原因/签收(课程需求6:
    # 退款原因不追问,提交退款单时从固定类目自选)。此断言曾缺失导致模型追问也放行。
    answer_text = ""
    for m in reversed(state.get("messages") or []):
        if getattr(m, "type", "") == "ai" and getattr(m, "content", ""):
            answer_text = str(m.content)
            break
    probing = ("原因" in answer_text and "请问" in answer_text) or "什么原因" in answer_text
    ok = bool(
        resumed.get("interrupt") is None
        and state.get("order_id") == orders[0]["order_id"]
        and (state.get("messages") or state.get("suggested_actions"))
    ) and not probing
    detail = (
        f"interrupt_orders={orders_desc} -> resume 选中 {orders[0]['order_id']};"
        f" 终态 order_id={state.get('order_id')} trace={list(state.get('trace', {}))}"
        f" actions={[a.get('type') for a in state.get('suggested_actions') or []]}"
        f" answer前60字={answer_text[:60]!r}"
    )
    return ok, detail


async def main() -> None:
    await runtime.init_graph()
    results: list[tuple[str, bool, str]] = []
    try:
        ok1, d1 = await acceptance_1_multi_turn_drift()
        results.append(("验收1 多轮意图漂移(物流→退款→物流)", ok1, d1))
        # 验收2(意图 JSON 稳定)在 scripts/eval_intent:八类标注 + 怪问题落「其他」,这里只提示
        ok3, d3 = await acceptance_3_refund_flow_with_order_id()
        results.append(("验收3 带单号退款诉求走 refund_flow", ok3, d3))
        ok4, d4 = await acceptance_4_interrupt_and_resume()
        results.append(("验收4 缺单号 interrupt 选单 → resume 走完", ok4, d4))
    finally:
        await runtime.close_graph()

    all_ok = True
    print("=" * 72)
    for name, ok, detail in results:
        all_ok &= ok
        print(f"{'✅ PASS' if ok else '❌ FAIL'} {name}")
        print(f"   {detail}")
    print("=" * 72)
    print("验收2(意图 JSON 稳定)= scripts/eval_intent 的 16+1 条标注(单独真跑)")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
