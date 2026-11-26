"""ch05 · 五验收端到端评估(直连图)。需 MySQL + Milvus + 真实上游可用。

本仓库 ch05 只保留 /api/chat/stream 一个 HTTP 入口,没有 /api/agent,
评估走 app/graph/runtime.run_turn 直连图(每题独立会话,conversation_id=None),
断言终态 State(intent/route/trace/steps/suggested_actions/messages 工具轨迹)。

DeepSeek 欠费 402 期间跑不了,记 dev-notes 待验证。
用法:
    .\\.venv\\Scripts\\python.exe -m scripts.eval_ch05
"""
import asyncio

from langchain_core.messages import AIMessage

from app.graph import runtime
from app.graph.nodes import CHITCHAT_REPLY, COMPLAINT_REPLY

# ch06 退役 chitchat_reply 节点:闲聊并入 script_reply,文案仍=CHITCHAT_REPLY 常量
_SCRIPT_CHITCHAT = CHITCHAT_REPLY


def tool_names(state: dict) -> set[str]:
    """收集整轮所有 AIMessage 发起的工具名(ReAct 轨迹)。"""
    names: set[str] = set()
    for message in state.get("messages", []):
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                name = call.get("name")
                if name:
                    names.add(name)
    return names


async def turn(message: str) -> dict:
    """跑一轮(独立会话),返回图终态 State。"""
    result = await runtime.run_turn(None, message, user_id="eval-ch05")
    return result["state"]


async def main() -> None:
    await runtime.init_graph()
    results: list[tuple[str, bool, str]] = []
    try:
        # 验收1:知识类问句走强制检索节点(trace.forced_rag=True)
        state = await turn("退货政策是什么")
        trace = state.get("trace", {})
        ok = state.get("route") == "knowledge" and trace.get("forced_rag") is True
        results.append(("验收1 强制检索(route=knowledge & trace.forced_rag)", ok,
                        f"route={state.get('route')} gate={trace.get('gate')}"))

        # 验收2:业务数据类,Agent 自己决定调工具(物流查询)
        state = await turn("订单1001的物流到哪了")
        names = tool_names(state)
        results.append(("验收2 业务路自调工具(query_logistics)", "query_logistics" in names,
                        sorted(names)))

        # 验收3:投诉出「转人工」「建工单」两可选项,后端不自动建单
        state = await turn("我要投诉,你们太差了")
        types = {action.get("type") for action in state.get("suggested_actions", [])}
        ok = {"transfer_human", "create_ticket"} <= types and state.get("answer") == COMPLAINT_REPLY
        results.append(("验收3 投诉出两可选项+固定话术", ok,
                        f"types={sorted(types)} answer前12字={str(state.get('answer', ''))[:12]}"))

        # 验收4:闲聊走脚本出口(ch06 script_reply),固定话术零工具
        state = await turn("你好呀")
        ok = state.get("answer") == _SCRIPT_CHITCHAT and not tool_names(state)
        results.append(("验收4 闲聊固定话术零工具", ok,
                        f"answer前12字={str(state.get('answer', ''))[:12]}"))

        # 验收5:复杂问 ReAct 顺序多步(query_order 产出 tracking_no → query_logistics)
        # 注:原用例「手机尾号1001」对 14B 属模糊指代,模型按规则1稳定选择追问(合规),
        # 改为明确订单号的复合问句,仍考察 order→logistics 数据依赖
        state = await turn("订单1001的物流到哪了?顺便告诉我这单里有什么商品")
        names = tool_names(state)
        ok = state.get("steps", 0) > 1 and {"query_order", "query_logistics"} <= names
        results.append(("验收5 ReAct 多步(steps>1 且 order→logistics 数据依赖)", ok,
                        f"steps={state.get('steps')} tools={sorted(names)}"))
    finally:
        await runtime.close_graph()

    print("=" * 60)
    for name, ok, detail in results:
        print(f"{'✅' if ok else '❌'} {name} -> {detail}")
    print("=" * 60)
    print("验收3(前端部分):浏览器问「我要投诉」→ 出「转人工」「建工单」两按钮;")
    print("  点转人工出已转接+客服小猫问候;点建工单填表提交写 tickets 表;都不点继续正常对话。")
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n五验收 {passed}/5 通过(上游非确定性,抖动如实重跑记录)")


if __name__ == "__main__":
    asyncio.run(main())
