"""ch05/ch06 · 分流规则与 ReAct 停止条件(纯函数,TDD)。

八类意图 → 五出口的映射写死在代码里:分流规则是产品决策,不该藏在 Prompt
里让模型即兴发挥。模型只负责"判断是哪类",不负责"判断去哪条路"。
"""
from langchain_core.messages import AIMessage

from app.graph.deps import GraphDeps

# 八类意图 → 五出口(ch06):商品咨询走知识路(先检索再生成);退款退货进
# 人工接管前的自动链(抓订单 → 检索退款政策 → 带证据判定);物流/订单/售后
# 交给主力 Agent(自调业务工具);投诉走确定性话术;闲聊/其他走脚本兜底。
INTENT_TO_ROUTE: dict[str, str] = {
    "物流": "business",
    "订单": "business",
    "商品咨询": "knowledge",
    "退款退货": "refund_flow",
    "售后": "business",
    "投诉": "complaint",
    "闲聊": "fallback_script",
    "其他": "business",
}

ROUTES = ("knowledge", "refund_flow", "business", "complaint", "fallback_script")

# 与 settings.max_agent_steps 的默认值一致;config 注入 deps 时以 settings 为准
DEFAULT_MAX_AGENT_STEPS = 6


def route_by_intent(state) -> str:
    """按意图分流;未知意图保守归 business(让主力 Agent 自己应对)。"""
    return INTENT_TO_ROUTE.get(state.get("intent", ""), "business")


def confidence_gate(state) -> str:
    """知识路生成前证据闸:强放行、弱兜底。"""
    return "strong" if state.get("evidence_strong") else "weak"


def should_continue(state, config=None) -> str:
    """ReAct 停止条件:无 tool_calls 收敛;步数封顶强停;否则继续环。

    防打转靠步数:一步一次模型调用,数得清、好解释——各家 Agent SDK 给的
    也是 max_turns / max_iterations 这类步数上限。token 花销只进
    state.tokens_used 记账(给 ch09 成本统计),不参与停止判断。
    """
    last = state["messages"][-1]
    has_tool_calls = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not has_tool_calls:
        return "stop"
    if state.get("steps", 0) >= _max_agent_steps(config):
        return "stop"
    return "continue"


def _max_agent_steps(config) -> int:
    if config:
        deps = (config.get("configurable") or {}).get("deps")
        if isinstance(deps, GraphDeps):
            return deps.settings.max_agent_steps
    return DEFAULT_MAX_AGENT_STEPS
