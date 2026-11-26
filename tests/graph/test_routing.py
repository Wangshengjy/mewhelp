"""ch05 · 分流规则 + ReAct 停止条件(确定性 → TDD)。"""
from langchain_core.messages import AIMessage

from app.graph.deps import GraphDeps
from app.graph.routing import (
    INTENT_TO_ROUTE,
    confidence_gate,
    route_by_intent,
    should_continue,
)
from app.settings import Settings


def test_intent_to_route_covers_eight_to_five():
    assert len(INTENT_TO_ROUTE) == 8
    assert set(INTENT_TO_ROUTE.values()) == {
        "knowledge",
        "refund_flow",
        "business",
        "complaint",
        "fallback_script",
    }
    cases = {
        "商品咨询": "knowledge",
        "退款退货": "refund_flow",
        "物流": "business",
        "订单": "business",
        "售后": "business",
        "投诉": "complaint",
        "闲聊": "fallback_script",
        "其他": "business",
    }
    for intent, route in cases.items():
        assert route_by_intent({"intent": intent}) == route


def test_route_unknown_intent_defaults_business():
    assert route_by_intent({"intent": "火星语"}) == "business"
    assert route_by_intent({}) == "business"


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"
    assert confidence_gate({}) == "weak"


def _tool_call_ai() -> AIMessage:
    return AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])


def test_should_continue_stops_when_no_tool_calls():
    state = {"messages": [AIMessage("答完了")], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_should_continue_continues_on_tool_calls_under_budget():
    state = {"messages": [_tool_call_ai()], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "continue"


def test_should_continue_stops_on_step_cap_from_settings():
    """步数上限以 config 注入的 settings.max_agent_steps 为准。"""
    deps = GraphDeps(
        session_factory=None,
        settings=Settings(
            chat_base_url="http://localhost.invalid/v1",
            chat_model="unavailable",
            chat_api_key="unavailable",
            max_agent_steps=2,
        ),
    )
    config = {"configurable": {"deps": deps}}
    assert should_continue({"messages": [_tool_call_ai()], "steps": 1}, config) == "continue"
    assert should_continue({"messages": [_tool_call_ai()], "steps": 2}, config) == "stop"


def test_should_continue_stops_on_default_cap_without_deps():
    """没有 deps 注入时用默认 6(与 settings 默认一致)。"""
    assert should_continue({"messages": [_tool_call_ai()], "steps": 5}) == "continue"
    assert should_continue({"messages": [_tool_call_ai()], "steps": 6}) == "stop"


def test_token_cost_is_not_a_stop_condition():
    """环里只按步数封顶;token 记账继续走(进 trace、给 ch09 统计)。"""
    state = {"messages": [_tool_call_ai()], "steps": 1, "tokens_used": 10_000_000}
    assert should_continue(state) == "continue"
