"""ch07 · 上下文预算:从窗口大小倒推滑窗与层 1/层 2 预算。

口径(与课程 Spec §6 一致):
    fixed   = system + evidence + summary + max_output + safety
    peak    = max_user_input + steps × (tool_result + step_ai)
    room    = window - fixed - peak
    sliding = min(turns × per_turn, room),取小者并记 limited_by
    层 1 预算 = sliding × layer1_ratio,层 2 预算 = sliding × (1 - ratio)

演示配置(18000/2000/2000/3 步/1200/5 条证据)应算出滑窗 5650、层 1 3954、层 2 1695,
tests/test_budget.py 用同一组数字钉死口径。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.settings import Settings

# 已知模型的上下文窗口;键为小写子串匹配。拿不准的模型查 Context7/官方文档后补表。
_KNOWN_WINDOWS: dict[str, int] = {
    "qwen2.5-14b-instruct": 32768,
    "qwen2.5-7b-instruct": 32768,
    "qwen2.5-72b-instruct": 32768,
    "qwen3-14b": 40960,
    "deepseek-v3": 65536,
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4.1-mini": 1000000,
    "gpt-4.1": 1000000,
}

# 查不到窗口时的保守值:宁可算小提前降级,也不要把请求顶爆。
_CONSERVATIVE_WINDOW = 8192


def lookup_window(model: str) -> tuple[int, bool]:
    """查已知模型上下文窗口,返回 (窗口大小, 是否命中已知表)。"""
    lowered = (model or "").lower()
    for prefix, window in _KNOWN_WINDOWS.items():
        if prefix in lowered:
            return window, True
    return _CONSERVATIVE_WINDOW, False


def resolve_window(settings: Settings | None) -> int:
    """显式 MODEL_CONTEXT_WINDOW 优先,否则查表;按命中与否决定是否告警由调用方看 healthy。"""
    resolved = settings or Settings.hermetic_defaults()
    if resolved.model_context_window > 0:
        return resolved.model_context_window
    return lookup_window(resolved.chat_model)[0]


def summary_tokens(settings: Settings | None) -> int:
    """摘要投影注入的 token 预算:段数 × 每段预算。"""
    resolved = settings or Settings.hermetic_defaults()
    return resolved.summary_inject_segments * resolved.summary_tokens_per_segment


def evidence_tokens(settings: Settings | None, top_k: int | None = None) -> int:
    """证据块的 token 预算:条数 × 每条预算。"""
    resolved = settings or Settings.hermetic_defaults()
    return (top_k if top_k is not None else resolved.rerank_top_k) * resolved.evidence_tokens_per_chunk


def turn_peak_tokens(settings: Settings | None) -> int:
    """单轮峰值开销:输入 + 每步(工具结果 + 一次 AI 输出)× 最大步数。"""
    resolved = settings or Settings.hermetic_defaults()
    return resolved.max_user_input_tokens + resolved.max_agent_steps * (
        resolved.tool_result_max_tokens + resolved.agent_step_ai_tokens
    )


def history_per_turn(settings: Settings | None) -> int:
    """每轮历史增量:输入 + 输出 + 层 2 压缩后的工具占位 × 步数。"""
    resolved = settings or Settings.hermetic_defaults()
    return (
        resolved.max_user_input_tokens
        + resolved.max_output_tokens
        + resolved.max_agent_steps * resolved.layer2_tool_keep_tokens
    )


@dataclass(frozen=True)
class ContextBudget:
    """一次预算倒推的完整结果,describe() 直接渲染成人话。"""

    window: int
    fixed: int
    peak: int
    per_turn: int
    turns: int
    sliding: int
    limited_by: str
    healthy: bool
    layer1_budget: int
    layer2_budget: int

    def describe(self) -> str:
        """一行人话,供启动自检与 model_ctx 日志引用。"""
        cap_note = f"{self.turns}轮×{self.per_turn}" if self.limited_by == "turns" else "窗口受限"
        return (
            f"window={self.window} fixed={self.fixed} peak={self.peak} "
            f"per_turn={self.per_turn} sliding={self.sliding}({cap_note}) "
            f"层1预算={self.layer1_budget} 层2预算={self.layer2_budget} healthy={self.healthy}"
        )


def compute(settings: Settings | None = None) -> ContextBudget:
    """从窗口大小倒推滑窗与分层预算。room≤0 记不健康,sliding 归零兜底。"""
    resolved = settings or Settings.hermetic_defaults()
    window = resolve_window(resolved)
    fixed = (
        resolved.system_prompt_tokens
        + evidence_tokens(resolved)
        + summary_tokens(resolved)
        + resolved.max_output_tokens
        + resolved.budget_safety_margin
    )
    peak = turn_peak_tokens(resolved)
    per_turn = history_per_turn(resolved)
    room = window - fixed - peak
    turn_cap = resolved.context_budget_turns * per_turn
    if room > 0:
        sliding = min(turn_cap, room)
        limited_by = "turns" if turn_cap <= room else "window"
    else:
        sliding = 0
        limited_by = "window"
    healthy = room > 0 and sliding > 0
    layer1_budget = int(sliding * resolved.layer1_ratio)
    layer2_budget = int(sliding * (1 - resolved.layer1_ratio))
    return ContextBudget(
        window=window,
        fixed=fixed,
        peak=peak,
        per_turn=per_turn,
        turns=resolved.context_budget_turns,
        sliding=sliding,
        limited_by=limited_by,
        healthy=healthy,
        layer1_budget=layer1_budget,
        layer2_budget=layer2_budget,
    )
