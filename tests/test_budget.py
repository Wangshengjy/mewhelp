"""ch07 · 预算倒推 compute()/lookup_window()/describe() 单测。

演示配置(课程验收 2):MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000
MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200
RERANK_TOP_K=5 → 滑窗 5650、层 1 预算 3954、层 2 预算 1695。
"""

import dataclasses

from app.core import budget
from app.settings import Settings


def _demo() -> Settings:
    return dataclasses.replace(
        Settings.hermetic_defaults(),
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        max_agent_steps=3,
        tool_result_max_tokens=1200,
        rerank_top_k=5,
    )


def test_lookup_window_known_prefix_and_unknown():
    window, known = budget.lookup_window("Qwen/Qwen2.5-14B-Instruct")
    assert window == 32768 and known
    assert budget.lookup_window("gpt-4o-mini")[0] == 128000
    window, known = budget.lookup_window("totally-unknown-model")
    assert window == 8192 and not known  # 保守值兜底


def test_resolve_window_env_override_beats_table():
    assert budget.resolve_window(_demo()) == 18000
    settings = dataclasses.replace(Settings.hermetic_defaults(), chat_model="Qwen/Qwen2.5-14B-Instruct")
    assert budget.resolve_window(settings) == 32768


def test_demo_config_matches_course_numbers():
    result = budget.compute(_demo())
    assert result.window == 18000
    assert result.fixed == 700 + 5 * 160 + 3 * 250 + 2000 + 1000  # 5250
    assert result.peak == 2000 + 3 * (1200 + 500)  # 7100
    assert result.per_turn == 2000 + 2000 + 3 * 200  # 6600
    assert result.sliding == 5650
    assert result.limited_by == "window"
    assert result.healthy
    assert result.layer1_budget == 3954  # int(5650*0.7) 浮点截断
    assert result.layer2_budget == 1695  # int(5650*(1-0.7))


def test_turns_cap_limits_sliding():
    settings = dataclasses.replace(_demo(), model_context_window=150000)  # room 远大于轮数上限
    result = budget.compute(settings)
    assert result.limited_by == "turns"
    assert result.sliding == 20 * result.per_turn  # 轮数上限先到(92000)


def test_unhealthy_budget_zeroes_sliding():
    settings = dataclasses.replace(_demo(), max_output_tokens=17000)  # fixed+peak 超过窗口
    result = budget.compute(settings)
    assert not result.healthy
    assert result.sliding == 0
    assert result.layer1_budget == 0 and result.layer2_budget == 0


def test_describe_renders_one_line():
    line = budget.compute(_demo()).describe()
    assert "window=18000" in line
    assert "sliding=5650" in line
    assert "层1预算=3954" in line
    assert "层2预算=1695" in line
    assert "healthy=True" in line
