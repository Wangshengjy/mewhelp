"""ch07 · memory 三段切窗/层 2 半压/token 口径 单测。"""

import dataclasses

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.core import memory
from app.settings import Settings


def _user(text: str, msg_id: int | None = None) -> HumanMessage:
    return HumanMessage(content=text, id=f"db-{msg_id}" if msg_id is not None else None)


def _tool(content: str, call_id: str = "t1", name: str = "query_faq") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


def _dialog() -> list:
    return [
        _user("我要查订单1001", 1),
        AIMessage(content="好的" * 40),  # 长答复 → 层 2 应被压头
        _user("再帮我看看物流", 2),
        AIMessage(content="物流已经到达北京分拨中心,预计明日送达,请留意查收。" * 3),  # 层 2 压头
        _tool("一大段工具结果" * 50),  # 350 字 ≫ 200 token → 层 2 占位
        _user("谢谢", 3),
        AIMessage(content="不客气"),
    ]


# ── 锚点解析 ──────────────────────────────────────────────────────────────────


def test_db_msg_id_parses_anchor_only():
    assert memory.db_msg_id(_user("hi", 5)) == 5
    assert memory.db_msg_id(AIMessage(content="hi")) == 0  # 自动 id
    assert memory.db_msg_id(HumanMessage(content="hi", id="db-abc")) == 0  # 非数字


def test_count_tokens_uses_zh_chars_per_token():
    message = HumanMessage(content="a" * 120)
    default = memory.count_tokens([message])  # zh_chars_per_token=1.2
    wide = memory.count_tokens([message], dataclasses.replace(Settings.hermetic_defaults(), zh_chars_per_token=4.0))
    assert abs(default - (120 / 1.2 + 3)) <= 8  # 口径含 role/id 开销,允许小幅偏差
    assert wide < default  # 口径越大,估算越小
    assert memory.chars_to_tokens("") == 0


# ── 三段切窗 ──────────────────────────────────────────────────────────────────


def test_build_window_three_segments():
    window = memory.build_window(_dialog(), summary_upto_msg_id=1, layer1_from_msg_id=2)
    texts = [m.content for m in window]
    # 已摘要段(锚点1)不出现
    assert all("订单1001" not in t for t in texts)
    # 层 2 = 锚点2 的整轮(用户原话不动 + 答复压头 + 工具占位)
    layer2_user = window[0]
    assert isinstance(layer2_user, HumanMessage) and layer2_user.id == "db-2"
    assert layer2_user.content == "再帮我看看物流"
    assert window[1].content.endswith("…(略)")  # 层 2 答复被压
    assert window[2].content.startswith("(已调用 query_faq")  # 层 2 工具占位
    # 层 1 = 锚点3 之后原样
    assert window[3].content == "谢谢"
    assert window[4].content == "不客气"
    assert [m.id for m in window[:1]] == ["db-2"]  # 锚点 id 保留


def test_build_window_summary_cut_only():
    window = memory.build_window(_dialog(), summary_upto_msg_id=2, layer1_from_msg_id=0)
    assert [m.content for m in window] == ["谢谢", "不客气"]


def test_build_window_no_anchors_degrades_to_trim():
    messages = [HumanMessage(content="早" * 400), AIMessage(content="答" * 400), HumanMessage(content="新问题"), AIMessage(content="新的回答")]
    window = memory.build_window(messages, 0, 0, max_tokens=60)
    assert window[0].content == "新问题"  # strategy=last + start_on=human,保住最近一轮


def test_build_window_trim_to_empty_falls_back():
    messages = _dialog()
    window = memory.build_window(messages, 0, 0, max_tokens=0)  # 预算不健康:滑窗=0
    assert len(window) == len(messages)  # 裁到空宁可超预算,回退原窗


def test_build_window_untrimmed_by_default():
    window = memory.build_window(_dialog(), 0, 0)
    assert len(window) == len(_dialog())


# ── 分层用量 ──────────────────────────────────────────────────────────────────


def test_layer_tokens_measures_both_layers():
    l2, l1 = memory.layer_tokens(_dialog(), 1, 2)
    assert l2 > 0 and l1 > 0
    # 层 2 半压后应小于该层原始体积
    raw_layer2 = memory.count_tokens(_dialog()[2:5])
    assert l2 < raw_layer2


def test_layer_tokens_zero_boundaries_means_single_layer():
    l2, l1 = memory.layer_tokens(_dialog(), 0, 0)
    assert l2 == 0
    assert l1 == memory.count_tokens(_dialog())


# ── 层 1 降级算新边界 ─────────────────────────────────────────────────────────


def test_next_layer1_from_walks_back_to_budget():
    messages = [
        _user("字" * 120, 1), AIMessage(content="字" * 120),
        _user("字" * 120, 2), AIMessage(content="字" * 120),
        _user("字" * 120, 3), AIMessage(content="字" * 120),
    ]
    # 每条约 107 token(role/id 计入),一轮 214;预算装下最后两轮 → 层 2 留锚点 1
    assert memory.next_layer1_from(messages, 0, 450) == 1
    assert memory.next_layer1_from(messages, 0, 300) == 2  # 只装下一轮
    assert memory.next_layer1_from(messages, 0, 100_000) == 0  # 全量不超,无需降级


def test_next_layer1_from_respects_summary_boundary_and_min_anchors():
    messages = [
        _user("字" * 120, 1), AIMessage(content="字" * 120),
        _user("字" * 120, 2), AIMessage(content="字" * 120),
        _user("字" * 120, 3), AIMessage(content="字" * 120),
    ]
    # 摘要已吃到锚点 1:层 1 只能在锚点 2/3 之间选,边界不会再落回 1
    assert memory.next_layer1_from(messages, 1, 300) == 2
    assert memory.next_layer1_from([_user("只有一轮", 1)], 0, 1) == 0  # 锚点不足两条,无法降级
    assert memory.next_layer1_from([AIMessage(content="无锚点")], 0, 1) == 0


def test_next_layer1_from_never_splits_current_turn():
    messages = [
        _user("字" * 120, 1), AIMessage(content="字" * 120),
        _user("字" * 120, 2), AIMessage(content="超长的一轮" + "字" * 600),
    ]
    # 连最新一轮都装不下:保底只留最新一轮,把锚点 1 推进层 2
    assert memory.next_layer1_from(messages, 0, 10) == 1


# ── 层 2 半压 ─────────────────────────────────────────────────────────────────


def test_compress_reply_keeps_head():
    long = "长答复" * 50
    assert memory.compress_reply(long) == long[:60].rstrip() + "…(略)"
    assert memory.compress_reply("短") == "短"
    assert memory.compress_reply(long, keep_chars=10) == long[:10] + "…(略)"


def test_compress_tool_result_placeholder():
    assert memory.compress_tool_result("query_faq", "短结果") == "短结果"
    assert memory.compress_tool_result("query_faq", "长" * 500) == "(已调用 query_faq,结果从略)"
    assert memory.compress_tool_result(None, "长" * 500) == "(已调用 工具,结果从略)"


def test_to_layer2_preserves_tool_calls_and_ids():
    ai = AIMessage(content="我来查一下", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "call-1", "type": "tool_call"}])
    out = memory.to_layer2([_user("订单1001到哪了", 7), ai, _tool("结果" * 200)])
    assert out[0] is not None and out[0].id == "db-7"  # 用户原话连锚点一起保留
    assert out[1].tool_calls == ai.tool_calls  # tool_calls 原样带上,否则 ToolMessage 配不上对 400
    assert out[1].id == ai.id
    assert out[2].content.startswith("(已调用")


# ── 摘要注入 ──────────────────────────────────────────────────────────────────


def test_summary_line_and_system():
    assert memory.summary_line(None) == ""
    assert memory.summary_line("   ") == ""
    line = memory.summary_line("用户咨询订单1001物流")
    assert line == "(早前对话摘要:用户咨询订单1001物流)"
    block = memory.summary_system("用户咨询订单1001物流")
    assert isinstance(block, SystemMessage) and block.content == line
    assert memory.summary_system("") is None


def test_window_budget_explicit_override_wins():
    settings = dataclasses.replace(Settings.hermetic_defaults(), context_window_max_tokens=1234)
    assert memory.window_budget(settings) == 1234
    assert memory.window_budget(None) == 0  # hermetic 配置窗口装不下 → 不健康,滑窗归零兜底
