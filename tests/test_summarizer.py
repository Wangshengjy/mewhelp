"""ch07 · 分段摘要器单测(全 mock,不碰上游真实模型)。"""

import asyncio
import dataclasses
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core import summarizer
from app.repositories import conversations as repo
from app.settings import Settings

HEALTHY = dataclasses.replace(
    Settings.hermetic_defaults(),
    model_context_window=32768,
    max_output_tokens=4000,
    tool_result_max_tokens=1200,
)


class _FakeModel:
    """记录输入并回罐头 JSON 的假模型。"""

    def __init__(self, content: str):
        self.content = content
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.content)


def _seed(session_factory, *, layer1_from=0, upto=0, reply="客服答复。" + "细节" * 50):
    """建会话并写入三轮 user/assistant,返回 (cid, [(u.id, a.id), ...])。"""
    with session_factory() as session:
        conversation = repo.create_conversation(session)
        pairs = []
        for index in range(1, 4):
            user = repo.append_message(session, conversation.id, role="user", content=f"第{index}轮,查订单100{index}")
            assistant = repo.append_message(session, conversation.id, role="assistant", content=reply)
            pairs.append((user.id, assistant.id))
        if layer1_from:
            repo.update_layer1_from(session, conversation.id, layer1_from)
        if upto:
            repo.update_conversation_summary(session, conversation.id, "旧摘要内容", upto)
        session.commit()
        return conversation.id, pairs


# ── JSON 解析 ─────────────────────────────────────────────────────────────────


def test_parse_summary_accepts_strict_json_and_fences():
    assert summarizer._parse_summary('{"summary": "用户咨询订单1001"}') == "用户咨询订单1001"
    assert summarizer._parse_summary('```json\n{"summary": "带围栏"}\n```') == "带围栏"
    assert summarizer._parse_summary('前言 {"summary": "正则兜底"} 后记') == "正则兜底"


def test_parse_summary_rejects_non_json():
    with pytest.raises(ValueError):
        summarizer._parse_summary("这不是JSON")
    with pytest.raises(ValueError):
        summarizer._parse_summary('{"summary": ""}')


# ── summarize_dialog ──────────────────────────────────────────────────────────


async def test_summarize_dialog_builds_prompt_and_returns_summary():
    model = _FakeModel('{"summary": "用户咨询订单1001物流,客服答复已发货"}')
    dialog = [HumanMessage(content="查订单1001"), AIMessage(content="已发货")]
    result = await summarizer.summarize_dialog("旧摘要", dialog, model=model)
    assert result == "用户咨询订单1001物流,客服答复已发货"
    system, human = model.calls[0]
    assert isinstance(system, SystemMessage) and system.content == summarizer.SUMMARY_SYSTEM
    assert "旧摘要" in human.content  # 旧摘要只作背景进 prompt
    assert "客服:已发货" in human.content


async def test_summarize_dialog_rejects_bad_output():
    model = _FakeModel("我不是JSON")
    with pytest.raises(ValueError):
        await summarizer.summarize_dialog("", [HumanMessage(content="你好")], model=model)
    with pytest.raises(ValueError):
        await summarizer.summarize_dialog("", [], model=model)  # 空对话


# ── 触发判断 ──────────────────────────────────────────────────────────────────


async def test_should_summarize_requires_boundary(session_factory):
    cid, _ = _seed(session_factory)
    assert not await summarizer._should_summarize(session_factory, cid, HEALTHY)  # 未降级,层2为空


async def test_should_summarize_under_budget(session_factory):
    cid, pairs = _seed(session_factory)
    # 层 1 边界压到第 2 轮起点(层 2 = 第 1 轮),远小于 HEALTHY 的 3995 预算
    with session_factory() as session:
        repo.update_layer1_from(session, cid, pairs[1][0])
        session.commit()
    assert not await summarizer._should_summarize(session_factory, cid, HEALTHY)


async def test_should_summarize_over_budget(session_factory):
    cid, pairs = _seed(session_factory)
    with session_factory() as session:
        repo.update_layer1_from(session, cid, pairs[1][0])
        session.commit()
    assert await summarizer._should_summarize(session_factory, cid, None)  # hermetic 预算 0,有内容即超


# ── 执行摘要 ──────────────────────────────────────────────────────────────────


async def test_run_summary_appends_segment_and_advances_upto(session_factory, monkeypatch):
    cid, pairs = _seed(session_factory)
    # 层 1 边界 = 第 2 轮起点用户消息 → 层 2 = 第 1 轮整轮
    with session_factory() as session:
        repo.update_layer1_from(session, cid, pairs[0][0])
        session.commit()

    async def _fake(old_summary, dialog, settings=None, model=None):
        return "新梗概:第1轮查了订单1001"

    monkeypatch.setattr(summarizer, "summarize_dialog", _fake)
    result = await summarizer.run_summary(session_factory, cid, HEALTHY)
    assert result == "新梗概:第1轮查了订单1001"
    with session_factory() as session:
        conversation = repo.get_conversation(session, cid)
        assert conversation.summary == "新梗概:第1轮查了订单1001"
        assert conversation.summary_upto_msg_id == pairs[0][1]  # 段末推进到第 1 轮答复
        assert repo.list_summary_segments(session, cid) == [result]


async def test_run_summary_returns_none_when_caught_up(session_factory):
    cid, pairs = _seed(session_factory, upto=10**9)  # 摘要已追平一切
    assert await summarizer.run_summary(session_factory, cid, HEALTHY) is None


async def test_run_summary_without_boundary(session_factory):
    cid, _ = _seed(session_factory)
    assert await summarizer.run_summary(session_factory, cid, HEALTHY) is None


# ── 轮后调度 ──────────────────────────────────────────────────────────────────


async def test_maybe_schedule_summary_triggers_background_run(session_factory, monkeypatch, caplog):
    cid, _ = _seed(session_factory)
    monkeypatch.setattr(summarizer, "_load_layer2_usage", lambda sf, c, s: (5000, 100))

    ran: list[str] = []

    async def _fake_run(sf, conv_id, settings=None):
        ran.append(conv_id)
        return "新段"

    monkeypatch.setattr(summarizer, "run_summary", _fake_run)
    caplog.set_level(logging.INFO, logger="app.core.summarizer")
    await summarizer.maybe_schedule_summary(cid, session_factory=session_factory, settings=HEALTHY)
    assert "summary trigger" in caplog.text
    assert cid in summarizer._pending
    await summarizer._pending[cid]
    assert ran == [cid]
    assert cid not in summarizer._pending  # 完成后清防抖表
    assert "summary done" in caplog.text


async def test_maybe_schedule_summary_skips_inflight_and_under_budget(session_factory, monkeypatch, caplog):
    cid, _ = _seed(session_factory)
    monkeypatch.setattr(summarizer, "_load_layer2_usage", lambda sf, c, s: (10, 100))
    caplog.set_level(logging.DEBUG, logger="app.core.summarizer")
    await summarizer.maybe_schedule_summary(cid, session_factory=session_factory, settings=HEALTHY)
    assert cid not in summarizer._pending

    monkeypatch.setattr(summarizer, "_load_layer2_usage", lambda sf, c, s: (5000, 100))
    blocker = asyncio.create_task(asyncio.sleep(3600))
    summarizer._pending[cid] = blocker
    ran: list[str] = []

    async def _fake_run(sf, conv_id, settings=None):
        ran.append(conv_id)
        return None

    monkeypatch.setattr(summarizer, "run_summary", _fake_run)
    await summarizer.maybe_schedule_summary(cid, session_factory=session_factory, settings=HEALTHY)
    assert ran == []  # 在途任务挡住,不起第二个
    assert "已有在途任务" in caplog.text
    blocker.cancel()
    summarizer._pending.pop(cid, None)


async def test_maybe_schedule_summary_swallows_errors(session_factory, monkeypatch, caplog):
    cid, _ = _seed(session_factory)

    def _boom(sf, c, s):
        raise RuntimeError("db down")

    monkeypatch.setattr(summarizer, "_load_layer2_usage", _boom)
    caplog.set_level(logging.WARNING, logger="app.core.summarizer")
    await summarizer.maybe_schedule_summary(cid, session_factory=session_factory, settings=HEALTHY)
    assert "summary schedule failed" in caplog.text
