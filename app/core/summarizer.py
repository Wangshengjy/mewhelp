"""ch07 · 异步分段摘要器:层 2 超预算就压,后台任务不阻塞当轮回复。

设计要点(Spec §7/§8):
- 触发看用量不数条数:层 2 半压渲染后的 token 超过 sliding×(1-layer1_ratio) 才压。
- 摘要一段一行只追加;压完的段落不回炉重压,一个事实只经历一次有损压缩。
- 旧摘要只作背景给模型看,不参与合并(SUMMARY_SYSTEM 约束)。
- maybe_schedule_summary 用 dict[cid→Task] 防抖:同一会话同时只允许一个在途任务;
  调度失败/任务失败只留痕,不外抛,不阻塞当轮回复。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Sequence
from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.core import budget as budget_mod
from app.core import memory
from app.prompts import SUMMARY_PROMPT, SUMMARY_SYSTEM
from app.repositories import conversations as repo
from app.settings import Settings

logger = logging.getLogger(__name__)

# 防抖表:同一会话最多一个在途摘要任务
_pending: dict[str, asyncio.Task] = {}


class _SummaryOut(BaseModel):
    """SUMMARY_SYSTEM 约束的严格 JSON 形状。"""

    summary: str


def _settings(settings: Settings | None) -> Settings:
    return settings or Settings.hermetic_defaults()


def _build_model(settings: Settings | None = None) -> ChatOpenAI:
    """摘要生成模型:SUMMARY_* 缺省回落 CHAT_*。"""
    s = _settings(settings)
    return ChatOpenAI(
        base_url=s.summary_base_url or s.chat_base_url,
        model=s.summary_model or s.chat_model,
        api_key=s.summary_api_key or s.chat_api_key,
        temperature=0,
        timeout=max(s.tool_timeout_seconds, 30),
    )


def _parse_summary(raw: str) -> str:
    """严格 JSON 解析,失败降级正则抓 summary 字段;都不行抛 ValueError。"""
    text = (raw or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
        summary = data.get("summary") if isinstance(data, dict) else None
        if isinstance(summary, str) and summary.strip():
            return summary.strip()
    except json.JSONDecodeError:
        pass
    match = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    if match and match.group(1).strip():
        return match.group(1).strip()
    raise ValueError(f"summary output is not valid JSON: {text[:80]!r}")


def _format_dialog(messages: Sequence[BaseMessage]) -> str:
    """对话渲染成 用户:/客服: 行;工具调用中间态不进摘要。"""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"用户:{message.content}")
        elif isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
            content = message.content if isinstance(message.content, str) else ""
            if content.strip():
                lines.append(f"客服:{content}")
    return "\n".join(lines)


async def summarize_dialog(
    old_summary: str,
    dialog: Sequence[BaseMessage],
    settings: Settings | None = None,
    model: object | None = None,
) -> str:
    """压一段对话成新梗概。旧摘要只作背景;输出非 JSON 抛 ValueError 由调用方留痕。"""
    text = _format_dialog(dialog)
    if not text.strip():
        raise ValueError("dialog is empty")
    model = model or _build_model(settings)
    old_section = (
        f"(背景参考,帮助你理解上下文,不要合并进新梗概)\n旧摘要:{old_summary.strip()}\n\n"
        if old_summary and old_summary.strip()
        else ""
    )
    prompt = SUMMARY_PROMPT.format(old_summary_section=old_section, dialog=text)
    response = await model.ainvoke(
        [SystemMessage(content=SUMMARY_SYSTEM), HumanMessage(content=prompt)]
    )
    raw = response.content if isinstance(response.content, str) else str(response.content)
    summary = _parse_summary(raw)
    if not 10 <= len(summary) <= 400:
        raise ValueError(f"summary length out of range: {len(summary)}")
    return summary


# ── 层 2 用量与触发 ───────────────────────────────────────────────────────────


def _load_layer2_usage(
    session_factory: Callable, conversation_id: str, settings: Settings
) -> tuple[int, int]:
    """同步读层 2 当前内容并按半压口径计 token,返回 (用量, 预算)。"""
    with session_factory() as session:
        conversation = repo.get_conversation(session, conversation_id)
        if conversation is None:
            return 0, 0
        layer1_from = conversation.layer1_from_msg_id or 0
        summary_upto = conversation.summary_upto_msg_id or 0
        rows = repo.list_dialog_messages(session, conversation_id)
    if layer1_from <= 0:
        return 0, budget_mod.compute(settings).layer2_budget
    user_after = [row.id for row in rows if row.role == "user" and row.id > layer1_from]
    next_user = user_after[0] if user_after else None
    layer2_rows = [
        row
        for row in rows
        if row.id > summary_upto and (next_user is None or row.id < next_user)
    ]
    messages = [
        HumanMessage(content=row.content) if row.role == "user" else AIMessage(content=row.content)
        for row in layer2_rows
    ]
    usage = memory.count_tokens(memory.to_layer2(messages, settings), settings)
    return usage, budget_mod.compute(settings).layer2_budget


async def _should_summarize(
    session_factory: Callable, conversation_id: str, settings: Settings | None = None
) -> bool:
    """层 2 用量超预算就压;看用量不数条数。"""
    usage, limit = await asyncio.to_thread(
        _load_layer2_usage, session_factory, conversation_id, _settings(settings)
    )
    return usage > limit


# ── 执行摘要 ──────────────────────────────────────────────────────────────────


def _load_summary_target(
    session_factory: Callable, conversation_id: str
) -> tuple[int, int, str, list[BaseMessage]] | None:
    """读出待压段落:覆盖区间 [from, upto] + 旧摘要 + 对话行。无可压返回 None。"""
    with session_factory() as session:
        conversation = repo.get_conversation(session, conversation_id)
        if conversation is None:
            return None
        layer1_from = conversation.layer1_from_msg_id or 0
        summary_upto = conversation.summary_upto_msg_id or 0
        old_summary = conversation.summary or ""
        rows = repo.list_dialog_messages(session, conversation_id)
    if layer1_from <= 0:
        return None  # 层 1 还没降级过,没有可压的层 2
    user_after = [row.id for row in rows if row.role == "user" and row.id > layer1_from]
    next_user = user_after[0] if user_after else None
    segment_rows = [
        row
        for row in rows
        if row.id > summary_upto and (next_user is None or row.id < next_user)
    ]
    if not segment_rows:
        return None  # 摘要已追平层 2 起点
    dialog = [
        HumanMessage(content=row.content) if row.role == "user" else AIMessage(content=row.content)
        for row in segment_rows
    ]
    return segment_rows[0].id, segment_rows[-1].id, old_summary, dialog


def _commit_segment(
    session_factory: Callable,
    conversation_id: str,
    from_msg_id: int,
    upto_msg_id: int,
    content: str,
    inject_segments: int,
) -> int:
    with session_factory() as session:
        seq = repo.append_summary_segment(
            session, conversation_id, from_msg_id, upto_msg_id, content,
            inject_segments=inject_segments,
        )
        session.commit()
        return seq


async def run_summary(
    session_factory: Callable, conversation_id: str, settings: Settings | None = None
) -> str | None:
    """把层 2 当前整段压成新摘要段落库,推进投影与 summary_upto;无可压返回 None。"""
    s = _settings(settings)
    target = await asyncio.to_thread(_load_summary_target, session_factory, conversation_id)
    if target is None:
        return None
    from_msg_id, upto_msg_id, old_summary, dialog = target
    summary = await summarize_dialog(old_summary, dialog, s)
    seq = await asyncio.to_thread(
        _commit_segment, session_factory, conversation_id, from_msg_id, upto_msg_id, summary, s.summary_inject_segments
    )
    logger.info(
        "summary append conv=%s 第%d段 覆盖[%d,%d] %d 字", conversation_id, seq, from_msg_id, upto_msg_id, len(summary)
    )
    return summary


# ── 轮后调度 ──────────────────────────────────────────────────────────────────


def _default_session_factory() -> Callable:
    from app.graph import runtime  # 局部导入避免 runtime→summarizer→runtime 循环

    return runtime.get_deps().session_factory


def _discard_task(conversation_id: str, task: asyncio.Task) -> None:
    if _pending.get(conversation_id) is task:
        _pending.pop(conversation_id, None)


async def _run_logged(
    conversation_id: str, session_factory: Callable, settings: Settings
) -> None:
    started = time.monotonic()
    logger.info("summary start conv=%s", conversation_id)
    try:
        summary = await run_summary(session_factory, conversation_id, settings)
        if summary is None:
            logger.info("summary skip conv=%s 无可压段落", conversation_id)
        else:
            logger.info(
                "summary done conv=%s 耗时 %.1fs 新段 %d 字",
                conversation_id,
                time.monotonic() - started,
                len(summary),
            )
    except Exception as exc:
        logger.warning(
            "summary failed conv=%s (%s: %s)", conversation_id, type(exc).__name__, exc
        )


async def maybe_schedule_summary(
    conversation_id: str,
    *,
    session_factory: Callable | None = None,
    settings: Settings | None = None,
) -> None:
    """轮后调度:层 2 超预算才起后台任务;防抖;异常只留痕不外抛。"""
    try:
        factory = session_factory or _default_session_factory()
        s = _settings(settings)
        task = _pending.get(conversation_id)
        if task is not None and not task.done():
            logger.info("summary skip conv=%s 已有在途任务", conversation_id)
            return
        usage, limit = await asyncio.to_thread(
            _load_layer2_usage, factory, conversation_id, s
        )
        if usage <= limit:
            logger.debug(
                "summary skip conv=%s 层2 %d token ≤ 预算 %d", conversation_id, usage, limit
            )
            return
        logger.info(
            "summary trigger conv=%s 层2 约 %d token > 预算 %d", conversation_id, usage, limit
        )
        task = asyncio.create_task(_run_logged(conversation_id, factory, s))
        _pending[conversation_id] = task
        task.add_done_callback(lambda done: _discard_task(conversation_id, done))
    except Exception as exc:
        logger.warning(
            "summary schedule failed conv=%s (%s: %s)",
            conversation_id,
            type(exc).__name__,
            exc,
        )
