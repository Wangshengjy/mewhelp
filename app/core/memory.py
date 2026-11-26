"""ch07 · 会话上下文管理核心:三段切窗 + 层 2 半压 + token 口径。

ch02 遗留的 trim_history(app/chat.py)演进为本模块 build_window 的兜底段;
app/chat.py 本身保留不动。本模块只依赖 langchain_core + Settings,纯函数可单测。

三个层次由两个锚点表达(不搬数据):
    id ≤ summary_upto_msg_id          已进摘要,不再出现在窗口里
    summary_upto < id ≤ layer1_from   层 2,渲染成半压形态
    id > layer1_from                  层 1,原样

边界只落在带 db-{msg_id} 锚点的用户消息上;锚点缺失(老会话、旧 checkpoint)
自动退化为单层 + trim 兜底,行为与 ch06 一致。
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from app.core.budget import compute as _compute_budget
from app.settings import Settings

_DB_ID_PREFIX = "db-"


def _settings(settings: Settings | None) -> Settings:
    return settings or Settings.hermetic_defaults()


# ── token 口径 ────────────────────────────────────────────────────────────────


def db_msg_id(message: BaseMessage) -> int:
    """从消息 id 解析 db-{msg_id} 锚点;无锚点(老消息/自动 id)返回 0。"""
    raw = getattr(message, "id", None)
    if isinstance(raw, str) and raw.startswith(_DB_ID_PREFIX):
        try:
            return int(raw[len(_DB_ID_PREFIX) :])
        except ValueError:
            return 0
    return 0


def count_tokens(messages: Sequence[BaseMessage], settings: Settings | None = None) -> int:
    """统一 token 口径:count_tokens_approximately + zh_chars_per_token(中文 1.2 字/token)。"""
    return count_tokens_approximately(
        list(messages), chars_per_token=_settings(settings).zh_chars_per_token
    )


def chars_to_tokens(text: str, settings: Settings | None = None) -> int:
    """纯文本的 token 估算(工具结果/压缩判断用)。"""
    if not text:
        return 0
    return count_tokens([HumanMessage(content=text)], settings)


# ── 三段切窗 ──────────────────────────────────────────────────────────────────


def _first_user_index_after(messages: list[BaseMessage], anchor: int) -> int | None:
    """第一条 db-id 大于 anchor 的用户消息下标;找不到返回 None。"""
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage) and db_msg_id(message) > anchor:
            return index
    return None


def _split_layers(
    messages: list[BaseMessage], summary_upto: int, layer1_from: int
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """按两个锚点把消息切成 (层 2, 层 1);锚点缺失时退化为 ([], 全部)。"""
    start = _first_user_index_after(messages, max(summary_upto, 0))
    window = messages[start:] if start is not None else list(messages)
    mid = _first_user_index_after(window, max(layer1_from, 0))
    if mid is None:
        return [], window
    return window[:mid], window[mid:]


def build_window(
    messages: Sequence[BaseMessage],
    summary_upto_msg_id: int = 0,
    layer1_from_msg_id: int = 0,
    max_tokens: int | None = None,
    settings: Settings | None = None,
) -> list[BaseMessage]:
    """三段切窗:已摘要的不出现,层 2 半压,层 1 原样,超滑窗再 trim 兜底。

    锚点缺失(老会话)退化纯裁剪,与 ch06 行为一致;裁到空宁可超预算回退原窗。
    """
    messages = list(messages)
    layer2, layer1 = _split_layers(messages, summary_upto_msg_id or 0, layer1_from_msg_id or 0)
    window = [*to_layer2(layer2, settings), *layer1]
    if max_tokens is not None and max_tokens > 0 and count_tokens(window, settings) > max_tokens:
        trimmed = trim_messages(
            window,
            token_counter=lambda msgs: count_tokens(msgs, settings),
            max_tokens=max_tokens,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )
        return list(trimmed) if trimmed else window
    return window


def layer_tokens(
    messages: Sequence[BaseMessage],
    summary_upto_msg_id: int = 0,
    layer1_from_msg_id: int = 0,
    settings: Settings | None = None,
) -> tuple[int, int]:
    """(层 2 token, 层 1 token),层 2 按半压后的渲染口径计。"""
    layer2, layer1 = _split_layers(list(messages), summary_upto_msg_id or 0, layer1_from_msg_id or 0)
    return count_tokens(to_layer2(layer2, settings), settings), count_tokens(layer1, settings)


def next_layer1_from(
    messages: Sequence[BaseMessage],
    summary_upto_msg_id: int = 0,
    layer1_budget: int = 0,
    settings: Settings | None = None,
) -> int:
    """层 1 超预算时算新边界:从最新往回累加到预算为止,一次到位。

    层 1 起点必须落在用户消息锚点(整轮原子移动);返回层 2 里最后一条
    带锚点的用户消息 db-id。层 1 已达标、锚点不足两条时返回 0(无需/无法降级)。
    """
    messages = list(messages)
    start = _first_user_index_after(messages, max(summary_upto_msg_id, 0))
    window = messages[start:] if start is not None else messages
    anchors = [
        (index, db_msg_id(message))
        for index, message in enumerate(window)
        if isinstance(message, HumanMessage) and db_msg_id(message) > 0
    ]
    if len(anchors) < 2:
        return 0
    per_message = [count_tokens([message], settings) for message in window]
    suffix = [0] * (len(window) + 1)
    for index in range(len(window) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + per_message[index]
    for rank, (index, msg_id) in enumerate(anchors):
        if suffix[index] <= layer1_budget:
            if rank == 0:
                return 0  # 全量留在层 1 也不超预算,无需降级
            return anchors[rank - 1][1]
    # 连最新一轮都装不下:保底只把最新一轮留在层 1
    return anchors[-2][1]


# ── 层 2 半压 ─────────────────────────────────────────────────────────────────


def compress_reply(text: str, keep_chars: int | None = None, settings: Settings | None = None) -> str:
    """层 2 里的客服答复:保留开头若干字 + 省略标记,语义足够支撑指代。"""
    keep = _settings(settings).layer2_reply_keep_chars if keep_chars is None else keep_chars
    if len(text) <= keep:
        return text
    return text[:keep].rstrip() + "…(略)"


def compress_tool_result(name: str | None, content: str, settings: Settings | None = None) -> str:
    """层 2 里的工具结果:超预算就整条压成一行占位。"""
    if chars_to_tokens(content, settings) <= _settings(settings).layer2_tool_keep_tokens:
        return content
    return f"(已调用 {name or '工具'},结果从略)"


def to_layer2(messages: Sequence[BaseMessage], settings: Settings | None = None) -> list[BaseMessage]:
    """把一批消息渲染成层 2 半压形态。

    用户原话不动(保留 db- 锚点 id);客服答复压头;工具消息压占位——
    AIMessage 的 tool_calls 必须原样带上,否则和 ToolMessage 配不上对会 400。
    """
    out: list[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            out.append(message)
        elif isinstance(message, AIMessage):
            content = message.content if isinstance(message.content, str) else ""
            out.append(
                AIMessage(
                    content=compress_reply(content, settings=settings),
                    tool_calls=list(getattr(message, "tool_calls", []) or []),
                    id=message.id,
                )
            )
        elif isinstance(message, ToolMessage):
            content = message.content if isinstance(message.content, str) else ""
            out.append(
                ToolMessage(
                    content=compress_tool_result(message.name, content, settings),
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )
        else:
            out.append(message)
    return out


# ── 摘要注入 ──────────────────────────────────────────────────────────────────


def summary_line(summary: str | None) -> str:
    """摘要投影渲染成一行文本;空摘要返回空串(不占位置)。"""
    if not summary or not str(summary).strip():
        return ""
    return f"(早前对话摘要:{str(summary).strip()})"


def summary_system(summary: str | None) -> SystemMessage | None:
    """摘要包成文本块;.content 会被取出并进 HumanMessage,不会单独占一条 system。"""
    line = summary_line(summary)
    return SystemMessage(content=line) if line else None


# ── 滑窗预算入口 ──────────────────────────────────────────────────────────────


def window_budget(settings: Settings | None = None) -> int:
    """滑窗 token 上限:显式覆盖优先,否则走预算倒推。"""
    resolved = _settings(settings)
    if resolved.context_window_max_tokens > 0:
        return resolved.context_window_max_tokens
    return _compute_budget(resolved).sliding
