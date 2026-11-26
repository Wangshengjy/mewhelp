"""Query understanding before retrieval: colloquial rewrite + synonym expansion.

Uses the CHAT model with structured output. Returns flat fields
``{"standard": str, "expanded": list[str]}`` — a scalar string list, never a
list of objects (the ch03 glm-502 lesson). ``expanded`` is only concatenated
into the BM25 search text; it never touches the dense query or the stored rows.
Any failure degrades to the original query (fail-open, availability first).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.settings import Settings

logger = logging.getLogger(__name__)

QUERY_REWRITE_SYSTEM = (
    "你是电商客服检索前的 Query 归一化器。把用户口语、模糊、带情绪的问法改写成简洁标准的问法,"
    "并给出同义词/近义扩展词(用于关键词召回)。\n"
    "- standard:一句话标准问法,去口语和情绪,保留关键实体(型号、品类、政策词)。\n"
    "- expanded:3-6 个与问题相关的同义词/近义词/别称(如「邮费↔运费」「多久到↔时效」),只列词,不含原词。\n"
    "- 不臆造原问题没有的实体或型号。\n"
    "严格只输出 JSON 对象: {\"standard\": \"...\", \"expanded\": [\"词1\", \"词2\"]},"
    "不要输出 markdown 代码块或其它字段。"
)


class RewriteOutput(BaseModel):
    standard: str = Field(default="", description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


def _parse_json_blob(text: str) -> dict | None:
    """Tolerant JSON parse: strip code fences, find the first {...} block."""
    import json
    import re

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return None
    return None


def understand(model_fn, query: str, temperature: float = 0.0) -> dict:
    """Rewrite one query; ``model_fn(messages) -> content`` is injected for tests.

    Returns flat ``{"standard": str, "expanded": list[str]}``; falls back to the
    original query with an empty expansion when the model call or parse fails.
    """
    fallback = {"standard": query, "expanded": []}
    if not query or not query.strip():
        return fallback
    try:
        response = model_fn(
            [
                SystemMessage(content=QUERY_REWRITE_SYSTEM),
                HumanMessage(content=f"用户问法:{query}"),
            ],
            temperature=temperature,
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        payload: dict | None = _parse_json_blob(str(content))
        if payload is None:
            logger.warning("query understanding returned non-JSON; using original query")
            return fallback
        try:
            parsed = RewriteOutput.model_validate(payload)
        except Exception:
            parsed = RewriteOutput()
            parsed.standard = str(payload.get("standard") or "")
            expanded_raw = payload.get("expanded")
            parsed.expanded = [str(word) for word in expanded_raw] if isinstance(expanded_raw, list) else []
        standard = parsed.standard.strip() or query
        expanded = [word.strip() for word in parsed.expanded if str(word).strip() and str(word).strip() != query]
        return {"standard": standard, "expanded": expanded}
    except Exception as exc:
        logger.warning("query understanding failed (%s); using original query", type(exc).__name__)
        return fallback


def default_model_fn(settings: Settings, temperature: float = 0.0):
    """Build the default CHAT-model callable used by the online pipeline.

    提供商解析：JUDGE_*（评估专用通道）优先，未配置时回退 CHAT_*。
    """

    def _invoke(messages: list, temperature: float = 0.0) -> Any:
        from langchain_openai import ChatOpenAI

        base_url = settings.judge_base_url or settings.chat_base_url
        api_key = settings.judge_api_key or settings.chat_api_key
        model = settings.judge_model if settings.judge_base_url else settings.chat_model
        chat = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        return chat.invoke(messages)

    return _invoke


# ── ch06 · 多路检索前的查询扩展(一条问句 → 严格 3 条检索查询) ────────────────────

EXPAND_QUERIES_SYSTEM = (
    "你是电商知识库的检索查询改写器。把用户的一句话改写成 3 条彼此不同、各自独立可搜的查询,"
    "用于多路召回同一段退款/售后政策。\n"
    "要求:\n"
    "- 只改写,不扩写:每条查询都是对原句的检索友好表达(关键词化、补全口语缩略、同义术语),"
    "不得引入原句没有的新条件或新问题;\n"
    "- 3 条从不同角度覆盖同一诉求(如政策名、适用条件、流程步骤);\n"
    "- 每条意思完整,长度不超过 20 字。"
)


class ExpandedQueries(BaseModel):
    """structured output 目标 schema:扁平字符串数组(ch03 glm-502 教训:不嵌对象)。"""

    queries: list[str] = Field(default_factory=list, description="严格 3 条检索查询")


async def expand_queries(
    query: str,
    settings: Settings | None = None,
    chat_model_factory=None,
) -> list[str]:
    """一条问句 → 最多 3 条去重后的检索查询;全失败回退 [query](检索照跑)。

    与 understand() 的差异:understand 服务单路 BM25 的拼接扩展,本函数为
    retrieve_policy 的多路召回服务——每条查询独立走一次检索。
    """
    if settings is None:
        settings = Settings.load()
    fallback = [query] if query.strip() else []
    if not fallback:
        return []
    try:
        if chat_model_factory is not None:
            model = chat_model_factory(settings)
        else:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                base_url=settings.chat_base_url,
                api_key=settings.chat_api_key,
                model=settings.chat_model,
                temperature=0.0,
                streaming=False,
                max_retries=1,
            )
        structured = model.with_structured_output(ExpandedQueries, method="function_calling")
        result = await structured.ainvoke(
            [
                SystemMessage(content=EXPAND_QUERIES_SYSTEM),
                HumanMessage(content=f"用户问句:{query}"),
            ]
        )
        cleaned: list[str] = []
        for item in result.queries:
            text = (item or "").strip()
            if text and text not in cleaned and text != query:
                cleaned.append(text)
        return (cleaned[:3] or [query])
    except Exception as exc:
        logger.warning("expand_queries failed (%s); fallback to original query", type(exc).__name__)
        return fallback
