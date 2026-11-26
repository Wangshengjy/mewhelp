"""Pre-generation self-check gate: does the recalled evidence answer the query?

Structured output with flat fields ``{"useful": bool, "reason": str}`` — never a
list of objects (the ch03 glm-502 lesson). ``useful=false`` means the tool
refuses to answer and the question falls into low_confidence_questions.
Call errors fail open (treat as useful) so the online path stays available.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.settings import Settings

logger = logging.getLogger(__name__)

SELF_CHECK_SYSTEM = (
    "你是检索质量评审员。给定用户问题和检索到的知识证据,判断这些证据是否足以准确回答该问题。\n"
    "- useful=true:证据包含回答该问题所需的关键信息。\n"
    "- useful=false:证据与问题无关、或缺少关键信息、或只能部分回答核心诉求。\n"
    "- reason:一句话说明判断依据。\n"
    "严格只看证据是否够答,不要脑补证据外的知识。\n"
    '严格只输出 JSON 对象: {"useful": true/false, "reason": "一句话"},不要输出 markdown 代码块或其它字段。'
)


def _parse_json_blob(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()).strip()
    import json

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


def check_sufficient(model_fn, query: str, evidence_texts: Sequence[str], temperature: float = 0.0) -> dict:
    """Judge evidence sufficiency; flat ``useful``/``reason`` output.

    ``model_fn(messages, temperature) -> response`` is injected for tests. Any
    failure returns ``useful=True`` with a note (fail-open, logged).
    """
    evidence = "\n".join(f"[{index + 1}] {text}" for index, text in enumerate(evidence_texts)) or "(无证据)"
    try:
        response = model_fn(
            [
                SystemMessage(content=SELF_CHECK_SYSTEM),
                HumanMessage(content=f"用户问题:{query}\n\n检索证据:\n{evidence}"),
            ],
            temperature=temperature,
        )
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        payload = _parse_json_blob(str(content))
        if payload is None:
            logger.warning("self-check returned non-JSON; fail-open")
            return {"useful": True, "reason": ""}
        useful = bool(payload.get("useful"))
        reason = str(payload.get("reason") or "")
        return {"useful": useful, "reason": reason}
    except Exception as exc:
        logger.warning("self-check failed (%s); fail-open", type(exc).__name__)
        return {"useful": True, "reason": f"自评调用失败:{type(exc).__name__}"}


def default_model_fn(settings: Settings):
    """Build the default CHAT-model callable used by the online pipeline."""

    def _invoke(messages: list, temperature: float = 0.0) -> Any:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            base_url=settings.chat_base_url,
            api_key=settings.chat_api_key,
            model=settings.chat_model,
            temperature=temperature,
        )
        return chat.invoke(messages)

    return _invoke
