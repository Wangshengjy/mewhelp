"""ch06 · 指代消解:把「这个能退吗」改写成「蓝牙耳机还能申请退货吗」。

规则先行,LLM 兜底:
- 规则短路:问句里已有订单号(`\d{4,}`),或历史里没有可指代的上下文时,
  原样返回——多数独立问句不值得花一次模型调用;
- LLM 改写:单次调用,输出即改写问句;输出为空/异常时回退原 query
  (宁可不改写,不可改丢)。resolve 返回原串时节点记 passthrough。
"""
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.prompts import COREF_REWRITE_SYSTEM
from app.settings import Settings

logger = logging.getLogger(__name__)

# 演示订单号 4 位起;命中即认为问句自带锚点,无需改写
_ORDER_ID_RE = re.compile(r"(?<!\d)\d{4,}(?!\d)")
# 常见指代词(中文):命中且历史有上文才走 LLM
_COREF_HINT_RE = re.compile(r"(这个|那个|它|这单|那单|该单|此单|刚才|上面|之前)")


def build_coref_model(settings: Settings) -> ChatOpenAI:
    """改写模型:与意图分类同一思路——temperature 0、不流式、独立构造。"""
    return ChatOpenAI(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        model=settings.intent_model or settings.chat_model,
        temperature=0.0,
        streaming=False,
        max_retries=1,
    )


def _needs_rewrite(query: str, history: str) -> bool:
    """规则短路:有订单号锚点,或无历史/无指代词,都不值得花调用。"""
    if _ORDER_ID_RE.search(query):
        return False
    if not history:
        return False
    return bool(_COREF_HINT_RE.search(query))


async def resolve(
    query: str,
    history: str,
    settings: Settings,
    chat_model_factory=None,
) -> str:
    """返回改写后的独立问句;无指代/失败时原样返回 query。"""
    if not _needs_rewrite(query, history):
        return query
    try:
        if chat_model_factory is not None:
            model = chat_model_factory(settings)
        else:
            model = build_coref_model(settings)
        rewritten = await model.ainvoke(
            [
                SystemMessage(content=COREF_REWRITE_SYSTEM),
                HumanMessage(content=f"最近对话:\n{history}\n\n用户最后一句话:{query}"),
            ]
        )
        text = (rewritten.content or "").strip() if isinstance(rewritten.content, str) else ""
        if not text or len(text) > len(query) * 4 + 50:
            # 空串/明显跑偏的超长输出都不要:回退原文
            return query
        return text
    except Exception:
        logger.exception("coref 改写失败,回退原问句")
        return query
