"""ch06 · 八类意图分类器:with_structured_output + 置信度 + 多轮历史注入。

要点:
- structured output 的 Literal 八类是硬约束,模型吐越界标签时 pydantic 直接报错;
- 任何异常(上游 4xx/5xx/超时)或越界一律回退「其他」(confidence=0)——「其他」走
  business 出口交给主力 Agent 应对,分类器挂了也不把用户炸进固定脚本话术;
- history 由节点拼好注入:多轮对话里的「这个/它」需要上文才能判准;
- 模型可独立于对话模型:显式按 settings.intent_model 构造(空则回落 chat_model),
  不走 build_chat_model 的 lru_cache——两处模型名可能不同,缓存键会串。
"""
import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.prompts import INTENT_CLASSIFY_SYSTEM
from app.settings import Settings

logger = logging.getLogger(__name__)

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他")

IntentLabelType = Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊", "其他"]


class IntentLabel(BaseModel):
    """structured output 的目标 schema:intent 硬约束 + 自报置信度。"""

    intent: IntentLabelType
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="判断把握,0~1")


def build_intent_model(settings: Settings) -> ChatOpenAI:
    """意图模型显式构造:temperature 0(分类求稳)、不流式、快速失败。"""
    return ChatOpenAI(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        model=settings.intent_model or settings.chat_model,
        temperature=0.0,
        streaming=False,
        max_retries=1,
    )


def _classify_prompt(query: str, history: str) -> list:
    content = f"用户这句话:{query}" if not history else f"最近对话:\n{history}\n\n用户最后一句话:{query}"
    return [
        SystemMessage(content=INTENT_CLASSIFY_SYSTEM),
        HumanMessage(content=content),
    ]


async def classify(
    query: str,
    settings: Settings,
    chat_model_factory=None,
    history: str = "",
) -> dict:
    """单标签八类分类,返回 {intent, confidence};异常/越界回退 {其他, 0.0}。"""
    try:
        if chat_model_factory is not None:
            model = chat_model_factory(settings)
        else:
            model = build_intent_model(settings)
        structured = model.with_structured_output(IntentLabel, method="function_calling")
        label = await structured.ainvoke(_classify_prompt(query, history))
        intent, confidence = label.intent, float(label.confidence)
    except Exception:
        # 上游挂了也要给用户一条路:「其他」→ business,主力 Agent 兜着
        logger.exception("意图分类失败,回退其他")
        return {"intent": "其他", "confidence": 0.0}
    if intent not in INTENTS:
        return {"intent": "其他", "confidence": 0.0}
    return {"intent": intent, "confidence": max(0.0, min(1.0, confidence))}
