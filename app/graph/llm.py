"""ch05 · 图内模型工厂。

课程参考的 ``app.core.llm.get_chat_model`` 是 FastAPI 依赖(请求级缓存);
图节点拿不到 request,这里改为按 Settings 构建并做进程内缓存——
同一个 Settings 反复构建 ChatOpenAI 会反复建 httpx 客户端,缓存住。
测试直接给 GraphDeps.chat_model_factory 注入 Fake,不经过本函数。
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.settings import Settings


@lru_cache(maxsize=8)
def build_chat_model(settings: Settings) -> ChatOpenAI:
    """按 Settings 构建流式 ChatOpenAI(frozen Settings 可哈希,可作缓存键)。

    ch09:usage 统一收口——stream_usage=True 让流式调用也回传 token 用量,
    Langfuse 按意图统计才不漏账(意图/自评/摘要等非流式调用不受影响)。"""
    return ChatOpenAI(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        model=settings.chat_model,
        temperature=settings.chat_temperature,
        streaming=True,
        stream_usage=True,  # ch09:流式也回传 usage(等效 stream_options.include_usage)
        max_retries=1,
    )
