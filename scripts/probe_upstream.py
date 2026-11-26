# -*- coding: utf-8 -*-
"""上游探针：验证候选免费模型是否支撑 ch05 全链路所需的三种能力。

用法：.venv\\Scripts\\python.exe -m scripts.probe_upstream
验证项：
  1. 普通对话（基础可用性）
  2. bind_tools 工具调用（ReAct 主力 Agent 依赖）
  3. with_structured_output 结构化输出（意图分类依赖）
密钥复用 .env 的 JUDGE_*（硅基流动），不改 .env。
"""
import asyncio
import sys

from pydantic import BaseModel, Field

from app.settings import Settings

BASE_URL = "https://api.siliconflow.cn/v1"
CANDIDATES = [
    "Qwen/Qwen2.5-14B-Instruct",
]

QUERY_TOOL = {
    "type": "function",
    "function": {
        "name": "query_order",
        "description": "按订单号查询订单信息",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "订单号"}},
            "required": ["order_id"],
        },
    },
}


class Intent(BaseModel):
    intent: str = Field(description="七类意图之一：物流/订单/商品咨询/退款退货/售后/投诉/闲聊")


async def probe(model: str, api_key: str) -> None:
    from langchain_openai import ChatOpenAI

    print(f"\n===== {model} =====")
    # 每个模型独立 Client,避免跨模型复用连接
    kwargs = dict(model=model, api_key=api_key, base_url=BASE_URL, timeout=30)

    # 1 普通对话
    try:
        llm = ChatOpenAI(**kwargs)
        r = await llm.ainvoke("回复两个字：好的")
        print(f"  chat         OK  {str(r.content)[:20]!r}")
    except Exception as e:
        print(f"  chat         FAIL {type(e).__name__}: {str(e)[:120]}")
        return

    # 2 bind_tools
    try:
        llm = ChatOpenAI(**kwargs, streaming=True)
        r = await llm.bind_tools([QUERY_TOOL]).ainvoke("订单1001的物流到哪了")
        calls = r.tool_calls if hasattr(r, "tool_calls") else []
        print(f"  bind_tools   {'OK ' if calls else 'NO-CALL'} calls={[(c.get('name'), c.get('args')) for c in calls]}")
    except Exception as e:
        print(f"  bind_tools   FAIL {type(e).__name__}: {str(e)[:120]}")

    # 3 with_structured_output
    try:
        llm = ChatOpenAI(**kwargs)
        r = await llm.with_structured_output(Intent).ainvoke("我想退货，怎么操作")
        print(f"  structured   OK  intent={r.intent!r}")
    except Exception as e:
        print(f"  structured   FAIL {type(e).__name__}: {str(e)[:120]}")


async def run_all(api_key: str) -> None:
    for model in CANDIDATES:
        await probe(model, api_key)


def main() -> None:
    settings = Settings.load()
    api_key = settings.judge_api_key
    if not api_key:
        print("缺 JUDGE_API_KEY（.env）")
        sys.exit(1)
    # 单一事件循环跑全部探测:多个 asyncio.run 会因连接池跨 loop 复用报 Event loop is closed
    asyncio.run(run_all(api_key))


if __name__ == "__main__":
    main()
