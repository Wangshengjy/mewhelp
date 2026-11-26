"""ch09 · Langfuse 链路冒烟:真实对话几轮(意图/闸/Agent),产生带 intent tag 的 trace。"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import observability  # noqa: E402
from app.graph import runtime  # noqa: E402

QUESTIONS = [
    "退货政策是什么样的",           # 商品咨询 → knowledge
    "帮查一下订单 MH20260101 的物流",  # 物流 → business/Agent
    "你们家猫粮适不适合幼猫吃呀",     # 商品咨询 → knowledge
]


async def main() -> int:
    await runtime.init_graph()
    print("langfuse_enabled:", observability.langfuse_enabled())
    for q in QUESTIONS:
        out = await runtime.run_turn(None, q)
        st = out["state"]
        print(f"  [{st.get('intent')}] {q[:20]} → route={st.get('route')} answer={len(st.get('answer') or '')}字")
    await runtime.close_graph()
    # OTel BatchSpanProcessor 有延迟,进程退出前显式 flush,保证 traces/observations 全量入库
    client = observability.get_langfuse()
    if client is not None:
        client.flush()
    print("traces 已上报并 flush")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
