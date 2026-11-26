"""ch06 · 查询扩展评估:3 条问句真跑 expand_queries,验收「3 条独立可搜」。

课程 Plan Task 6 的评估,适配本仓库入口
app/kb/query_understanding.expand_queries(query, settings)
(with_structured_output 扁平 list[str],失败回退 [query])。需真实上游可用。
用法:
    .\\.venv\\Scripts\\python.exe -m scripts.eval_expand
"""
import asyncio

from app.kb.query_understanding import expand_queries
from app.settings import Settings

QUERIES = [
    "蓝牙耳机能退吗",
    "订单1001想退款",
    "7天无理由包含运费吗",
]


async def main() -> None:
    settings = Settings.load()
    all_ok = True
    for query in QUERIES:
        expanded = await expand_queries(query, settings)
        ok = len(expanded) == 3 and all(expanded) and len(set(expanded)) == 3
        all_ok &= ok
        print(f"{'✅' if ok else '❌'} {query!r} -> {expanded}")
    print("\n验收口径:每条恰好 3 条、彼此不同、不引入原句没有的新条件;"
          "模型抖动时个别条目合并属可接受,连续为空则查上游。")


if __name__ == "__main__":
    asyncio.run(main())
