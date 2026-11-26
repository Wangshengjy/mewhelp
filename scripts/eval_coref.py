"""ch06 · 指代消解评估:3 条问句真跑 coref 改写率。

课程 Plan Task 5 的评估问句,适配本仓库入口 app/graph/coref.resolve(query, history, settings)
(规则短路 + LLM 改写,失败回退原句)。需真实上游可用。
用法:
    .\\.venv\\Scripts\\python.exe -m scripts.eval_coref
"""
import asyncio

from app.graph.coref import resolve
from app.settings import Settings

# (历史, 当前问句, 期望含关键词)
CASES = [
    ("用户:蓝牙耳机多少钱\n客服:199元,音质口碑不错", "这个能退吗", "蓝牙耳机"),
    ("用户:订单1001什么时候发货\n客服:48小时内发出", "那它现在到哪了", "1001"),
    ("", "发票怎么开", "发票"),  # 无历史无指代:应原样返回(passthrough)
]


async def main() -> None:
    settings = Settings.load()
    rewritten = 0
    for history, query, keyword in CASES:
        resolved = await resolve(query, history, settings)
        mode = "rewrite" if resolved != query else "passthrough"
        ok = keyword in resolved
        rewritten += mode == "rewrite"
        print(f"{'✅' if ok else '❌'} [{mode}] {query!r} -> {resolved!r}")
    print(f"\n改写 {rewritten}/{len(CASES)}(第3条设计上就是 passthrough;"
          "改写失败会回退原句——宁可不改写,不可改丢)")


if __name__ == "__main__":
    asyncio.run(main())
