"""ch06 · 意图分类标注评估:16 条标注样例核对 classify 八类判对率 + 2 条怪问题。

课程 Plan Task 4 的标注清单,ch06 v2 口径:八类 + confidence(0~1)+ 多轮历史注入。
入口:app/graph/intent.classify(query, settings)(with_structured_output Literal 八类,
异常/越界回退「其他」)。需真实上游可用。
用法:
    .\\.venv\\Scripts\\python.exe -m scripts.eval_intent
"""
import asyncio

from app.graph.intent import classify
from app.settings import Settings

# (用户问法, 期望意图)——八类各 2 条
SAMPLES = [
    ("订单1001的快递到哪了", "物流"),
    ("我买的东西发货了吗", "物流"),
    ("订单2002现在什么状态", "订单"),
    ("我上周下的单多少钱来着", "订单"),
    ("这款猫粮多少钱一包", "商品咨询"),
    ("智能猫砂盆怎么用啊", "商品咨询"),
    ("我想退货", "退款退货"),
    ("退款一般几天到账", "退款退货"),
    ("我的猫爬架坏了能保修吗", "售后"),
    ("换货进度到哪了", "退款退货"),  # 提示词口径:换货诉求归退款退货类
    ("你们这什么破服务,我要投诉", "投诉"),
    ("太差了给我个说法", "投诉"),
    ("你好呀", "闲聊"),
    ("在吗", "闲聊"),
    ("帮我写一首关于猫的诗", "其他"),
    ("珍珠港事件是哪一年", "其他"),
]

# 多轮消歧:「它」指上一轮的订单,应判物流而非其他
HISTORY_CASE = {
    "history": "用户:订单1001多久能到\n客服:预计2天内送达",
    "query": "它怎么还没到啊",
    "expect": "物流",
}


async def main() -> None:
    settings = Settings.load()
    passed = 0
    for query, expect in SAMPLES:
        verdict = await classify(query, settings)
        got, confidence = verdict["intent"], verdict["confidence"]
        ok = got == expect
        passed += ok
        print(f"{'✅' if ok else '❌'} {query!r} -> {got}(conf={confidence:.2f}) 期望={expect}")

    verdict = await classify(HISTORY_CASE["query"], settings, history=HISTORY_CASE["history"])
    got, confidence = verdict["intent"], verdict["confidence"]
    ok = got == HISTORY_CASE["expect"]
    passed += ok
    print(f"{'✅' if ok else '❌'} [多轮] {HISTORY_CASE['query']!r} -> {got}(conf={confidence:.2f}) "
          f"期望={HISTORY_CASE['expect']}(历史:{HISTORY_CASE['history']})")

    total = len(SAMPLES) + 1
    print(f"\n判对 {passed}/{total}(上游非确定性,抖动如实重跑记录;"
          "「售后↔退款退货」边界容许个别抖动,明显偏差则调 INTENT_CLASSIFY_SYSTEM 重跑)")


if __name__ == "__main__":
    asyncio.run(main())
