r"""ch07 · 冒烟(真跑上游模型 + dev MySQL 落库,唯一真跑位置之一):

  演示配置 25 轮 → 层1降级 + summary trigger/done + 「最开始那个订单」靠摘要/半压答对
                + conversations 三字段落库 + conversation_summaries 有段;
  默认配置 20 轮 → 全程不降级不摘要,三字段保持 NULL。

跑法:.\.venv\Scripts\python.exe -m scripts.smoke_ch07
演示配置环境变量(脚本内自动设置):MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000
    MAX_USER_INPUT_TOKENS=2000 MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5
"""

import asyncio
import logging
import os
import sys
import time

from app.graph import runtime
from app.settings import Settings

# ── 演示配置(压窗触发分层降级与摘要) ──────────────────────────────────────────
DEMO_ENV = {
    "MODEL_CONTEXT_WINDOW": "18000",
    "MAX_OUTPUT_TOKENS": "2000",
    "MAX_USER_INPUT_TOKENS": "2000",
    "MAX_AGENT_STEPS": "3",
    "TOOL_RESULT_MAX_TOKENS": "1200",
    "RERANK_TOP_K": "5",
}
CH07_ENV_KEYS = list(DEMO_ENV)

ORDER_QUESTION = "订单 1001 的物流到哪了?"
# 引导语让模型直接用对话历史(摘要行/半压层 2)作答,而不是再调工具走兜底
FINAL_QUESTION = "请按我们前面聊过的内容回答,不用查询新信息:最开始那个订单,后来怎么样了?"

# 演示遍中间轮全用「帮我查一下订单 XXX」长句式物流查询:
# 1)每次 query_order+query_logistics 两段 demo JSON,层 1 每轮都超预算、边界持续推进;
# 2)user 原文在层 2 半压渲染里原样保留,问题必须够长(130+ 字),层 2 半压用量才能
#   稳超 summary 触发预算(1695)——短问题会卡在 1500 左右,模型步数波动一截断就不触发;
# 3)句式统一避开「环节/退款/退货」等词(意图分类会把「环节」误判成退款流程造成 interrupt);
# 4)mock_data 是确定性 RNG,任意单号都有完整 demo 数据,1001 反复出现与「最开始那个订单」语义一致
_LOGISTICS_TAIL = (
    "。另外帮我确认一下这个包裹有没有派送异常或者卡件的记录,如果卡在某个站点超过两天,"
    "帮我标记一下,我周末可能不在家,需要的话想改约派送时间,也想顺便问下超时赔付的规则是怎么算的,辛苦客服了。"
)
LOGISTICS_TOPICS = [
    "帮我查一下订单 1001 的最新物流状态,昨天显示运输中今天还没更新,想看看包裹现在到哪了" + _LOGISTICS_TAIL,
    "再帮我看看订单 2002 的物流,昨天显示发货了,今天想确认下包裹位置和预计送达时间" + _LOGISTICS_TAIL,
    "订单 1001 的包裹现在到哪里了?物流有点久我有点着急,帮我查一下最新轨迹" + _LOGISTICS_TAIL,
    "麻烦查一下订单 3003 的配送状态,我想知道现在到哪个城市了,大概什么时候能送到" + _LOGISTICS_TAIL,
    "帮我查一下订单 1001 的物流轨迹,想确认包裹是不是已经在派送了,昨天说今天到但没动静" + _LOGISTICS_TAIL,
    "订单 4004 现在的物流怎么样了?之前显示运输中,想看看有没有更新" + _LOGISTICS_TAIL,
    "再查一下订单 1001 的包裹位置,麻烦把最新的物流轨迹发给我看看" + _LOGISTICS_TAIL,
    "帮我查一下订单 5005 的物流信息,想确认包裹是否正常运输,预计还要多久送到" + _LOGISTICS_TAIL,
    "订单 1001 的快递现在到哪一站了?帮我查查最新轨迹,我想安排时间收货" + _LOGISTICS_TAIL,
    "麻烦帮我查一下订单 2002 的包裹状态,想看看是不是已经开始派送了" + _LOGISTICS_TAIL,
    "帮我查一下订单 1001 还有多久送到,物流轨迹帮我拉一下最新的" + _LOGISTICS_TAIL,
    "订单 6006 的物流帮我查一下,想确认包裹有没有卡件,现在到哪个城市了" + _LOGISTICS_TAIL,
    "再帮我看看订单 1001 的配送进度,昨天查还在运输中,今天想确认下最新位置" + _LOGISTICS_TAIL,
    "帮我查一下订单 7007 的物流,包裹显示发货好几天了,想看看现在到哪里了" + _LOGISTICS_TAIL,
    "订单 1001 的包裹帮我跟踪一下,想确认是否已经在派送途中" + _LOGISTICS_TAIL,
    "麻烦查一下订单 8008 的物流状态,想看看包裹有没有异常,大概什么时候能到" + _LOGISTICS_TAIL,
    "帮我查一下订单 1001 的最新位置,物流轨迹麻烦发一下,想预估下送达时间" + _LOGISTICS_TAIL,
    "再查一下订单 2002 的包裹到哪里了,想确认配送是否正常" + _LOGISTICS_TAIL,
    "订单 1001 的物流有更新吗?帮我查一下最新轨迹,想看看是不是快到了" + _LOGISTICS_TAIL,
    "帮我查一下订单 9009 的配送进度,想确认包裹现在到哪里了" + _LOGISTICS_TAIL,
    "麻烦帮我看看订单 1001 的快递轨迹,想确认包裹是否已经在派送" + _LOGISTICS_TAIL,
    "帮我查一下订单 2002 的最新物流,想看看包裹位置和预计送达时间" + _LOGISTICS_TAIL,
    "订单 1001 的物流状态帮我刷新一下,想看最新轨迹,麻烦客服查一下" + _LOGISTICS_TAIL,
]
FAQ_TOPICS = [
    "发票如何开具?支持电子发票吗?",
    "发货时效多久?下单后几天发出?",
    "你们发什么快递?一般几天能到?",
    "支持哪些支付方式?",
    "会员等级和积分规则是怎样的?",
    "价格保护怎么申请?",
    "保修说明是怎样的?保修多久?",
]
CHITCHAT_TOPICS = [
    "你好呀,你们客服是真人还是机器人?",
    "好的,我了解一下。",
    "谢谢!",
    "嗯嗯,明白了。",
    "你们周末也上班吗?",
]


class _Collector(logging.Handler):
    """把 INFO 以上日志收进内存,断言「层1 降级」「summary trigger/done」用。"""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(record.getMessage())
        except Exception:  # noqa: BLE001 - 留痕失败不影响主流程
            pass


def _set_env(mode: str) -> None:
    for key in CH07_ENV_KEYS:
        os.environ.pop(key, None)
    if mode == "demo":
        os.environ.update(DEMO_ENV)


def _answer_of(state: dict) -> str:
    text = (state.get("answer") or "").strip()
    if text:
        return text
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            content = message.content if isinstance(message.content, str) else ""
            if content.strip():
                return content.strip()
    return ""


def _questions(mode: str, turns: int) -> list[str]:
    if mode == "demo":
        # 首轮钉死订单 1001(最终问题靠它验证摘要),中间全物流长问题压总量,末轮总结
        middle = [LOGISTICS_TOPICS[i % len(LOGISTICS_TOPICS)] for i in range(turns - 2)]
        return [ORDER_QUESTION, *middle, FINAL_QUESTION]
    # 默认遍:7 FAQ + 13 闲聊,总上下文远低于层 1 预算,不应降级
    questions = []
    faq_i = chitchat_i = 0
    for i in range(turns):
        want_faq = i < 7
        if want_faq:
            questions.append(FAQ_TOPICS[faq_i % len(FAQ_TOPICS)])
            faq_i += 1
        else:
            questions.append(CHITCHAT_TOPICS[chitchat_i % len(CHITCHAT_TOPICS)])
            chitchat_i += 1
    return questions


async def _run_pass(label: str, mode: str, turns: int) -> tuple[str, list[str], list[str]]:
    """跑完一整段会话,返回 (cid, 回答列表, 日志行)。"""
    collector = _Collector()
    root = logging.getLogger()
    root.addHandler(collector)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    try:
        _set_env(mode)
        await runtime.close_graph()
        await runtime.init_graph(settings=Settings.load())
        cid = None
        answers: list[str] = []
        questions = _questions(mode, turns)
        for index, question in enumerate(questions, 1):
            result = await runtime.run_turn(cid, question)
            cid = result["conversation_id"]
            if result["interrupt"]:
                raise AssertionError(f"[{label}] 第{index}轮意外进入 interrupt: {result['interrupt']}")
            answer = _answer_of(result["state"])
            answers.append(answer)
            print(f"[{label}] 轮 {index}/{len(questions)}:{question} → {answer[:36]}…")
            await asyncio.sleep(0.2)
        return cid, answers, collector.records
    finally:
        root.removeHandler(collector)


async def _wait_summary_done(cid: str, records: list[str], timeout: float = 240.0) -> bool:
    """后台摘要任务不阻塞轮次;等它落库完成再问最终问题。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any("summary done conv=" in line and cid in line for line in records):
            return True
        await asyncio.sleep(2)
    return False


def _db_snapshot(conversation_id: str) -> dict:
    import pymysql

    settings = Settings.load()
    connection = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT summary, summary_upto_msg_id, layer1_from_msg_id "
                "FROM conversations WHERE id=%s",
                (conversation_id,),
            )
            row = cursor.fetchone()
            assert row is not None, f"会话 {conversation_id} 不在 dev 库"
            cursor.execute(
                "SELECT COUNT(*), COALESCE(MAX(seq), 0) FROM conversation_summaries WHERE conversation_id=%s",
                (conversation_id,),
            )
            segment_row = cursor.fetchone()
        return {
            "summary": row[0] or "",
            "summary_upto": row[1],
            "layer1_from": row[2],
            "segment_count": int(segment_row[0]),
            "max_seq": int(segment_row[1]),
        }
    finally:
        connection.close()


def _check(condition: bool, message: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}: {message}")
    if not condition:
        raise AssertionError(message)


def _verify_demo(cid: str, answers: list[str], records: list[str]) -> None:
    print(f"[demo] 断言会话 {cid}")
    _check(any("层1 降级 conv=" in line and cid in line for line in records), "日志出现「层1 降级」")
    _check(any("summary trigger conv=" in line and cid in line for line in records), "日志出现「summary trigger」")
    _check(any("summary done conv=" in line and cid in line for line in records), "日志出现「summary done」")
    final_answer = answers[-1]
    _check("1001" in final_answer, f"最终问题「最开始那个订单」回答含 1001(实际:{final_answer[:60]}…)")
    snapshot = _db_snapshot(cid)
    print(f"  DB: summary {len(snapshot['summary'])} 字, upto={snapshot['summary_upto']}, "
          f"layer1_from={snapshot['layer1_from']}, 摘要段数={snapshot['segment_count']}(max_seq={snapshot['max_seq']})")
    _check(bool(snapshot["summary"].strip()), "conversations.summary 已落库")
    _check(snapshot["summary_upto"] is not None, "conversations.summary_upto_msg_id 已落库")
    _check(snapshot["layer1_from"] is not None, "conversations.layer1_from_msg_id 已落库")
    _check(snapshot["segment_count"] >= 1, "conversation_summaries 至少一段")


def _verify_default(cid: str, records: list[str]) -> None:
    print(f"[default] 断言会话 {cid}")
    _check(
        not any("层1 降级 conv=" in line and cid in line for line in records),
        "日志无「层1 降级」",
    )
    _check(
        not any("summary trigger conv=" in line and cid in line for line in records),
        "日志无「summary trigger」",
    )
    snapshot = _db_snapshot(cid)
    print(f"  DB: summary {len(snapshot['summary'])} 字, upto={snapshot['summary_upto']}, "
          f"layer1_from={snapshot['layer1_from']}, 摘要段数={snapshot['segment_count']}")
    _check(snapshot["summary_upto"] is None, "summary_upto_msg_id 保持 NULL")
    _check(snapshot["layer1_from"] is None, "layer1_from_msg_id 保持 NULL")
    _check(snapshot["segment_count"] == 0, "conversation_summaries 无段")


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    print("=== 第一遍:演示配置 25 轮(压窗,应触发层1降级 + 分段摘要) ===")
    demo_cid, demo_answers, demo_records = await _run_pass("demo", "demo", 25)
    done = await _wait_summary_done(demo_cid, demo_records)
    if not done:
        print("  警告:等待 summary done 超时,继续断言(预期失败)")
    _verify_demo(demo_cid, demo_answers, demo_records)

    print("\n=== 第二遍:默认配置 20 轮(不应触发降级与摘要) ===")
    default_cid, _, default_records = await _run_pass("default", "default", 20)
    _verify_default(default_cid, default_records)

    print("\nSMOKE PASS:两遍冒烟全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
