"""ch07 · 摘要 prompt 三样例真模型验证(eval 脚本,唯一真跑上游的位置)。

样例:
  1. 事实保留 + 寒暄不留(订单号/手机尾号/诉求在,客套话不在);
  2. 无编造(摘要里的数字/专名必须都能在原文找到);
  3. 滚动合并(旧摘要只作背景,新段覆盖本轮对话,旧事实不依赖回炉合并)。

跑法:.env 的 CHAT_* 指向硅基 Qwen2.5-14B(代金券计费,只有这里和 smoke 真跑)。
    .\\.venv\\Scripts\\python.exe -m scripts.eval_ch07
"""

import asyncio
import re
import sys

from langchain_core.messages import AIMessage, HumanMessage

from app.core import summarizer
from app.settings import Settings


def _dialog(*pairs) -> list:
    messages = []
    for user, assistant in pairs:
        messages.append(HumanMessage(content=user))
        messages.append(AIMessage(content=assistant))
    return messages


SAMPLES = [
    {
        "name": "① 事实保留 + 寒暄不留",
        "old": "",
        "dialog": _dialog(
            ("你好,在吗?", "您好,在的,请问有什么可以帮您?"),
            (
                "我订单10086的快递三天了还没动,手机尾号6677,帮我催一下,再不动我就投诉了",
                "已为您登记催单,查询到订单10086物流滞后,我们会在24小时内给您答复",
            ),
            ("好的,麻烦了,谢谢", "不客气,有消息会第一时间通知您"),
        ),
        "must": ["10086", "6677", "投诉"],
        "must_not": ["在吗", "麻烦", "谢谢", "不客气"],
    },
    {
        "name": "② 无编造",
        "old": "",
        "dialog": _dialog(
            (
                "订单20015显示已发货,但我想改地址,收件人张三,麻烦改成李四",
                "订单20015当前状态为已发货,无法修改收货地址,建议拒收后申请退回",
            ),
            ("那退货的话运费谁出?", "若是质量问题退货,运费由平台承担;非质量问题需您自行承担"),
        ),
        "must": ["20015", "运费"],
        "must_not": ["李四改成功", "已修改地址"],
    },
    {
        "name": "③ 滚动合并:旧摘要只作背景",
        "old": "用户咨询订单10086物流滞后,已登记催单,客服承诺24小时内答复",
        "dialog": _dialog(
            (
                "昨天说订单10086会24小时内答复,现在有结果了吗?",
                "订单10086物流已恢复更新,预计明天送达,无需另行催单",
            ),
            ("行,那不用打你们投诉电话了吧", "是的,若明天仍未送达再联系我们即可"),
        ),
        "must": ["10086", "明天"],
        "must_not": ["已登记催单,客服承诺24小时内答复"],  # 不复述旧摘要
    },
]

_LENGTH = (20, 250)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d{3,}", text))


async def _run_sample(sample: dict, settings: Settings, model) -> list[str]:
    problems: list[str] = []
    summary = await summarizer.summarize_dialog(sample["old"], sample["dialog"], settings, model=model)
    print(f"--- {sample['name']}")
    print(f"    摘要({len(summary)}字):{summary}")
    if not (_LENGTH[0] <= len(summary) <= _LENGTH[1]):
        problems.append(f"长度 {len(summary)} 不在 {_LENGTH}")
    for token in sample["must"]:
        if token not in summary:
            problems.append(f"关键事实缺失:{token}")
    for token in sample["must_not"]:
        if token in summary:
            problems.append(f"不应出现:{token}")
    # 无编造:摘要里 3 位以上的数字必须全部能在对话原文找到
    dialog_text = " ".join(m.content for m in sample["dialog"])
    ghost = _numbers(summary) - _numbers(sample["old"]) - _numbers(dialog_text)
    if ghost:
        problems.append(f"编造数字:{sorted(ghost)}")
    return problems


async def main() -> int:
    settings = Settings.load()
    model = summarizer._build_model(settings)
    print(f"摘要模型:{settings.summary_model or settings.chat_model}")
    all_problems: list[str] = []
    for sample in SAMPLES:
        try:
            all_problems.extend(await _run_sample(sample, settings, model))
        except Exception as exc:  # 真跑脚本:逐样例留痕,继续跑下一个
            all_problems.append(f"{sample['name']} 执行失败:{type(exc).__name__}: {exc}")
    if all_problems:
        print("\nFAIL:")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1
    print("\nPASS: 3/3")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
