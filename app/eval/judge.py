"""ch04 · 评估裁判与生成提示词（判据全文落地：七类豁免 + 两类真编造 + 边界判例）。

裁判输出全部是扁平字段 + 标量字符串数组（ch03 教训：不用嵌套对象数组）。
`temperature=0` 由调用方显式传入（0.3 下重放会抖动，抖动会写进分数）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

RAG_ANSWER_SYSTEM = (
    "你是 MewHelp 电商平台客服。只依据给定的知识证据回答用户问题。\n"
    "- 证据足够：给出简洁、准确的回答，关键结论后标注来源角标 [n]"
    "（n 为证据编号，从 1 开始，多来源可并列如 [1][2]）。\n"
    "- 证据不足或与问题无关：只回复「暂时没有查到相关信息，您可以通过 App 内「我的」-「联系客服」"
    "咨询人工客服」，不得编造。\n"
    "- 禁止承诺具体时效/赔偿金额；平台规则类问题统一表述「以平台实际处理进度为准」。\n"
    "- 不要输出 markdown 代码块。"
)

# 忠实度 + 答案覆盖度合并裁判（一次调用判两件事，省一半上游调用）
JUDGE_SYSTEM = """你是 RAG 评估裁判。给定「用户问题」「标准答案要点」「检索证据」和「客服回答」，输出两个判断：
一、covered_points：标准答案要点中，客服回答实际覆盖了哪些（原样摘抄被覆盖的要点，逐条列入数组）。
二、faithful：回答中的事实性主张是否都能被证据支撑。

忠实度判数字的三问次序：
1. 把回答逐条拆成事实主张（数字、天数、金额、比例、次数、条件档次、路径、正反结论各算一条）。
2. 逐条在证据里找**直接支撑**：主张的主体、条件、数字必须与证据对应得上——证据里有相似规则但条件或主体不同，等于没有支撑。
3. 找不到直接支撑的主张，再看是否落在七类豁免之一；豁免只豁「措辞形式」（兜底话术、寒暄、常识注解），
   **不豁「无据的具体主张」**——凡主张了某个具体数值/条件/路径/正反结论而证据没有，哪怕表述含糊也不算豁免。

七类豁免（只豁措辞，不豁无据主张）：
1. 引用角标本身（[1] 之类编号不是事实主张）；
2. 规则兜底表述（「以平台实际处理进度为准」「以商品页公示为准」等）——仅当它兜的是证据未给出的维度，
   且句子没有主张「存在某个具体规则/数值」；
3. 与问题无关的礼貌寒暄、引导话术（「请问还有什么可以帮您」「可在 App 内联系人工客服」）；
4. 跨证据合并（把两条证据各自有据的事实合在一句里）；
5. 不确定语气的提醒（「可能」「建议您留意」）；
6. 无害安全提示（常识性提醒，如「注意用电安全」）；
7. 通用常识注解（不涉及平台事实的常识，如「工作日不含周末」）。

两类真编造（faithful=false）：
1. 凭空数值：证据中不存在的金额/天数/比例/次数/型号参数被写成确定事实；
2. 条件张冠李戴：把 A 条件（档次/型号/会员等级/时间窗）下成立的事实安到 B 条件上。

边界判例（few-shot）：
- 例1 证据「单笔订单满 99 元包邮」→ 回答「满 99 元包邮」：faithful=true（同义复述）。
- 例2 证据「退款 1-3 个工作日原路退回」→ 回答「退款 24 小时内到账」：faithful=false（凭空数值）。
- 例3 证据「金卡每月 3 张免运费券」→ 回答「普通会员每月 3 张免运费券」：faithful=false（条件张冠李戴）。
- 例4 证据未给时效 → 回答「具体到账时间以平台实际处理进度为准」：faithful=true（规则兜底）。
- 例4b 证据只有「100 积分抵 1 元」、未给单笔上限 → 回答「一笔订单最多抵扣的现金金额以平台实际处理进度为准」：
  faithful=true（规则兜底：兜的是证据未给出的维度，句子未主张存在具体上限数值）。
- 例5 证据与问题无关 → 回答「暂时没有查到相关信息…」：faithful=true（合理拒答）。
- 例6 证据A「收取 10 元运费」+证据B「偏远地区每单加收附加运费」→ 回答「偏远地区每单加收 10 元附加运费」：
  faithful=true（跨证据合并；若证据只有「收取 10 元运费」而无偏远表述，则 faithful=false）。
- 例7 证据「积分可在结算时抵扣,100 积分抵 1 元」→ 回答「积分不是用来兑换商品的」：
  faithful=false（凭空断言无据——无据否定与凭空数值同罪，且不在七类豁免内）。
- 例8 证据「电子发票将于 3 个工作日内发送至邮箱」→ 回答「开票申请的处理时限为 3 个工作日」：
  faithful=false（时限换说法：发送时限与申请处理时限是两个主张，主体换了即无据）。
- 例9 证据A「银卡会员退换货运费权益 1 次/月」+证据B「金卡每月 3 张免运费券,当月未用完不结转」→
  回答「银卡退换货运费权益当月未用完不结转」：faithful=false（条件张冠李戴：B 证据的规则安到 A 条件上，
  两条证据各自成立也救不了交叉主张）。

输出 JSON 对象（无 markdown、无多余字段）：
{"covered_points": ["要点原样", ...], "faithful": true/false, "fabricated_points": ["编造主张", ...], "reason": "一句话"}

说明：
- covered_points 只列回答确实覆盖的要点；覆盖不了就留空数组；
- fabricated_points 列编造主张（faithful=true 时为空数组）；
- 回答是合理拒答时 covered_points 为空数组、faithful=true、reason 说明是拒答。"""

JUDGE_HUMAN = "用户问题:{query}\n\n标准答案要点:\n{points}\n\n检索证据:\n{evidence}\n\n客服回答:\n{answer}"


def normalize(text: str | None) -> str:
    """去空白（与 metrics.normalize 同口径）。"""
    return "".join((text or "").split())


def _parse_json_blob(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip()).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return None
    return None


def judge(model_fn, query: str, expect_points: list[str], evidence_texts: list[str], answer: str) -> dict:
    """One combined coverage+faithfulness verdict; flat fields + scalar arrays.

    Returns {"covered_points": [...], "faithful": bool, "fabricated_points": [...], "reason": str}.
    Parse/调用失败返回 None（调用方记「调用失败」并跳过）。
    """
    points = "\n".join(f"- {point}" for point in expect_points) or "-（无要点）"
    evidence = "\n".join(f"[{index + 1}] {text}" for index, text in enumerate(evidence_texts)) or "(无证据)"
    human = (
        f"用户问题:{query}\n\n标准答案要点:\n{points}\n\n检索证据:\n{evidence}\n\n客服回答:\n{answer}"
    )
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = model_fn([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=human)], temperature=0.0)
    except Exception as exc:
        logger.warning("judge call failed (%s)", type(exc).__name__)
        return None
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    payload = _parse_json_blob(str(content))
    if payload is None:
        logger.warning("judge returned non-JSON; skipping this verdict")
        return None
    covered = payload.get("covered_points")
    fabricated = payload.get("fabricated_points")
    return {
        "covered_points": [str(item) for item in covered] if isinstance(covered, list) else [],
        "faithful": bool(payload.get("faithful")),
        "fabricated_points": [str(item) for item in fabricated] if isinstance(fabricated, list) else [],
        "reason": str(payload.get("reason") or ""),
    }


def answer_coverage(expect_points: list[str], covered_points: list[str]) -> float | None:
    """答案覆盖度：covered_points 与 expect_points（去空白精确/子串匹配）交集占比。"""
    points = [point for point in (expect_points or []) if point]
    if not points:
        return None
    normalized_covered = {normalize(point) for point in (covered_points or [])}
    hit = 0
    for point in points:
        normalized = normalize(point)
        if normalized in normalized_covered or any(normalized in item for item in normalized_covered):
            hit += 1
    return hit / len(points)


def _answer_with(model_fn, system_prompt: str, query: str, evidence_texts: list[str]) -> str:
    """用证据生成客服回答（带 [n] 角标规则）；证据为空则明确传「(无证据)」。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    evidence = "\n".join(f"[{index + 1}] {text}" for index, text in enumerate(evidence_texts)) or "(无证据)"
    response = model_fn(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"用户问题:{query}\n\n知识证据:\n{evidence}"),
        ],
        temperature=0.0,
    )
    content = getattr(response, "content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content).strip()
