"""ch10 预标：大模型照权威术语表给问题打多标签。折中路线的前半——预标，后半人工抽审。

多标签口径写死在 prompt（字面提到几个打几个，一个不多一个不少）；
单测 mock `_build_model`，真跑走仓库现有聊天上游（CHAT_*）。
"""
import asyncio

from pydantic import BaseModel, Field

from app.core.taxonomy import LABEL2ID, terminology_table


class _Labeled(BaseModel):
    labels: list[str] = Field(description="命中的类目名数组，取术语表类目名原文")


PRELABEL_PROMPT = """你是电商客服主题标注员。对照下面这份 17 类权威归并术语表，给用户问题打主题标签。

术语表（类目：边界说明（示例））：
{terminology}

标注铁律：
1. 多标签口径：一句话字面提到几个诉求就打几个标签，一个不多一个不少。「买大了想退」→ 尺码+退换货；\
「这鞋买大了一码」→ 只标尺码，不许因为「可能要退」就脑补退换货。
2. 近邻边界：修归保修维修、退归退换货；运费管钱、物流管货；价保是补差价、优惠活动是券和满减。
3. 方言、口语、错别字要看穿字面认诉求：「俺买的那玩意儿咋还没到俺这疙瘩」是物流；「退活」是「退货」的错字。
4. 都对不上就标「其他」；标签必须取术语表的类目名原文。

用户问题：{text}"""


def _build_model():
    """真跑的模型工厂（单测 monkeypatch 这里）。"""
    from app.graph.llm import build_chat_model
    from scripts._common import load_settings

    return build_chat_model(load_settings()).with_structured_output(_Labeled)


async def prelabel_one(text: str, model=None) -> list[str]:
    model = model or _build_model()
    try:
        r: _Labeled = await model.ainvoke(
            PRELABEL_PROMPT.format(terminology=terminology_table(), text=text))
    except Exception:
        return ["其他"]
    if isinstance(r, dict):
        r = _Labeled(**r)
    labels = [lb for lb in r.labels if lb in LABEL2ID]
    return labels or ["其他"]


async def prelabel_batch(texts: list[str], concurrency: int = 8) -> list[list[str]]:
    model = _build_model()
    sem = asyncio.Semaphore(concurrency)

    async def one(t: str) -> list[str]:
        async with sem:
            return await prelabel_one(t, model)

    return list(await asyncio.gather(*[one(t) for t in texts]))
