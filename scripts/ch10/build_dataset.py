"""ch10 数据集：分层划分 80/10/10 + 训练集增强（同义词替换/句式微调）。

增强只扩训练集——验证/测试是考题，不许照练习题变。
增强走批量改写（每批 10 条一次调用）控成本；变体撞上任何原句（含考题）就丢弃。
运行：python -m scripts.ch10.build_dataset（需上游可用）。
"""
import asyncio
import json
import pathlib

from pydantic import BaseModel, Field

from app.core.taxonomy import TOPIC_NAMES
from scripts.ch10.corpus_lib import split_dataset

SRC = pathlib.Path("data/ch10/corpus_labeled.jsonl")
OUT = pathlib.Path("data/ch10/dataset")
AUG_BATCH = 10

AUGMENT_PROMPT = """把下面 {n} 句电商客服用户问题各改写一个变体：换同义词、微调句式\
（比如改成「我想问一下……」的口气），不改原意、不增删诉求（多标签句改写后诉求也要原样保留）。\
按原顺序输出改写后的句子。

{sentences}"""


class _AugOut(BaseModel):
    variants: list[str] = Field(description="与输入顺序一一对应的改写句")


def _build_structured_model():
    from app.graph.llm import build_chat_model
    from scripts._common import load_settings

    return build_chat_model(load_settings()).with_structured_output(_AugOut)


async def augment(samples: list[dict], structured_model=None) -> list[dict]:
    """批量改写训练集，每批一次调用；单批失败丢弃该批（不挡流水线）。"""
    structured_model = structured_model or _build_structured_model()
    sem = asyncio.Semaphore(4)

    async def one_batch(batch: list[dict]) -> list[dict]:
        async with sem:
            numbered = "\n".join(f"{i+1}. {s['text']}" for i, s in enumerate(batch))
            try:
                r = await structured_model.ainvoke(
                    AUGMENT_PROMPT.format(n=len(batch), sentences=numbered))
                if isinstance(r, dict):
                    r = _AugOut(**r)
                if len(r.variants) == len(batch):
                    return [{"text": t.strip(), "labels": s["labels"], "origin": "augmented"}
                            for s, t in zip(batch, r.variants) if t.strip()]
            except Exception:
                pass
            return []

    outs: list[dict] = []
    batches = [samples[i:i + AUG_BATCH] for i in range(0, len(samples), AUG_BATCH)]
    results = await asyncio.gather(*[one_batch(b) for b in batches])
    for r in results:
        outs.extend(r)
    return outs


def _dump(path: pathlib.Path, samples: list[dict]) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"text": s["text"], "labels": s["labels"]}, ensure_ascii=False)
            for s in samples),
        encoding="utf-8")


def _dist(name: str, samples: list[dict]) -> None:
    counts = {n: 0 for n in TOPIC_NAMES}
    for s in samples:
        for lb in s["labels"]:
            counts[lb] += 1
    print(f"{name}（{len(samples)} 条）: " + " ".join(f"{k}={v}" for k, v in counts.items()))


async def main() -> int:
    samples = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    train, val, test = split_dataset(samples)
    aug = await augment(train)
    seen = {s["text"] for s in samples}          # 变体撞上任何原句（含考题）就丢弃
    train = train + [a for a in aug if a["text"] not in seen]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, ds in (("train", train), ("val", val), ("test", test)):
        _dump(OUT / f"{name}.jsonl", ds)
        _dist(name, ds)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
