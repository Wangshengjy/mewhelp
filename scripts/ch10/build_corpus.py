"""ch10 语料流水线：捞池 → 脱敏去重 → LLM 修错别字 → LLM 预标 → 模拟补足 → 导出人工抽审。

运行：python -m scripts.ch10.build_corpus [--target-per-class 15]（需 mysql + 上游可用）。
语料 = 池里真实低置信度问题 + 大模型照术语表造模拟问题补足（默认 17 类 × 15 条封顶控成本）；
多标签口径与刁钻样本比例（方言约 15%、错别字约 10%、多诉求约 15%）写死在造数 prompt。
产物落 data/ch10/：corpus_raw.jsonl / corpus_clean.jsonl / corpus_labeled.jsonl / sample_review.md。
"""
import argparse
import asyncio
import json
import pathlib
import random
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.core.taxonomy import LABEL2ID, TOPIC_CLASSES, terminology_table
from scripts._common import load_settings, session_factory_from
from scripts.ch10.corpus_lib import dedupe, desensitize

OUT = pathlib.Path("data/ch10")
TARGET_PER_CLASS = 15   # 用户口径：17 类 × 约 15 条封顶，真实 Qwen 短句控成本
SIM_BATCH = 15          # 单次造数条数，小批控质量
CLEAN_BATCH = 10        # 清洗批量

CLEAN_PROMPT = """修正下面 {n} 句电商客服用户问题里的错别字和乱格式：不改语义、不改口语风格、\
不增删诉求，没有错误就原样返回。按原顺序输出修正后的句子。

{sentences}"""


class _CleanOut(BaseModel):
    fixed: list[str] = Field(description="与输入顺序一一对应的修正后句子")


class _SimItem(BaseModel):
    text: str
    labels: list[str]


class _SimBatch(BaseModel):
    items: list[_SimItem]


SIMULATE_PROMPT = """你是电商客服语料造数员。为猫用品电商「喵喵优选」（卖猫粮、冻干、猫零食、猫砂盆、\
猫抓板、猫窝、猫爬架、猫碗、逗猫棒、项圈等）生成 {n} 条模拟用户问题，全部命中主题类目「{name}」。

17 类权威术语表：
{terminology}

要求：
1. 每条一句独立、语义完整的口语化用户问题，长短语气多样，贴近真实客服提问，彼此不重复。
2. 大约 15% 用方言土话（如「俺买的那玩意儿咋还没到俺这疙瘩」），10% 故意带错别字（如「退活」=退货），\
15% 是多诉求句——除「{name}」外再字面提到一个其他类目的诉求，labels 把两个类都打上。
3. 其余条目只含「{name}」一个诉求，labels 只有它。
4. 标签铁律：多标签口径——字面提到几个诉求打几个标签，一个不多一个不少；标签取术语表类目名原文。
5. 每条不超过 25 个字（短句控成本）。"""


def _dump(path: pathlib.Path, samples: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in samples), encoding="utf-8")


def _build_models():
    """真跑的模型工厂（清洗用裸模型，造数用结构化输出）。"""
    from app.graph.llm import build_chat_model

    settings = load_settings()
    model = build_chat_model(settings)
    return model, model.with_structured_output(_SimBatch), model.with_structured_output(_CleanOut)


async def clean_texts(raw_model, structured, texts: list[str]) -> list[str]:
    sem = asyncio.Semaphore(4)

    async def one_batch(batch: list[str]) -> list[str]:
        async with sem:
            numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(batch))
            try:
                r = await structured.ainvoke(CLEAN_PROMPT.format(n=len(batch), sentences=numbered))
                if isinstance(r, dict):
                    r = _CleanOut(**r)
                if len(r.fixed) == len(batch):
                    return [t.strip() or batch[i] for i, t in enumerate(r.fixed)]
            except Exception:
                pass
            return batch  # 清洗失败原样保留，不挡流水线

    outs: list[str] = []
    for i in range(0, len(texts), CLEAN_BATCH):
        outs.extend(await one_batch(texts[i:i + CLEAN_BATCH]))
    return outs


async def simulate(structured_model, name: str, need: int) -> list[dict]:
    out: list[dict] = []
    misses = 0
    while len(out) < need and misses < 5:
        n = min(SIM_BATCH, need - len(out))
        try:
            r = await structured_model.ainvoke(SIMULATE_PROMPT.format(
                n=n, name=name, terminology=terminology_table()))
            if isinstance(r, dict):
                r = _SimBatch(**r)
        except Exception:
            misses += 1
            continue
        got = 0
        for it in r.items:
            labels = [lb for lb in it.labels if lb in LABEL2ID]
            if name not in labels:
                labels = [name] + labels
            if it.text.strip():
                out.append({"text": it.text.strip(), "labels": labels, "origin": "simulated"})
                got += 1
        misses = misses + 1 if got == 0 else 0
    return out[:need]


async def run(target_per_class: int) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_model, sim_model, clean_structured = _build_models()
    settings = load_settings()
    session_factory = session_factory_from(settings)

    # 1) 捞池（标准化问法优先，question_id 溯源）
    from app.repositories.topic_classification import list_pool_texts

    with session_factory() as session:
        pool = [
            {"text": p["text"], "labels": [], "origin": "pool", "question_id": p["question_id"]}
            for p in list_pool_texts(session)
        ]
    _dump(OUT / "corpus_raw.jsonl", pool)
    print(f"捞池 {len(pool)} 条")

    # 2) 清洗：脱敏 → 去重 → LLM 修错别字 → 再去重
    cleaned = dedupe([{**s, "text": desensitize(s["text"])} for s in pool])
    fixed = await clean_texts(raw_model, clean_structured, [s["text"] for s in cleaned])
    cleaned = dedupe([{**s, "text": t} for s, t in zip(cleaned, fixed)])
    _dump(OUT / "corpus_clean.jsonl", cleaned)
    print(f"清洗后 {len(cleaned)} 条")

    # 3) 预标真实问题
    from scripts.ch10.prelabel import prelabel_batch

    labels = await prelabel_batch([s["text"] for s in cleaned])
    labeled = [{**s, "labels": lb} for s, lb in zip(cleaned, labels)]

    # 4) 模拟补足：按标签计数补到每类 target（多诉求句给命中的每类都记数）
    counts = {c.name: 0 for c in TOPIC_CLASSES}
    for s in labeled:
        for lb in s["labels"]:
            counts[lb] += 1
    for c in TOPIC_CLASSES:
        need = target_per_class - counts[c.name]
        if need <= 0:
            print(f"{c.name}: 已够 {counts[c.name]} 条，不补")
            continue
        sims = await simulate(sim_model, c.name, need)
        labeled.extend(sims)
        for s in sims:
            for lb in s["labels"]:
                counts[lb] += 1
        print(f"{c.name}: 补造 {len(sims)} 条（当前 {counts[c.name]}）")
    labeled = dedupe(labeled)
    _dump(OUT / "corpus_labeled.jsonl", labeled)
    n_pool = sum(1 for s in labeled if s["origin"] == "pool")
    print(f"语料总量 {len(labeled)} 条（真实池 {n_pool} / 模拟 {len(labeled)-n_pool}）；"
          f"按标签计各类:{counts}")

    # 5) 抽审导出：真实池全量 + 每类模拟抽 5
    rng = random.Random(42)
    lines = ["# ch10 语料人工抽审（预标 + 模拟）", "",
             "> 格式：问题 → 标签。看到错标直接指出行号或原句。", "",
             "## 真实池问题（全量）", ""]
    for s in labeled:
        if s["origin"] == "pool":
            lines.append(f"- {s['text']} → {'、'.join(s['labels'])}")
    lines += ["", "## 模拟问题（每类抽 5）", ""]
    for c in TOPIC_CLASSES:
        sims = [s for s in labeled if s["origin"] == "simulated" and c.name in s["labels"]]
        lines.append(f"### {c.name}")
        lines += [f"- {s['text']} → {'、'.join(s['labels'])}"
                  for s in rng.sample(sims, min(5, len(sims)))]
        lines.append("")
    (OUT / "sample_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"抽审文件已导出：{OUT / 'sample_review.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ch10 语料流水线")
    ap.add_argument("--target-per-class", type=int, default=TARGET_PER_CLASS,
                    help=f"每类目标条数（默认 {TARGET_PER_CLASS}，封顶控成本）")
    args = ap.parse_args()
    return asyncio.run(run(args.target_per_class))


if __name__ == "__main__":
    import sys

    sys.exit(main())
