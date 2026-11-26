"""ch10 阈值扫描重演：验证集经 :8110 打分一次、分数表固定，九个候选线套同一张表各算一遍
micro-F1；consistent 记「重演当选线 == threshold.json 在用线」，证明阈值是扫出来的不是拍的。

打平时先到的留任（严格大于），与训练侧 scan_threshold 同口径。
运行：python -m scripts.ch10.scan_threshold_replay（需分类器服务 :8110）。
"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime, timezone

import httpx
import numpy as np
from sklearn.metrics import f1_score

from app.core.taxonomy import LABEL2ID, NUM_CLASSES

VAL = pathlib.Path("data/ch10/dataset/val.jsonl")
ONNX_DIR = pathlib.Path("data/ch10/onnx")
REPORTS = pathlib.Path("data/ch10/reports")
SERVICE = "http://127.0.0.1:8110"
CANDIDATES = [round(float(t), 2) for t in np.arange(0.30, 0.71, 0.05)]  # 0.30~0.70 步进 0.05


async def main() -> int:
    samples = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [s["text"] for s in samples]
    gold = np.zeros((len(samples), NUM_CLASSES), dtype=int)
    for i, s in enumerate(samples):
        for lb in s["labels"]:
            gold[i][LABEL2ID[lb]] = 1

    async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
        r = await client.post(f"{SERVICE}/classify",
                              json={"texts": texts})
        r.raise_for_status()
        results = r.json()["results"]
    # 服务回的是过线 labels（阈值已应用），scores 才是重演的原料——按类目 id 对齐列序
    col_order = [name for name in sorted(results[0]["scores"], key=lambda n: LABEL2ID[n])]
    probs = np.array([[res["scores"][n] for n in col_order] for res in results])
    id_order = [LABEL2ID[n] for n in col_order]
    if id_order != list(range(NUM_CLASSES)):
        gold = gold[:, id_order]

    scan = []
    best_t, best_f1 = CANDIDATES[0], -1.0
    for t in CANDIDATES:
        preds = (probs >= t).astype(int)
        # 兜底口径与服务一致：全不过线取最高分一类
        for row in range(len(preds)):
            if preds[row].sum() == 0:
                preds[row][probs[row].argmax()] = 1
        tp = int(((preds == 1) & (gold == 1)).sum())
        fp = int(((preds == 1) & (gold == 0)).sum())
        fn = int(((preds == 0) & (gold == 1)).sum())
        f1 = f1_score(gold, preds, average="micro", zero_division=0)
        scan.append({"threshold": t, "micro_f1": round(float(f1), 4), "tp": tp, "fp": fp, "fn": fn})
        if f1 > best_f1:
            best_t, best_f1 = t, float(f1)

    in_use = json.loads((ONNX_DIR / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    consistent = best_t == in_use
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "threshold_scan.json").write_text(json.dumps({
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "val_size": len(samples),
        "scan": scan,
        "best_threshold": best_t,
        "best_micro_f1": round(best_f1, 4),
        "in_use_threshold": in_use,
        "consistent": consistent,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"九候选线扫描完成：best={best_t}（micro-F1 {best_f1:.4f}）；"
          f"在用线 {in_use}；一致 {'✅' if consistent else '🔴 不一致，查 threshold.json 来源'}")
    return 0 if consistent else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
