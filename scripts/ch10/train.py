"""ch10 训练：RoBERTa-wwm-ext 全参微调，17 类多标签（BCEWithLogitsLoss）。

完整形态（拿到 GPU 即跑正式规模）：--full 或默认 —— Trainer + 早停（盯验证集 micro-F1，
patience=2）+ weight_decay 正则 + 每 epoch 评验证集 + 训后验证集阈值扫描（0.30~0.70 步进 0.05）。
本机无 GPU：--smoke 用 8-16 条、1 epoch、max_length=64 的 CPU 冒烟跑通 train→export→serve→classify_pool
全链路，产物 threshold.json 标注 mode=smoke「plumbing 冒烟权重，非正式模型」。

运行：python -m scripts.ch10.train [--smoke]
HF 下载不通时：$env:HF_ENDPOINT="https://hf-mirror.com" 后再跑。
transformers v5 参数口径：eval_strategy（非 evaluation_strategy）、processing_class（非 tokenizer=）。
"""
import argparse
import json
import pathlib
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainingArguments,
                          default_data_collator)

from app.core.taxonomy import ID2LABEL, LABEL2ID, NUM_CLASSES

BASE = "hfl/chinese-roberta-wwm-ext"
DATA = pathlib.Path("data/ch10/dataset")
OUT = pathlib.Path("data/ch10/model")

SMOKE_NOTE = "plumbing 冒烟权重，非正式模型：CPU 链路验收用，不代表模型质量"


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(samples: list[dict], tokenizer, max_length: int) -> list[dict]:
    enc = tokenizer([s["text"] for s in samples], truncation=True,
                    padding="max_length", max_length=max_length)
    items = []
    for i, s in enumerate(samples):
        vec = [0.0] * NUM_CLASSES                 # float 向量：BCEWithLogitsLoss 要求
        for lb in s["labels"]:
            vec[LABEL2ID[lb]] = 1.0
        items.append({"input_ids": enc["input_ids"][i],
                      "attention_mask": enc["attention_mask"][i],
                      "token_type_ids": enc["token_type_ids"][i],
                      "labels": vec})
    return items


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
    return {"micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def scan_threshold(probs: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    """验证集上扫全局最优阈值（0.30~0.70 步进 0.05）；打平时先到的留任（严格大于）。"""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.71, 0.05):
        f1 = f1_score(gold, (probs >= t).astype(int), average="micro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), float(f1)
    return best_t, best_f1


def main(smoke: bool) -> None:
    import torch

    device = pick_device()
    mode = "smoke" if smoke else "full"
    print(f"设备：{device}；模式：{mode}"
          + (f"（{SMOKE_NOTE}）" if smoke else "（正式规模：全参微调 8 epoch 上限 + 早停）"))

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=NUM_CLASSES, problem_type="multi_label_classification",
        id2label=ID2LABEL, label2id=LABEL2ID)

    max_length = 64 if smoke else 128
    train_samples = load_jsonl(DATA / "train.jsonl")
    val_samples = load_jsonl(DATA / "val.jsonl")
    if smoke:                                     # 冒烟：训练 16 条 / 验证 8 条
        train_samples = train_samples[:16]
        val_samples = val_samples[:8]
    train_ds = encode(train_samples, tokenizer, max_length)
    val_ds = encode(val_samples, tokenizer, max_length)

    args = TrainingArguments(
        output_dir="data/ch10/checkpoints",
        eval_strategy="epoch",                    # v5 参数名，不是 evaluation_strategy
        save_strategy="epoch",                    # load_best_model_at_end 要求与 eval 一致
        learning_rate=2e-5,
        per_device_train_batch_size=8 if smoke else 16,
        per_device_eval_batch_size=32,
        num_train_epochs=1 if smoke else 8,
        weight_decay=0.01,                        # 正则化防过拟合
        load_best_model_at_end=not smoke,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=5 if smoke else 20,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),    # 显式走 CPU/GPU，本机 CPU 冒烟（v5 已移除 no_cuda）
    )
    callbacks = [] if smoke else [EarlyStoppingCallback(early_stopping_patience=2)]
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,               # v5：tokenizer= 已改名
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer.train()

    # 验证集上扫全局最优阈值
    logits = trainer.predict(val_ds).predictions
    probs = 1 / (1 + np.exp(-logits))
    gold = np.array([d["labels"] for d in val_ds])
    best_t, best_f1 = scan_threshold(probs, gold)

    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    (OUT / "threshold.json").write_text(json.dumps({
        "threshold": best_t,
        "val_micro_f1": round(best_f1, 4),
        "mode": mode,
        "note": SMOKE_NOTE if smoke else "正式模型",
        "base_model": BASE,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "max_length": max_length,
        "device": device,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False), encoding="utf-8")
    print(f"最优阈值 {best_t}，验证集 micro-F1 {best_f1:.4f}；模型已存 {OUT}"
          + (f"（{SMOKE_NOTE}）" if smoke else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ch10 训练：RoBERTa-wwm-ext 全参微调")
    ap.add_argument("--smoke", action="store_true",
                    help="CPU 冒烟：16 条 1 epoch max_length=64，跑通链路（非正式模型）")
    a = ap.parse_args()
    main(a.smoke)
