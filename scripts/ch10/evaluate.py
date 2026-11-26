"""ch10 评测：留出测试集上每类 P/R/F1 + 每类混淆矩阵 + 容错红线 + 判错样本导出。

运行：python -m scripts.ch10.evaluate
评测集扎在自家电商场景（dataset/test.jsonl），不引公开榜单。
红线按档位：严 F1≥0.9、中 ≥0.8、宽不设线（taxonomy.SEVERITY_RED_LINE）。
报告一式两份：data/ch10/reports/eval_report.{md,json} + error_samples.md。
冒烟权重（threshold.json mode=smoke）跑出的真实（差）数字照实落报告，质量结论等 GPU。
"""
import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, SEVERITY, SEVERITY_RED_LINE, TOPIC_NAMES

MODEL_DIR = pathlib.Path("data/ch10/model")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def predict(model, tokenizer, texts: list[str], threshold: float, device: str,
            max_length: int = 128) -> np.ndarray:
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=max_length, return_tensors="pt").to(device)
            probs_all.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    probs = np.concatenate(probs_all)
    preds = (probs >= threshold).astype(int)
    for r in range(len(preds)):                    # 全不过线取最高分兜底，不产出空标签
        if preds[r].sum() == 0:
            preds[r][probs[r].argmax()] = 1
    return preds


def main() -> int:
    device = pick_device()
    threshold_meta = json.loads((MODEL_DIR / "threshold.json").read_text(encoding="utf-8"))
    threshold = threshold_meta["threshold"]
    smoke = threshold_meta.get("mode") == "smoke"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    samples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [s["text"] for s in samples]
    gold = np.zeros((len(samples), NUM_CLASSES), dtype=int)
    for i, s in enumerate(samples):
        for lb in s["labels"]:
            gold[i][LABEL2ID[lb]] = 1
    preds = predict(model, tokenizer, texts, threshold, device,
                    max_length=int(threshold_meta.get("max_length", 128)))

    p, r, f1, support = precision_recall_fscore_support(gold, preds, zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0)
    cms = multilabel_confusion_matrix(gold, preds)

    classes, errors = [], []
    total_fp = total_fn = total_matrix_cells = 0
    for i, name in enumerate(TOPIC_NAMES):
        tn, fp = int(cms[i][0][0]), int(cms[i][0][1])
        fn, tp = int(cms[i][1][0]), int(cms[i][1][1])
        total_fp += fp
        total_fn += fn
        total_matrix_cells += fp + fn + tp + tn
        red_line = SEVERITY_RED_LINE[SEVERITY[name]] or None
        passed = None if red_line is None else bool(f1[i] >= red_line)
        classes.append({"name": name, "severity": SEVERITY[name],
                        "p": round(float(p[i]), 4), "r": round(float(r[i]), 4),
                        "f1": round(float(f1[i]), 4), "support": int(support[i]),
                        "red_line": red_line, "passed": passed,
                        "tn": tn, "fp": fp, "fn": fn, "tp": tp})

    for i, s in enumerate(samples):
        pred_labels = [TOPIC_NAMES[j] for j in range(NUM_CLASSES) if preds[i][j]]
        if set(pred_labels) != set(s["labels"]):
            missed = [lb for lb in s["labels"] if lb not in pred_labels]
            extra = [lb for lb in pred_labels if lb not in s["labels"]]
            kind = "错位" if missed and extra else ("漏打" if missed else "多打")
            errors.append({"text": s["text"], "gold": list(s["labels"]), "pred": pred_labels,
                           "missed": missed, "extra": extra, "kind": kind,
                           "matrix_entries": len(missed) + len(extra)})
    red_line_passed = all(c["passed"] is not False for c in classes)

    REPORTS.mkdir(parents=True, exist_ok=True)
    ran_at = datetime.now(timezone.utc).isoformat()
    report = {
        "ran_at": ran_at,
        "test_size": len(samples),
        "threshold": threshold,
        "red_lines": {"严": SEVERITY_RED_LINE["严"], "中": SEVERITY_RED_LINE["中"], "宽": None},
        "micro": {"p": round(float(micro_p), 4), "r": round(float(micro_r), 4),
                  "f1": round(float(micro_f1), 4)},
        "macro": {"p": round(float(macro_p), 4), "r": round(float(macro_r), 4),
                  "f1": round(float(macro_f1), 4)},
        "classes": classes,
        "errors": errors,
        "total_cells": total_matrix_cells,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "red_line_passed": red_line_passed,
        "model_mode": threshold_meta.get("mode"),
        "model_note": threshold_meta.get("note"),
        "device": device,
    }
    (REPORTS / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    tag = f"（{threshold_meta.get('note')}）" if smoke else ""
    lines = ["# ch10 分类器评测报告（留出测试集）", "",
             f"- 运行时间：{ran_at}",
             f"- 测试集 {len(samples)} 条；判定阈值 {threshold}（验证集扫描所得）{tag}",
             f"- **micro**：P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}  |  "
             f"**macro**：P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}",
             f"- 红线：严档 F1≥{SEVERITY_RED_LINE['严']}、中档 ≥{SEVERITY_RED_LINE['中']}、宽档不设线"
             f"；红线总闸 {'✅ 全过' if red_line_passed else '🔴 有不达标（先回头搞数据）'}", "",
             "## 每类指标", "",
             "| 类目 | 档位 | P | R | F1 | support | 红线 |",
             "|---|---|---|---|---|---|---|"]
    for c in classes:
        if c["passed"] is None:
            flag = "—"
        elif c["passed"]:
            flag = "✅"
        else:
            flag = "🔴 不达标，先回头搞数据"
        lines.append(f"| {c['name']} | {c['severity']} | {c['p']:.3f} | {c['r']:.3f} "
                     f"| {c['f1']:.3f} | {c['support']} | {flag} |")
    lines += ["", "## 每类混淆矩阵（TN FP / FN TP）", ""]
    for c in classes:
        lines.append(f"- **{c['name']}**：TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")
    lines += ["", "## 错误方向记账", "",
              f"错例 {len(errors)} 条；矩阵错笔（FP+FN）{total_fp + total_fn} 笔；"
              "差额全在「错位」（一次记 2 笔：放跑 + 冤枉）。", ""]
    for e in errors:
        lines.append(f"- [{e['kind']}] {e['text']}：标准 {e['gold']} / 预测 {e['pred']}"
                     f"（放跑 {e['missed']}、冤枉 {e['extra']}）")
    (REPORTS / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    err = ["# ch10 判错样本（人工复核：错在哪一类？标注本身有没有毛病？）", ""]
    for e in errors:
        err.append(f"- [{e['kind']}] {e['text']}\n  标准: {e['gold']}  预测: {e['pred']}")
    (REPORTS / "error_samples.md").write_text("\n".join(err), encoding="utf-8")

    print(f"micro-F1 {micro_f1:.4f} / macro-F1 {macro_f1:.4f}；红线总闸 "
          f"{'过' if red_line_passed else '不过'}；错例 {len(errors)} 条；"
          f"报告已落 {REPORTS}/" + ("（冒烟权重，非正式模型）" if smoke else ""))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
