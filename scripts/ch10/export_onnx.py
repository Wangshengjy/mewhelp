"""ch10 ONNX 导出：torch 模型 → onnx（变长 batch/seq），导出后跑测试集校验与 torch 预测完全一致。

注意：torch 2.9+ 的 torch.onnx.export 默认 dynamo=True，导 HF 模型此处显式 dynamo=False
走 TorchScript 导出器 + dynamic_axes 稳定路线。运行：python -m scripts.ch10.export_onnx。
产物：data/ch10/onnx/（model.onnx + tokenizer 文件 + threshold.json，服务只吃这个目录）。
导出报告 export_report.json 失败也要写（mismatch != 0 先写报告再退出），验收页才能看到「上次导出没过」。
"""
import json
import pathlib
import shutil
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = pathlib.Path("data/ch10/model")
OUT = pathlib.Path("data/ch10/onnx")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")
OPSET = 17


class LogitsWrapper(torch.nn.Module):
    """HF 模型输出是 ModelOutput dict，导出前包一层只回 logits 张量。"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def _write_report(report: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    wrapper = LogitsWrapper(model)
    sample = tokenizer(["买大了想退", "快递到哪了"], padding=True, return_tensors="pt")
    dyn = {0: "batch", 1: "seq"}
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": dyn, "attention_mask": dyn,
                      "token_type_ids": dyn, "logits": {0: "batch"}},
        dynamo=False,
        opset_version=OPSET,
    )
    tokenizer.save_pretrained(OUT)          # 带出 tokenizer.json 给轻运行时用
    shutil.copy(MODEL_DIR / "threshold.json", OUT / "threshold.json")

    # 一致性校验：测试集全量，ONNX 与 torch 的过线标签必须完全一致。
    # 用重新从磁盘加载的干净实例——刚被 torch.onnx.export 追踪过的实例在 torch 2.15 上
    # 状态会被污染（追踪后再前向与 ONNX 输出出现大幅偏差），重载后两者一致到 1e-6。
    threshold = json.loads((OUT / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    texts = [json.loads(l)["text"]
             for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    sess = ort.InferenceSession(str(OUT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    check_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    check_model.eval()
    check_wrapper = LogitsWrapper(check_model)
    mismatch = 0
    max_logit_diff = 0.0
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt")
            t_logits = check_wrapper(**enc).numpy()
            (o_logits,) = sess.run(["logits"], {k: v.numpy() for k, v in enc.items()})
            diff = float(np.max(np.abs(t_logits - o_logits)))
            max_logit_diff = max(max_logit_diff, diff)
            assert diff <= 1e-3, f"logits 超容差：{diff}"
            t_pred = (1 / (1 + np.exp(-t_logits)) >= threshold)
            o_pred = (1 / (1 + np.exp(-o_logits)) >= threshold)
            mismatch += int((t_pred != o_pred).any(axis=1).sum())

    onnx_path = OUT / "model.onnx"
    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "checked": len(texts),
        "mismatch": mismatch,
        "passed": mismatch == 0,
        "onnx_path": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "opset": OPSET,
        "max_logit_diff": round(max_logit_diff, 6),
    }
    _write_report(report)
    if mismatch:
        print(f"ONNX 与 torch 预测不一致 {mismatch} 条，导出失败（报告已落）")
        return 1
    print(f"ONNX 导出并校验通过（{len(texts)} 条预测完全一致，logits 最大偏差 "
          f"{max_logit_diff:.2e}）：{onnx_path}（{report['onnx_bytes']} 字节）")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
