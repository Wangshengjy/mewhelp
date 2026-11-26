"""ch10 推理服务：ONNX + FastAPI 独立进程 :8110，轻运行时（onnxruntime + tokenizers，不背 torch）。

起停：python -m scripts.ch10.classifier_ctl start|stop|status（Windows 无 nohup，pid 文件 data/classifier.pid）。
只 import app.core.taxonomy（无 settings 依赖）；阈值用训练产物 threshold.json。
classifier_ctl 经 CIM 按文件路径直起本文件（cwd 不可靠），故这里自举仓库根进 sys.path。
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tokenizers import Tokenizer

from app.core.taxonomy import TOPIC_NAMES

DIR = pathlib.Path("data/ch10/onnx")

app = FastAPI(title="mewhelp ch10 topic classifier")
_sess = ort.InferenceSession(str(DIR / "model.onnx"), providers=["CPUExecutionProvider"])
_tok = Tokenizer.from_file(str(DIR / "tokenizer.json"))
_tok.enable_truncation(max_length=128)
_tok.enable_padding(pad_id=0, pad_token="[PAD]")
_threshold: float = json.loads((DIR / "threshold.json").read_text(encoding="utf-8"))["threshold"]
_meta: dict = json.loads((DIR / "threshold.json").read_text(encoding="utf-8"))


class ClassifyIn(BaseModel):
    texts: list[str]


@app.get("/healthz")
def healthz():
    return {"ok": True, "mode": _meta.get("mode"), "threshold": _threshold,
            "note": _meta.get("note")}


@app.post("/classify")
def classify(body: ClassifyIn):
    if not body.texts:
        return {"results": []}
    encs = _tok.encode_batch(body.texts)
    feed = {
        "input_ids": np.array([e.ids for e in encs], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
        "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64),
    }
    (logits,) = _sess.run(["logits"], feed)
    probs = 1 / (1 + np.exp(-logits))
    results = []
    for row in probs:
        labels = [TOPIC_NAMES[i] for i, p in enumerate(row) if p >= _threshold]
        if not labels:                       # 全不过线取最高分兜底，不产出空标签
            labels = [TOPIC_NAMES[int(row.argmax())]]
        results.append({"labels": labels,
                        "scores": {TOPIC_NAMES[i]: round(float(p), 4)
                                   for i, p in enumerate(row)}})
    return {"results": results}


if __name__ == "__main__":
    # classifier_ctl 经 CIM + powershell 包装直起本进程（不经 cmd），日志由进程自己落盘；
    # pid 文件由本进程写（pid 文件必须指向真正的 python 进程，包装进程会随其退出）。
    import logging
    import os

    _LOG = pathlib.Path(__file__).resolve().parents[2] / "log" / "classifier.log"
    _LOG.parent.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=str(_LOG), level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "classifier.pid").write_text(str(os.getpid()), encoding="utf-8")
    uvicorn.run(app, host="127.0.0.1", port=8110, log_config=None)
