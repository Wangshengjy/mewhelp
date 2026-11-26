"""ch10 · 预标模块纯逻辑（LLM 一律 mock，不打真上游）。"""
import asyncio

import pytest

import scripts.ch10.prelabel as prelabel_module
from scripts.ch10.prelabel import prelabel_batch, prelabel_one


class _FakeModel:
    """按文本关键词回标签的假模型；raise 起来时模拟上游故障。"""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on

    async def ainvoke(self, prompt: str):
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("upstream down")
        text = prompt.rsplit("用户问题：", 1)[1].strip()
        if "买大了想退" in text:
            labels = ["尺码", "退换货", "不存在的类目"]   # 非法标签须被过滤
        elif "快递" in text:
            labels = ["物流"]
        elif "闲聊" in text:
            labels = []                                   # 空 → 兜底「其他」
        else:
            labels = ["其他"]
        return prelabel_module._Labeled(labels=labels)


@pytest.fixture(autouse=True)
def _fake_build_model(monkeypatch):
    holder = {}

    def _build(fail_on=None):
        holder.setdefault("model", _FakeModel(fail_on))
        return holder["model"]

    monkeypatch.setattr(prelabel_module, "_build_model", lambda: _build())


def test_prelabel_one_filters_invalid_labels():
    labels = asyncio.run(prelabel_one("猫窝买大了想退"))
    assert labels == ["尺码", "退换货"]


def test_prelabel_one_empty_falls_back_to_other():
    assert asyncio.run(prelabel_one("闲聊一下")) == ["其他"]


def test_prelabel_one_upstream_failure_falls_back():
    model = _FakeModel(fail_on="error")
    assert asyncio.run(prelabel_one("触发故障error", model=model)) == ["其他"]


def test_prelabel_batch_order_preserved():
    texts = ["我的快递到哪了", "猫窝买大了想退", "闲聊一下"]
    out = asyncio.run(prelabel_batch(texts, concurrency=3))
    assert out == [["物流"], ["尺码", "退换货"], ["其他"]]
