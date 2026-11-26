"""ch06 · 八类意图分类器:structured output + 置信度 + 异常/越界回退。"""
import asyncio

from app.graph import intent


class FakeStructuredModel:
    def __init__(self, label: str, confidence: float = 0.9):
        self.label = label
        self.confidence = confidence

    async def ainvoke(self, messages):
        return intent.IntentLabel(intent=self.label, confidence=self.confidence)


class FakeChatModel:
    def __init__(self, label: str, confidence: float = 0.9):
        self.label = label
        self.confidence = confidence

    def with_structured_output(self, schema, method=None):
        return FakeStructuredModel(self.label, self.confidence)


class RaisingChatModel:
    """模拟上游不可用(如欠费 402):构建结构化输出即抛异常。"""

    def with_structured_output(self, schema, method=None):
        raise RuntimeError("upstream 402")


def test_classify_returns_label_and_confidence(settings):
    out = asyncio.run(
        intent.classify("退货政策是什么", settings, lambda s: FakeChatModel("退款退货", 0.83))
    )
    assert out == {"intent": "退款退货", "confidence": 0.83}


def test_classify_passes_history_for_coreference(settings):
    """多轮消歧:history 非空时拼进 HumanMessage(「那这个呢」才有上下文可判)。"""
    seen: dict = {}

    class Capturing(FakeStructuredModel):
        async def ainvoke(self, messages):
            seen["content"] = messages[-1].content
            return intent.IntentLabel(intent="订单", confidence=0.9)

    class CapturingChat(FakeChatModel):
        def with_structured_output(self, schema, method=None):
            return Capturing(self.label)

    asyncio.run(
        intent.classify(
            "那这个呢",
            settings,
            lambda s: CapturingChat("订单"),
            history="用户:订单1001到哪了\n客服:已发货",
        )
    )
    assert "订单1001到哪了" in seen["content"]
    assert "那这个呢" in seen["content"]


def test_classify_out_of_range_label_falls_back_to_other(settings):
    """Literal 八类是硬约束:越界标签过不了 pydantic 校验 → 回退「其他」。"""
    out = asyncio.run(intent.classify("你好", settings, lambda s: FakeChatModel("火星语")))
    assert out == {"intent": "其他", "confidence": 0.0}


def test_classify_upstream_failure_falls_back_to_other(settings):
    out = asyncio.run(intent.classify("查订单", settings, lambda s: RaisingChatModel()))
    assert out == {"intent": "其他", "confidence": 0.0}
