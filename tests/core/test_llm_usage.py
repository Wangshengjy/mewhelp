"""ch09 Cost Control 前提：所有模型调用（含流式）都回传 usage，按意图统计才不漏账。"""
from app.graph.llm import build_chat_model
from app.settings import Settings


def _settings() -> Settings:
    return Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
    )


def test_streaming_model_requests_usage():
    m = build_chat_model(_settings())
    assert m.stream_usage is True


def test_non_streaming_model_also_flags_usage():
    # 本仓库 build_chat_model 统一 streaming=True；断言无论如何 usage 都开着
    assert build_chat_model(_settings()).stream_usage is True


def test_factory_is_cached_per_settings():
    """同一 Settings 反复构建拿到同一实例（既有 lru_cache 行为不回归）。"""
    s = _settings()
    assert build_chat_model(s) is build_chat_model(s)
