"""ch09 观测挂载：缺配置不挂回调、不碰 langfuse；齐配置才 with_config。

本仓库 Settings 是 frozen dataclass，observability 经 configure() 注入——
测试直接 monkeypatch 模块级 _settings（等价于换一套配置）。
"""
from types import SimpleNamespace

from app.core import observability


def _settings_with(public_key="", secret_key="", base_url=""):
    return SimpleNamespace(
        langfuse_public_key=public_key,
        langfuse_secret_key=secret_key,
        langfuse_base_url=base_url,
    )


def test_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr(observability, "_settings", _settings_with(public_key="", secret_key="sk", base_url="http://x"))
    monkeypatch.setattr(observability, "_client", None)
    assert observability.langfuse_enabled() is False


def test_disabled_when_no_settings_injected(monkeypatch):
    monkeypatch.setattr(observability, "_settings", None)
    assert observability.langfuse_enabled() is False


def test_attach_returns_graph_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr(observability, "_settings", _settings_with())
    monkeypatch.setattr(observability, "_client", None)
    sentinel = object()
    assert observability.attach_observability(sentinel) is sentinel


def test_tag_intent_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(observability, "_settings", _settings_with())
    monkeypatch.setattr(observability, "_client", None)
    observability.tag_intent("商品咨询", 0.9)  # 不许抛异常、不许要求 langfuse 服务在跑


def test_get_langfuse_none_when_disabled(monkeypatch):
    monkeypatch.setattr(observability, "_settings", _settings_with())
    assert observability.get_langfuse() is None


def test_attach_wraps_with_callbacks_when_enabled(monkeypatch):
    monkeypatch.setattr(
        observability, "_settings",
        _settings_with(public_key="pk", secret_key="sk", base_url="http://localhost:3000"),
    )
    calls = {}

    class FakeGraph:
        def with_config(self, cfg):
            calls["cfg"] = cfg
            return "wrapped"

    monkeypatch.setattr(observability, "_init_client", lambda: object())
    monkeypatch.setattr(observability, "_make_handler", lambda: "HANDLER")
    assert observability.attach_observability(FakeGraph()) == "wrapped"
    assert calls["cfg"] == {"callbacks": ["HANDLER"]}


def test_tag_intent_swallows_sdk_errors(monkeypatch):
    """Langfuse 启用但 SDK 报错（如服务没起）时静默——观测失败不许影响业务流。"""
    monkeypatch.setattr(
        observability, "_settings",
        _settings_with(public_key="pk", secret_key="sk", base_url="http://localhost:3000"),
    )

    class BoomClient:
        def update_current_trace(self, **kwargs):
            raise RuntimeError("langfuse 未起")

    monkeypatch.setattr(observability, "_init_client", lambda: BoomClient())
    observability.tag_intent("物流", 0.99)  # 不抛
