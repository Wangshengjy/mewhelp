"""ch09 · runtime 每轮 config 带 langfuse_session_id（Langfuse 会话归属，多轮串成一组）。"""
from app.graph import runtime


def test_graph_config_carries_session_metadata(monkeypatch, graph_deps):
    monkeypatch.setattr(runtime, "_deps", graph_deps)
    config = runtime._graph_config("conv-1")
    assert config["metadata"]["langfuse_session_id"] == "conv-1"
    assert config["configurable"]["deps"] is graph_deps
    assert config["configurable"]["thread_id"] == "conv-1"
