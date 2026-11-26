"""ch05 图测试公共 fixture:hermetic settings + 真实 SQLite session_factory。"""
import pytest

from app.graph.deps import GraphDeps
from app.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings.hermetic_defaults()


@pytest.fixture
def graph_deps(session_factory, settings) -> GraphDeps:
    return GraphDeps(session_factory=session_factory, settings=settings)


def cfg(deps: GraphDeps) -> dict:
    """构造 LangGraph config:deps 经 configurable 注入(与 runtime 一致)。"""
    return {"configurable": {"deps": deps}}
