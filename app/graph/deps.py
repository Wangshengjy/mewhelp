"""ch05 · 图节点的依赖注入(GraphDeps)。

课程参考用模块级单例(``app.config.settings`` + repository 直连);本仓库的
Settings 是 ``load()`` 出来的 frozen dataclass、repository 是同步 Session
风格,于是把 ``session_factory / settings / chat_model_factory`` 打包成
GraphDeps,经 LangGraph 的 ``config["configurable"]["deps"]`` 注入——
这是 LangGraph 官方推荐的运行期传参方式;checkpointer 只持久化 State,
不会把依赖序列化进 checkpoint。
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.graph.llm import build_chat_model

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.settings import Settings


@dataclass(frozen=True)
class GraphDeps:
    """图节点运行所需的外部依赖(不可变,随 config 走)。"""

    session_factory: Callable[[], "Session"]
    settings: "Settings"
    # 模型工厂单独抽出:测试注入 Fake 模型、换模型只动这一处
    chat_model_factory: Callable[["Settings"], Any] = build_chat_model


def deps_from_config(config: dict | None) -> GraphDeps:
    """从 LangGraph config 里取 GraphDeps;缺了就明确报错,不静默兜底。"""
    deps = (config or {}).get("configurable", {}).get("deps")
    if not isinstance(deps, GraphDeps):
        raise RuntimeError(
            "图节点缺依赖:应在 config['configurable']['deps'] 注入 GraphDeps"
            "(由 app.graph.runtime 构造)"
        )
    return deps
