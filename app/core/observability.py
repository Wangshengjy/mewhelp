"""ch09 可观测：Langfuse 挂载与 trace 标注，全部可选降级。

课程 README 姿势：设三个环境变量 + 图编译时挂一次回调，节点业务代码零侵入。
本仓库 Settings 是 load() 出来的 frozen dataclass（无模块级单例），故经
``configure(settings)`` 在 runtime.init_graph 时注入；所有对外函数在未配置
Langfuse 时必须是安全 no-op——观测是增强，不是依赖（测试环境不装 Langfuse 服务）。
"""
import logging

logger = logging.getLogger(__name__)

_settings = None  # 由 runtime.init_graph 注入的 Settings（frozen dataclass）
_client = None


def configure(settings) -> None:
    """注入应用 Settings（init_graph 时调用；测试可直接替换模块级 _settings）。"""
    global _settings, _client
    _settings = settings
    _client = None  # settings 换了则单例重建


def langfuse_enabled() -> bool:
    if _settings is None:
        return False
    return bool(_settings.langfuse_public_key and _settings.langfuse_secret_key
                and _settings.langfuse_base_url)


def _init_client():
    """初始化（或复用）Langfuse 单例。langfuse==3.15.0 构造参名为 host。"""
    global _client
    if _client is None:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=_settings.langfuse_public_key,
            secret_key=_settings.langfuse_secret_key,
            host=_settings.langfuse_base_url,
        )
    return _client


def _make_handler():
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def get_langfuse():
    """启用时返回已初始化 client（cost 脚本用）；未启用返回 None。"""
    if not langfuse_enabled():
        return None
    return _init_client()


def attach_observability(graph):
    """编译后挂一次 Langfuse 回调，全图自动 trace（课程 README：「编译时挂一次，节点里一行不用动」）。

    未配置时原样返回——不挂、不 import langfuse。"""
    if not langfuse_enabled():
        return graph
    _init_client()
    return graph.with_config({"callbacks": [_make_handler()]})


def tag_intent(intent: str, confidence: float) -> None:
    """把意图写进当前 trace 的 metadata + tag（Cost Control 按意图分堆的钩子）。

    课程姿势：classify_intent 节点内调 update_current_trace；回调挂全图时
    当前执行上下文即该轮 trace。任何异常静默降级——观测失败不许影响业务流。"""
    if not langfuse_enabled():
        return
    try:
        client = _init_client()
        client.update_current_trace(
            metadata={"intent": intent, "intent_confidence": float(confidence)},
            tags=[f"intent:{intent}"],
        )
    except Exception:
        logger.debug("langfuse tag_intent 失败（已忽略）", exc_info=True)
