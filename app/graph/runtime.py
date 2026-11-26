"""ch05/ch06 · 图运行时:checkpointer 生命周期 + 图单例 + run/resume/stream。

- init_graph/close_graph 由 FastAPI lifespan 管理;session_factory/settings/checkpointer
  均可注入(测试、eval 脚本直连图用)。
- prepare_turn:开流前完成会话解析/创建 + user 消息落库(MySQL 审计)——
  404 / DB 错误在 SSE 开始前抛出,路由层才能给出 404/503 响应。
- stream_turn/stream_resume:驱动图 astream(stream_mode=["messages","updates"]),把
  节点更新映射为 SSE 事件 dict;帧契约保留 ch04(conversation/tool/token/citations/
  done/error),ch05 新增 actions,ch06 新增 interrupt(图暂停,本次流到此为止)。
"""
import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.core import budget as budget_mod
from app.core import memory
from app.core import observability
from app.core import summarizer
from app.graph.build import build_graph
from app.graph.deps import GraphDeps
from app.graph.llm import build_chat_model
from app.kb import citations as citations_store
from app.repositories import conversations as conversations_repository
from app.repositories import low_confidence as low_confidence_repository
from app.settings import Settings

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点:意图分类器等内部 LLM 调用的 token 不进 delta(R1 防泄露)
ANSWER_NODES = {"agent_llm"}
# 确定性出口节点把答复写在 state['answer'],整块作为 delta 吐出(无逐 token 流)
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply"}
# 产出编号引用元数据的节点:流结束后统一 resolve(只发真正被引用的子集)
CITATION_NODES = {"forced_rag", "retrieve_policy"}
# 拦截型工具:转成前端提议,不发 running/completed 工具帧
INTERCEPTED_TOOLS = {"create_ticket", "submit_refund"}

_graph = None
_deps: GraphDeps | None = None
_cm = None  # AsyncSqliteSaver 上下文管理器,持引用防 GC


class ConversationNotFound(Exception):
    """conversation_id 不存在(404)。"""


class TurnNotFound(Exception):
    """checkpoint 里没有该轮次(回捞越界,400)。"""


async def init_graph(*, session_factory=None, settings=None, checkpointer=None, chat_model_factory=None) -> None:
    """编译图并持有依赖。不传依赖时按 Settings.load() 自建(脚本/eval 用)。"""
    global _graph, _deps, _cm
    settings = settings or Settings.load()
    if session_factory is None:
        session_factory = _build_default_session_factory(settings)
    if checkpointer is None:
        db_path = settings.checkpointer_db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _cm = AsyncSqliteSaver.from_conn_string(db_path)
        checkpointer = await _cm.__aenter__()
        await checkpointer.setup()
    _deps = GraphDeps(
        session_factory=session_factory,
        settings=settings,
        chat_model_factory=chat_model_factory or build_chat_model,
    )
    # ch09:Langfuse 三配置齐全才挂回调(编译时一次,全图零侵入);缺省原样返回照常跑
    observability.configure(settings)
    _graph = observability.attach_observability(build_graph(checkpointer=checkpointer))
    logger.info(
        "ch05 图已编译(checkpointer=%s, langfuse=%s)",
        type(checkpointer).__name__, observability.langfuse_enabled(),
    )


async def close_graph() -> None:
    global _graph, _deps, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None
    _deps = None


def get_graph():
    if _graph is None:
        raise RuntimeError("ch05 图未初始化:请先在 lifespan 里 await runtime.init_graph(...)")
    return _graph


def get_deps() -> GraphDeps:
    if _deps is None:
        raise RuntimeError("ch05 图未初始化:请先在 lifespan 里 await runtime.init_graph(...)")
    return _deps


def _build_default_session_factory(settings: Settings):
    """脚本/eval 直连图用:按 settings 自建引擎(生产由 lifespan 注入,不走这里)。"""
    from app.database import build_engine, build_session_factory, initialize_database

    engine = build_engine(settings.database_url)
    factory = build_session_factory(engine)
    initialize_database(factory)
    return factory


async def prepare_turn(conversation_id: str | None, message: str, user_id: str = "demo-user") -> tuple[str, bool]:
    """开流前:解析/创建会话并落 user 消息(MySQL 审计)。404/DB 错在 SSE 前抛出。"""
    deps = get_deps()

    def _work() -> tuple[str, bool]:
        with deps.session_factory() as session:
            if conversation_id is None:
                conversation = conversations_repository.create_conversation(session, user_id=user_id)
                session.commit()
                cid, created = conversation.id, True
            else:
                if conversations_repository.get_conversation(session, conversation_id) is None:
                    raise ConversationNotFound(conversation_id)
                cid, created = conversation_id, False
            conversations_repository.append_message(session, cid, role="user", content=message)
            session.commit()
        return cid, created

    return await asyncio.to_thread(_work)


async def run_turn(conversation_id: str | None, message: str, user_id: str = "demo-user") -> dict:
    """非流式整轮(测试/eval 脚本直连图用):prepare + ainvoke + 轮末分层结算。

    图被 fetch_order 的 interrupt 暂停时,返回多带一个 ``interrupt`` 键
    (待选订单列表);用 resume_turn(或 stream_resume)选中订单后继续。"""
    cid, _ = await prepare_turn(conversation_id, message, user_id)
    summary, upto, layer1_from, msg_id = await _ensure_conversation(cid)
    final = await get_graph().ainvoke(
        _graph_input(cid, message, summary=summary, summary_upto=upto, layer1_from=layer1_from, msg_id=msg_id),
        _graph_config(cid),
    )
    await _after_turn(cid, final)
    return {"conversation_id": cid, "state": final, "interrupt": _interrupt_payload(final)}


async def resume_turn(conversation_id: str, resume_value) -> dict:
    """interrupt 后续跑(测试/eval 直连图用):Command(resume=值) 继续被暂停的图。

    resume 不是新的用户消息,不落库;被中断节点从头重跑,interrupt() 这次
    返回 resume 值(冒烟钉死的语义)。分层字段沿用上轮入图时的值(checkpoint
    里就有),续跑结束照做 settle_layers + 摘要触发检查。"""
    await ensure_conversation(conversation_id)
    final = await get_graph().ainvoke(Command(resume=resume_value), _graph_config(conversation_id))
    await _after_turn(conversation_id, final)
    return {"conversation_id": conversation_id, "state": final, "interrupt": _interrupt_payload(final)}


async def stream_turn(conversation_id: str, message: str) -> AsyncIterator[dict]:
    """新消息整轮流式运行(供 /api/chat/stream 转 SSE 帧)。

    done 帧吐给前端之后、流收尾前做轮末分层结算与摘要调度;interrupt 暂停
    (本轮没走完)则跳过,等 resume 结束再结算。"""
    summary, upto, layer1_from, msg_id = await _ensure_conversation(conversation_id)
    graph_input = _graph_input(
        conversation_id, message, summary=summary, summary_upto=upto, layer1_from=layer1_from, msg_id=msg_id
    )
    interrupted = False
    async for event in _stream_events(conversation_id, graph_input):
        if event.get("type") == "interrupt":
            interrupted = True
        yield event
    if not interrupted:
        await _after_turn(conversation_id, None)


async def stream_resume(conversation_id: str, resume_value) -> AsyncIterator[dict]:
    """interrupt 后续轮流式运行(供 /api/actions/resume 转 SSE 帧)。"""
    await ensure_conversation(conversation_id)
    interrupted = False
    async for event in _stream_events(conversation_id, Command(resume=resume_value)):
        if event.get("type") == "interrupt":
            interrupted = True
        yield event
    if not interrupted:
        await _after_turn(conversation_id, None)


async def ensure_conversation(conversation_id: str) -> None:
    """resume 前置校验:会话不存在按 404 语义抛出(必须在开流前,路由层才能回 404)。"""
    deps = get_deps()
    exists = await asyncio.to_thread(_conversation_exists, deps.session_factory, conversation_id)
    if not exists:
        raise ConversationNotFound(conversation_id)


async def _ensure_conversation(conversation_id: str) -> tuple[str, int, int, int]:
    """读会话分层上下文与锚点消息 id:(summary, summary_upto, layer1_from, msg_id)。

    msg_id 取该会话最新一条 user 消息(prepare_turn 刚落库的那条),构造
    ``db-{msg_id}`` 锚点进图;会话不存在按 404 语义抛出。"""
    deps = get_deps()

    def _work() -> tuple[str, int, int, int]:
        with deps.session_factory() as session:
            conversation = conversations_repository.get_conversation(session, conversation_id)
            if conversation is None:
                raise ConversationNotFound(conversation_id)
            msg_id = conversations_repository.latest_user_message_id(session, conversation_id)
            return (
                conversation.summary or "",
                conversation.summary_upto_msg_id or 0,
                conversation.layer1_from_msg_id or 0,
                int(msg_id or 0),
            )

    return await asyncio.to_thread(_work)


def _conversation_exists(session_factory, conversation_id: str) -> bool:
    with session_factory() as session:
        return conversations_repository.get_conversation(session, conversation_id) is not None


def _interrupt_payload(state: dict) -> dict | None:
    """从终态抽 interrupt 载荷(ch06 选单 / ch08 工单确认卡;无中断时 None)。"""
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", None)
    return value if isinstance(value, dict) else None


async def _stream_events(conversation_id: str, graph_input: dict) -> AsyncIterator[dict]:
    """驱动图流式运行,产出事件 dict。

    事件:type=delta|tool|citations|actions|interrupt|done
    - delta:agent_llm 的流式 token 按 langgraph_node 过滤(R1);确定性节点整块吐
    - tool:agent_llm 更新发 running(跳过拦截型工具),agent_tools 更新发终态
    - citations:流结束后按 [n] 角标 resolve(ch04 语义:只发真正被引用的子集)
    - actions:转人工/建工单/退款表单提议(SSE 新增帧)
    - interrupt:图暂停等待人工输入后本次流直接结束——没有 done 帧,前端凭
      interrupt 帧按 payload.type 渲染卡片(ch06 选单 / ch08 工单确认)
    """
    config = _graph_config(conversation_id)
    answer_parts: list[str] = []
    citations: list | None = None
    actions: list = []

    async for mode, chunk in get_graph().astream(graph_input, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    answer_parts.append(text)
                    yield {"type": "delta", "text": text}
            continue
        # ch06/ch08:interrupt 以 updates chunk 形式出现(chunk={"__interrupt__": (Interrupt, …)});
        # 载荷透传(select_order 选单卡 / confirm_ticket 工单确认卡),前端按 kind 渲染
        if isinstance(chunk, dict) and "__interrupt__" in chunk:
            payload = getattr(chunk["__interrupt__"][0], "value", None) or {}
            yield {"type": "interrupt", "payload": payload}
            return
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            if node in DETERMINISTIC_ANSWER_NODES and update.get("answer"):
                yield {"type": "delta", "text": update["answer"]}
            if node == "agent_llm":
                for call in _ai_tool_calls(update):
                    if call.get("name") not in INTERCEPTED_TOOLS:
                        yield {
                            "type": "tool",
                            "tool_call_id": call.get("id") or "",
                            "name": call.get("name"),
                            "status": "running",
                        }
            if node in CITATION_NODES and update.get("citations") is not None:
                citations = update["citations"]
            if node == "agent_tools":
                for message_ in update.get("messages", []):
                    name = getattr(message_, "name", None)
                    if name and name not in INTERCEPTED_TOOLS:
                        status = "failed" if getattr(message_, "status", "success") == "error" else "completed"
                        yield {
                            "type": "tool",
                            "tool_call_id": getattr(message_, "tool_call_id", ""),
                            "name": name,
                            "status": status,
                        }
            if update.get("suggested_actions"):
                actions = update["suggested_actions"]

    if citations:
        cited = citations_store.resolve_citations("".join(answer_parts), citations)
        if cited:
            yield {"type": "citations", "conversation_id": conversation_id, "items": cited}
    if actions:
        yield {"type": "actions", "items": actions}
    yield {"type": "done", "conversation_id": conversation_id}


async def get_turn_snapshot(conversation_id: str, turn_index: int) -> dict:
    """ch09 · 快照回捞:从 checkpoint 读第 turn_index 轮的 State 投影。

    前端把每轮回合号随 👎 传回来,后台凭 (conversation_id, turn_index) 从
    AsyncSqliteSaver checkpoint 里捞出当时的召回片段/引用/拒答来源——
    不依赖日志文件,也不需要把快照再存一遍。
    找不到会话/轮次抛 ConversationNotFound / TurnNotFound。
    (async:服务端运行在事件循环内,checkpoint 读取必须 await,不能用 asyncio.run)
    """
    checkpoint = await _load_checkpoint(conversation_id)
    if checkpoint is None:
        raise ConversationNotFound(conversation_id)
    # aget_tuple 返回 CheckpointTuple(对象属性,非 dict)
    state = (checkpoint.checkpoint.get("channel_values")
             if getattr(checkpoint, "checkpoint", None) else None) or {}
    messages = [m for m in (state.get("messages") or []) if isinstance(m, HumanMessage)]
    if not 1 <= turn_index <= len(messages):
        raise TurnNotFound(f"turn_index={turn_index} 超出范围(共 {len(messages)} 轮)")
    return {
        "query": state.get("query") or messages[turn_index - 1].text,
        "retrieved_snapshot": state.get("retrieved_snapshot") or [],
        "citations": state.get("citations") or [],
        "fallback_source": state.get("fallback_source") or "",
        "evidence_confidence": state.get("evidence_confidence") or 0.0,
        "answer": _extract_answer(messages, turn_index),
    }


async def _load_checkpoint(conversation_id: str) -> dict | None:
    """取该会话最新 checkpoint（含 channel_values）。"""
    checkpointer = _graph.checkpointer if _graph is not None else None
    if checkpointer is None:
        return None
    config = {"configurable": {"thread_id": conversation_id}}
    try:
        return await checkpointer.aget_tuple(config)
    except Exception as exc:
        logger.warning("checkpoint 读取失败(%s): %s", type(exc).__name__, exc)
        return None


def _extract_answer(messages: list[AnyMessage], turn_index: int) -> str:
    """该轮 HumanMessage 之后第一条 AI 消息文本(该轮的回答)。"""
    target = messages[turn_index - 1]
    seen = False
    for message in messages[turn_index - 1:]:
        if message is target:
            seen = True
            continue
        if seen and isinstance(message, AIMessage):
            return message.text()
    return ""


# ── ch09 · 用户反馈入口 ──────────────────────────────────────────────────────

async def record_feedback(
    conversation_id: str,
    turn_index: int,
    rating: str,
    comment: str = "",
) -> tuple[dict, bool]:
    """记一条 👍/👎:落池 source=user_feedback(飞轮第三入口),查重去重由飞轮流水线做。

    落池 reason 带 rating/comment/turn_index;快照从 checkpoint 回捞随条存
    retrieved_chunks。返回 (snapshot, created);会话不存在/轮次越界抛对应异常。
    """
    snapshot = await get_turn_snapshot(conversation_id, turn_index)
    raw_question = snapshot["query"]
    retrieved = snapshot["retrieved_snapshot"] or None
    reason = f"rating={rating} turn={turn_index}" + (f" comment={comment}" if comment else "")
    created = await asyncio.to_thread(
        _insert_low_confidence_sync,
        conversation_id, raw_question, "user_feedback", reason, retrieved,
    )
    return snapshot, created


def _insert_low_confidence_sync(
    conversation_id: str, raw_question: str, source: str, reason: str, snapshot: list | None
) -> bool:
    with get_deps().session_factory() as session:
        conversations_repository.get_conversation(session, conversation_id)  # FK:会话必须存在
        row = low_confidence_repository.fetch_latest_by_conversation(
            session, conversation_id, raw_question, source)
        if row is not None:
            return False  # 同轮重复点 👎 不重复入池
        low_confidence_repository.insert_low_confidence(
            session,
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
            retrieved_chunks=snapshot,
        )
        session.commit()
        return True


def _graph_config(conversation_id: str) -> dict:
    # ch09:metadata 带 langfuse_session_id——同一会话多轮在 Langfuse 界面串成一组
    # (LangChain/LangGraph 集成的 config.metadata 会成为 trace metadata;未启用 Langfuse 时多余键无害)
    return {
        "configurable": {"thread_id": conversation_id, "deps": get_deps()},
        "metadata": {"langfuse_session_id": str(conversation_id)},
    }


def _graph_input(conversation_id: str, message: str, *, summary: str = "", summary_upto: int = 0, layer1_from: int = 0, msg_id: int | None = None) -> dict:
    return {
        # ch07 · 锚点:HumanMessage 带 db-{msg_id},三段切窗靠它认轮次;
        # add_messages 同 id 覆盖语义靠 db-{msg_id} 的全局唯一性兜底
        "messages": [HumanMessage(message, id=f"db-{msg_id}" if msg_id else None)],
        "conversation_id": conversation_id,
        "query": message,
        # ch07 · 分层字段每轮从会话行重读(最后写入者胜)
        "summary": summary,
        "summary_upto_msg_id": summary_upto,
        "layer1_from_msg_id": layer1_from,
        # ch06 多轮字段每轮清零:改写/置信度/订单语境不复用上一轮
        "resolved_query": "",
        "intent_confidence": 0.0,
        "order_id": "",
        "order_data": {},
        "steps": 0,
        "tokens_used": 0,
        # ch09 闸与飞轮字段每轮清零:置信度/拒答来源/召回快照不复用上一轮
        "evidence_confidence": 0.0,
        "fallback_source": "",
        "retrieved_snapshot": [],
    }


# ── ch07 · 轮末分层结算 ───────────────────────────────────────────────────────


def _update_layer1_from(session_factory, conversation_id: str, msg_id: int) -> None:
    with session_factory() as session:
        conversations_repository.update_layer1_from(session, conversation_id, msg_id)
        session.commit()


async def settle_layers(conversation_id: str, state: dict | None = None) -> None:
    """轮末分层结算:层 1 超预算就整轮降级进层 2(只挪 layer1_from 锚点,不搬数据)。

    state=None 时自取 checkpoint 终态;新边界>旧边界才写库(边界只进不退);
    异常吞掉只留痕——分层是优化,不能影响当轮回复。
    """
    try:
        values = state
        if values is None:
            snapshot = await get_graph().aget_state(_graph_config(conversation_id))
            values = snapshot.values
        messages = list(values.get("messages") or [])
        if not messages:
            return
        summary_upto = values.get("summary_upto_msg_id") or 0
        layer1_from = values.get("layer1_from_msg_id") or 0
        settings = get_deps().settings
        result = budget_mod.compute(settings)
        _, l1_tokens = memory.layer_tokens(messages, summary_upto, layer1_from, settings)
        if l1_tokens <= result.layer1_budget:
            return
        new_boundary = memory.next_layer1_from(messages, summary_upto, result.layer1_budget, settings)
        if not new_boundary or new_boundary <= layer1_from:
            return
        await asyncio.to_thread(
            _update_layer1_from, get_deps().session_factory, conversation_id, new_boundary
        )
        logger.info(
            "层1 降级 conv=%s %d→%d 层1约%d token>预算%d",
            conversation_id, layer1_from, new_boundary, l1_tokens, result.layer1_budget,
        )
    except Exception as exc:
        logger.warning(
            "settle_layers 失败 conv=%s (%s: %s)", conversation_id, type(exc).__name__, exc
        )


async def _after_turn(conversation_id: str, state: dict | None) -> None:
    """轮末收尾:settle 先挪边界,摘要调度才能看到新鲜边界。均不阻塞当轮回复。

    摘要调度必须显式传应用 settings——不传会落 hermetic 默认(窗 0/模型 unavailable),
    预算恒 0 造成假阳性触发、真摘要必败。"""
    await settle_layers(conversation_id, state)
    await summarizer.maybe_schedule_summary(conversation_id, settings=get_deps().settings)


def _ai_tool_calls(update: dict) -> list[dict]:
    for message in reversed(update.get("messages", [])):
        calls = getattr(message, "tool_calls", None)
        if calls:
            return calls
    return []
