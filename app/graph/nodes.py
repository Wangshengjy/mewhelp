"""ch05/ch06 · 图节点:意图/检索/Agent ReAct 环 + 确定性出口 + 退款接管链 + 日志审计。

依赖注入:节点统一 ``(state, config=None)``,从 ``config["configurable"]["deps"]``
取 GraphDeps(session_factory/settings/chat_model_factory)——本仓库 Settings 是
load() 出来的 dataclass、repository 是同步 Session 风格,不能像课程参考那样用
模块级单例直连。同步 DB/检索调用一律 ``asyncio.to_thread`` 包裹,不阻塞事件循环。

ch06 新增:coref 改写(resolved_query)、八类意图+置信度、fetch_order(interrupt
选单)、retrieve_policy(多路扩展检索)、script_reply 脚本出口、submit_refund
拦截(前端退款表单确认后落库)。
"""
import asyncio
import json
import logging
import re
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from app.core import memory
from app.core.confidence import compute_evidence_confidence, snapshot_from_hits
from app.core.observability import tag_intent
from app.graph.coref import resolve as coref_resolver
from app.graph.deps import GraphDeps, deps_from_config
from app.graph.intent import classify as classify_intent_label
from app.graph.routing import INTENT_TO_ROUTE
from app.kb import retrieval as kb_retrieval
from app.kb import selfcheck as selfcheck_module
from app.kb.query_understanding import expand_queries as query_expander
from app.prompts import (
    CHAT_SYSTEM_PROMPT,
    CHITCHAT_REPLY_TEXT,
    COMPLAINT_REPLY_TEXT,
    FALLBACK_REPLY_TEXT,
    REFUND_JUDGE_HINT,
    SCRIPT_REPLY_CHITCHAT,
    SCRIPT_REPLY_OTHER,
)
from app.repositories import conversations as conversations_repository
from app.repositories import low_confidence as low_confidence_repository
from app.tools import engine
from app.tools import mock_data
from app.tools import registry as tool_registry

logger = logging.getLogger(__name__)

# 三条固定话术(nodes.py 暴露常量,测试与前端文案单一来源在 app/prompts.py)
CHITCHAT_REPLY = CHITCHAT_REPLY_TEXT
COMPLAINT_REPLY = COMPLAINT_REPLY_TEXT
FALLBACK_REPLY = FALLBACK_REPLY_TEXT

# ch04 在线证据条数上限(首尾组装后的条数),照搬进图
MAX_ONLINE_MATCHES = 5

# 知识路进 agent_llm 时上下文已带编号证据:证据已给,别再检索;仍可自调业务工具
_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据(请据此作答,每个关键结论后标注来源编号如[1];"
    "证据已给,不要再调用 query_faq;仍可按需调用订单/物流等工具)\n"
)

# ch06 · 演示订单号是 4 位起:取文中第一个前后都不是数字的 4+ 位数字串
# (\b 在中文邻接数字时无效——「单1001」两侧都是 \w 没有词边界)
_ORDER_ID_RE = re.compile(r"(?<!\d)\d{4,}(?!\d)")


# ── 入口段 ────────────────────────────────────────────────────────────────────


def _user_text(state) -> str:
    """取最后一条 human 文本(整轮只处理一句;多意图留 ch06)。"""
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _history_text(state, max_messages: int = 6) -> str:
    """摘要行 + 窗内最近几条 human/ai 文本,coref/意图分类共用。

    ch07:历史先过 build_window(已摘要段不出现、层 2 半压),再取尾部;
    摘要行放最前,让「最开始那个订单」这类远期指代有据可改。全链路留痕 history_ctx。
    """
    window = memory.build_window(
        list(state.get("messages", [])),
        state.get("summary_upto_msg_id") or 0,
        state.get("layer1_from_msg_id") or 0,
    )
    history_parts: list[str] = []
    for message in window:
        content = message.content
        text = content if isinstance(content, str) else ""
        text = text.strip()
        if not text:
            continue
        if isinstance(message, HumanMessage):
            history_parts.append(f"用户:{text}")
        elif isinstance(message, AIMessage):
            parts_text = f"客服:{text}"
            history_parts.append(parts_text)
    tail = history_parts[-max_messages:]
    summary_line = memory.summary_line(state.get("summary"))
    if summary_line:
        tail = [summary_line, *tail]
    text_out = "\n".join(tail)
    logger.info(
        "history_ctx conv=%s window=%d 输出=%d字 text=%s",
        state.get("conversation_id"),
        len(window),
        len(text_out),
        text_out[:300],
    )
    return text_out


async def coref(state, config=None) -> dict:
    """指代消解(ch06 做实):LLM 把「这个能退吗」改写成独立问句,写 resolved_query。

    query 保持原文;改写失败/无指代时 resolved_query == query(节点内部规则短路)。"""
    deps = deps_from_config(config)
    query = state.get("query") or _user_text(state)
    resolved = await coref_resolver(query, _history_text(state), deps.settings, deps.chat_model_factory)
    mode = "rewrite" if resolved != query else "passthrough"
    return {"resolved_query": resolved, "trace": {"coref": mode}}


async def classify_intent(state, config=None) -> dict:
    """意图识别(ch06 v2):吃 coref 改写后的问句 + 最近历史,输出八类标签与置信度。

    分类器异常/越界由 intent.classify 兜底回「其他」。route 也在此落进 State
    (条件边函数只选边不写状态,下游节点要用 route 判路)。"""
    deps = deps_from_config(config)
    query = state.get("resolved_query") or _user_text(state)
    verdict = await classify_intent_label(
        query, deps.settings, deps.chat_model_factory, history=_history_text(state)
    )
    intent, confidence = verdict["intent"], verdict["confidence"]
    # ch09:意图进当前 trace 的 metadata+tag(未配 Langfuse 时静默跳过),Cost Control 按意图分堆
    tag_intent(intent, confidence)
    route = INTENT_TO_ROUTE.get(intent, "business")
    return {
        "intent": intent,
        "intent_confidence": confidence,
        "route": route,
        "trace": {"intent": intent, "intent_confidence": confidence, "route": route},
    }


# ── 确定性出口(零模型调用) ──────────────────────────────────────────────────
# ch06 退役 chitchat_reply 节点:闲聊并入 script_reply(见下);
# CHITCHAT_REPLY 常量保留(eval_ch05 与文案单一来源仍引用)。


async def complaint_reply(state, config=None) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个可选项(后端不执行,交前端自选)。"""
    actions = [
        {"type": "transfer_human"},
        {
            "type": "create_ticket",
            "draft": {"description": _user_text(state), "ticket_type": "投诉"},
        },
    ]
    return {
        "answer": COMPLAINT_REPLY,
        "suggested_actions": actions,
        "trace": {"route": "complaint"},
    }


async def fallback_reply(state, config=None) -> dict:
    """置信度兜底:证据弱回固定话术。

    低置信落池由 forced_rag 的两道闸负责(retrieval_low_conf / self_check 各一行,
    带 reason 与 evidence_top),这里不重复落池,只补 route 痕迹。
    """
    return {"answer": FALLBACK_REPLY, "trace": {"route": "fallback"}}


async def script_reply(state, config=None) -> dict:
    """脚本出口(ch06):闲聊/其他零模型调用,按意图二选一固定话术。

    取代 ch05 的 chitchat_reply:新增「其他」类,给礼貌兜底 + 引导二选一,
    不承诺任何业务办理。"""
    if state.get("intent") == "闲聊":
        return {"answer": SCRIPT_REPLY_CHITCHAT, "trace": {"route": "fallback_script", "script": "chitchat"}}
    return {"answer": SCRIPT_REPLY_OTHER, "trace": {"route": "fallback_script", "script": "other"}}


# ── 知识路:强制检索 + 生成前证据闸 ───────────────────────────────────────────


async def forced_rag(state, config=None) -> dict:
    """知识类强制检索(复用 ch03/04 检索器):产出编号证据 + 证据强弱信号。

    复刻 query_faq 的两道生成前证据闸(ch09 把机械闸升级为四信号置信度闸):
    - 置信度闸:evidence_confidence(四信号加权,阈值评估集校准)低于
      settings.evidence_confidence_threshold → evidence_strong=False,
      落池 source=retrieval_low_conf,reason 带四信号值;
    - 语义闸:selfcheck 自评"证据够不够回答" → 不够则 evidence_strong=False,
      落池 source=self_check;
    - 快照规则:只要走了检索就把 Top3 快照写进 state.retrieved_snapshot——
      闸不过随落池写 retrieved_chunks;闸都过留给用户事后点 👎 时快照回捞;
    - 双闸都过:首尾组装 + 编号证据 + 引用元数据进 state.citations。
    """
    deps = deps_from_config(config)
    query_raw = state.get("query") or _user_text(state)
    conversation_id = state.get("conversation_id")
    trace: dict = {"forced_rag": True, "evidence_top": 0.0}

    try:
        hits = await asyncio.to_thread(
            kb_retrieval.search_knowledge,
            deps.session_factory,
            deps.settings,
            query_raw,
            strategy=deps.settings.retrieval_strategy,
            top_k=MAX_ONLINE_MATCHES,
        )
    except kb_retrieval.RetrievalNotConfiguredError:
        logger.info("hybrid retrieval unconfigured; keyword fallback")
        return await _degraded_rag(deps, query_raw, trace, "检索未配置,已降级关键词查表")
    except Exception as exc:
        logger.warning("hybrid retrieval failed (%s); keyword fallback", type(exc).__name__)
        return await _degraded_rag(deps, query_raw, trace, f"检索暂不可用({type(exc).__name__}),已降级关键词查表")

    conf = compute_evidence_confidence(hits)
    snapshot = snapshot_from_hits(hits)
    trace["evidence_top"] = conf.signals["top1_score"]
    trace["evidence_confidence"] = conf.score
    trace["confidence_signals"] = conf.signals

    def _pool(source: str, reason: str) -> None:
        if conversation_id:
            asyncio.get_running_loop().run_in_executor(
                None, _insert_low_confidence_snapshot,
                deps.session_factory, conversation_id, query_raw,
                source, reason, snapshot or None,  # 无召回空快照存 NULL,与 DDL 注释一致
            )

    # 置信度闸(ch09 原 ch05 机械闸升级):四信号加权总分低于校准阈值 → 证据弱,拒答落池
    if not hits or conf.score < deps.settings.evidence_confidence_threshold:
        reason = (
            f"evidence_confidence={conf.score:.3f} 低于阈值 {deps.settings.evidence_confidence_threshold:.2f} "
            f"signals={json.dumps(conf.signals, ensure_ascii=False)}"
            if hits else f"检索无召回（strategy={deps.settings.retrieval_strategy}）"
        )
        _pool("retrieval_low_conf", reason)
        return {
            "evidence": "",
            "evidence_strong": False,
            "citations": [],
            "evidence_confidence": conf.score,
            "fallback_source": "retrieval_low_conf",
            "retrieved_snapshot": snapshot,
            "trace": {**trace, "gate": "mechanical", "gate_reason": reason},
        }

    # 语义闸:生成前结构化自评(扁平字段 useful + reason)
    evidence_texts = [f"{hit['section_path']}：{hit['question']} {hit['answer']}" for hit in hits]
    verdict = await asyncio.to_thread(
        selfcheck_module.check_sufficient,
        selfcheck_module.default_model_fn(deps.settings),
        query_raw,
        evidence_texts,
        temperature=deps.settings.self_check_temperature,
    )
    if not verdict["useful"]:
        reason = (
            f"{verdict['reason'] or '生成前自评判定证据不足以回答'} "
            f"evidence_confidence={conf.score:.3f} "
            f"signals={json.dumps(conf.signals, ensure_ascii=False)}"
        )
        _pool("self_check", reason)
        return {
            "evidence": "",
            "evidence_strong": False,
            "citations": [],
            "evidence_confidence": conf.score,
            "fallback_source": "self_check",
            "retrieved_snapshot": snapshot,
            "trace": {**trace, "gate": "self_check", "gate_reason": reason},
        }

    # 双闸都过:首尾组装 + 编号证据 + 引用角标元数据(角标渲染在后端 resolve)
    matches_raw = kb_retrieval.arrange_head_tail(hits[:MAX_ONLINE_MATCHES])
    evidence = "\n".join(
        f"[{index + 1}] {hit['section_path']}：{hit['question']} {hit['answer']}"
        for index, hit in enumerate(matches_raw)
    )
    citations = [_citation_entry(index + 1, hit) for index, hit in enumerate(matches_raw)]
    return {
        "evidence": evidence,
        "evidence_strong": True,
        "citations": citations,
        "evidence_confidence": conf.score,
        "retrieved_snapshot": snapshot,
        "trace": {**trace, "gate": "pass"},
    }


async def _degraded_rag(deps: GraphDeps, query_raw: str, trace: dict, note: str) -> dict:
    """检索不可用/未配置时的降级路径(对齐 ch04 query_faq):关键词查表,
    结果直接作证据;ch04 降级不走两道闸也不落池,这里保持一致。"""
    result = await asyncio.to_thread(
        kb_retrieval.retrieve, deps.session_factory, deps.settings, query_raw, 3
    )
    matches = result.matches
    if not matches:
        return {
            "evidence": "",
            "evidence_strong": False,
            "citations": [],
            "trace": {**trace, "gate": "degraded_empty", "gate_reason": note},
        }
    evidence = "\n".join(
        f"[{index + 1}] {match['question']} {match['answer']}"
        for index, match in enumerate(matches)
    )
    citations = [
        {
            "n": index + 1,
            "chunk_id": match.get("id"),
            "title": match.get("section_path") or match.get("category") or "知识库",
            "question": match["question"],
            "answer": match["answer"],
            "content_type": match.get("content_type") or "",
            "category": match.get("category") or "",
            "score": match.get("rerank_score") if match.get("rerank_score") is not None else match.get("score"),
        }
        for index, match in enumerate(matches)
    ]
    return {
        "evidence": evidence,
        "evidence_strong": True,
        "citations": citations,
        "trace": {**trace, "gate": "degraded", "gate_reason": note},
    }


async def confidence_check(state, config=None) -> dict:
    """生成前证据闸的实体节点:留痕门控决策供可观测;强/弱的实际分流在其后的
    条件边 confidence_gate(读 evidence_strong)。"""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": {"confidence": decision}}


def _citation_entry(number: int, hit: dict) -> dict:
    """编号引用元数据(与 ch04 query_faq 的 stash 结构保持一致)。"""
    return {
        "n": number,
        "chunk_id": hit.get("id"),
        "title": hit.get("section_path") or hit.get("category") or "知识库",
        "question": hit["question"],
        "answer": hit["answer"],
        "content_type": hit.get("content_type") or "",
        "category": hit.get("category") or "",
        "score": hit.get("rerank_score") if hit.get("rerank_score") is not None else hit.get("score"),
    }


def _insert_low_confidence(session_factory, conversation_id, raw_question, source, reason) -> None:
    with session_factory() as session:
        low_confidence_repository.insert_low_confidence(
            session,
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
        )
        session.commit()


def _insert_low_confidence_snapshot(
    session_factory, conversation_id, raw_question, source, reason, snapshot
) -> None:
    """ch09:落池带召回快照(retrieved_chunks 列,审核页给人看;线程池执行,同步 Session 风格)。"""
    with session_factory() as session:
        low_confidence_repository.insert_low_confidence(
            session,
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
            retrieved_chunks=snapshot,
        )
        session.commit()


# ── 退款接管链(ch06):抓订单 → 多路检索政策 → 带证据判定 ─────────────────────


def _extract_order_id(text: str) -> str | None:
    """从问句抽演示订单号:第一个前后都不是数字的 4+ 位数字串。"""
    match = _ORDER_ID_RE.search(text or "")
    return match.group(0) if match else None


async def fetch_order(state, config=None) -> dict:
    """退款路第一步:定位订单号并取快照。

    顺序:state.order_id → resolved_query 里抽数字;都缺则 ``interrupt`` 暂停图、
    把 list_user_orders 的卡片数据抛给前端选单(人工接管);resume 后本节点从头
    重跑,interrupt() 返回所选单号,继续取快照写 order_id/order_data。"""
    order_id = state.get("order_id") or _extract_order_id(
        state.get("resolved_query") or _user_text(state)
    )
    interrupted = False
    if not order_id:
        orders = await asyncio.to_thread(mock_data.list_user_orders, state.get("user_id") or "")
        order_id = interrupt({"type": "select_order", "orders": orders})
        interrupted = True
    order_id = str(order_id)
    order_data = await asyncio.to_thread(mock_data.build_order, order_id)
    return {
        "order_id": order_id,
        "order_data": order_data,
        "trace": {"fetch_order": {"order_id": order_id, "interrupted": interrupted}},
    }


async def retrieve_policy(state, config=None) -> dict:
    """退款路第二步:多路扩展检索退款政策,产出编号证据(比 forced_rag 多两件事)。

    - 查询扩展:expand_queries 把「能退吗」改成 3 条检索友好问法,多路召回;
    - 去重合并:按 hit id 去重、rerank_score(缺则 score)最高保留,首尾组装;
    - 不做证据闸:退款判定宁可给足证据让 agent 判,闸门拒答在这里等于把用户
      直接推进兜底,不如把「证据不足」留给判定提示词如实输出。
    """
    deps = deps_from_config(config)
    base_query = state.get("resolved_query") or _user_text(state)
    order_data = state.get("order_data") or {}
    seed = f"{base_query} {order_data.get('status', '')}".strip()
    queries = await query_expander(seed, deps.settings, deps.chat_model_factory)

    merged: dict[str, dict] = {}
    merged_score: dict[str, float] = {}
    for query in queries:
        try:
            hits = await asyncio.to_thread(
                kb_retrieval.search_knowledge,
                deps.session_factory,
                deps.settings,
                query,
                strategy=deps.settings.retrieval_strategy,
                top_k=MAX_ONLINE_MATCHES,
            )
        except Exception as exc:
            # 单路挂了不炸链:另一路照样有召回(可用性优先)
            logger.warning("retrieve_policy 查询 %r 检索失败(%s);跳过该路", query, type(exc).__name__)
            continue
        for hit in hits:
            score = hit.get("rerank_score") if hit.get("rerank_score") is not None else float(hit.get("score") or 0.0)
            if hit["id"] not in merged_score or score > merged_score[hit["id"]]:
                merged[hit["id"]] = hit
                merged_score[hit["id"]] = score

    ranked = [merged[hit_id] for hit_id in sorted(merged_score, key=merged_score.get, reverse=True)]
    arranged = kb_retrieval.arrange_head_tail(ranked)
    citations = [_citation_entry(index + 1, hit) for index, hit in enumerate(arranged)]
    evidence = "\n".join(f"[{entry['n']}] {entry['question']}：{entry['answer']}" for entry in citations)
    return {
        "evidence": evidence,
        "citations": citations,
        "trace": {"retrieve_policy": {"queries": queries, "hits": len(arranged)}},
    }


# ── 主力 Agent:ReAct 环 ──────────────────────────────────────────────────────


def _agent_messages(state, settings=None) -> list:
    """ch07 组装:system 人设红线打头(整条只此一条 system)→ 三段切窗 → 本轮材料。

    可变内容(摘要/证据/订单快照)一律不进 system——上游模板会把所有 system
    上提合并渲染,前缀缓存实测归零。本轮材料合成一条 turn_ctx HumanMessage,
    插在窗内最后一条用户消息之后:模型视角里"用户附上的参考资料"。
    """
    window = memory.build_window(
        list(state.get("messages", [])),
        state.get("summary_upto_msg_id") or 0,
        state.get("layer1_from_msg_id") or 0,
        max_tokens=memory.window_budget(settings),
        settings=settings,
    )
    return [SystemMessage(content=CHAT_SYSTEM_PROMPT), *_with_turn_context(window, _turn_context(state))]


def _turn_context(state) -> str:
    """本轮材料文本块:摘要投影 + 编号证据 + 订单快照/判定指令(有啥拼啥)。"""
    parts: list[str] = []
    summary_line = memory.summary_line(state.get("summary"))
    if summary_line:
        parts.append(summary_line)
    if state.get("evidence"):
        parts.append(_KNOWLEDGE_EVIDENCE_HINT.strip() + "\n" + state["evidence"])
    if state.get("route") == "refund_flow":
        order_data = state.get("order_data") or {}
        snapshot = json.dumps(order_data, ensure_ascii=False) if order_data else "(未取到订单快照)"
        parts.append(f"## 订单快照\n{snapshot}")
        parts.append(REFUND_JUDGE_HINT.strip())
    return "\n\n".join(parts)


def _with_turn_context(window: list[BaseMessage], turn_ctx: str) -> list[BaseMessage]:
    """把本轮材料插到窗内最后一条用户消息之后(没有用户消息就补在尾部)。"""
    if not turn_ctx:
        return window
    material = HumanMessage(content=turn_ctx)
    last_human = None
    for index, message in enumerate(window):
        if isinstance(message, HumanMessage):
            last_human = index
    if last_human is None:
        return [*window, material]
    return [*window[: last_human + 1], material, *window[last_human + 1 :]]


def _log_model_context(state, messages: list[BaseMessage]) -> None:
    """模型调用前留痕:窗口规模 + 逐条预览,log/app.log 可 grep model_ctx。"""
    previews = " | ".join(
        f"{type(message).__name__}:{(message.content if isinstance(message.content, str) else '')[:40]}"
        for message in messages
    )
    logger.info(
        "model_ctx conv=%s summary=%r msgs=%d items=[%s]",
        state.get("conversation_id"),
        (state.get("summary") or "")[:80],
        len(messages),
        previews,
    )


async def agent_llm(state, config=None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。

    ch08:工具清单 = 内置 + MCP 现问现拿合并(get_all_specs),主力 Agent 一视同仁;
    MCP Server 不可达时自动只上内置。"""
    deps = deps_from_config(config)
    specs = await tool_registry.get_all_specs(deps.settings)
    model = deps.chat_model_factory(deps.settings).bind_tools([spec.tool for spec in specs.values()])
    messages = _agent_messages(state, deps.settings)
    _log_model_context(state, messages)
    ai: AIMessage = await model.ainvoke(messages, config)
    used = (getattr(ai, "usage_metadata", None) or {}).get("total_tokens", 0)
    return {
        "messages": [ai],
        "steps": state.get("steps", 0) + 1,
        "tokens_used": state.get("tokens_used", 0) + used,
    }


async def agent_tools(state, config=None) -> dict:
    """ReAct 行动步:普通工具走 ch08 统一执行引擎(校验/权限门/超时重试/分诊/审计);
    create_ticket 走「信息追问 → interrupt 工单预览 → 确认放行/取消拒绝」确认流;
    submit_refund(ch06)拦截为『提议』转前端退款表单。

    确认流复用 ch06 interrupt/resume 钉形状:interrupt() 暂停图,前端确认/取消经
    /api/actions/resume 回灌 Command(resume={"confirmed": bool})。放行时带
    confirmed=True 过引擎权限门(模型绕不过);取消走权限拒绝路径并落审计。"""
    deps = deps_from_config(config)
    last = state["messages"][-1]
    specs = await tool_registry.get_all_specs(deps.settings)
    conversation_id = state.get("conversation_id") or ""
    tool_msgs: list[ToolMessage] = []
    actions = list(state.get("suggested_actions", []))
    for call in last.tool_calls:
        if call.get("name") == "create_ticket":
            tool_msgs.append(await _confirm_ticket(call, conversation_id, deps, specs))
            continue
        if call.get("name") == "submit_refund":
            args = call.get("args") or {}
            actions.append({"type": "refund_form", "order_id": str(args.get("order_id") or "")})
            tool_msgs.append(
                ToolMessage(
                    content=(
                        "已把『退款确认表单』呈现给用户,用户在前端确认后才会真正提交退款申请。"
                        "请用一句话说明判定依据(引用证据编号)并停止,不要再调用任何工具。"
                    ),
                    tool_call_id=call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    name="submit_refund",
                )
            )
            continue
        usable = _with_usable_call_id(call)
        run = await engine.execute_tool_call(
            usable,
            conversation_id,
            specs,
            session_factory=deps.session_factory,
            settings=deps.settings,
        )
        tool_msgs.append(run.tool_message)
    out: dict = {"messages": tool_msgs}
    if actions:
        out["suggested_actions"] = actions
    return out


async def _confirm_ticket(call: dict, conversation_id: str, deps: GraphDeps, specs: dict) -> ToolMessage:
    """ch08 · create_ticket 确认流(单次调用完整走完,返回 ToolMessage)。

    ① 参数不齐:校验错误回灌,让模型向用户追问,不瞎编;
    ② 参数齐:interrupt 推工单预览卡(类型+描述),等用户确认/取消;
    ③ 确认:引擎带确认令牌放行落 tickets(回复带工单号);
       取消/未确认:不带令牌走权限门,拒绝并落审计(权限拒绝)。"""
    tc_id = call.get("id") or f"call_{uuid.uuid4().hex[:24]}"
    args = dict(call.get("args") or {})
    spec = specs.get("create_ticket")
    error = engine.validate_args(spec, args) if spec else "create_ticket 未登记"
    if error is not None:
        return ToolMessage(
            content=(
                f"工单创建失败:参数校验未通过({error})。"
                "请向用户追问缺失信息(问题描述/工单类型),不要编造后再调用。"
            ),
            tool_call_id=tc_id,
            name="create_ticket",
            status="error",
        )

    answer = interrupt(
        {
            "type": "confirm_ticket",
            "preview": {
                "ticket_type": args.get("ticket_type", "咨询"),
                "description": args.get("description", ""),
            },
        }
    )
    confirmed = isinstance(answer, dict) and answer.get("confirmed") is True
    run = await engine.execute_tool_call(
        {"name": "create_ticket", "args": args, "id": tc_id},
        conversation_id,
        specs,
        session_factory=deps.session_factory,
        settings=deps.settings,
        confirmed=confirmed,
        deny_note="用户未确认建工单,已按权限拒绝;请如实告知用户工单未创建。",
    )
    if confirmed:
        logger.info("ch08 工单确认放行 conv=%s tc=%s", conversation_id, tc_id)
    return run.tool_message


def _with_usable_call_id(call: dict) -> dict:
    """个别上游可能给空 tool_call_id;executor 校验非空,这里兜底生成。"""
    call = dict(call)
    if not call.get("id"):
        call["id"] = f"call_{uuid.uuid4().hex[:24]}"
    return call


# ── 日志审计 ─────────────────────────────────────────────────────────────────


def resolve_answer(state) -> str:
    """最终答复:确定性节点写在 state['answer'];Agent 答复取最后一条 AIMessage 文本。"""
    if state.get("answer"):
        return state["answer"]
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            content = message.content
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _append_assistant_message(session_factory, conversation_id: str, answer: str) -> None:
    with session_factory() as session:
        conversations_repository.append_message(
            session, conversation_id, role="assistant", content=answer or ""
        )
        session.commit()


async def log_node(state, config=None) -> dict:
    """日志记录:留痕 intent/route/trace(可观测地基),并落库一条 assistant 消息(审计)。

    ch04 的 tool 轨迹 MySQL 行不再单独落:完整消息链(含工具往返)由
    AsyncSqliteSaver 承载,MySQL messages 表只保最终答复审计。
    """
    logger.info(
        "ch05 turn conv=%s intent=%s route=%s steps=%s tokens=%s trace=%s",
        state.get("conversation_id"),
        state.get("intent"),
        state.get("route"),
        state.get("steps"),
        state.get("tokens_used"),
        state.get("trace", {}),
    )
    deps = deps_from_config(config)
    conversation_id = state.get("conversation_id")
    if conversation_id:
        await asyncio.to_thread(
            _append_assistant_message, deps.session_factory, conversation_id, resolve_answer(state)
        )
    return {}
