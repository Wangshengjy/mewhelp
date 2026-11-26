"""ch08 · query_faq(ch04 RAG 管线工具原样迁入)。

走检索管线(Query 理解成本不在此,检索+重排+自评多次上游往返),因此:
timeout=30s、max_retries=0(超时多半是上游卡死,重试徒增等待)。
运行期依赖(session_factory/settings/tool_call_id/conversation_id)由引擎注入。
"""

from typing import Annotated, Any

from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.kb import citations as citations_store
from app.kb import retrieval as kb_retrieval
from app.kb import selfcheck as selfcheck_module
from app.repositories import low_confidence as low_confidence_repository
from app.settings import Settings
from app.tools import registry

# ch04 · 在线证据条数上限(首尾组装后的 matches 数量)
MAX_ONLINE_MATCHES = 5

REFUSAL_LOW_CONFIDENCE_MESSAGE = (
    "知识库证据不足，无法可靠回答。请向用户如实说明「暂时没有查到相关信息」，"
    "并引导用户联系人工客服（App 内「我的」-「联系客服」）。禁止编造任何政策细节。"
)

MAX_KEYWORD_LENGTH = 128


class FaqArgs(BaseModel):
    """模型可见口径只有 keyword/strategy;其余全是运行期注入参数。"""

    model_config = ConfigDict(extra="forbid")

    keyword: Annotated[str, Field(min_length=1, max_length=MAX_KEYWORD_LENGTH, description="FAQ 检索问题或关键词")]
    strategy: Annotated[
        str | None,
        Field(
            default=None,
            pattern="^(vector|bm25|hybrid|hybrid_rerank)$",
            description="检索策略（可选，默认线上混合重排 hybrid_rerank）",
        ),
    ]
    conversation_id: Annotated[str | None, InjectedToolArg] = Field(default=None, description="当前会话 ID,由运行时注入,模型不可见")
    tool_call_id: Annotated[str | None, InjectedToolCallId] = Field(default=None, description="模型工具调用 ID,由运行时注入,模型不可见")
    session_factory: Annotated[Any, InjectedToolArg] = Field(default=None, description="DB 会话工厂,由运行时注入,模型不可见")
    settings: Annotated[Any, InjectedToolArg] = Field(default=None, description="应用配置,由运行时注入,模型不可见")

    @field_validator("keyword")
    @classmethod
    def keyword_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("visible tool arguments must not be blank")
        return value.strip()


@tool(args_schema=FaqArgs)
def query_faq(
    keyword: str,
    strategy: str | None = None,
    conversation_id: str | None = None,
    tool_call_id: str | None = None,
    session_factory=None,
    settings=None,
) -> str:
    """检索客服知识库（政策、规则、时效、费用、商品规格等）。用于政策规则类问题；可传完整问句。证据不足时返回 found=false，必须如实拒答并引导人工客服。"""
    # ch04:混合检索管线(Query 理解 → BM25+dense 双路 → RRF → 重排)+ 两道闸。
    # 出参 JSON 结构:matches 恒为 question/answer/category 三键,
    # 引用角标 [n] 的映射由后端按 tool_call_id 暂存,随 SSE citations 事件下发。
    kb_settings = settings if settings is not None else Settings.hermetic_defaults()
    effective_strategy = strategy or kb_settings.retrieval_strategy
    try:
        hits = kb_retrieval.search_knowledge(
            session_factory,
            kb_settings,
            keyword,
            strategy=effective_strategy,
            top_k=MAX_ONLINE_MATCHES,
        )
    except kb_retrieval.RetrievalNotConfiguredError:
        result = kb_retrieval.retrieve(session_factory, kb_settings, keyword, top_k=3)
        return _degraded_payload(keyword, result)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning("hybrid retrieval failed (%s); keyword fallback", type(exc).__name__)
        result = kb_retrieval.retrieve(session_factory, kb_settings, keyword, top_k=3)
        return _degraded_payload(keyword, result)

    # 机械闸:无命中 / 最高相关度低于阈值 → 拒答落池
    gate_score = _gate_score(hits, effective_strategy)
    if not hits or (gate_score is not None and gate_score < kb_settings.rerank_min_score):
        with session_factory() as session:
            low_confidence_repository.insert_low_confidence(
                session,
                conversation_id=conversation_id,
                raw_question=keyword,
                source="retrieval_low_conf",
                reason=_mechanical_reason(hits, gate_score, effective_strategy, kb_settings),
            )
            session.commit()
        citations_store.stash(tool_call_id, {"citations": [], "refused": True})
        payload: dict = {"keyword": keyword, "found": False, "matches": []}
        payload["message"] = f"（{effective_strategy}）{REFUSAL_LOW_CONFIDENCE_MESSAGE}"
        return _dumps(payload)

    # 语义闸:生成前结构化自评(扁平字段 useful + reason)
    evidence_texts = [f"{hit['section_path']}：{hit['question']} {hit['answer']}" for hit in hits]
    verdict = selfcheck_module.check_sufficient(
        selfcheck_module.default_model_fn(kb_settings),
        keyword,
        evidence_texts,
        temperature=kb_settings.self_check_temperature,
    )
    if not verdict["useful"]:
        with session_factory() as session:
            low_confidence_repository.insert_low_confidence(
                session,
                conversation_id=conversation_id,
                raw_question=keyword,
                source="self_check",
                reason=verdict["reason"] or "生成前自评判定证据不足以回答",
            )
            session.commit()
        citations_store.stash(tool_call_id, {"citations": [], "refused": True})
        payload = {"keyword": keyword, "found": False, "matches": []}
        payload["message"] = f"（self_check:{verdict['reason'] or '证据不足'}）{REFUSAL_LOW_CONFIDENCE_MESSAGE}"
        return _dumps(payload)

    # 通过两道闸:首尾组装 + 引用角标映射暂存(n = matches 顺序,1 起)
    matches_raw = kb_retrieval.arrange_head_tail(hits[:MAX_ONLINE_MATCHES])
    stash_payload = {
        "citations": [
            {
                "n": index + 1,
                "chunk_id": hit["id"],
                "title": hit["section_path"] or hit["category"] or "知识库",
                "question": hit["question"],
                "answer": hit["answer"],
                "content_type": hit["content_type"],
                "category": hit["category"],
                "score": hit.get("rerank_score") if hit.get("rerank_score") is not None else hit["score"],
            }
            for index, hit in enumerate(matches_raw)
        ],
        "refused": False,
    }
    citations_store.stash(tool_call_id, stash_payload)
    payload = {
        "keyword": keyword,
        "found": True,
        "matches": [
            {"question": hit["question"], "answer": hit["answer"], "category": hit["category"]}
            for hit in matches_raw
        ],
    }
    return _dumps(payload)


def _dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


def _gate_score(hits: list[dict], strategy: str) -> float | None:
    """可比量纲的闸门分数:重排分 > dense 余弦 > 无(BM25/RRF 分数不可比)。"""
    if not hits:
        return None
    first = hits[0]
    if first.get("rerank_score") is not None:
        return float(first["rerank_score"])
    if strategy == "vector":
        return float(first.get("score") or 0.0)
    return None  # bm25 / hybrid(RRF 融合分)只做「无命中」闸


def _mechanical_reason(hits: list[dict], gate_score: float | None, strategy: str, settings) -> str:
    if not hits:
        return f"检索无召回（strategy={strategy}）"
    return f"检索证据低置信：top={gate_score:.2f} 低于阈值 {settings.rerank_min_score:.2f}（strategy={strategy}）"


def _degraded_payload(keyword: str, result) -> str:
    matches = [
        {"question": match["question"], "answer": match["answer"], "category": match["category"]}
        for match in result.matches
    ]
    payload: dict = {"keyword": keyword, "found": bool(matches), "matches": matches}
    payload["message"] = (result.note or "向量检索不可用，已降级为关键词检索。") + (
        "未找到匹配的演示 FAQ，请勿编造政策。" if not matches else ""
    )
    return _dumps(payload)


registry.register(registry.spec_from_langchain_tool(query_faq, source="builtin", timeout=30.0, max_retries=0))
