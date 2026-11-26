"""ch09 · 数据飞轮:标准化+查重一次输出(app/core/flywheel.py) + 审核服务函数。

流水线(课程定稿):知识缺口(待审队列)的价值 = 让「用户已经问过的」不再重复拒答。
一图三入口落池(ch05 闸落池 / ch06 interrupt 弃单 / ch09 👎),本模块消费:
  1. fetch_unmatched 捞未归并原话(matched_review_id IS NULL = 飞轮处理游标);
  2. LLM 结构化:标准化成 FAQ 问法 + 在候选缺口里查重归并,一次输出;
  3. 归并:命中缺口累次数不新建行(驳回即终审);新缺口建 review_queue 行;
  4. 审核通过写回知识库(复用 ch03 add_manual_entry:切块+指纹去重+pending+向量化)。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.repositories import knowledge as knowledge_repository
from app.repositories import review_queue as review_queue_repository
from app.settings import Settings

logger = logging.getLogger(__name__)


class FlywheelUnavailable(RuntimeError):
    """审核/飞轮服务不可用(依赖未初始化或写回失败),映射 503。"""


class FlywheelVerdict(BaseModel):
    """标准化+查重一次输出:模型只决定「问法」和「归并到谁」,落库由代码管。"""

    normalized_question: str = Field(min_length=1, max_length=120, description="标准化后的 FAQ 式问题")
    matched_id: int | None = Field(default=None, description="命中的候选缺口 ID,无则 null")
    suggested_answer: str = Field(default="", max_length=2000, description="示例答案备查")


NORMALIZE_PROMPT = """你是客服知识库运营。把用户的原始提问标准化成一条 FAQ 式问题,并在已有缺口列表里查重:
- 标准化:去掉口语/错别字/指代,保留关键实体(商品/政策点),不超过 60 字,以问号结尾;
- 查重:候选列表是历史缺口(ID + 标准化问题)。语义相同/同一直播主题就返回对应 matched_id;
  没有匹配的返回 null;
- suggested_answer:根据原话猜一个简短示例答案备查(30 字内,不确定就留空)。

历史缺口候选:
{candidates}

用户原话:{raw_question}

只输出 JSON。"""


def normalize_and_match(settings: Settings, raw_question: str, candidates: list[dict]) -> FlywheelVerdict:
    """真模型调用:结构化输出标准化问题+归并目标(单测 mock 本函数)。"""
    from langchain_core.prompts import ChatPromptTemplate

    from app.graph.llm import build_chat_model

    model = build_chat_model(settings)
    prompt = ChatPromptTemplate.from_template(NORMALIZE_PROMPT)
    chain = prompt | model.with_structured_output(FlywheelVerdict)
    verdict = chain.invoke({
        "candidates": "\n".join(f"- ID {c['id']}: {c['normalized_question']}" for c in candidates) or "（暂无）",
        "raw_question": raw_question,
    })
    if isinstance(verdict, dict):
        verdict = FlywheelVerdict(**verdict)
    return verdict


def process_one(session: Session, settings: Settings, raw) -> str:
    """处理一条原话:标准化+查重→归并/建缺口→set_matched_review。返回动作标签。"""
    candidates = review_queue_repository.list_review_candidates(session)
    verdict = normalize_and_match(settings, raw.raw_question, candidates)
    if not verdict.normalized_question.strip():
        raise ValueError("模型返回空标准化问题")
    if verdict.matched_id is not None:
        valid_ids = {c["id"] for c in candidates}
        if verdict.matched_id not in valid_ids:
            verdict.matched_id = None  # 模型幻觉 ID→按新缺口处理
    if verdict.matched_id is not None:
        review_queue_repository.increment_occurrence(session, verdict.matched_id)
        review_queue_repository.set_matched_review(session, raw.id, verdict.matched_id)
        return "merged"
    item = review_queue_repository.insert_review_item(
        session,
        normalized_question=verdict.normalized_question.strip(),
        ai_suggested_answer=(verdict.suggested_answer or "").strip() or None,
    )
    review_queue_repository.set_matched_review(session, raw.id, item.id)
    return "created"


def run_pipeline(
    session_factory: Callable[[], Session],
    settings: Settings,
    limit: int = 50,
) -> dict:
    """批处理入口(scripts/flywheel_pipeline.py 与 API 共用):游标推进,失败项留给下轮。"""
    merged = created = failed = 0
    failures: list[dict] = []
    with session_factory() as session:
        pending = review_queue_repository.fetch_unmatched(session, limit=limit)
        for raw in pending:
            try:
                action = process_one(session, settings, raw)
            except Exception as exc:  # 单条失败不拖垮整批,游标不动留给下轮
                failed += 1
                failures.append({"id": raw.id, "error": f"{type(exc).__name__}: {exc}"})
                session.rollback()
                logger.warning("flywheel 单条处理失败 id=%s: %s", raw.id, exc)
                continue
            if action == "merged":
                merged += 1
            else:
                created += 1
        session.commit()
    return {"processed": merged + created, "merged": merged, "created": created,
            "failed": failed, "failures": failures}


# ── 审核服务(review API 用;session 操作统一 to_thread,同步 Session 风格) ──────

def deps_from_app(app) -> tuple:
    deps = getattr(app.state, "graph_deps", None)
    if deps is None:
        from app.graph.runtime import get_deps
        deps = get_deps()
    return deps.session_factory, deps.settings


async def list_queue(deps, status: str | None = None) -> list[dict]:
    session_factory, _ = deps

    def _run() -> list[dict]:
        with session_factory() as session:
            rows = review_queue_repository.list_review_queue(session, status)
            return [
                {
                    "id": row.id,
                    "normalized_question": row.normalized_question,
                    "ai_suggested_answer": row.ai_suggested_answer,
                    "occurrence_count": row.occurrence_count,
                    "review_status": row.review_status,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    return await asyncio.to_thread(_run)


async def get_detail(deps, review_id: int):
    session_factory, _ = deps

    def _run():
        with session_factory() as session:
            return review_queue_repository.get_review_detail(session, review_id)

    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        raise FlywheelUnavailable(f"审核详情查询失败: {exc}") from exc


async def reject(deps, review_id: int, note: str) -> bool:
    session_factory, _ = deps

    def _run() -> bool:
        with session_factory() as session:
            updated = review_queue_repository.update_review_status(session, review_id, "驳回", note)
            session.commit()
            return updated

    return await asyncio.to_thread(_run)


async def approve_and_writeback(deps, review_id: int, approved_answer: str) -> dict:
    """通过 + 写回知识库:状态流转 → add_manual_entry(faq, vectorize) 复用 ch03 全链路。

    - 写回失败(chunk/向量化异常):状态已流转但知识库没有→视为审核失败,
      状态回滚为待审,抛 FlywheelUnavailable(503),审核人可重试;
    - 指纹幂等:重复通过同一缺口,add_manual_entry 指纹查重 skipped=1,不重复入库。"""
    session_factory, settings = deps

    def _load():
        with session_factory() as session:
            detail = review_queue_repository.get_review_detail(session, review_id)
            if detail is None:
                return None
            item = detail[0]
            return {"id": item.id, "question": item.normalized_question, "status": item.review_status}

    target = await asyncio.to_thread(_load)
    if target is None:
        raise FlywheelUnavailable("缺口不存在")
    if target["status"] != "待审":
        raise FlywheelUnavailable(f"仅「待审」可流转,当前 {target['status']}")

    def _flip_status():
        with session_factory() as session:
            updated = review_queue_repository.update_review_status(
                session, review_id, "通过", approved_answer)
            session.commit()
            return updated

    if not await asyncio.to_thread(_flip_status):
        raise FlywheelUnavailable("仅「待审」可流转")

    def _writeback() -> dict:
        from app.kb import pipeline as kb_pipeline
        content = f"### {target['question']}\n\n{approved_answer}"
        return kb_pipeline.add_manual_entry(
            session_factory, settings, content, "faq", vectorize=True
        )

    try:
        kb_result = await asyncio.to_thread(_writeback)
    except Exception as exc:
        logger.exception("审核写回知识库失败 review_id=%s", review_id)

        def _rollback():
            with session_factory() as session:
                row = session.get(_review_queue_model(), review_id)
                if row is not None:
                    row.review_status = "待审"
                    row.approved_answer = None
                session.commit()

        await asyncio.to_thread(_rollback)
        raise FlywheelUnavailable(f"知识库写回失败: {type(exc).__name__}: {exc}") from exc

    return {
        "status": "approved",
        "id": review_id,
        "kb": {
            "inserted": kb_result.get("inserted", 0),
            "skipped": kb_result.get("skipped", 0),
            "vectorized": kb_result.get("vectorized", 0),
            "note": kb_result.get("note"),
            "message": kb_result.get("message"),
        },
    }


def _review_queue_model():
    from app.models import ReviewQueue
    return ReviewQueue
