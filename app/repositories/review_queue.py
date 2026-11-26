"""ch09 · 数据飞轮 repository：待审队列（review_queue）+ 评估轮次（eval_runs）。

本仓库 repository 是同步 Session 风格（与 low_confidence.py 一致），由节点/脚本
拿 session 调用并自行 commit。归并语义（用户拍板）：查重候选 = review_queue 全部行；
命中任何状态的行都只累加 occurrence_count、状态不变（驳回即终审，不复活）。
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import EvalRun, LowConfidenceQuestion, ReviewQueue

STATUSES = ("待审", "通过", "驳回")
DEFAULT_CANDIDATE_LIMIT = 200


def fetch_unmatched(session: Session, limit: int = 50) -> list[LowConfidenceQuestion]:
    """飞轮处理游标：尚未归并（matched_review_id IS NULL）的池内问题，按 id 升序。"""
    return list(
        session.scalars(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id)
            .limit(limit)
        ).all()
    )


def list_review_candidates(session: Session, limit: int = DEFAULT_CANDIDATE_LIMIT) -> list[dict]:
    """查重候选：全部状态的缺口行（比全部，驳回即终审），updated_at 倒序截断防 token 爆。"""
    rows = session.execute(
        select(ReviewQueue.id, ReviewQueue.normalized_question)
        .order_by(ReviewQueue.updated_at.desc(), ReviewQueue.id.desc())
        .limit(limit)
    ).all()
    return [{"id": row.id, "normalized_question": row.normalized_question} for row in rows]


def insert_review_item(
    session: Session, *, normalized_question: str, ai_suggested_answer: str | None
) -> ReviewQueue:
    row = ReviewQueue(normalized_question=normalized_question, ai_suggested_answer=ai_suggested_answer)
    session.add(row)
    session.flush()
    return row


def increment_occurrence(session: Session, review_id: int) -> None:
    """查重命中：只累加次数，状态不动（命中已驳回/已通过也一样——驳回即终审）。"""
    row = session.get(ReviewQueue, review_id)
    if row is not None:
        row.occurrence_count = row.occurrence_count + 1
        session.flush()


def set_matched_review(session: Session, lcq_id: int, review_id: int) -> None:
    row = session.get(LowConfidenceQuestion, lcq_id)
    if row is not None:
        row.matched_review_id = review_id
        session.flush()


def list_review_queue(session: Session, status: str | None = None) -> list[ReviewQueue]:
    """审核页列表：按出现次数降序（频次=优先级），同频新的在前；status=None 全量。"""
    query = select(ReviewQueue).order_by(ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc())
    if status:
        query = query.where(ReviewQueue.review_status == status)
    return list(session.scalars(query).all())


def get_review_detail(session: Session, review_id: int) -> tuple[ReviewQueue, list[LowConfidenceQuestion]] | None:
    """详情：缺口行 + 归并进来的原话流水（带 source/快照，审核人判断真缺还是没检到）。"""
    item = session.get(ReviewQueue, review_id)
    if item is None:
        return None
    raws = list(
        session.scalars(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.id)
        ).all()
    )
    return item, raws


def update_review_status(
    session: Session, review_id: int, status: str, approved_answer: str | None = None
) -> bool:
    """仅「待审」可流转（通过/驳回都是终态）；返回是否真的更新了。"""
    row = session.get(ReviewQueue, review_id)
    if row is None or row.review_status != "待审":
        return False
    row.review_status = status
    if approved_answer is not None:
        row.approved_answer = approved_answer
    session.flush()
    return True


def insert_eval_run(session: Session, *, triggered_by: str, dataset_size: int, metrics: dict) -> EvalRun:
    row = EvalRun(triggered_by=triggered_by, dataset_size=dataset_size, metrics=metrics)
    session.add(row)
    session.flush()
    return row


def list_eval_runs(session: Session, limit: int = 10) -> list[EvalRun]:
    """评估趋势：最近 N 轮，最新在前。"""
    return list(
        session.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit)).all()
    )
