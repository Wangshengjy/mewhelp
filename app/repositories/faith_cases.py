"""ch04 · faith_cases：编造个案台账的确定性方法（TDD）。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import FaithCase

STATUSES = ("未解决", "已解决", "无需解决")


def record_case(
    session: Session,
    *,
    eval_id: str,
    bucket: str,
    query: str,
    strategy: str,
    answer: str,
    reason: str,
    citations: list | None,
    judge_model: str | None,
    seen: int | None = None,
) -> FaithCase:
    """Upsert one fabricated case keyed by eval_id (repeat = update + count++).

    A resolved case judged fabricated again flips back to 未解决 (recurrence):
    ``resolution`` is cleared, ``resolved_at`` is kept to mark the recurrence.
    """
    row = session.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id))
    now = datetime.now(timezone.utc)
    if row is None:
        row = FaithCase(
            eval_id=eval_id,
            bucket=bucket,
            query=query,
            strategy=strategy,
            answer=answer,
            reason=reason,
            citations=citations,
            judge_model=judge_model,
            status="未解决",
            seen_count=1,
            first_seen_at=now,
            last_seen_at=now,
        )
        session.add(row)
        session.flush()
        return row

    row.bucket = bucket
    row.query = query
    row.strategy = strategy
    row.answer = answer
    row.reason = reason
    row.citations = citations
    row.judge_model = judge_model
    row.seen_count = max(1, int(seen) if seen is not None else row.seen_count + 1)
    row.last_seen_at = now
    if row.status != "未解决":
        row.status = "未解决"
        row.resolution = None
    session.flush()
    return row


def resolve_case(session: Session, *, eval_id: str, status: str, resolution: str) -> FaithCase:
    """Mark a case 已解决 / 无需解决 with a mandatory one-line explanation."""
    if status not in STATUSES or status == "未解决":
        raise ValueError(f"处置状态必须是 {'/'.join(STATUSES[1:])}，收到 {status!r}")
    row = session.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id))
    if row is None:
        raise ValueError(f"个案 {eval_id} 不存在")
    if not resolution or not str(resolution).strip():
        raise ValueError("处置必须填写一句说明")
    row.status = status
    row.resolution = str(resolution).strip()
    row.resolved_at = datetime.now(timezone.utc)
    session.flush()
    return row


def list_cases(session: Session, status: str | None = None, limit: int | None = None) -> list[FaithCase]:
    query = select(FaithCase).order_by(FaithCase.bucket.asc(), FaithCase.eval_id.asc())
    if status:
        query = query.where(FaithCase.status == status)
    if limit:
        query = query.limit(limit)
    return list(session.scalars(query).all())


def status_counts(session: Session) -> dict[str, int]:
    from sqlalchemy import func

    rows = session.execute(select(FaithCase.status, func.count()).group_by(FaithCase.status)).all()
    counts = {status: int(count) for status, count in rows}
    return {status: counts.get(status, 0) for status in STATUSES}


def case_to_dict(case: FaithCase) -> dict[str, Any]:
    """JSON-serializable projection for the /rag-eval console."""
    return {
        "id": case.id,
        "eval_id": case.eval_id,
        "bucket": case.bucket,
        "query": case.query,
        "strategy": case.strategy,
        "answer": case.answer,
        "reason": case.reason,
        "citations": case.citations or [],
        "judge_model": case.judge_model,
        "status": case.status,
        "seen_count": case.seen_count,
        "first_seen_at": case.first_seen_at.isoformat() if case.first_seen_at else None,
        "last_seen_at": case.last_seen_at.isoformat() if case.last_seen_at else None,
        "resolution": case.resolution,
        "resolved_at": case.resolved_at.isoformat() if case.resolved_at else None,
    }
