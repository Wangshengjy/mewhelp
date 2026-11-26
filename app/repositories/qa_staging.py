"""Repository for qa_extraction_staging: batched QA mining with human review."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import QaExtractionStaging

STATUS_EXTRACTED = "extracted"
STATUS_KEPT = "kept"
STATUS_DISCARDED = "discarded"
STATUS_REJECTED = "rejected"
STATUS_APPROVED = "approved"

ALL_STATUSES: tuple[str, ...] = (
    STATUS_EXTRACTED,
    STATUS_KEPT,
    STATUS_DISCARDED,
    STATUS_REJECTED,
    STATUS_APPROVED,
)


def insert_extracted(
    session: Session, *, batch_no: str, source_ref: str | None, question: str, answer: str
) -> QaExtractionStaging:
    """Insert one mined QA pair as ``extracted`` inside the given batch."""
    row = QaExtractionStaging(batch_no=batch_no, source_ref=source_ref, question=question, answer=answer)
    session.add(row)
    session.flush()
    return row


def get_staging_row(session: Session, row_id: int) -> QaExtractionStaging | None:
    return session.get(QaExtractionStaging, row_id)


def latest_batch_no(session: Session) -> str | None:
    """Newest batch number (mine-YYYYmmddHHMMSS sorts lexically by time)."""
    return session.execute(
        select(QaExtractionStaging.batch_no).order_by(QaExtractionStaging.batch_no.desc()).limit(1)
    ).scalar_one_or_none()


def list_by_batch(session: Session, batch_no: str, *, status: str | None = None) -> list[QaExtractionStaging]:
    """Rows of exactly one batch (dedup must never sweep stale batches in)."""
    statement = select(QaExtractionStaging).where(QaExtractionStaging.batch_no == batch_no)
    if status:
        statement = statement.where(QaExtractionStaging.status == status)
    statement = statement.order_by(QaExtractionStaging.id)
    return list(session.scalars(statement).all())


def list_staging(session: Session, *, statuses: Sequence[str] | None = None) -> list[QaExtractionStaging]:
    """Rows for the review card, newest batch first."""
    statement = select(QaExtractionStaging).order_by(QaExtractionStaging.batch_no, QaExtractionStaging.id)
    if statuses:
        statement = statement.where(QaExtractionStaging.status.in_(list(statuses)))
    return list(session.scalars(statement).all())


def set_status(session: Session, row_id: int, status: str) -> QaExtractionStaging | None:
    """Transition one staging row (kept/discarded/rejected/approved)."""
    row = session.get(QaExtractionStaging, row_id)
    if row is None:
        return None
    row.status = status
    session.flush()
    return row


def staging_stats(session: Session) -> dict[str, int]:
    counts = dict(
        session.execute(
            select(QaExtractionStaging.status, func.count()).group_by(QaExtractionStaging.status)
        ).all()
    )
    return {status: int(counts.get(status) or 0) for status in ALL_STATUSES}
