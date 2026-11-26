"""ch04 · low_confidence_questions：拒答落池的确定性写入（TDD）。"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import LowConfidenceQuestion

SOURCES = ("retrieval_low_conf", "self_check", "user_feedback")


def fetch_latest_by_conversation(
    session: Session, conversation_id: str, raw_question: str, source: str
) -> LowConfidenceQuestion | None:
    """同会话+同问题+同来源最近一条(重复反馈防重:同轮连点两次 👎 只记一次)。"""
    return session.scalars(
        select(LowConfidenceQuestion)
        .where(
            LowConfidenceQuestion.conversation_id == conversation_id,
            LowConfidenceQuestion.raw_question == raw_question,
            LowConfidenceQuestion.source == source,
        )
        .order_by(LowConfidenceQuestion.id.desc())
        .limit(1)
    ).first()


def list_by_conversation(session: Session, conversation_id: str) -> list[LowConfidenceQuestion]:
    return list(
        session.scalars(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.conversation_id == conversation_id)
            .order_by(LowConfidenceQuestion.id)
        ).all()
    )


def insert_low_confidence(
    session: Session,
    *,
    conversation_id: str | None,
    raw_question: str,
    source: str,
    reason: str | None,
    retrieved_chunks: list | None = None,
) -> LowConfidenceQuestion:
    """Insert one refusal record; ``source`` must be one of the DDL enum values.

    ch09: ``retrieved_chunks`` 带上落池当时的召回片段快照（Top 几条原文与得分），
    没走检索的入口存 NULL，给审核页当人工审核依据。
    """
    if source not in SOURCES:
        raise ValueError(f"source 必须是 {'/'.join(SOURCES)} 之一，收到 {source!r}")
    row = LowConfidenceQuestion(
        conversation_id=conversation_id,
        raw_question=raw_question,
        source=source,
        reason=reason,
        retrieved_chunks=retrieved_chunks,
    )
    session.add(row)
    session.flush()
    return row


def recent_rows(session: Session, limit: int = 50) -> list[LowConfidenceQuestion]:
    """Newest first, for consoles and quick triage."""
    return list(session.scalars(select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id.desc()).limit(limit)).all())


def counts_by_source(session: Session) -> dict[str, int]:
    rows = session.execute(
        select(LowConfidenceQuestion.source, func.count()).group_by(LowConfidenceQuestion.source)
    ).all()
    return {source: int(count) for source, count in rows}
