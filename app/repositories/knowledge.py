"""Repository for knowledge_chunks: dual-write state machine and fingerprints.

The authoritative text lives in MySQL; Milvus only stores vectors keyed by the
chunk id. ``vectorize_status`` drives the pending→done state machine so any
interrupted vectorization run can be resumed by re-reading the pending rows.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import KnowledgeChunk

STATUS_PENDING = "pending"
STATUS_DONE = "done"

DOC_CONTENT_TYPES: frozenset[str] = frozenset({"faq", "policy", "manual", "spec"})

_PUNCTUATION_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def chunk_fingerprint(category: str, questions: str, answer: str) -> str:
    """sha256 over the three vectorization source fields (category+questions+answer)."""
    joined = f"{category}\n{questions}\n{answer}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def normalized_text(text: str) -> str:
    """问法归一化：去空白/标点、小写，中文保留。"""
    return _PUNCTUATION_RE.sub("", text).lower()


def qa_fingerprint(question: str, answer: str) -> str:
    """问法+正文指纹（归一化后），用于暂存表与库内存量的查重。"""
    joined = f"{normalized_text(question)}\n{normalized_text(answer)}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def insert_chunk(
    session: Session,
    *,
    category: str,
    questions: str,
    answer: str,
    section_path: str | None = None,
    content_type: str | None = None,
    is_key_clause: bool = False,
    is_table_block: bool = False,
    vectorize_status: str = STATUS_PENDING,
) -> KnowledgeChunk:
    """Insert one chunk as pending (or done) and return it with its id."""
    chunk = KnowledgeChunk(
        category=category,
        questions=questions,
        answer=answer,
        section_path=section_path,
        content_type=content_type,
        is_key_clause=is_key_clause,
        vectorize_status=vectorize_status,
    )
    session.add(chunk)
    session.flush()
    return chunk


def get_by_fingerprint(session: Session, fingerprint: str) -> KnowledgeChunk | None:
    """Scan chunks for one matching sha256(category+questions+answer).

    The DDL intentionally has no fingerprint column, so matching happens in
    Python over the authoritative fields; the corpus is small.
    """
    for chunk in session.scalars(select(KnowledgeChunk).order_by(KnowledgeChunk.id)).all():
        if chunk_fingerprint(chunk.category, chunk.questions, chunk.answer) == fingerprint:
            return chunk
    return None


def list_fingerprints(session: Session) -> set[str]:
    """All chunk fingerprints currently stored (for bulk dedup checks)."""
    rows = session.scalars(select(KnowledgeChunk)).all()
    return {chunk_fingerprint(row.category, row.questions, row.answer) for row in rows}


def list_pending(session: Session) -> list[KnowledgeChunk]:
    """Chunks waiting to be vectorized, oldest first."""
    statement = (
        select(KnowledgeChunk)
        .where(KnowledgeChunk.vectorize_status == STATUS_PENDING)
        .order_by(KnowledgeChunk.id)
    )
    return list(session.scalars(statement).all())


def get_chunks_by_ids(session: Session, chunk_ids: Sequence[int]) -> list[KnowledgeChunk]:
    """Load chunks by primary key preserving the input order (missing ids dropped)."""
    if not chunk_ids:
        return []
    rows = {
        row.id: row
        for row in session.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.id.in_([int(value) for value in chunk_ids]))
        ).all()
    }
    return [rows[int(value)] for value in chunk_ids if int(value) in rows]


def mark_done(session: Session, chunk_id: int, vector_id: str) -> KnowledgeChunk | None:
    """Flip one chunk to done and back-fill its Milvus vector id."""
    chunk = session.get(KnowledgeChunk, chunk_id)
    if chunk is None:
        return None
    chunk.vectorize_status = STATUS_DONE
    chunk.vector_id = vector_id
    session.flush()
    return chunk


def stats(session: Session) -> dict[str, int]:
    """pending/done/total ledger of the dual-write state machine."""
    pending = session.scalar(
        select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.vectorize_status == STATUS_PENDING)
    )
    done = session.scalar(
        select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.vectorize_status == STATUS_DONE)
    )
    pending = int(pending or 0)
    done = int(done or 0)
    return {"pending": pending, "done": done, "total": pending + done}


def count_doc_chunks(session: Session, content_types: Iterable[str] = DOC_CONTENT_TYPES) -> int:
    """Number of stored chunks whose content_type belongs to the document set."""
    types = list(content_types)
    if not types:
        return 0
    return int(
        session.scalar(
            select(func.count()).select_from(KnowledgeChunk).where(KnowledgeChunk.content_type.in_(types))
        )
        or 0
    )


def delete_doc_chunks(session: Session, content_types: Iterable[str] = DOC_CONTENT_TYPES) -> list[KnowledgeChunk]:
    """Delete document chunks (kb_build --reset) and return what was removed."""
    types = list(content_types)
    if not types:
        return []
    rows = list(
        session.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.content_type.in_(types)).order_by(KnowledgeChunk.id)
        ).all()
    )
    for row in rows:
        session.delete(row)
    session.flush()
    return rows


def list_doc_chunks(
    session: Session, section_prefix: str, doc_content_type: str
) -> list[KnowledgeChunk]:
    """Chunks of one document (section_path equals title or starts with ``title >``)."""
    rows = list(
        session.scalars(
            select(KnowledgeChunk)
            .where(
                KnowledgeChunk.content_type == doc_content_type,
                (KnowledgeChunk.section_path == section_prefix)
                | (KnowledgeChunk.section_path.like(f"{section_prefix} > %")),
            )
            .order_by(KnowledgeChunk.id)
        ).all()
    )
    return rows


def relink_document(session: Session, section_prefix: str, doc_content_type: str) -> int:
    """Two-pass pointer back-fill: rewire prev/next across one document's chunks.

    A chunk belongs to the document when its section_path equals the document
    title or starts with ``title + " > "``. Called after all chunks of the
    document were inserted (so ids exist), which is the course-required second
    pass.
    """
    rows = list(
        session.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.content_type == doc_content_type).order_by(KnowledgeChunk.id)
        ).all()
    )
    doc_rows = [
        row
        for row in rows
        if row.section_path == section_prefix or (row.section_path or "").startswith(f"{section_prefix} > ")
    ]
    for position, row in enumerate(doc_rows):
        row.prev_chunk_id = doc_rows[position - 1].id if position > 0 else None
        row.next_chunk_id = doc_rows[position + 1].id if position + 1 < len(doc_rows) else None
    session.flush()
    return len(doc_rows)
