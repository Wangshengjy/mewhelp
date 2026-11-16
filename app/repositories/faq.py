"""Parameterized SQL LIKE lookups over the FAQ table."""

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Faq

_ESCAPE_CHARACTER = "\\"


def _escape_like(value: str) -> str:
    """Escape LIKE wildcards so the keyword is always matched literally."""
    return (
        value.replace(_ESCAPE_CHARACTER, _ESCAPE_CHARACTER * 2)
        .replace("%", _ESCAPE_CHARACTER + "%")
        .replace("_", _ESCAPE_CHARACTER + "_")
    )


def search_faq(session: Session, keyword: str) -> list[Faq]:
    """Return FAQ rows whose question, answer or category contains the keyword."""
    pattern = f"%{_escape_like(keyword.strip())}%"
    statement = (
        select(Faq)
        .where(
            or_(
                Faq.question.like(pattern, escape=_ESCAPE_CHARACTER),
                Faq.answer.like(pattern, escape=_ESCAPE_CHARACTER),
                Faq.category.like(pattern, escape=_ESCAPE_CHARACTER),
            )
        )
        .order_by(Faq.id)
    )
    return list(session.scalars(statement).all())
