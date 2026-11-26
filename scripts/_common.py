"""Shared bootstrap helpers for the ch03 script modules (make-command equivalents)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import build_engine, build_session_factory  # noqa: E402
from app.models import Base  # noqa: E402
from app.settings import Settings  # noqa: E402


def load_settings() -> Settings:
    """Load settings from mewhelp/.env."""
    return Settings.load()


def session_factory_from(settings: Settings):
    """Build a session factory against the configured MySQL.

    Runs an idempotent ``create_all`` so the ch02 tables (faq/conversations/
    messages/tickets) exist on a fresh database; the ch03 tables come from the
    DDL and are left untouched (checkfirst skips them).
    """
    engine = build_engine(settings.database_url)
    Base.metadata.create_all(engine)
    return build_session_factory(engine)


def embed_ready(settings: Settings) -> bool:
    """Guard shared by every script that needs real embedding calls."""
    if settings.embed_api_key:
        return True
    print("[skip] EMBED_API_KEY 未配置，无法调用嵌入服务（去硅基流动申请后重试）")
    return False


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")
