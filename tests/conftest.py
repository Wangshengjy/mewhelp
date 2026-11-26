import pytest

from app.main import create_app
from app.settings import Settings


@pytest.fixture
def session_factory():
    """A standalone SQLite session factory for repository-level tests."""
    from app.database import build_engine, build_session_factory
    from app.models import Base

    engine = build_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


@pytest.fixture
def test_app():
    """A fully initialized application backed by one isolated SQLite database."""
    settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        checkpointer_db_path=":memory:",  # ch05:测试里 checkpointer 也走内存,不落盘
    )
    return create_app(settings_override=settings, database_url_override="sqlite+pysqlite:///:memory:")
