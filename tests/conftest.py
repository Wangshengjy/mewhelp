import pytest

from app.main import create_app
from app.settings import Settings


@pytest.fixture
def test_app():
    """A fully initialized application backed by one isolated SQLite database."""
    settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
    )
    return create_app(settings_override=settings, database_url_override="sqlite+pysqlite:///:memory:")
