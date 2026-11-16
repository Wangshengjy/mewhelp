"""Database engine and session helpers."""

from collections.abc import Callable

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.demo_seed import seed_demo_data
from app.models import Base


def build_engine(database_url: str | URL) -> Engine:
    """Build an engine suitable for production MySQL or isolated SQLite tests."""
    if isinstance(database_url, URL):
        # ``str(URL)`` hides the password as ``***``; render the real one so the
        # MySQL driver authenticates with the configured credentials.
        database_url = database_url.render_as_string(hide_password=False)
    if database_url.startswith("sqlite"):
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        @event.listens_for(engine, "connect")
        def enable_sqlite_foreign_keys(dbapi_connection: object, _: object) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    return create_engine(database_url, pool_pre_ping=True)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def initialize_database(session_factory: Callable[[], Session]) -> None:
    """Create the schema and insert the fixed demo data exactly once."""
    Base.metadata.create_all(session_factory.kw["bind"])
    seed_demo_data(session_factory)
