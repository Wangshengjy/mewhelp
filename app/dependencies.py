"""FastAPI dependencies that supply the configured model and database session factory."""

from collections.abc import Callable
from threading import Lock
from typing import Annotated

from fastapi import Depends, Request
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from app.settings import Settings

_chat_model_lock = Lock()


def get_settings() -> Settings:
    """Load local settings lazily so importing the application needs no secrets."""
    return Settings.load()


def get_chat_model(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BaseChatModel:
    """Lazily build and reuse one OpenAI-compatible chat client per process."""
    application = request.app
    if application.state.chat_model is None:
        with _chat_model_lock:
            if application.state.chat_model is None:
                application.state.chat_model = ChatOpenAI(
                    base_url=settings.chat_base_url,
                    api_key=settings.chat_api_key,
                    model=settings.chat_model,
                    temperature=settings.chat_temperature,
                )
    return application.state.chat_model


def get_session_factory(request: Request) -> Callable[[], Session]:
    """Return the application-owned SQLAlchemy session factory from lifespan state."""
    return request.app.state.session_factory