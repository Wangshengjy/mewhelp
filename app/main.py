"""FastAPI application, chat page, and after-sales extraction endpoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.language_models import BaseChatModel

from app.api.chat import router as chat_router
from app.database import build_engine, build_session_factory, initialize_database
from app.dependencies import get_chat_model
from app.extraction import build_extraction_chain
from app.schemas import AfterSalesExtraction, ExtractionRequest
from app.settings import Settings

logger = logging.getLogger(__name__)
_CHAT_PAGE_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"


def create_app(
    settings_override: Settings | None = None,
    database_url_override: str | None = None,
) -> FastAPI:
    """Create an application with an injectable database for tests."""
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        settings = settings_override or Settings.load()
        database_url = database_url_override or settings.database_url
        engine = build_engine(database_url)
        session_factory = build_session_factory(engine)
        application.state.settings = settings
        application.state.engine = engine
        application.state.session_factory = session_factory
        initialize_database(session_factory)
        try:
            yield
        finally:
            engine.dispose()

    application = FastAPI(title="MewHelp Customer Support", version="0.1.0", lifespan=lifespan)
    application.state.chat_model = None
    application.include_router(chat_router)
    application.get("/", response_class=HTMLResponse, include_in_schema=False)(chat_page)
    application.post("/api/after-sales/extract", response_model=AfterSalesExtraction)(extract_after_sales)
    return application


async def chat_page() -> HTMLResponse:
    """Serve the single-file browser chat client."""
    return HTMLResponse(content=_CHAT_PAGE_PATH.read_text(encoding="utf-8"))


async def extract_after_sales(
    request: ExtractionRequest,
    model: Annotated[BaseChatModel, Depends(get_chat_model)],
) -> AfterSalesExtraction:
    try:
        chain = build_extraction_chain(model)
        return await chain.ainvoke({"description": request.description})
    except Exception as exc:
        logger.warning("Structured extraction failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="Structured extraction is unavailable for the configured model",
        ) from exc


app = create_app()
