"""FastAPI application, chat page, and after-sales extraction endpoint."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.language_models import BaseChatModel

from app.api.actions import router as actions_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router
from app.api.feedback import router as feedback_router
from app.api.kb import router as kb_router
from app.api.rag_eval import PAGE_PATH as _RAG_EVAL_PAGE_PATH
from app.api.rag_eval import router as rag_eval_router
from app.api.review import PAGE_PATH as _REVIEW_PAGE_PATH
from app.api.review import router as review_router
from app.api.observability import PAGE_PATH as _OBSERVABILITY_PAGE_PATH
from app.api.observability import router as observability_router
from app.api.topics import router as topics_router
from app.database import build_engine, build_session_factory, initialize_database
from app.dependencies import get_chat_model
from app.extraction import build_extraction_chain
from app.schemas import AfterSalesExtraction, ExtractionRequest
from app.settings import Settings
from app.graph import runtime

logger = logging.getLogger(__name__)
_CHAT_PAGE_PATH = Path(__file__).resolve().parents[1] / "static" / "index.html"
_KB_PAGE_PATH = Path(__file__).resolve().parents[1] / "static" / "kb.html"
_LOG_PATH = Path(__file__).resolve().parents[1] / "log" / "app.log"
_TOPICS_PAGE_PATH = Path(__file__).resolve().parents[1] / "static" / "topics.html"
_TOPIC_QUESTIONS_PAGE_PATH = Path(__file__).resolve().parents[1] / "static" / "topic-questions.html"


def _setup_file_logging() -> None:
    """ch07:model_ctx / history_ctx / summary / 层1降级 落 log/app.log,验收可 grep。

    幂等:同一文件已有 FileHandler 就不重复挂;根级别提到 INFO 保证 INFO 级留痕生效。"""
    _LOG_PATH.parent.mkdir(exist_ok=True)
    root = logging.getLogger()
    target = str(_LOG_PATH)
    already = any(
        isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == target
        for handler in root.handlers
    )
    if not already:
        handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


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
        # ch08:内置工具服务启动时登记(扫描 app/tools/builtin/ 包,模块 import 即注册)
        from app.tools import registry as tool_registry

        tool_registry.scan_builtin()
        # ch05:图与 AsyncSqliteSaver 随应用起止(依赖注入便于测试替换)
        await runtime.init_graph(session_factory=session_factory, settings=settings)
        # ch09:审核/飞轮 API 经 app.state.graph_deps 拿 session_factory+settings
        application.state.graph_deps = runtime.get_deps()
        try:
            yield
        finally:
            await runtime.close_graph()
            engine.dispose()

    application = FastAPI(title="MewHelp Customer Support", version="0.1.0", lifespan=lifespan)
    application.state.chat_model = None
    _setup_file_logging()
    application.include_router(chat_router)
    application.include_router(actions_router)
    application.include_router(conversations_router)
    application.include_router(kb_router)
    application.include_router(rag_eval_router)
    application.include_router(feedback_router)  # ch09 · 用户反馈入口
    application.include_router(review_router)  # ch09 · 审核后台
    application.include_router(observability_router)  # ch09 · 观测与成本页
    application.include_router(topics_router)  # ch10 · 主题分布（微调分类器旁路产物）
    application.get("/", response_class=HTMLResponse, include_in_schema=False)(chat_page)
    application.get("/kb", response_class=HTMLResponse, include_in_schema=False)(kb_page)
    application.get("/rag-eval", response_class=HTMLResponse, include_in_schema=False)(rag_eval_page)
    application.get("/topics", response_class=HTMLResponse, include_in_schema=False)(topics_page)
    application.get("/topics/questions", response_class=HTMLResponse, include_in_schema=False)(topic_questions_page)
    application.post("/api/after-sales/extract", response_model=AfterSalesExtraction)(extract_after_sales)
    return application


async def chat_page() -> HTMLResponse:
    """Serve the single-file browser chat client."""
    return HTMLResponse(content=_CHAT_PAGE_PATH.read_text(encoding="utf-8"))


async def kb_page() -> HTMLResponse:
    """Serve the ch03 knowledge-base console page."""
    return HTMLResponse(content=_KB_PAGE_PATH.read_text(encoding="utf-8"))


async def rag_eval_page() -> HTMLResponse:
    """Serve the ch04 RAG evaluation console page."""
    return HTMLResponse(content=_RAG_EVAL_PAGE_PATH.read_text(encoding="utf-8"))


async def topics_page() -> HTMLResponse:
    """ch10 · 主题分布页（17 类条形图，数据只读 /api/topics/distribution）。"""
    return HTMLResponse(content=_TOPICS_PAGE_PATH.read_text(encoding="utf-8"))


async def topic_questions_page() -> HTMLResponse:
    """ch10 · 类目问题列表页（数据只读 /api/topics/questions）。"""
    return HTMLResponse(content=_TOPIC_QUESTIONS_PAGE_PATH.read_text(encoding="utf-8"))


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
