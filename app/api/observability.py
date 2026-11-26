"""ch09 · 观测与成本报表 API(只读)+ 三张报表聚合 + 白名单作业运行器。

三张报表(课程定稿,数据源各就各位):
  1. 知识缺口 TopN   —— review_queue( occurrence_count 降序)
  2. 反馈拒答统计    —— low_confidence_questions 按 source 计数 + 最近样本
  3. Langfuse 成本   —— GET /api/public/traces(sessionId 过滤),按 intent tag 分堆算 token

作业重跑:白名单(flywheel_pipeline / calibrate_confidence / cost_by_intent)经
core.jobs 起子进程,前端轮询 GET /api/observability/jobs/{name} 读日志尾。
"""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from app.core import jobs as jobs_core
from app.core.observability import get_langfuse, langfuse_enabled

logger = logging.getLogger(__name__)

router = APIRouter()
_PAGE_PATH = Path(__file__).resolve().parents[2] / "static" / "observability.html"
PAGE_PATH = _PAGE_PATH


@router.get("/api/observability/overview")
async def overview(request: Request) -> dict:
    """观测页三块一次拉齐(前端一个接口渲染三张报表);单块失败不影响其余。"""
    from app.core import flywheel as flywheel_core

    session_factory, settings = flywheel_core.deps_from_app(request.app)
    blocks: dict = {"gap_top": None, "refusal_stats": None, "cost_report": None,
                    "langfuse_enabled": langfuse_enabled()}
    try:
        blocks["gap_top"] = await jobs_core.gap_top_report(session_factory)
    except Exception as exc:
        logger.warning("缺口报表失败: %s", exc)
    try:
        blocks["refusal_stats"] = await jobs_core.refusal_report(session_factory)
    except Exception as exc:
        logger.warning("拒答统计失败: %s", exc)
    try:
        blocks["cost_report"] = await jobs_core.cost_report(session_factory, settings)
    except Exception as exc:
        logger.warning("成本报表失败: %s", exc)
    return blocks


@router.get("/api/observability/gaps")
async def gaps(request: Request) -> dict:
    from app.core import flywheel as flywheel_core

    session_factory, _ = flywheel_core.deps_from_app(request.app)
    return await jobs_core.gap_top_report(session_factory)


@router.get("/api/observability/cost")
async def cost(request: Request) -> dict:
    """按意图 token 账:直读 cost_by_intent 产物 json(作业产物,不实时拉)。"""
    report = jobs_core.read_cost_artifact()
    if report is None:
        raise HTTPException(status_code=404, detail="cost_by_intent 产物不存在,先跑一次作业")
    return report


@router.get("/api/observability/jobs-data/{name}")
async def jobs_data(name: str) -> dict:
    """读作业产物 json(eval_trend 等),观测页趋势表数据源。"""
    data = jobs_core.read_artifact(name)
    if data is None:
        raise HTTPException(status_code=404, detail=f"{name} 产物不存在,先跑对应作业")
    return data


@router.post("/api/observability/jobs/{name}/run")
async def run_job(name: str) -> dict:
    """白名单作业重跑(异步起子进程,前端轮询日志)。"""
    try:
        proc = jobs_core.start_job(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未知作业: {name}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started", "name": name, "pid": proc.pid}


@router.get("/api/observability/jobs/{name}")
async def job_status(name: str) -> dict:
    try:
        return jobs_core.job_status(name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"未知作业: {name}") from exc


@router.get("/observability", response_class=HTMLResponse, include_in_schema=False)
async def observability_page() -> HTMLResponse:
    return HTMLResponse(content=_PAGE_PATH.read_text(encoding="utf-8"))
