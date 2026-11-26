"""ch04 · /rag-eval 评估台 API：报告只读、台账处置、就地重跑（后台线程回显日志）。"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from app.dependencies import get_session_factory
from app.repositories import faith_cases as faith_cases_repository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rag-eval", tags=["rag-eval"])

ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / "data" / "ch04" / "reports" / "rag_eval.json"
PAGE_PATH = ROOT / "static" / "rag_eval.html"

# 就地重跑：单进程内只允许一轮；日志逐行缓冲供前端轮询
_RERUN_LOCK = threading.Lock()
_RERUN_STATE: dict = {"running": False, "lines": [], "exit_code": None, "started_at": None}


class FaithCaseResolution(BaseModel):
    """处置动作：已解决/无需解决 必填一句说明。"""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(pattern="^(已解决|无需解决)$", description="处置状态")
    resolution: str = Field(min_length=1, max_length=300, description="处置说明（必填）")


@router.get("/report")
def get_report() -> dict:
    """读取最近一轮评估报告；未跑过返回 404。"""
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="还没有评估报告，先跑一轮 scripts.eval_rag")
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@router.get("/faith-cases")
def list_faith_cases(
    status: str | None = Query(default=None, pattern="^(未解决|已解决|无需解决)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    session_factory=Depends(get_session_factory),
) -> dict:
    """台账分页（状态页签）+ 两个口径的幻觉率。"""
    with session_factory() as session:
        all_cases = faith_cases_repository.list_cases(session, status=status)
        counts = faith_cases_repository.status_counts(session)
    total = len(all_cases)
    start = (page - 1) * page_size
    rows = [faith_cases_repository.case_to_dict(case) for case in all_cases[start : start + page_size]]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "items": rows,
        "status_counts": counts,
    }


@router.post("/faith-cases/{eval_id}/resolve")
def resolve_faith_case(
    eval_id: str,
    payload: FaithCaseResolution,
    session_factory=Depends(get_session_factory),
) -> dict:
    try:
        with session_factory() as session:
            case = faith_cases_repository.resolve_case(
                session, eval_id=eval_id, status=payload.status, resolution=payload.resolution
            )
            result = faith_cases_repository.case_to_dict(case)
            session.commit()
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/rerun")
def rerun_eval(request: Request) -> dict:
    """后台线程起子进程跑 scripts.eval_rag，日志逐行回显，产物落盘后前端刷新。

    子进程自行从 .env 读配置（scripts._common.load_settings）；此处不注入 settings，
    与 /api/kb 一致用 request: Request 风格，避免 Annotated 依赖的前向引用解析问题。
    """
    with _RERUN_LOCK:
        if _RERUN_STATE["running"]:
            raise HTTPException(status_code=409, detail="已有一轮评估在跑")
        _RERUN_STATE.update({"running": True, "lines": [], "exit_code": None, "started_at": None})

    def _run() -> None:
        from datetime import datetime

        _RERUN_STATE["started_at"] = datetime.now().isoformat(timespec="seconds")
        process = subprocess.Popen(
            [sys.executable, "-m", "scripts.eval_rag"],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            _RERUN_STATE["lines"].append(line.rstrip("\n"))
        process.wait()
        _RERUN_STATE["exit_code"] = process.returncode
        _RERUN_STATE["running"] = False

    thread = threading.Thread(target=_run, daemon=True, name="rag-eval-rerun")
    thread.start()
    return {"ok": True, "message": "评估已在后台启动，轮询 /api/rag-eval/rerun/status 查看进度"}


@router.get("/rerun/status")
def rerun_status() -> dict:
    return {
        "running": _RERUN_STATE["running"],
        "lines": _RERUN_STATE["lines"][-400:],
        "exit_code": _RERUN_STATE["exit_code"],
        "started_at": _RERUN_STATE["started_at"],
    }


@router.get("/page", include_in_schema=False)
def rag_eval_page() -> "HTMLResponse":  # noqa: F821
    """Serve the /rag-eval console page (registered in main.py at /rag-eval)."""
    from fastapi.responses import HTMLResponse

    return HTMLResponse(content=PAGE_PATH.read_text(encoding="utf-8"))
