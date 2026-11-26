"""ch09 · 白名单作业运行器 + 三张报表聚合(观测页数据层)。

- start_job:只允许 WHITELIST 里的作业名,subprocess 起 `python -m scripts.<name>`,
  日志落 data/ch09/jobs/<name>.log;同名作业同一时刻只跑一个(信号文件)。
- 三张报表:缺口 TopN / 拒答按来源统计 / cost_by_intent 产物读取。
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

from app.core.observability import get_langfuse, langfuse_enabled

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "data" / "ch09" / "jobs"
REPORT_DIR = ROOT / "data" / "ch09" / "reports"

WHITELIST = {
    "flywheel_pipeline": ["python", "-m", "scripts.flywheel_pipeline", "--limit", "50"],
    "calibrate_confidence": ["python", "-m", "scripts.calibrate_confidence"],
    "cost_by_intent": ["python", "-m", "scripts.cost_by_intent"],
}
MAX_LOG_BYTES = 64 * 1024


def _log_path(name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{name}.log"


def start_job(name: str) -> subprocess.Popen:
    """起白名单作业;同名作业在跑则 RuntimeError(前端提示先等当前完成)。"""
    if name not in WHITELIST:
        raise KeyError(name)
    log = _log_path(name)
    lock = LOG_DIR / f"{name}.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime < 30 * 60:
        raise RuntimeError(f"作业 {name} 正在运行,请稍后再试")
    lock.write_text(str(time.time()), encoding="utf-8")
    cmd = [sys.executable if part == "python" else part for part in WHITELIST[name]]
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT),
        stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def job_status(name: str) -> dict:
    """作业状态:轮询读日志尾(已完成/运行中由锁文件时间与日志判断,尽力而为)。"""
    if name not in WHITELIST:
        raise KeyError(name)
    log = _log_path(name)
    lock = LOG_DIR / f"{name}.lock"
    running = lock.exists() and time.time() - lock.stat().st_mtime < 30 * 60
    tail = log.read_text(encoding="utf-8", errors="replace")[-MAX_LOG_BYTES:] if log.exists() else ""
    if not running and lock.exists():
        try:
            lock.unlink()
        except OSError:
            pass
    return {"name": name, "running": running, "log_tail": tail}


# ── 三张报表聚合 ─────────────────────────────────────────────────────────────

async def gap_top_report(session_factory, top_n: int = 10) -> dict:
    """报表 1 · 知识缺口 TopN:出现次数最高的未决缺口(该补库的优先级)。"""
    from app.repositories import review_queue as review_queue_repository

    def _run():
        with session_factory() as session:
            rows = review_queue_repository.list_review_queue(session, "待审")[:top_n]
            return [
                {
                    "id": row.id,
                    "normalized_question": row.normalized_question,
                    "occurrence_count": row.occurrence_count,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    items = await asyncio.to_thread(_run)
    return {"title": "知识缺口 TopN", "items": items}


async def refusal_report(session_factory, sample_n: int = 8) -> dict:
    """报表 2 · 反馈/拒答统计:low_confidence_questions 按 source 计数 + 最近样本。"""
    from sqlalchemy import func, select

    from app.models import LowConfidenceQuestion

    def _run():
        with session_factory() as session:
            counts = dict(session.execute(
                select(LowConfidenceQuestion.source, func.count(LowConfidenceQuestion.id))
                .group_by(LowConfidenceQuestion.source)
            ).all())
            samples = session.scalars(
                select(LowConfidenceQuestion).order_by(LowConfidenceQuestion.id.desc()).limit(sample_n)
            ).all()
            return {
                "counts": counts,
                "total": sum(counts.values()),
                "recent": [
                    {
                        "id": row.id,
                        "raw_question": row.raw_question,
                        "source": row.source,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }
                    for row in samples
                ],
            }

    data = await asyncio.to_thread(_run)
    return {"title": "反馈与拒答统计", **data}


async def cost_report(session_factory, settings) -> dict | None:
    """报表 3 · Langfuse 成本:读 cost_by_intent 产物;无产物返回 None(提示先跑作业)。"""
    report = read_cost_artifact()
    if report is None and langfuse_enabled():
        # 有 Langfuse 但还没跑过作业:现场聚合一次(小数据量,直接拉)
        from scripts.cost_by_intent import aggregate
        try:
            if get_langfuse() is not None:
                return await asyncio.to_thread(
                    aggregate, settings.langfuse_base_url,
                    settings.langfuse_public_key, settings.langfuse_secret_key, None,
                )
        except Exception:
            return None
    return report


def read_cost_artifact() -> dict | None:
    path = REPORT_DIR / "cost_by_intent.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_artifact(name: str) -> dict | None:
    """按产物名读 data/ch09/reports/<name>.json(eval_trend 等);白名单防路径穿越。"""
    allowed = {"eval_trend", "cost_by_intent", "confidence_calibration", "flywheel_samples",
               "flywheel_pipeline_last"}
    if name not in allowed:
        return None
    path = REPORT_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
