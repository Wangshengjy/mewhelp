"""ch10 · 主题分布 API：飞轮后台看各类目问题量，决定先补哪块知识。只读。

页面只读这里与仓库 json 产物，不在接口里重算——同一个指标只有一个真相。
"""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.taxonomy import TOPIC_NAMES
from app.dependencies import get_session_factory
from app.repositories import topic_classification as topic_repository

router = APIRouter()


@router.get("/api/topics/distribution")
async def distribution(
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
) -> dict:
    """17 类各自问题量 + 每类样例（样例按文本去重）+ 总量/最近归类时间。"""
    with session_factory() as session:
        return topic_repository.topic_distribution(session)


@router.get("/api/topics/questions")
async def questions(
    label: str,
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
    page: int = 1,
    size: int = 10,
) -> dict:
    """某一类目下归类到的全部问题，分页；多标签问题在它命中的每个类目下都出现。

    类目名不在权威表里回 400 说清楚，不静静回一页空列表——那会被当成「这类没有问题」。
    """
    if label not in TOPIC_NAMES:
        raise HTTPException(
            status_code=400, detail=f"未知类目 {label!r}；类目必须取 17 类权威表原文")
    if page < 1 or size < 1 or size > 100:
        raise HTTPException(status_code=400, detail="page/size 取值不合法")
    with session_factory() as session:
        return topic_repository.topic_questions(session, label=label, page=page, size=size)
