"""ch07 · 会话侧栏两个只读接口:列表(新在前+首问预览+摘要徽标)与历史消息。"""

import asyncio
import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_session_factory
from app.repositories import conversations as repo

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
    user_id: str = "demo-user",
) -> dict:
    """会话列表:新会话在前,带首问预览与 has_summary 徽标位。"""

    def _work() -> list[dict]:
        with session_factory() as session:
            return repo.list_conversations(session, user_id=user_id)

    items = await asyncio.to_thread(_work)
    return {"items": items}


@router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    session_factory: Annotated[Callable[[], Session], Depends(get_session_factory)],
    user_id: str = "demo-user",
) -> dict:
    """会话历史消息(user/assistant 按 id 升序);会话不存在按 404 处理。"""

    def _work() -> tuple[bool, list[dict]]:
        with session_factory() as session:
            conversation = repo.get_conversation(session, conversation_id)
            if conversation is None:
                return False, []
            # tool 行与带 tool_calls 的空答复行(模型先调工具的那条)不进历史展示
            rows = repo.list_dialog_messages(session, conversation_id)
            return True, [
                {"role": row.role, "content": row.content}
                for row in rows
                if row.content
            ]

    found, items = await asyncio.to_thread(_work)
    if not found:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"items": items}
