"""SQLAlchemy repository functions for conversations and message history."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.models import Conversation, ConversationSummary, Message


def create_conversation(session: Session, user_id: str = "demo-user") -> Conversation:
    """Create and persist a new conversation with the given user."""
    conversation = Conversation(user_id=user_id)
    session.add(conversation)
    session.flush()
    return conversation


def get_conversation(session: Session, conversation_id: str) -> Conversation | None:
    """Return the conversation with the given ID, or None when it is missing."""
    return session.get(Conversation, conversation_id)


def append_message(
    session: Session,
    conversation_id: str,
    *,
    role: str,
    content: str,
    tool_calls: dict | list | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> Message:
    """Append a message to a conversation and return the persisted row."""
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
    )
    session.add(message)
    session.flush()
    return message


def load_history(session: Session, conversation_id: str) -> list[BaseMessage]:
    """Load a conversation as LangChain messages in chronological order."""
    messages = session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    ).all()

    history: list[BaseMessage] = []
    for message in messages:
        if message.role == "user":
            history.append(HumanMessage(content=message.content))
        elif message.role == "assistant":
            history.append(
                AIMessage(
                    content=message.content,
                    tool_calls=message.tool_calls or [],
                )
            )
        elif message.role == "tool":
            history.append(
                ToolMessage(
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                    name=message.tool_name,
                )
            )
        else:
            raise ValueError(f"Unknown message role: {message.role}")
    return history


# ── ch07 · 分层上下文:锚点更新 + 分段摘要 + 会话列表 ─────────────────────────


def latest_user_message_id(session: Session, conversation_id: str) -> int | None:
    """最新一条 user 消息的 id(轮入口构造 db-{msg_id} 锚点用)。"""
    return session.scalar(
        select(Message.id)
        .where(Message.conversation_id == conversation_id, Message.role == "user")
        .order_by(Message.id.desc())
        .limit(1)
    )


def list_conversations(session: Session, user_id: str = "demo-user", limit: int = 50) -> list[dict]:
    """ch07 · 会话侧栏列表:新会话在前,带首问预览与 has_summary 标记。"""
    rows = session.execute(
        select(Conversation.id, Conversation.summary)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
        .limit(limit)
    ).all()
    items: list[dict] = []
    for conversation_id, summary in rows:
        preview = session.scalar(
            select(Message.content)
            .where(Message.conversation_id == conversation_id, Message.role == "user")
            .order_by(Message.id)
            .limit(1)
        )
        items.append(
            {
                "id": conversation_id,
                "preview": preview or "",
                "has_summary": bool(summary and summary.strip()),
            }
        )
    return items


def count_messages_after(session: Session, conversation_id: str, after_id: int | None) -> int:
    """统计某会话中 id 大于 after_id 的消息条数(None 视为 0,即全量)。"""
    query = select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
    if after_id is not None:
        query = query.where(Message.id > after_id)
    return int(session.scalar(query) or 0)


def list_dialog_messages(session: Session, conversation_id: str) -> list[Message]:
    """按 id 升序取 user/assistant 对话行(过滤 tool 行,摘要与用量统计用)。"""
    return list(
        session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id, Message.role.in_(("user", "assistant")))
            .order_by(Message.id)
        ).all()
    )


def update_conversation_summary(session: Session, conversation_id: str, summary: str, upto_msg_id: int) -> None:
    """摘要投影与覆盖锚点原子更新(只在摘要成功落库后调用)。"""
    conversation = get_conversation(session, conversation_id)
    if conversation is None:
        raise ValueError(f"Unknown conversation_id: {conversation_id}")
    conversation.summary = summary
    conversation.summary_upto_msg_id = upto_msg_id
    session.flush()


def update_layer1_from(session: Session, conversation_id: str, msg_id: int) -> None:
    """层 1 起点锚点更新(settle_layers 降级专用,只挪 id 不搬数据)。"""
    conversation = get_conversation(session, conversation_id)
    if conversation is None:
        raise ValueError(f"Unknown conversation_id: {conversation_id}")
    conversation.layer1_from_msg_id = msg_id
    session.flush()


def append_summary_segment(
    session: Session,
    conversation_id: str,
    from_msg_id: int,
    upto_msg_id: int,
    content: str,
    *,
    inject_segments: int = 3,
) -> int:
    """追加一段摘要并重拼最近 N 段投影到 conversations.summary,返回段号。

    inject_segments 由调用方传 settings.summary_inject_segments(默认 3 与起步值一致)。"""
    conversation = get_conversation(session, conversation_id)
    if conversation is None:
        raise ValueError(f"Unknown conversation_id: {conversation_id}")
    next_seq = int(
        session.scalar(
            select(func.max(ConversationSummary.seq)).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
        or 0
    ) + 1
    session.add(
        ConversationSummary(
            conversation_id=conversation_id,
            seq=next_seq,
            from_msg_id=from_msg_id,
            upto_msg_id=upto_msg_id,
            content=content,
        )
    )
    segments = list_summary_segments(session, conversation_id, limit=inject_segments)
    conversation.summary = "\n".join(segments)
    conversation.summary_upto_msg_id = upto_msg_id
    session.flush()
    return next_seq


def list_summary_segments(session: Session, conversation_id: str, limit: int | None = None) -> list[str]:
    """最近几段摘要正序返回(limit=2 取第 N-1、N 段;None 取全部)。"""
    query = (
        select(ConversationSummary.content)
        .where(ConversationSummary.conversation_id == conversation_id)
        .order_by(ConversationSummary.seq.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    rows = list(session.scalars(query).all())
    return list(reversed(rows))
