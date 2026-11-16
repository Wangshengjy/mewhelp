"""SQLAlchemy repository functions for conversations and message history."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.models import Conversation, Message


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
