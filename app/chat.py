"""In-memory conversations and estimated token-budget history trimming."""

from threading import RLock
from uuid import uuid4
from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately


class PromptBudgetExceeded(ValueError):
    """The fixed prompt and current user message do not fit the configured budget."""


class ConversationStore:
    """A process-local store that only appends complete user/assistant turns."""

    def __init__(self) -> None:
        self._conversations: dict[str, list[BaseMessage]] = {}
        self._lock = RLock()

    def create(self) -> str:
        conversation_id = str(uuid4())
        with self._lock:
            self._conversations[conversation_id] = []
        return conversation_id

    def get(self, conversation_id: str) -> list[BaseMessage] | None:
        with self._lock:
            messages = self._conversations.get(conversation_id)
            return list(messages) if messages is not None else None

    def commit_turn(self, conversation_id: str, user_text: str, assistant_text: str) -> None:
        with self._lock:
            messages = self._conversations.get(conversation_id)
            if messages is None:
                raise KeyError(f"Unknown conversation_id: {conversation_id}")
            messages.extend((HumanMessage(content=user_text), AIMessage(content=assistant_text)))


def trim_history(
    system_text: str,
    history: Sequence[BaseMessage],
    current_text: str,
    token_budget: int,
    safety_margin: int,
) -> list[BaseMessage]:
    """Keep the newest complete user turns within the estimated budget.

    A user turn runs from a HumanMessage through its final assistant
    AIMessage, including any ToolMessage and the assistant tool request that
    produced it. Whole turns are removed so a tool result is never kept
    without the assistant request that asked for it.
    """
    usable_budget = token_budget - safety_margin
    if usable_budget <= 0:
        raise PromptBudgetExceeded("The fixed prompt and current message exceed the token budget")

    system_message = SystemMessage(content=system_text)
    current_message = HumanMessage(content=current_text)

    def estimated(messages: Sequence[BaseMessage]) -> int:
        return count_tokens_approximately([system_message, *messages, current_message])

    if estimated([]) > usable_budget:
        raise PromptBudgetExceeded("The fixed prompt and current message exceed the token budget")

    turns = _group_complete_turns(history)
    while turns and estimated(_flatten_turns(turns)) > usable_budget:
        turns.pop(0)
    return _flatten_turns(turns)


def _flatten_turns(turns: Sequence[Sequence[BaseMessage]]) -> list[BaseMessage]:
    """Flatten grouped turns back into a single message list."""
    return [message for turn in turns for message in turn]


def _group_complete_turns(history: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """Split history into complete user turns.

    Each turn starts at a HumanMessage and ends at the final assistant
    AIMessage before the next HumanMessage (or the end of the history).
    """
    if not history:
        return []

    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []
    for message in history:
        if message.type == "human":
            if current_turn:
                _ensure_complete_turn(current_turn)
                turns.append(current_turn)
            current_turn = [message]
        else:
            current_turn.append(message)

    if current_turn:
        _ensure_complete_turn(current_turn)
        turns.append(current_turn)
    return turns


def _ensure_complete_turn(turn: Sequence[BaseMessage]) -> None:
    """Validate that a turn starts with a user message and ends with an answer."""
    if turn[0].type != "human":
        raise ValueError("Conversation history must start each turn with a human message")
    if turn[-1].type != "ai":
        raise ValueError("Conversation history must end each turn with an assistant message")
