import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.chat import ConversationStore, PromptBudgetExceeded, trim_history


def _char_counter(messages):
    return sum(len(message.content) + 1 for message in messages)


def _tool_turn(question, tool_id, answer):
    return [
        HumanMessage(content=question),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_faq", "args": {"keyword": "退货"}, "id": tool_id, "type": "tool_call"}
            ],
        ),
        ToolMessage(content="工具结果", tool_call_id=tool_id),
        AIMessage(content=answer),
    ]


def test_oldest_complete_tool_turn_is_removed(monkeypatch):
    monkeypatch.setattr("app.chat.count_tokens_approximately", _char_counter)
    history = [
        *_tool_turn("旧问题", "old-call", "旧答案"),
        *_tool_turn("新问题", "new-call", "新答案"),
    ]

    kept = trim_history("sys", history, "cur", token_budget=25, safety_margin=0)

    assert kept == history[4:]


def test_tool_result_is_never_returned_without_its_assistant_request(monkeypatch):
    monkeypatch.setattr("app.chat.count_tokens_approximately", _char_counter)
    history = [
        *_tool_turn("旧问题", "old-call", "旧答案"),
        HumanMessage(content="新问题"),
        AIMessage(content="新答案"),
    ]

    kept = trim_history("sys", history, "cur", token_budget=20, safety_margin=0)

    assert kept == history[4:]
    for index, message in enumerate(kept):
        if message.type == "tool":
            request_ids = {
                call["id"]
                for previous in kept[:index]
                if previous.type == "ai"
                for call in (previous.tool_calls or [])
            }
            assert message.tool_call_id in request_ids


def test_system_prompt_is_never_trimmed(monkeypatch):
    seen = []

    def counter(messages):
        seen.append(messages)
        return _char_counter(messages)

    monkeypatch.setattr("app.chat.count_tokens_approximately", counter)
    history = [HumanMessage(content="旧问题"), AIMessage(content="旧答案")]

    kept = trim_history("固定系统提示", history, "cur", token_budget=15, safety_margin=0)

    assert kept == []
    assert any(message.type == "system" and message.content == "固定系统提示" for message in seen[0])


def test_fixed_prompt_over_budget_raises():
    with pytest.raises(PromptBudgetExceeded):
        trim_history("很长的固定系统提示", [], "当前输入", token_budget=2, safety_margin=0)


def test_safety_margin_reduces_available_budget(monkeypatch):
    monkeypatch.setattr("app.chat.count_tokens_approximately", _char_counter)
    history = [HumanMessage(content="旧问题"), AIMessage(content="旧答案")]

    kept = trim_history("sys", history, "cur", token_budget=20, safety_margin=5)

    assert kept == []


def test_incomplete_turn_is_rejected():
    with pytest.raises(ValueError):
        trim_history(
            "sys",
            [HumanMessage(content="问题"), ToolMessage(content="工具结果", tool_call_id="call-1")],
            "cur",
            token_budget=100,
            safety_margin=0,
        )


def test_conversation_store_commits_only_complete_turns():
    store = ConversationStore()
    conversation_id = store.create()

    store.commit_turn(conversation_id, "question", "answer")

    messages = store.get(conversation_id)
    assert [(message.type, message.content) for message in messages] == [
        ("human", "question"),
        ("ai", "answer"),
    ]
    assert store.get("unknown") is None
