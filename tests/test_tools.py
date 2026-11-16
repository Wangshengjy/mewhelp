"""Tests for the five customer-support tools and their explicit registry."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.database import build_engine, build_session_factory
from app.demo_seed import seed_demo_data
from app.models import Base, Faq, Ticket
from app.repositories.conversations import create_conversation
from app.repositories.faq import search_faq
from app.tools import mock_data
from app.tools.registry import build_tool_registry

SQLITE_URL = "sqlite+pysqlite:///:memory:"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_VISIBLE_SCHEMAS = {
    "query_order": {"order_id"},
    "query_product": {"product_id"},
    "query_logistics": {"order_id"},
    "query_faq": {"keyword"},
    "create_ticket": {"description", "ticket_type"},
}

INJECTED_ARGUMENTS = {"conversation_id", "tool_call_id"}
TICKET_TYPES = {"refund", "exchange", "logistics_query", "quality_issue", "other"}


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    seed_demo_data(factory)
    yield factory
    engine.dispose()


@pytest.fixture
def registry(session_factory):
    return build_tool_registry(session_factory)


@pytest.fixture
def conversation_ids(session_factory):
    with session_factory() as session:
        first = create_conversation(session)
        second = create_conversation(session)
        session.commit()
        return first.id, second.id


def invoke(tool, args, *, call_id="call-1", conversation_id="conv-1"):
    merged = {**args, "conversation_id": conversation_id}
    return tool.invoke({"name": tool.name, "args": merged, "id": call_id, "type": "tool_call"})


def content(tool, args, **kwargs):
    return json.loads(invoke(tool, args, **kwargs).content)


def test_registry_exposes_exactly_the_approved_tools(registry):
    assert set(registry) == set(EXPECTED_VISIBLE_SCHEMAS)
    assert len(registry) == 5
    assert all(registry[name].name == name for name in registry)
    for unexpected in ("delete_ticket", "query_orders", "execute_sql"):
        assert unexpected not in registry


def test_model_visible_schemas_are_exact_and_hide_runtime_arguments(registry):
    for name, expected in EXPECTED_VISIBLE_SCHEMAS.items():
        schema = registry[name].tool_call_schema.model_json_schema()
        assert set(schema["properties"]) == expected
        assert set(schema.get("required", [])) == expected
        assert set(schema["properties"]).isdisjoint(INJECTED_ARGUMENTS)
        internal = registry[name].args_schema.model_json_schema()["properties"]
        assert INJECTED_ARGUMENTS <= set(internal)


def test_visible_inputs_are_bounded_non_empty_strings(registry):
    for name, expected in EXPECTED_VISIBLE_SCHEMAS.items():
        properties = registry[name].tool_call_schema.model_json_schema()["properties"]
        for field in expected - {"ticket_type"}:
            assert properties[field]["type"] == "string"
            assert properties[field]["minLength"] == 1
            assert properties[field]["maxLength"] > 1

    ticket_schema = registry["create_ticket"].tool_call_schema.model_json_schema()["properties"]
    assert set(ticket_schema["ticket_type"]["enum"]) == TICKET_TYPES

    with pytest.raises(ValidationError):
        invoke(registry["query_order"], {"order_id": "   "})
    with pytest.raises(ValidationError):
        invoke(registry["query_faq"], {"keyword": ""})
    with pytest.raises(ValidationError):
        invoke(registry["query_faq"], {"keyword": "k" * 200})
    with pytest.raises(ValidationError):
        invoke(registry["query_product"], {"product_id": "P-1", "unexpected": "x"})
    with pytest.raises(ValidationError):
        invoke(registry["create_ticket"], {"description": "工单", "ticket_type": "not_a_type"})


def test_demo_data_is_deterministic_across_calls_and_registries(session_factory, registry):
    rebuilt = build_tool_registry(session_factory)
    for name, args in (
        ("query_order", {"order_id": "1001"}),
        ("query_product", {"product_id": "P-1001"}),
        ("query_logistics", {"order_id": "1001"}),
    ):
        first = content(registry[name], args)
        repeat = content(registry[name], args, call_id="call-2")
        separately_built = content(rebuilt[name], args)
        assert first == repeat == separately_built
        assert json.loads(json.dumps(first, ensure_ascii=False)) == first

    variants = {
        json.dumps(content(registry["query_order"], {"order_id": str(number)}), sort_keys=True)
        for number in range(1001, 1008)
    }
    assert len(variants) > 1


def test_demo_data_is_stable_in_a_fresh_python_process():
    program = (
        "import json;"
        "from app.tools.mock_data import build_order, build_logistics;"
        "print(json.dumps([build_order('1001'), build_logistics('1001')], sort_keys=True, ensure_ascii=False))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    expected = json.dumps(
        [mock_data.build_order("1001"), mock_data.build_logistics("1001")],
        sort_keys=True,
        ensure_ascii=False,
    )
    assert completed.stdout.strip() == expected


def test_all_five_tools_return_json_objects(registry, conversation_ids):
    conversation_id, _ = conversation_ids
    calls = (
        ("query_order", {"order_id": "1001"}),
        ("query_product", {"product_id": "P-1001"}),
        ("query_logistics", {"order_id": "1001"}),
        ("query_faq", {"keyword": "退货"}),
        ("create_ticket", {"description": "需要人工协助", "ticket_type": "other"}),
    )
    for index, (name, args) in enumerate(calls):
        payload = content(registry[name], args, call_id=f"call-{index}", conversation_id=conversation_id)
        assert isinstance(payload, dict)
        assert json.dumps(payload, ensure_ascii=False)


def test_faq_like_finds_seeded_return_policy(registry):
    for keyword in ("退货", "七天"):
        payload = content(registry["query_faq"], {"keyword": keyword})
        assert payload["found"] is True
        assert any(
            match["question"] == "演示商品可以退货吗？" and "七天" in match["answer"]
            for match in payload["matches"]
        )


def test_faq_postage_returns_explicit_no_match(registry):
    payload = content(registry["query_faq"], {"keyword": "邮费"})
    assert payload["matches"] == []
    assert payload["found"] is False


def test_query_faq_echoes_the_stripped_keyword_it_searches_with(registry):
    payload = content(registry["query_faq"], {"keyword": "  退货  "})
    assert payload["keyword"] == "退货"
    assert payload["found"] is True


def test_faq_like_escapes_wildcard_characters(session_factory, registry):
    with session_factory() as session:
        session.add(Faq(question="折扣 100% 全含", answer="演示：仅演示用途", category="演示"))
        session.add(Faq(question="编号 A_B 示例", answer="演示：仅演示用途", category="演示"))
        session.commit()

        percent_matches = [row.question for row in search_faq(session, "%")]
        underscore_matches = [row.question for row in search_faq(session, "_")]

    assert percent_matches == ["折扣 100% 全含"]
    assert underscore_matches == ["编号 A_B 示例"]


def test_create_ticket_persists_and_retries_idempotently(session_factory, registry, conversation_ids):
    conversation_id, other_conversation_id = conversation_ids
    arguments = {"description": "需要退货工单", "ticket_type": "refund"}

    first = content(registry["create_ticket"], arguments, call_id="call-1", conversation_id=conversation_id)
    retry = content(registry["create_ticket"], arguments, call_id="call-1", conversation_id=conversation_id)

    assert first["ticket_no"] == retry["ticket_no"]
    assert first["status"] == "open"
    assert first["ticket_type"] == "refund"

    new_call = content(registry["create_ticket"], arguments, call_id="call-2", conversation_id=conversation_id)
    new_conversation = content(
        registry["create_ticket"], arguments, call_id="call-1", conversation_id=other_conversation_id
    )

    assert new_call["ticket_no"] != first["ticket_no"]
    assert new_conversation["ticket_no"] != first["ticket_no"]

    with session_factory() as session:
        persisted = session.scalars(select(Ticket).where(Ticket.conversation_id == conversation_id)).all()

    assert len(persisted) == 2
    assert {ticket.ticket_no for ticket in persisted} == {first["ticket_no"], new_call["ticket_no"]}
