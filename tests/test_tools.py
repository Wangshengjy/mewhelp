"""ch08 · 内置工具行为测试(query_logistics 已由 MCP 接管,不在内置清单)。

注册中心三要素/可见口径精确性 → tests/tools/test_registry.py;
工具行为(mock 数据确定性、faq 检索、工单幂等)保留在这里。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

from app.database import build_engine, build_session_factory
from app.demo_seed import seed_demo_data
from app.models import Base, Faq, Ticket
from app.repositories.conversations import create_conversation
from app.repositories.faq import search_faq
from app.settings import Settings
from app.tools import mock_data
from app.tools.builtin.faq import query_faq
from app.tools.builtin.orders import query_order, query_product
from app.tools.builtin.tickets import create_ticket

SQLITE_URL = "sqlite+pysqlite:///:memory:"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    seed_demo_data(factory)
    yield factory
    engine.dispose()


@pytest.fixture
def conversation_ids(session_factory):
    with session_factory() as session:
        first = create_conversation(session)
        second = create_conversation(session)
        session.commit()
        return first.id, second.id


def invoke(tool, args, *, call_id="call-1", conversation_id="conv-1", session_factory=None, settings=None):
    """内置工具直调:运行期注入参数由测试扮演引擎注入(tool_call_id 由调用 id 自动注入)。"""
    merged = {
        **args,
        "conversation_id": conversation_id,
        "session_factory": session_factory,
        "settings": settings,
    }
    return tool.invoke({"name": tool.name, "args": merged, "id": call_id, "type": "tool_call"})


def content(tool, args, **kwargs):
    return json.loads(invoke(tool, args, **kwargs).content)


def test_demo_data_is_deterministic_across_calls(session_factory):
    order_payload = content(query_order, {"order_id": "1001"})
    for tool, args in (
        (query_order, {"order_id": "1001"}),
        (query_product, {"product_id": "P-1001"}),
    ):
        first = content(tool, args)
        repeat = content(tool, args, call_id="call-2")
        assert first == repeat
        assert json.loads(json.dumps(first, ensure_ascii=False)) == first

    variants = {
        json.dumps(content(query_order, {"order_id": str(number)}), sort_keys=True)
        for number in range(1001, 1008)
    }
    assert len(variants) > 1


def test_demo_data_is_stable_in_a_fresh_python_process():
    program = (
        "import json;"
        "from app.tools.mock_data import build_order, build_logistics;"
        "order = build_order('1001');"
        "print(json.dumps([order, build_logistics(order['tracking_no'])], sort_keys=True, ensure_ascii=False))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    demo_order = mock_data.build_order("1001")
    expected = json.dumps(
        [demo_order, mock_data.build_logistics(demo_order["tracking_no"])],
        sort_keys=True,
        ensure_ascii=False,
    )
    assert completed.stdout.strip() == expected


def test_all_five_builtin_tools_return_json_objects(session_factory, conversation_ids):
    conversation_id, _ = conversation_ids
    calls = (
        (query_order, {"order_id": "1001"}),
        (query_product, {"product_id": "P-1001"}),
        (query_faq, {"keyword": "退货"}),
        (create_ticket, {"description": "需要人工协助", "ticket_type": "售后"}),
    )
    for index, (tool, args) in enumerate(calls):
        payload = content(
            tool, args, call_id=f"call-{index}", conversation_id=conversation_id,
            session_factory=session_factory, settings=Settings.hermetic_defaults(),
        )
        assert isinstance(payload, dict)
        assert json.dumps(payload, ensure_ascii=False)


def test_faq_like_finds_seeded_return_policy(session_factory):
    settings = Settings.hermetic_defaults()
    for keyword in ("退货", "七天"):
        payload = content(query_faq, {"keyword": keyword}, session_factory=session_factory, settings=settings)
        assert payload["found"] is True
        assert any(
            match["question"] == "演示商品可以退货吗？" and "七天" in match["answer"]
            for match in payload["matches"]
        )


def test_faq_postage_returns_explicit_no_match(session_factory):
    payload = content(
        query_faq, {"keyword": "邮费"},
        session_factory=session_factory, settings=Settings.hermetic_defaults(),
    )
    assert payload["matches"] == []
    assert payload["found"] is False


def test_query_faq_echoes_the_stripped_keyword_it_searches_with(session_factory):
    payload = content(
        query_faq, {"keyword": "  退货  "},
        session_factory=session_factory, settings=Settings.hermetic_defaults(),
    )
    assert payload["keyword"] == "退货"
    assert payload["found"] is True


def test_faq_like_escapes_wildcard_characters(session_factory):
    with session_factory() as session:
        session.add(Faq(question="折扣 100% 全含", answer="演示：仅演示用途", category="演示"))
        session.add(Faq(question="编号 A_B 示例", answer="演示：仅演示用途", category="演示"))
        session.commit()

        percent_matches = [row.question for row in search_faq(session, "%")]
        underscore_matches = [row.question for row in search_faq(session, "_")]

    assert percent_matches == ["折扣 100% 全含"]
    assert underscore_matches == ["编号 A_B 示例"]


def test_create_ticket_persists_and_retries_idempotently(session_factory, conversation_ids):
    conversation_id, other_conversation_id = conversation_ids
    arguments = {"description": "需要退货工单", "ticket_type": "售后"}

    first = content(
        create_ticket, arguments, call_id="call-1", conversation_id=conversation_id,
        session_factory=session_factory, settings=Settings.hermetic_defaults(),
    )
    retry = content(
        create_ticket, arguments, call_id="call-1", conversation_id=conversation_id,
        session_factory=session_factory, settings=Settings.hermetic_defaults(),
    )

    assert first["ticket_no"] == retry["ticket_no"]  # 同一申请单重试幂等
    assert first["status"] == "已转人工"
    assert first["ticket_type"] == "售后"

    new_call = content(
        create_ticket, arguments, call_id="call-2", conversation_id=conversation_id,
        session_factory=session_factory, settings=Settings.hermetic_defaults(),
    )
    new_conversation = content(
        create_ticket, arguments, call_id="call-1", conversation_id=other_conversation_id,
        session_factory=session_factory, settings=Settings.hermetic_defaults(),
    )

    assert new_call["ticket_no"] != first["ticket_no"]
    assert new_conversation["ticket_no"] != first["ticket_no"]

    with session_factory() as session:
        persisted = session.scalars(select(Ticket).where(Ticket.conversation_id == conversation_id)).all()

    assert len(persisted) == 2
    assert {ticket.ticket_no for ticket in persisted} == {first["ticket_no"], new_call["ticket_no"]}
