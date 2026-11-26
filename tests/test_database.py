from fastapi.testclient import TestClient
from sqlalchemy import inspect, select
from sqlalchemy.engine import URL

from app.database import build_engine, build_session_factory, initialize_database
from app.main import create_app
from app.models import Base, Conversation, Faq, Message, Ticket
from app.settings import Settings


SQLITE_URL = "sqlite+pysqlite:///:memory:"
TEST_SETTINGS = Settings(
    chat_base_url="https://example.test/v1",
    chat_model="test-model",
    chat_api_key="test-key",
)


def test_sqlite_application_factory_creates_exact_schema_and_constraints():
    app = create_app(settings_override=TEST_SETTINGS, database_url_override=SQLITE_URL)

    with TestClient(app):
        inspector = inspect(app.state.engine)
        assert set(inspector.get_table_names()) == {
            "faq",
            "conversations",
            "messages",
            "tickets",
            "knowledge_chunks",
            "qa_extraction_staging",
            "low_confidence_questions",
            "faith_cases",
            "conversation_summaries",
            "tool_audit_logs",  # ch08 · 工具调用审计留痕
            "review_queue",  # ch09 · 飞轮待审队列
            "eval_runs",  # ch09 · 评估流水线轮次
            "topic_classifications",  # ch10 · 微调分类器旁路归类结果
        }
        assert {column["name"] for column in inspector.get_columns("conversation_summaries")} == {
            "id",
            "conversation_id",
            "seq",
            "from_msg_id",
            "upto_msg_id",
            "content",
            "created_at",
        }
        assert {column["name"] for column in inspector.get_columns("faq")} == {
            "id",
            "question",
            "answer",
            "category",
        }
        assert {column["name"] for column in inspector.get_columns("conversations")} == {
            "id",
            "user_id",
            "status",
            "summary",
            "summary_upto_msg_id",
            "layer1_from_msg_id",
            "created_at",
        }
        assert {column["name"] for column in inspector.get_columns("messages")} == {
            "id",
            "conversation_id",
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "tool_name",
            "created_at",
        }
        assert {column["name"] for column in inspector.get_columns("tickets")} == {
            "ticket_no",
            "conversation_id",
            "description",
            "ticket_type",
            "status",
            "created_at",
            "source_tool_call_id",
        }
        assert inspector.get_pk_constraint("faq")["constrained_columns"] == ["id"]
        assert inspector.get_pk_constraint("conversations")["constrained_columns"] == ["id"]
        assert inspector.get_pk_constraint("messages")["constrained_columns"] == ["id"]
        assert inspector.get_pk_constraint("tickets")["constrained_columns"] == ["ticket_no"]
        assert inspector.get_pk_constraint("knowledge_chunks")["constrained_columns"] == ["id"]
        assert inspector.get_pk_constraint("qa_extraction_staging")["constrained_columns"] == ["id"]
        assert {tuple(item["column_names"]) for item in inspector.get_unique_constraints("faq")} == {
            ("question", "category")
        }
        assert ("conversation_id", "source_tool_call_id") in {
            tuple(item["column_names"]) for item in inspector.get_unique_constraints("tickets")
        }
        assert inspector.get_foreign_keys("messages")[0]["referred_table"] == "conversations"
        assert inspector.get_foreign_keys("tickets")[0]["referred_table"] == "conversations"
        knowledge_chunk_fks = {
            (tuple(item["constrained_columns"]), item["referred_table"]) for item in inspector.get_foreign_keys("knowledge_chunks")
        }
        assert (("prev_chunk_id",), "knowledge_chunks") in knowledge_chunk_fks
        assert (("next_chunk_id",), "knowledge_chunks") in knowledge_chunk_fks
        assert app.state.settings.database_url.drivername == "mysql+pymysql"


def test_initialization_seeds_one_idempotent_tool_trace_and_ticket():
    engine = build_engine(SQLITE_URL)
    session_factory = build_session_factory(engine)
    Base.metadata.create_all(engine)

    initialize_database(session_factory)
    initialize_database(session_factory)

    with session_factory() as session:
        faqs = session.scalars(select(Faq)).all()
        conversations = session.scalars(select(Conversation)).all()
        messages = session.scalars(select(Message).order_by(Message.id)).all()
        tickets = session.scalars(select(Ticket)).all()

    assert len(faqs) >= 2
    assert all("邮费" not in f"{faq.question}{faq.answer}{faq.category}" for faq in faqs)
    assert len(conversations) == 1
    assert conversations[0].user_id == "demo-user"
    assert conversations[0].status == "open"
    assert [(message.role, message.tool_name) for message in messages] == [
        ("user", None),
        ("assistant", "query_faq"),
        ("tool", "query_faq"),
        ("assistant", None),
    ]
    assert messages[1].tool_calls
    assert messages[2].tool_call_id == messages[1].tool_call_id
    assert len(tickets) == 1
    assert tickets[0].conversation_id == conversations[0].id
    assert tickets[0].status == "open"
    assert tickets[0].source_tool_call_id


def test_mysql_engine_keeps_the_configured_credentials():
    settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        mysql_host="127.0.0.1",
        mysql_port=3307,
        mysql_user="demo-user",
        mysql_password="p@ss:w/ord",
    )

    engine = build_engine(settings.database_url)
    _, connect_args = engine.dialect.create_connect_args(engine.url)

    assert engine.url.port == 3307
    assert engine.url.database == "mewhelp"
    assert engine.url.password == "p@ss:w/ord"
    assert connect_args["user"] == "demo-user"
    assert connect_args["password"] == "p@ss:w/ord"


def test_application_factory_hands_the_mysql_url_to_the_engine(monkeypatch):
    captured = {}

    def capture(database_url):
        captured["database_url"] = database_url
        return build_engine(SQLITE_URL)

    monkeypatch.setattr("app.main.build_engine", capture)
    settings = Settings(
        chat_base_url="https://example.test/v1",
        chat_model="test-model",
        chat_api_key="test-key",
        mysql_host="127.0.0.1",
        mysql_port=3307,
        mysql_user="demo-user",
        mysql_password="s3cret-value",
    )
    app = create_app(settings_override=settings)

    with TestClient(app):
        pass

    database_url = captured["database_url"]
    assert isinstance(database_url, URL), "the lifespan must pass the URL object, not its masked string form"
    assert database_url.port == 3307
    assert database_url.password == "s3cret-value"
