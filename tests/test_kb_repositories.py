"""Tests for the knowledge_chunks repository and the dual-write state machine."""

import pytest
from sqlalchemy import select

from app.database import build_engine, build_session_factory
from app.models import Base, KnowledgeChunk
from app.repositories import knowledge as knowledge_repository
from app.repositories import qa_staging as staging_repository

SQLITE_URL = "sqlite+pysqlite:///:memory:"


@pytest.fixture
def session_factory():
    engine = build_engine(SQLITE_URL)
    Base.metadata.create_all(engine)
    factory = build_session_factory(engine)
    yield factory
    engine.dispose()


def insert_chunk(session_factory, questions="问法", **overrides):
    payload = {
        "category": "分类",
        "questions": questions,
        "answer": "正文答案",
        "section_path": "分类 > 小节",
        "content_type": "policy",
        "is_key_clause": False,
        "is_table_block": False,
    }
    payload.update(overrides)
    with session_factory() as session:
        row = knowledge_repository.insert_chunk(session, **payload)
        session.commit()
        return row.id


def test_insert_chunk_defaults_to_pending_and_gets_an_autoincrement_id(session_factory):
    first = insert_chunk(session_factory)
    second = insert_chunk(session_factory, questions="另一个问法")

    assert first is not None and second is not None and first != second
    with session_factory() as session:
        row = session.get(KnowledgeChunk, first)
        assert row.vectorize_status == "pending"
        assert row.vector_id is None


def test_fingerprint_is_sha256_of_category_questions_answer(session_factory):
    import hashlib

    expected = hashlib.sha256("分类\n问法\n正文答案".encode("utf-8")).hexdigest()
    assert knowledge_repository.chunk_fingerprint("分类", "问法", "正文答案") == expected


def test_get_by_fingerprint_finds_exact_match_only(session_factory):
    chunk_id = insert_chunk(session_factory)
    fingerprint = knowledge_repository.chunk_fingerprint("分类", "问法", "正文答案")

    with session_factory() as session:
        found = knowledge_repository.get_by_fingerprint(session, fingerprint)
        assert found is not None and found.id == chunk_id
        assert knowledge_repository.get_by_fingerprint(session, "no-such-fingerprint") is None


def test_list_fingerprints_covers_all_rows(session_factory):
    insert_chunk(session_factory)
    insert_chunk(session_factory, questions="另一个问法")

    with session_factory() as session:
        fingerprints = knowledge_repository.list_fingerprints(session)

    assert len(fingerprints) == 2


def test_pending_done_state_machine_marks_done_with_vector_id(session_factory):
    chunk_id = insert_chunk(session_factory)

    with session_factory() as session:
        assert [row.id for row in knowledge_repository.list_pending(session)] == [chunk_id]
        stats = knowledge_repository.stats(session)
        assert stats == {"pending": 1, "done": 0, "total": 1}

        knowledge_repository.mark_done(session, chunk_id, str(chunk_id))
        session.commit()

        assert knowledge_repository.list_pending(session) == []
        stats = knowledge_repository.stats(session)
        assert stats == {"pending": 0, "done": 1, "total": 1}
        row = session.get(KnowledgeChunk, chunk_id)
        assert row.vector_id == str(chunk_id)


def test_mark_done_on_missing_chunk_is_safe(session_factory):
    with session_factory() as session:
        assert knowledge_repository.mark_done(session, 999, "v") is None


def test_get_chunks_by_ids_preserves_request_order(session_factory):
    first = insert_chunk(session_factory)
    second = insert_chunk(session_factory, questions="第二个")

    with session_factory() as session:
        rows = knowledge_repository.get_chunks_by_ids(session, [second, first, 999, second])

    assert [row.id for row in rows] == [second, first, second]


def test_count_doc_chunks_distinguishes_document_and_mined_types(session_factory):
    insert_chunk(session_factory, content_type="policy")
    insert_chunk(session_factory, questions=" mined", content_type="mined")

    with session_factory() as session:
        assert knowledge_repository.count_doc_chunks(session) == 1
        assert knowledge_repository.count_doc_chunks(session, {"mined"}) == 1
        assert knowledge_repository.count_doc_chunks(session, {"manual"}) == 0


def test_delete_doc_chunks_removes_only_requested_types(session_factory):
    policy_id = insert_chunk(session_factory, content_type="policy")
    mined_id = insert_chunk(session_factory, questions=" mined", content_type="mined")

    with session_factory() as session:
        removed = knowledge_repository.delete_doc_chunks(session)
        session.commit()

    assert [row.id for row in removed] == [policy_id]
    with session_factory() as session:
        remaining = session.scalars(select(KnowledgeChunk)).all()
    assert [row.id for row in remaining] == [mined_id]


def test_relink_document_backfills_prev_next_pointers_in_second_pass(session_factory):
    ids = [insert_chunk(session_factory, questions=f"块{index}") for index in range(4)]

    with session_factory() as session:
        linked = knowledge_repository.relink_document(session, "分类", "policy")
        session.commit()
        rows = {
            row.id: row
            for row in session.scalars(select(KnowledgeChunk)).all()
        }

    assert linked == 4
    assert rows[ids[0]].prev_chunk_id is None and rows[ids[0]].next_chunk_id == ids[1]
    assert rows[ids[1]].prev_chunk_id == ids[0] and rows[ids[1]].next_chunk_id == ids[2]
    assert rows[ids[3]].prev_chunk_id == ids[2] and rows[ids[3]].next_chunk_id is None


def test_relink_document_matches_section_prefix_not_substring(session_factory):
    insert_chunk(session_factory)  # 分类 > 小节
    insert_chunk(session_factory, questions="其它文档", section_path="分类二 > 小节", category="分类二")

    with session_factory() as session:
        linked = knowledge_repository.relink_document(session, "分类", "policy")

    assert linked == 1


def test_qa_fingerprint_normalizes_punctuations_case_and_keeps_chinese(session_factory):
    import hashlib

    first = knowledge_repository.qa_fingerprint("邮费是多少？", "满 99 包邮！")
    second = knowledge_repository.qa_fingerprint("邮费是多少", "满99包邮")
    assert first == second

    expected = hashlib.sha256("邮费是多少\n满99包邮".encode("utf-8")).hexdigest()
    assert first == expected


def test_staging_rows_flow_through_statuses(session_factory):
    with session_factory() as session:
        row = staging_repository.insert_extracted(
            session, batch_no="mine-1", source_ref="conv-1", question="怎么退货", answer="七天无理由"
        )
        session.commit()
        row_id = row.id

        assert staging_repository.latest_batch_no(session) == "mine-1"
        assert [item.id for item in staging_repository.list_by_batch(session, "mine-1")] == [row_id]
        assert staging_repository.list_by_batch(session, "mine-2") == []
        assert staging_repository.staging_stats(session) == {
            "extracted": 1,
            "kept": 0,
            "discarded": 0,
            "rejected": 0,
            "approved": 0,
        }

        staging_repository.set_status(session, row_id, "kept")
        assert staging_repository.staging_stats(session)["kept"] == 1
        assert staging_repository.list_staging(session, statuses=["kept"])[0].id == row_id
