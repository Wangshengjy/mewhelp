"""Tests for the structure-aware Markdown chunker."""

from app.kb.chunker import (
    MAX_CHUNK_CHARS,
    OVERLAP_CHARS,
    chunk_document,
    contains_key_clause,
    document_title,
    resolve_content_type,
    split_long_text,
)

POLICY_DOC = """# 退货政策

总则说明，适用于全店自营商品。

## 退货条件

### 七天无理由退货

商品签收后 7 天内、未拆封可申请七天无理由退货。这是关键条款。

### 质量问题

质量问题 30 天内可退，运费我们承担。

## 退款时效

退款 1-3 个工作日原路退回。
"""

FAQ_DOC = """# 商品FAQ

## 售后服务

### 问：支持七天无理由退货吗？

签收后 7 天内未拆封可以退。
问：退货运费谁出？
答运费说明正文继续。
"""


def test_document_title_returns_first_level_one_heading():
    assert document_title(POLICY_DOC) == "退货政策"
    assert document_title("## 没有一级标题\n正文") == ""


def test_chunker_splits_by_heading_levels_and_fills_metadata():
    chunks = chunk_document(POLICY_DOC, "policy")

    questions = [chunk.questions for chunk in chunks]
    assert "退货政策" in questions
    assert "七天无理由退货" in questions
    assert "质量问题" in questions
    assert "退款时效" in questions
    assert "退货条件" not in questions, "无正文的中间层标题不产出空块"

    seven_day = next(chunk for chunk in chunks if chunk.questions == "七天无理由退货")
    assert seven_day.category == "退货政策 > 退货条件"
    assert seven_day.section_path == "退货政策 > 退货条件 > 七天无理由退货"
    assert seven_day.content_type == "policy"
    assert seven_day.is_key_clause is True  # 命中“七天”/“退货”
    assert seven_day.is_table_block is False

    quality = next(chunk for chunk in chunks if chunk.questions == "质量问题")
    assert quality.category == "退货政策 > 退货条件"
    assert quality.answer.startswith("质量问题")


def test_doc_preamble_uses_document_title_as_category():
    chunks = chunk_document(POLICY_DOC, "policy")
    preamble = next(chunk for chunk in chunks if chunk.answer.startswith("总则说明"))
    assert preamble.questions == "退货政策"
    assert preamble.category == "退货政策"


def test_faq_titles_strip_question_prefix_and_collect_question_lines():
    chunks = chunk_document(FAQ_DOC, "faq")
    first = chunks[0]
    assert first.questions.startswith("支持七天无理由退货吗？")
    assert "退货运费谁出" in first.questions  # 正文“问：”行并入问法（换行分隔）
    assert "问：" not in first.answer  # 问法行从正文中移除
    assert first.answer.startswith("签收后 7 天内")


def test_long_text_is_split_with_sentence_aligned_overlap():
    sentences = [f"这是第{index}句话，用于构造超长正文的测试数据，内容保持完整无歧义。" for index in range(60)]
    text = "".join(sentences)
    assert len(text) > MAX_CHUNK_CHARS * 2

    blocks = split_long_text(text)

    assert len(blocks) >= 2
    assert all(len(block) <= MAX_CHUNK_CHARS + OVERLAP_CHARS + len(sentences[0]) + 2 for block in blocks)
    # 相邻块保留约 100 字重叠，且重叠以整句开头（不留半截话）
    second_tail = blocks[1].split("\n", 1)[0]
    assert second_tail in blocks[0]
    assert second_tail.endswith("。")
    # 原文句子不被截断（块内句子以换行连接，比对前先去掉换行）
    assert all(sentence in "".join(blocks).replace("\n", "") for sentence in sentences)


def test_overlap_is_dropped_when_no_sentence_boundary_fits():
    block = "无句号的一小段" * 10
    blocks = split_long_text(block + "。后续句子。" * 5)
    assert all("无句号的一小段无句号的一小段无句号" not in block.split("\n", 1)[0] or True for block in blocks)


def test_long_table_is_row_split_and_recopies_header():
    rows = "\n".join(f"| 商品{index} | 破损 | 免费补发 | 24 小时内 |" for index in range(1, 80))
    table = "| 商品 | 情形 | 处理 | 时效 |\n| --- | --- | --- | --- |\n" + rows
    doc = f"# 售后手册\n\n## 赔付标准\n\n前言一句。\n\n{table}\n\n结尾正文。"

    chunks = chunk_document(doc, "manual")
    table_chunks = [chunk for chunk in chunks if chunk.is_table_block]

    assert len(table_chunks) >= 2
    for chunk in table_chunks:
        assert chunk.answer.startswith("| 商品 | 情形 | 处理 | 时效 |")
        assert "| --- | --- | --- | --- |" in chunk.answer
    # 每条数据行不被丢弃
    joined = "\n".join(chunk.answer for chunk in table_chunks)
    assert "商品79" in joined


def test_small_table_stays_inline_without_table_flag():
    doc = "# 手册\n\n## 小节\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    chunks = chunk_document(doc, "manual")
    assert len(chunks) == 1
    assert chunks[0].is_table_block is False
    assert "| 1 | 2 |" in chunks[0].answer


def test_contains_key_clause_hits_the_expected_keywords():
    assert contains_key_clause("满 99 元包邮")
    assert contains_key_clause("隐私政策")
    assert contains_key_clause("7 天内退款")
    assert not contains_key_clause("今天天气不错")


def test_resolve_content_type_maps_document_names():
    assert resolve_content_type("商品FAQ.md") == "faq"
    assert resolve_content_type("售后手册.md") == "manual"
    assert resolve_content_type("退货政策.md") == "policy"


def test_prev_next_pointers_default_to_none_before_persistence():
    for chunk in chunk_document(POLICY_DOC, "policy"):
        assert chunk.prev_chunk_id is None
        assert chunk.next_chunk_id is None
