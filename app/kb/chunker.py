"""Structure-aware Markdown chunking for the knowledge base (ch03).

Rules implemented per the ch03 spec:
- split by Markdown heading levels (#/##/###), questions = section title,
  category = parent heading path;
- a section longer than ~800 chars is recursively split by paragraphs and then
  sentences;
- adjacent split chunks keep ~100 chars of overlap, trimmed to the nearest
  sentence boundary so no half sentence is left;
- long Markdown tables are split by rows and every block re-copies the header
  row;
- FAQ-style bodies may carry question lines (``问：``/``Q：``) which are moved
  into ``questions``.

Metadata only (never vectorized): ``section_path``, ``content_type``,
``is_key_clause``, ``is_table_block`` and the prev/next chunk pointers that are
back-filled at insert time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 100

KEY_CLAUSE_KEYWORDS: tuple[str, ...] = (
    "时效",
    "免费",
    "包邮",
    "退款",
    "赔付",
    "违约",
    "隐私",
    "7天",
    "30天",
    "七天",
    "三十天",
    "运费",
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_QUESTION_LINE_RE = re.compile(r"^[-*]?\s*(?:问|Q|q)\s*[:：]\s*(.+)$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?])\s*|\n+")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}")
_UNCLASSIFIED = "未分类"


@dataclass(slots=True)
class Chunk:
    """One knowledge chunk ready for the dual write (MySQL + Milvus)."""

    category: str
    questions: str
    answer: str
    section_path: str
    content_type: str
    is_key_clause: bool
    is_table_block: bool
    prev_chunk_id: int | None = None
    next_chunk_id: int | None = None


def contains_key_clause(*texts: str) -> bool:
    """Return whether any text hits one of the key-clause keywords."""
    joined = "\n".join(texts)
    return any(keyword in joined for keyword in KEY_CLAUSE_KEYWORDS)


def _split_sentences(text: str) -> list[str]:
    """Split text into whole sentences (delimiters kept, no half sentences)."""
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text)]
    return [part for part in parts if part]


def _split_oversized_sentence(sentence: str, max_chars: int) -> list[str]:
    """Hard-cut a single sentence that alone exceeds the budget."""
    return [sentence[start : start + max_chars] for start in range(0, len(sentence), max_chars)]


def _pack_units(units: list[str], max_chars: int, separator: str) -> list[str]:
    """Greedily pack atomic units into blocks no larger than max_chars."""
    blocks: list[str] = []
    current: list[str] = []
    current_length = 0
    for unit in units:
        extra = len(unit) + (len(separator) if current else 0)
        if current and current_length + extra > max_chars:
            blocks.append(separator.join(current))
            current, current_length = [unit], len(unit)
        else:
            current.append(unit)
            current_length += extra
    if current:
        blocks.append(separator.join(current))
    return blocks


def _sentence_aligned_overlap(previous_block: str, overlap_chars: int) -> str:
    """Return the tail of the previous block that starts at a sentence boundary."""
    tail = previous_block[-overlap_chars:]
    boundary = re.search(r"[。！？!?]", tail)
    if boundary is None:
        return ""
    return tail[boundary.end() :].strip()


def split_long_text(
    text: str, max_chars: int = MAX_CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[str]:
    """Recursively split long text: paragraphs, then sentences, then hard cuts."""
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            for sentence in _split_sentences(paragraph):
                if len(sentence) <= max_chars:
                    units.append(sentence)
                else:
                    units.extend(_split_oversized_sentence(sentence, max_chars))
    blocks = _pack_units(units, max_chars, "\n")
    if len(blocks) <= 1:
        return blocks
    overlapped: list[str] = [blocks[0]]
    for block in blocks[1:]:
        tail = _sentence_aligned_overlap(overlapped[-1], overlap_chars)
        overlapped.append(f"{tail}\n{block}" if tail else block)
    return overlapped


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2


def _split_table_rows(lines: list[str], max_chars: int) -> list[str]:
    """Split a long table by rows; every block re-copies the header row."""
    header = lines[0].rstrip()
    separator = lines[1].rstrip() if len(lines) > 1 and _TABLE_SEPARATOR_RE.match(lines[1]) else None
    prefix = [header] + ([separator] if separator else [])
    prefix_length = len("\n".join(prefix))
    rows = lines[2:] if separator else lines[1:]
    blocks: list[str] = []
    current: list[str] = []
    current_length = prefix_length
    for row in rows:
        row = row.rstrip()
        extra = len(row) + (1 if current else 0)
        if current and current_length + extra > max_chars:
            blocks.append("\n".join(prefix + current))
            current, current_length = [row], prefix_length + len(row)
        else:
            current.append(row)
            current_length += extra
    if current:
        blocks.append("\n".join(prefix + current))
    return blocks


def _split_section_body(
    body: str, max_chars: int = MAX_CHUNK_CHARS, overlap_chars: int = OVERLAP_CHARS
) -> list[tuple[str, bool]]:
    """Split one section body into (text, is_table_block) parts."""
    if len(body) <= max_chars:
        return [(body.strip(), False)]
    parts: list[tuple[str, bool]] = []
    lines = body.splitlines()
    paragraph: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if _is_table_line(line):
            if paragraph:
                parts.extend(
                    (block, False) for block in split_long_text("\n".join(paragraph), max_chars, overlap_chars)
                )
                paragraph = []
            table_lines: list[str] = []
            while index < len(lines) and _is_table_line(lines[index]):
                table_lines.append(lines[index])
                index += 1
            table_text = "\n".join(table_lines)
            if len(table_text) <= max_chars:
                parts.append((table_text, False))
            else:
                parts.extend((block, True) for block in _split_table_rows(table_lines, max_chars))
            continue
        paragraph.append(line)
        index += 1
    if paragraph:
        parts.extend((block, False) for block in split_long_text("\n".join(paragraph), max_chars, overlap_chars))
    return parts


@dataclass(slots=True)
class _Section:
    """One heading with its heading path and collected body lines."""

    title: str
    path: tuple[str, ...]
    body_lines: list[str]


def _parse_sections(markdown_text: str) -> tuple[str, list[_Section]]:
    """Parse the document into (document title, ordered body sections)."""
    title = ""
    stack: list[tuple[int, str]] = []
    sections: list[_Section] = []

    for line in markdown_text.splitlines():
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            if level == 1 and not title:
                title = heading_text
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            sections.append(
                _Section(title=heading_text, path=tuple(name for _, name in stack), body_lines=[])
            )
            continue
        if stack:
            sections[-1].body_lines.append(line)

    return title, [section for section in sections if "\n".join(section.body_lines).strip()]


def _clean_faq_title(title: str) -> str:
    stripped = re.sub(r"^[-*]\s*", "", title.strip())
    match = re.match(r"^(?:问|Q|q)\s*[:：]\s*(.+)$", stripped)
    return match.group(1).strip() if match else stripped


def document_title(markdown_text: str) -> str:
    """Return the first level-1 heading, or an empty string when absent."""
    return _parse_sections(markdown_text)[0]


def chunk_document(markdown_text: str, content_type: str) -> list[Chunk]:
    """Chunk one Markdown document into knowledge chunks."""
    _doc_title, sections = _parse_sections(markdown_text)
    chunks: list[Chunk] = []
    for section in sections:
        path = list(section.path) or [_UNCLASSIFIED]
        section_path = " > ".join(path)
        category = " > ".join(path[:-1]) or path[0]

        questions: list[str] = []
        if section.title:
            questions.append(_clean_faq_title(section.title) if content_type == "faq" else section.title)

        question_lines: list[str] = []
        body_lines: list[str] = []
        for line in section.body_lines:
            match = _QUESTION_LINE_RE.match(line.strip())
            if match and content_type == "faq":
                question_lines.append(match.group(1).strip())
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        for extra_question in question_lines:
            if extra_question not in questions:
                questions.append(extra_question)

        for part_text, is_table_block in _split_section_body(body):
            chunks.append(
                Chunk(
                    category=category,
                    questions="\n".join(questions),
                    answer=part_text,
                    section_path=section_path,
                    content_type=content_type,
                    is_key_clause=contains_key_clause(part_text, "\n".join(questions), category),
                    is_table_block=is_table_block,
                )
            )
    return chunks


def resolve_content_type(document_name: str) -> str:
    """Map a document file name to its content type."""
    lowered = document_name.lower()
    if "faq" in lowered:
        return "faq"
    if "手册" in document_name or "manual" in lowered:
        return "manual"
    if "spec" in lowered:
        return "spec"
    return "policy"
