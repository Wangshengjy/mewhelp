"""SQLAlchemy models for the MewHelp customer support data."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """The application's single declarative base."""


class Faq(Base):
    __tablename__ = "faq"
    __table_args__ = (UniqueConstraint("question", "category"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(String(255), nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="demo-user")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    # ch07 · 三层上下文:最近几段梗概投影 + 两个分层锚点(降级只挪锚点不搬数据)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_upto_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    layer1_from_msg_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class ConversationSummary(Base):
    """ch07 · 分段摘要,一段一行只追加:压完的段落不回炉重压,一个事实只经历一次有损压缩。

    层次边界用消息 id 表达:seq 从 1 递增,[from_msg_id, upto_msg_id] 为该段覆盖的闭区间。
    注意:conversations.id 在本仓库是 UUID 字符串(课程参考是 BIGINT),故此处为 String(36)。
    """

    __tablename__ = "conversation_summaries"
    __table_args__ = (UniqueConstraint("conversation_id", "seq", name="uk_conv_seq"),)

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    from_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    upto_msg_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tool_calls: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class Ticket(Base):
    __tablename__ = "tickets"
    __table_args__ = (UniqueConstraint("conversation_id", "source_tool_call_id"),)

    ticket_no: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    source_tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class KnowledgeChunk(Base):
    """ch03 · 知识库 chunk 原文权威源，双写状态机 pending→done。"""

    __tablename__ = "knowledge_chunks"

    # SQLite 测试库只对 INTEGER 主键自增，故 BigInteger 加 sqlite 变体（MySQL 仍为 BIGINT）
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    category: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    questions: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    is_key_clause: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    prev_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id"), nullable=True
    )
    next_chunk_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("knowledge_chunks.id"), nullable=True
    )
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vectorize_status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class QaExtractionStaging(Base):
    """ch03 · 历史对话抽 QA 的离线中转暂存表：分批抽取、去重、人工审核。"""

    __tablename__ = "qa_extraction_staging"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    batch_no: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="extracted", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)


class ReviewQueue(Base):
    """ch09 · 飞轮待审队列：一行 = 一个去重后的知识缺口；查重命中累加 occurrence_count 不新建行。

    review_status 是中文值 ENUM（'待审'/'通过'/'驳回'，MySQL 侧由 sql/ch09-ddl.sql 约束），
    驳回即终审：查重命中已驳回/已通过的行只累加次数、状态不变。
    """

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    normalized_question: Mapped[str] = mapped_column(String(512), nullable=False)
    ai_suggested_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    review_status: Mapped[str] = mapped_column(
        Enum("待审", "通过", "驳回"), nullable=False, default="待审", server_default="待审", index=True
    )
    approved_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now
    )


class EvalRun(Base):
    """ch09 · 自动化评估流水线：一行 = 一轮评估；metrics JSON 收各指标分数，按时间连成趋势线。"""

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    triggered_by: Mapped[str] = mapped_column(Enum("定时", "手动"), nullable=False, default="定时", server_default="定时")
    dataset_size: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class LowConfidenceQuestion(Base):
    """ch04 · 低置信度问题池：检索证据低 / 生成自评不足时拒答落池，ch09 飞轮入口。

    MySQL 侧由 sql/ch04-ddl.sql 建表（ENUM 精确约束）；ORM 用于 SQLite 测试与写入。
    ch09 追加两列：retrieved_chunks 落池时召回快照（审核页给人看）、
    matched_review_id 查重后归并落点（飞轮处理游标：IS NULL 即待处理）。
    """

    __tablename__ = "low_confidence_questions"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id"), nullable=True, index=True
    )
    raw_question: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)  # ch09 · 落池时召回快照
    matched_review_id: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), ForeignKey("review_queue.id"), nullable=True, index=True
    )  # ch09 · 查重归并落点
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)


class TopicClassification(Base):
    """ch10 · 主题分类结果：旁路批量归类，一行 = 一条问题的一次归类；labels 存命中类目名数组。

    对齐 sql/ch10-ddl.sql（uk_question_id 保证一问一归类，重跑幂等靠「先查未归类再写」）。
    实时对话主链路不写这张表。
    """

    __tablename__ = "topic_classifications"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    question_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("low_confidence_questions.id"),
        nullable=False,
        unique=True,
    )
    labels: Mapped[list] = mapped_column(JSON, nullable=False)
    classified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=utc_now
    )


class FaithCase(Base):
    """ch04 · 忠实度编造个案台账：一题一行（uk_eval_id），跨轮累计 seen_count。

    已处置个案再次被判编造 → 状态自动退回「未解决」（复发），resolved_at 保留以标记复发。
    """

    __tablename__ = "faith_cases"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    eval_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)
    bucket: Mapped[str] = mapped_column(String(24), nullable=False)
    query: Mapped[str] = mapped_column(String(512), nullable=False)
    strategy: Mapped[str] = mapped_column(String(24), nullable=False, default="hybrid_rerank")
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list | None] = mapped_column(JSON, nullable=True)
    judge_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="未解决", index=True)
    seen_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now, index=True)
    resolution: Mapped[str | None] = mapped_column(String(300), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ToolAuditLog(Base):
    """ch08 · 工具调用审计留痕:统一执行引擎每次调用落一条(含被权限拒、被校验拦的)。

    对齐 sql/ch08-ddl.sql。不挂外键:审计写入不能被引用约束拦住;conversation_id
    本仓库是 UUID 字符串(课程参考是 BIGINT),故 String(64) 只建普通索引。
    """

    __tablename__ = "tool_audit_logs"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    conversation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    tool_source: Mapped[str] = mapped_column(Enum("builtin", "mcp"), nullable=False)
    mcp_server: Mapped[str | None] = mapped_column(String(64), nullable=True)
    arguments: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        Enum("成功", "失败", "超时", "校验拦下", "权限拒绝"), nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(String(512), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utc_now)
