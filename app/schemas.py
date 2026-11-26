"""Validated API request and structured extraction schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, description="本轮用户消息")
    conversation_id: str | None = Field(default=None, description="已有会话 ID；省略时创建新会话")

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value.strip()


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1, description="用户描述的售后问题")

    @field_validator("description")
    @classmethod
    def description_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank")
        return value.strip()


class AfterSalesExtraction(BaseModel):
    """Three fixed fields extracted from a customer after-sales description."""

    model_config = ConfigDict(extra="forbid")

    order_number: str | None = Field(description="订单号原文；描述未提供时为 null")
    request_type: Literal["refund", "exchange", "logistics_query", "quality_issue"] | None = Field(
        description=(
            "用户希望客服办理的售后事项，不是商品故障原因；明确退款时填 refund，"
            "明确换新时填 exchange，查询发货或物流时填 logistics_query；只描述商品问题时填 quality_issue；无法判断时为 null"
        )
    )
    expected_solution: str | None = Field(
        description=(
            "用户明确提出的处理方式，简短规范地概括且不重复诉求类型；例如“换一副新的”概括为“换新”，"
            "“最好退回原支付方式”概括为“退回原支付方式”；未提出时为 null"
        )
    )


# ---------------------------------------------------------------------------
# ch03 · /kb 知识库页面请求模型
# ---------------------------------------------------------------------------


class KbPreviewRequest(BaseModel):
    """对粘贴的 Markdown 正文就地切块预览（只切不写）。"""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="粘贴的 Markdown 正文")
    content_type: Literal["faq", "policy", "manual"] = Field(default="policy", description="内容类型")


class KbEntryRequest(BaseModel):
    """手工录入：切块 + 指纹查重 + 入库，可选顺手向量化。"""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, description="Markdown 正文（需要标题 + 正文结构）")
    content_type: Literal["faq", "policy", "manual"] = Field(default="policy", description="内容类型")
    vectorize: bool = Field(default=False, description="入库后是否立即向量化写入 Milvus")


class KbSearchRequest(BaseModel):
    """检索自测；ch04 起支持四策略，mode=dense 保留为 ch03 兼容入口。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="检索问题")
    top_k: int = Field(default=5, ge=1, le=20, description="召回条数")
    mode: Literal["dense", "vector", "bm25", "hybrid", "hybrid_rerank"] = Field(
        default="dense", description="检索模式：dense=ch03 密集单路；其余为 ch04 四策略"
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class KbStagingActionRequest(BaseModel):
    """暂存行审核动作（approve=采纳入库 / reject=弃用留痕）。"""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1, description="qa_extraction_staging 行 ID")


class FeedbackRequest(BaseModel):
    """ch09 · 用户反馈入口（👍/👎）：rating 落 low_confidence_questions（source=user_feedback）。"""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1, description="会话 UUID（关卡点对账）")
    turn_index: int = Field(ge=1, description="第几轮问答（1 起，checkpoint 里数 HumanMessage）")
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=2000, description="可选补充说明")

    @field_validator("conversation_id")
    @classmethod
    def conversation_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("conversation_id must not be blank")
        return value.strip()
