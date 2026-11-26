"""Environment-backed configuration for the customer support service."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv
from sqlalchemy import URL


@dataclass(frozen=True, slots=True)
class Settings:
    chat_base_url: str
    chat_model: str
    chat_api_key: str
    chat_temperature: float = 0.2
    chat_context_token_budget: int = 6000
    chat_token_safety_margin: int = 500
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_database: str = "mewhelp"
    mysql_user: str = "mewhelp"
    mysql_password: str = "mewhelp"
    tool_timeout_seconds: int = 10
    # ch03 · RAG：嵌入与向量库配置。EMBED_API_KEY 允许为空（空值时检索自动降级为关键词查表）。
    embed_api_key: str = ""
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_model: str = "BAAI/bge-m3"
    milvus_uri: str = "http://localhost:19530"
    # ch04 · RAG 进阶：混合检索（Milvus 2.5 BM25 + dense）、重排、两道闸与评估。
    retrieval_strategy: str = "hybrid_rerank"  # 在线默认检索策略 vector/bm25/hybrid/hybrid_rerank
    recall_top_k: int = 50  # BM25 与 dense 单路各召回的条数
    rrf_rank_constant: int = 60  # RRF 融合平滑参数 k
    rerank_api_key: str = ""  # 为空时重排降级为混合序（不中断在线链路）
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 10  # 精排保留条数
    rerank_min_score: float = 0.3  # 机械闸：最高相关度低于该值判证据不足
    self_check_temperature: float = 0.0  # 语义闸自评温度
    judge_model: str = "deepseek-v4-pro"  # 忠实度/覆盖度裁判模型
    judge_base_url: str = ""  # 裁判/评估生成段专用提供商；空=复用 CHAT 提供商
    judge_api_key: str = ""  # 同上；空=复用 CHAT_API_KEY
    eval_top_k: int = 5  # 评估指标 K（MRR@5 / Recall@5）
    # ch05 · Workflow + Agent 混合架构（LangGraph 骨架）
    max_agent_steps: int = 6  # ReAct 环最大步数（封顶，超则强停走兜底）
    # token 花销不在环里卡：那是成本控制，归 ch09 跟 Langfuse 的账一起看
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"  # LangGraph AsyncSqliteSaver 持久化文件
    # ch06 · 意图工程与人工接管
    intent_model: str = ""  # 意图分类专用模型；空=回落 chat_model（可与对话模型不同名）
    intent_small_model: str = ""  # 轻量意图模型占位（ch08+ 成本分层用，本章不参与分流）
    intent_mode: str = "accuracy"  # accuracy（求准，本章）| fast（轻量模型，ch08+）
    intent_conf_threshold: float = 0.6  # 置信度阈值（低置信留痕/落池参考）
    # ch07 · 会话上下文管理（预算倒推 + 三层分级 + 异步分段摘要）
    model_context_window: int = 0  # 模型上下文窗口；0=按 chat_model 查表（拿不准的模型查 Context7 补表）
    context_budget_turns: int = 20  # 滑窗按多少轮历史设计（超出交给分层降级）
    max_user_input_tokens: int = 2000  # 单条用户输入硬上限
    max_output_tokens: int = 4000  # 单轮 AI 输出上限（课程表 10000；本仓库 32768 窗口 + 6 步峰值下按健康预算校准，见 dev-notes/ch07.md）
    system_prompt_tokens: int = 700  # AGENT_SYSTEM + 工具 schema 的实测开销
    evidence_tokens_per_chunk: int = 160  # 单条证据块预算
    rerank_top_k: int = 5  # 证据条数口径（与 nodes MAX_ONLINE_MATCHES 同源）
    layer1_ratio: float = 0.7  # 层 1 占滑窗比例
    layer2_reply_keep_chars: int = 60  # 层 2 客服答复保留前多少字
    layer2_tool_keep_tokens: int = 200  # 层 2 工具占位预算
    tool_result_max_tokens: int = 1200  # 单条工具结果硬上限（课程表 3000，同上校准）
    agent_step_ai_tokens: int = 500  # 每步 AI 输出峰值
    summary_inject_segments: int = 3  # 注入最近几段摘要
    summary_tokens_per_segment: int = 250  # 每段摘要的 token 预算
    budget_safety_margin: int = 1000  # 预算安全垫
    zh_chars_per_token: float = 1.2  # 中文 token 换算口径
    context_window_max_tokens: int = 0  # 滑窗 token 上限显式覆盖；0=走预算倒推
    summary_model: str = ""  # 摘要生成模型；空=回落 chat_model
    summary_base_url: str = ""  # 摘要生成提供商；空=复用 CHAT
    summary_api_key: str = ""  # 同上；空=复用 CHAT_API_KEY
    # ch08 · 即插即用工具系统（注册中心 + 执行引擎 + MCP 接入）
    mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"  # 物流 MCP Server（Streamable HTTP）
    mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"  # 售后 MCP Server
    tool_default_timeout: float = 5.0  # 内置工具默认超时（秒）
    mcp_tool_timeout: float = 10.0  # MCP 工具默认超时（走 HTTP，放宽）
    tool_max_retries: int = 2  # 只读工具暂时性故障最大重试次数；写操作恒不重试
    demo_ticket_delay_seconds: float = 0.0  # >0 时 create_ticket 人为变慢（验收6 写超时演示）
    # ch09 · 可观测（Langfuse 自部署；三者齐全才挂回调，缺省时系统照常跑、测试环境不依赖）
    # 环境变量名与课程 README 一致：LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = ""  # 如 http://localhost:3000（自部署地址，数据不出门）
    # ch09 · 置信度闸（正式版）：阈值由校准脚本在 ch04 评估集上实跑校准后回填，不拍脑袋
    # 校准报告 data/ch09/reports/confidence_calibration.txt：0.138 = 正样本 P10 / 负样本 P80 上探，
    # 拦 86.7% 负样本、放行 90% 正样本（占位 0.5 会误拦 58/240 可答题，数据见报告）
    evidence_confidence_threshold: float = 0.138

    @property
    def database_url(self) -> URL:
        """Build a database URL without hand-encoding credentials."""
        return URL.create(
            "mysql+pymysql",
            username=self.mysql_user,
            password=self.mysql_password,
            host=self.mysql_host,
            port=self.mysql_port,
            database=self.mysql_database,
        )

    @classmethod
    def hermetic_defaults(cls) -> "Settings":
        """Offline-safe defaults for tools built without explicit settings (tests).

        EMBED_API_KEY stays empty so dense retrieval degrades to the keyword
        lookup without touching the network; chat fields are inert placeholders.
        """
        return cls(
            chat_base_url="http://localhost.invalid/v1",
            chat_model="unavailable",
            chat_api_key="unavailable",
        )

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from mewhelp/.env, allowing process variables to win."""
        env_file = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_file, override=False)

        required = {
            "CHAT_BASE_URL": os.getenv("CHAT_BASE_URL"),
            "CHAT_MODEL": os.getenv("CHAT_MODEL"),
            "CHAT_API_KEY": os.getenv("CHAT_API_KEY"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("Missing required environment variable(s): " + ", ".join(missing))

        try:
            temperature = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
            token_budget = int(os.getenv("CHAT_CONTEXT_TOKEN_BUDGET", "6000"))
            safety_margin = int(os.getenv("CHAT_TOKEN_SAFETY_MARGIN", "500"))
            mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
            tool_timeout_seconds = int(os.getenv("TOOL_TIMEOUT_SECONDS", "10"))
            recall_top_k = int(os.getenv("RECALL_TOP_K", "50"))
            rrf_rank_constant = int(os.getenv("RRF_RANK_CONSTANT", "60"))
            rerank_top_n = int(os.getenv("RERANK_TOP_N", "10"))
            rerank_min_score = float(os.getenv("RERANK_MIN_SCORE", "0.3"))
            self_check_temperature = float(os.getenv("SELF_CHECK_TEMPERATURE", "0.0"))
            eval_top_k = int(os.getenv("EVAL_TOP_K", "5"))
            max_agent_steps = int(os.getenv("MAX_AGENT_STEPS", "6"))
            intent_conf_threshold = float(os.getenv("INTENT_CONF_THRESHOLD", "0.6"))
            model_context_window = int(os.getenv("MODEL_CONTEXT_WINDOW", "0"))
            context_budget_turns = int(os.getenv("CONTEXT_BUDGET_TURNS", "20"))
            max_user_input_tokens = int(os.getenv("MAX_USER_INPUT_TOKENS", "2000"))
            max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "4000"))
            system_prompt_tokens = int(os.getenv("SYSTEM_PROMPT_TOKENS", "700"))
            evidence_tokens_per_chunk = int(os.getenv("EVIDENCE_TOKENS_PER_CHUNK", "160"))
            rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))
            layer1_ratio = float(os.getenv("LAYER1_RATIO", "0.7"))
            layer2_reply_keep_chars = int(os.getenv("LAYER2_REPLY_KEEP_CHARS", "60"))
            layer2_tool_keep_tokens = int(os.getenv("LAYER2_TOOL_KEEP_TOKENS", "200"))
            tool_result_max_tokens = int(os.getenv("TOOL_RESULT_MAX_TOKENS", "1200"))
            agent_step_ai_tokens = int(os.getenv("AGENT_STEP_AI_TOKENS", "500"))
            summary_inject_segments = int(os.getenv("SUMMARY_INJECT_SEGMENTS", "3"))
            summary_tokens_per_segment = int(os.getenv("SUMMARY_TOKENS_PER_SEGMENT", "250"))
            budget_safety_margin = int(os.getenv("BUDGET_SAFETY_MARGIN", "1000"))
            zh_chars_per_token = float(os.getenv("ZH_CHARS_PER_TOKEN", "1.2"))
            context_window_max_tokens = int(os.getenv("CONTEXT_WINDOW_MAX_TOKENS", "0"))
            tool_default_timeout = float(os.getenv("TOOL_DEFAULT_TIMEOUT", "5.0"))
            mcp_tool_timeout = float(os.getenv("MCP_TOOL_TIMEOUT", "10.0"))
            tool_max_retries = int(os.getenv("TOOL_MAX_RETRIES", "2"))
            demo_ticket_delay_seconds = float(os.getenv("DEMO_TICKET_DELAY_SECONDS", "0"))
            evidence_confidence_threshold = float(os.getenv("EVIDENCE_CONFIDENCE_THRESHOLD", "0.138"))
        except ValueError as exc:
            raise ValueError("numeric settings must contain valid numbers") from exc

        if token_budget <= 0:
            raise ValueError("CHAT_CONTEXT_TOKEN_BUDGET must be greater than zero")
        if safety_margin < 0 or safety_margin >= token_budget:
            raise ValueError("CHAT_TOKEN_SAFETY_MARGIN must be non-negative and smaller than the token budget")
        if mysql_port <= 0:
            raise ValueError("MYSQL_PORT must be greater than zero")
        if tool_timeout_seconds <= 0:
            raise ValueError("TOOL_TIMEOUT_SECONDS must be greater than zero")
        if max_agent_steps < 1:
            raise ValueError("MAX_AGENT_STEPS must be at least 1")
        if layer1_ratio <= 0 or layer1_ratio >= 1:
            raise ValueError("LAYER1_RATIO must be between 0 and 1")
        if zh_chars_per_token <= 0:
            raise ValueError("ZH_CHARS_PER_TOKEN must be greater than zero")
        if summary_inject_segments < 0:
            raise ValueError("SUMMARY_INJECT_SEGMENTS must be non-negative")
        if tool_default_timeout <= 0 or mcp_tool_timeout <= 0:
            raise ValueError("TOOL_DEFAULT_TIMEOUT and MCP_TOOL_TIMEOUT must be greater than zero")
        if tool_max_retries < 0:
            raise ValueError("TOOL_MAX_RETRIES must be non-negative")
        if demo_ticket_delay_seconds < 0:
            raise ValueError("DEMO_TICKET_DELAY_SECONDS must be non-negative")
        if not 0.0 <= evidence_confidence_threshold <= 1.0:
            raise ValueError("EVIDENCE_CONFIDENCE_THRESHOLD must be between 0 and 1")

        return cls(
            chat_base_url=required["CHAT_BASE_URL"].rstrip("/"),
            chat_model=required["CHAT_MODEL"],
            chat_api_key=required["CHAT_API_KEY"],
            chat_temperature=temperature,
            chat_context_token_budget=token_budget,
            chat_token_safety_margin=safety_margin,
            mysql_host=os.getenv("MYSQL_HOST", "127.0.0.1"),
            mysql_port=mysql_port,
            mysql_database=os.getenv("MYSQL_DATABASE", "mewhelp"),
            mysql_user=os.getenv("MYSQL_USER", "mewhelp"),
            mysql_password=os.getenv("MYSQL_PASSWORD", "mewhelp"),
            tool_timeout_seconds=tool_timeout_seconds,
            embed_api_key=os.getenv("EMBED_API_KEY", "").strip(),
            embed_base_url=os.getenv("EMBED_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
            embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            milvus_uri=os.getenv("MILVUS_URI", "http://localhost:19530"),
            retrieval_strategy=os.getenv("RETRIEVAL_STRATEGY", "hybrid_rerank").strip(),
            recall_top_k=recall_top_k,
            rrf_rank_constant=rrf_rank_constant,
            rerank_api_key=os.getenv("RERANK_API_KEY", "").strip(),
            rerank_base_url=os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/"),
            rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            rerank_top_n=rerank_top_n,
            rerank_min_score=rerank_min_score,
            self_check_temperature=self_check_temperature,
            judge_model=os.getenv("JUDGE_MODEL", "deepseek-v4-pro").strip(),
            judge_base_url=os.getenv("JUDGE_BASE_URL", "").strip(),
            judge_api_key=os.getenv("JUDGE_API_KEY", "").strip(),
            eval_top_k=eval_top_k,
            max_agent_steps=max_agent_steps,
            checkpointer_db_path=os.getenv("CHECKPOINTER_DB_PATH", "data/ch05_checkpoints.sqlite").strip(),
            intent_model=os.getenv("INTENT_MODEL", "").strip(),
            intent_small_model=os.getenv("INTENT_SMALL_MODEL", "").strip(),
            intent_mode=os.getenv("INTENT_MODE", "accuracy").strip(),
            intent_conf_threshold=intent_conf_threshold,
            model_context_window=model_context_window,
            context_budget_turns=context_budget_turns,
            max_user_input_tokens=max_user_input_tokens,
            max_output_tokens=max_output_tokens,
            system_prompt_tokens=system_prompt_tokens,
            evidence_tokens_per_chunk=evidence_tokens_per_chunk,
            rerank_top_k=rerank_top_k,
            layer1_ratio=layer1_ratio,
            layer2_reply_keep_chars=layer2_reply_keep_chars,
            layer2_tool_keep_tokens=layer2_tool_keep_tokens,
            tool_result_max_tokens=tool_result_max_tokens,
            agent_step_ai_tokens=agent_step_ai_tokens,
            summary_inject_segments=summary_inject_segments,
            summary_tokens_per_segment=summary_tokens_per_segment,
            budget_safety_margin=budget_safety_margin,
            zh_chars_per_token=zh_chars_per_token,
            context_window_max_tokens=context_window_max_tokens,
            summary_model=os.getenv("SUMMARY_MODEL", "").strip(),
            summary_base_url=os.getenv("SUMMARY_BASE_URL", "").strip(),
            summary_api_key=os.getenv("SUMMARY_API_KEY", "").strip(),
            mcp_logistics_url=os.getenv("MCP_LOGISTICS_URL", "http://127.0.0.1:8101/mcp").strip(),
            mcp_aftersales_url=os.getenv("MCP_AFTERSALES_URL", "http://127.0.0.1:8102/mcp").strip(),
            tool_default_timeout=tool_default_timeout,
            mcp_tool_timeout=mcp_tool_timeout,
            tool_max_retries=tool_max_retries,
            demo_ticket_delay_seconds=demo_ticket_delay_seconds,
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
            langfuse_base_url=os.getenv("LANGFUSE_BASE_URL", "").strip(),
            evidence_confidence_threshold=evidence_confidence_threshold,
        )
