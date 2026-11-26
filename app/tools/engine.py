"""ch08 · 统一执行引擎:六段管道(查工具→校验→权限门→执行→分诊→格式化+审计)。

- 校验拦下/权限拒绝不抛异常,坏消息如实回灌模型让它自己纠正;
- 重试只给暂时性故障(超时/网络抖动);写操作恒不重试(超时未必没执行,重复更糟);
  query_faq 这类 RAG 管线工具按 spec.max_retries=0 也不重试;
- 结果挑字段/枚举翻人话走 spec.format_result 钩子,未登记透传;JSON 中文不转义;
- 每次调用(含被拦的)落 tool_audit_logs,审计写失败只 log,绝不反拦工具执行。
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx
from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from langchain_core.messages import ToolMessage

from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# 暂时性故障(值得重试):超时、网络抖动。业务异常/ToolException 不在列——重试不解决问题。
_TRANSIENT = (asyncio.TimeoutError, TimeoutError, ConnectionError, httpx.TransportError)
_SUMMARY_LIMIT = 500

# settings 缺位时的引擎默认值(与 settings.py 的 ch08 字段同源)
_DEFAULT_TIMEOUT_BUILTIN = 5.0
_DEFAULT_TIMEOUT_MCP = 10.0
_DEFAULT_MAX_RETRIES = 2


@dataclass
class ToolRun:
    tool_call_id: str
    name: str
    ok: bool
    tool_message: ToolMessage
    status: str  # 成功 / 失败 / 超时 / 校验拦下 / 权限拒绝
    retry_count: int = 0
    duration_ms: int = 0


def validate_args(spec: ToolSpec, args: dict) -> str | None:
    """按 JSON Schema 校验模型给的参数;返回人类可读错误(None=通过)。

    agent_tools 复用它判「create_ticket 参数齐不齐、要不要弹确认卡」。
    """
    err = best_match(Draft202012Validator(spec.json_schema).iter_errors(args))
    if err is None:
        return None
    where = f"(字段 {err.json_path})" if err.json_path != "$" else ""
    return f"{err.message}{where}"


def _timeout_of(spec: ToolSpec, settings) -> float:
    if spec.timeout is not None:
        return float(spec.timeout)
    if settings is not None:
        return float(settings.mcp_tool_timeout if spec.source == "mcp" else settings.tool_default_timeout)
    return _DEFAULT_TIMEOUT_MCP if spec.source == "mcp" else _DEFAULT_TIMEOUT_BUILTIN


def _retries_of(spec: ToolSpec, settings) -> int:
    if spec.permission == "write":
        return 0  # 写操作恒不重试:超时未必没执行,重复执行比失败更糟
    if spec.max_retries is not None:
        return int(spec.max_retries)
    if settings is not None:
        return int(settings.tool_max_retries)
    return _DEFAULT_MAX_RETRIES


def _summarize(content: str) -> str:
    return content if len(content) <= _SUMMARY_LIMIT else content[:_SUMMARY_LIMIT] + "…(截断)"


def _unwrap_content_blocks(result):
    """MCP adapters 工具的原始返回是 [content-block] 列表:取唯一 text 块解回数据,
    其余形状原样透传(这是「结果格式化我们侧治理」的一部分)。"""
    if (
        isinstance(result, list)
        and len(result) == 1
        and isinstance(result[0], dict)
        and result[0].get("type") == "text"
        and isinstance(result[0].get("text"), str)
    ):
        return result[0]["text"]
    return result


def _format_content(spec: ToolSpec, result) -> str:
    """结果格式化:MCP 工具返回内容块列表→解包 text,再解 JSON 文本为 dict;
    format_result 钩子挑字段/枚举翻人话(未登记透传);统一 ensure_ascii=False(中文不转义)。"""
    result = _unwrap_content_blocks(result)
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            return result  # 纯文本结果原样回灌
    if spec.format_result is not None and isinstance(result, dict):
        result = spec.format_result(result)
    return json.dumps(result, ensure_ascii=False, default=str)


def _injected_value(name: str, conversation_id, tool_call_id, session_factory, settings):
    """注入参数取值:身份类来自调用上下文,依赖类来自 GraphDeps——模型填的不算数。"""
    return {
        "conversation_id": conversation_id,
        "tool_call_id": tool_call_id,
        "session_factory": session_factory,
        "settings": settings,
    }.get(name)


def _write_audit_row(session_factory, record: dict) -> None:
    """审计落库(同步,在线程池跑);测试经 monkeypatch 本函数捕获。"""
    from app.repositories import tool_audit as tool_audit_repository

    with session_factory() as session:
        tool_audit_repository.insert_tool_audit(session, **record)
        session.commit()


async def _audit(session_factory, record: dict) -> None:
    if session_factory is None:
        return
    try:  # 审计失败不许反拦工具执行
        await asyncio.to_thread(_write_audit_row, session_factory, record)
    except Exception:  # noqa: BLE001
        logger.exception(
            "审计写入失败(不影响工具执行) tool=%s status=%s", record.get("tool_name"), record.get("status")
        )


async def execute_tool_call(
    tool_call: dict,
    conversation_id,
    specs: dict[str, ToolSpec],
    *,
    session_factory=None,
    settings=None,
    confirmed: bool = False,
    deny_note: str | None = None,
) -> ToolRun:
    """一切工具调用的唯一通道。六段管道详见模块 docstring。"""
    name = tool_call.get("name") or ""
    tc_id = tool_call.get("id") or ""
    args = dict(tool_call.get("args") or {})
    started = time.monotonic()

    def _run(ok: bool, content: str, status: str, retry_count: int = 0) -> ToolRun:
        return ToolRun(
            tool_call_id=tc_id,
            name=name,
            ok=ok,
            status=status,
            retry_count=retry_count,
            duration_ms=int((time.monotonic() - started) * 1000),
            tool_message=ToolMessage(
                content=content,
                tool_call_id=tc_id,
                name=name or "unknown",
                status="success" if ok else "error",
            ),
        )

    spec = specs.get(name)
    # ① 查工具:查不到如实回灌,不猜
    if spec is None:
        run = _run(False, f"工具执行失败:未知工具 {name}", "失败")
        await _audit(
            session_factory,
            {
                "conversation_id": conversation_id or None,
                "tool_call_id": tc_id or None,
                "tool_name": name or "unknown",
                "tool_source": "builtin",
                "mcp_server": None,
                "arguments": args or None,
                "result_summary": None,
                "status": "失败",
                "error_message": f"未知工具 {name}",
                "retry_count": 0,
                "duration_ms": run.duration_ms,
            },
        )
        return run

    # ② JSON Schema 校验:拦下不抛异常,错误说明回灌模型(让它追问用户或重组调用)
    verr = validate_args(spec, args)
    if verr is not None:
        run = _run(
            False,
            f"参数校验未通过:{verr}。请修正参数重新调用;缺少的信息请先向用户追问,不要编造。",
            "校验拦下",
        )
        await _audit(
            session_factory,
            {
                "conversation_id": conversation_id or None,
                "tool_call_id": tc_id or None,
                "tool_name": name,
                "tool_source": spec.source,
                "mcp_server": spec.mcp_server,
                "arguments": args or None,
                "result_summary": None,
                "status": "校验拦下",
                "error_message": verr,
                "retry_count": 0,
                "duration_ms": run.duration_ms,
            },
        )
        return run

    # ③ 权限门:写操作必须带确认令牌(interrupt 确认后由 agent_tools 传入),模型绕不过
    if spec.permission == "write" and not confirmed:
        note = deny_note or "该写操作需要用户确认,未确认前拒绝执行。请勿再次发起,除非用户明确要求。"
        run = _run(False, f"建工单未执行:{note}", "权限拒绝")
        await _audit(
            session_factory,
            {
                "conversation_id": conversation_id or None,
                "tool_call_id": tc_id or None,
                "tool_name": name,
                "tool_source": spec.source,
                "mcp_server": spec.mcp_server,
                "arguments": args or None,
                "result_summary": None,
                "status": "权限拒绝",
                "error_message": note,
                "retry_count": 0,
                "duration_ms": run.duration_ms,
            },
        )
        return run

    # ④ 执行:校验通过后注入运行期参数;超时 + 暂时性故障重试(写操作恒 0 次)
    model_args = dict(args)  # 审计记模型原样参数(注入参数不可 JSON 序列化,且属运行期上下文)
    for injected in spec.injected_args:
        if injected == "tool_call_id":
            continue  # tool_call_id 由 ToolCall 形式原生注入(见下),不重复塞进 args
        value = _injected_value(injected, conversation_id, tc_id, session_factory, settings)
        if value is not None:
            args[injected] = value
    retries = _retries_of(spec, settings)
    tool_timeout = _timeout_of(spec, settings)
    # 带 InjectedToolCallId 的工具必须按完整 ToolCall 形式调用,langchain 才会注入 tool_call_id
    if "tool_call_id" in spec.injected_args:
        invocation = {"args": args, "name": name, "type": "tool_call", "id": tc_id}
    else:
        invocation = args
    attempt = 0
    while True:
        try:
            result = await asyncio.wait_for(spec.tool.ainvoke(invocation), timeout=tool_timeout)
            content = _format_content(spec, result)
            run = _run(True, content, "成功", attempt)
            await _audit(
                session_factory,
                {
                    "conversation_id": conversation_id or None,
                    "tool_call_id": tc_id or None,
                    "tool_name": name,
                    "tool_source": spec.source,
                    "mcp_server": spec.mcp_server,
                    "arguments": model_args or None,
                    "result_summary": _summarize(content),
                    "status": "成功",
                    "error_message": None,
                    "retry_count": attempt,
                    "duration_ms": run.duration_ms,
                },
            )
            return run
        except _TRANSIENT as e:
            if attempt < retries:
                attempt += 1
                logger.warning("工具暂时性故障,第 %s 次重试 name=%s err=%s", attempt, name, type(e).__name__)
                await asyncio.sleep(0.2 * attempt)
                continue
            # ⑤ 分诊:真故障如实回灌,不装没事
            is_timeout = isinstance(e, (asyncio.TimeoutError, TimeoutError))
            status = "超时" if is_timeout else "失败"
            run = _run(
                False,
                f"工具暂时不可用:{'执行超时' if is_timeout else type(e).__name__},请稍后再试或如实告知用户。",
                status,
                attempt,
            )
            await _audit(
                session_factory,
                {
                    "conversation_id": conversation_id or None,
                    "tool_call_id": tc_id or None,
                    "tool_name": name,
                    "tool_source": spec.source,
                    "mcp_server": spec.mcp_server,
                    "arguments": model_args or None,
                    "result_summary": None,
                    "status": status,
                    "error_message": type(e).__name__,
                    "retry_count": attempt,
                    "duration_ms": run.duration_ms,
                },
            )
            return run
        except Exception as e:  # noqa: BLE001 业务/未知异常:不重试,如实回灌
            logger.exception("工具执行失败 name=%s", name)
            run = _run(
                False,
                f"工具暂时不可用:{type(e).__name__}。请如实告知用户,不要编造结果。",
                "失败",
                attempt,
            )
            await _audit(
                session_factory,
                {
                    "conversation_id": conversation_id or None,
                    "tool_call_id": tc_id or None,
                    "tool_name": name,
                    "tool_source": spec.source,
                    "mcp_server": spec.mcp_server,
                    "arguments": model_args or None,
                    "result_summary": None,
                    "status": "失败",
                    "error_message": f"{type(e).__name__}: {e}"[:500],
                    "retry_count": attempt,
                    "duration_ms": run.duration_ms,
                },
            )
            return run
