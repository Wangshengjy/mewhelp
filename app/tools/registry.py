"""ch08 工具注册中心:内置(启动扫描 builtin/ 包)+ MCP(mcp_client 现拉)统一登记 ToolSpec。

三样必齐:工具名、用途描述、JSON Schema 参数定义。权限只认我们侧 WRITE_TOOLS,
不看 Server 声明、不让模型临场判断。注册即定义:内置工具丢进 builtin/ 包即被扫描登记,
MCP Server 加工具重启即被发现(get_all_specs 现问现拿)——清单合并内置优先。
"""

import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.tools import BaseTool
from pydantic import BaseModel
from langchain_core.tools import InjectedToolArg

logger = logging.getLogger(__name__)

# 权限我们侧独裁:写清单按名字定;未登记(含所有 MCP)工具一律按只读放行。
# 生产环境接不受信 Server 时应默认拒绝未知写操作;本章两台自建 Server 都是查询类,只读放行足够。
WRITE_TOOLS: set[str] = {"create_ticket"}


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict  # 参数 JSON Schema(校验用;模型可见口径,不含注入参数)
    tool: BaseTool
    permission: str  # "read" | "write"
    source: str  # "builtin" | "mcp"
    mcp_server: str | None = None
    timeout: float | None = None  # None → engine 按来源取默认(内置 5s / MCP 10s)
    max_retries: int | None = None  # None → engine 按默认;0 = 不重试(如 query_faq RAG 管线)
    injected_args: tuple[str, ...] = field(default=())  # 运行期注入参数(InjectedToolArg 字段)
    format_result: Callable[[dict], dict] | None = None  # 挑字段/枚举翻人话,None 透传


_BUILTIN: dict[str, ToolSpec] = {}
_scanned = False


def permission_for(name: str) -> str:
    return "write" if name in WRITE_TOOLS else "read"


def _json_schema_of(tool: BaseTool) -> dict:
    """三样齐的 schema 口径:MCP 工具 args_schema 本就是 dict;内置 pydantic 模型用
    tool_call_schema(排除 InjectedToolArg,模型可见口径)转 JSON Schema。"""
    raw = getattr(tool, "args_schema", None)
    if isinstance(raw, dict):
        return raw
    tcs = getattr(tool, "tool_call_schema", None) or raw
    return tcs.model_json_schema()


def _injected_args_of(tool) -> tuple[str, ...]:
    """探测 args_schema 里标注 InjectedToolArg 的运行期注入字段(引擎校验通过后注入)。"""
    schema = getattr(tool, "args_schema", None)
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        return ()
    names: list[str] = []
    for name, f in schema.model_fields.items():
        if any(
            isinstance(m, InjectedToolArg) or (isinstance(m, type) and issubclass(m, InjectedToolArg))
            for m in f.metadata
        ):
            names.append(name)
    return tuple(names)


def spec_from_langchain_tool(
    tool: BaseTool,
    *,
    source: str,
    mcp_server: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    format_result: Callable | None = None,
) -> ToolSpec:
    return ToolSpec(
        name=tool.name,
        description=tool.description or "",
        json_schema=_json_schema_of(tool),
        tool=tool,
        permission=permission_for(tool.name),
        source=source,
        mcp_server=mcp_server,
        timeout=timeout,
        max_retries=max_retries,
        injected_args=_injected_args_of(tool),
        format_result=format_result,
    )


def register(spec: ToolSpec) -> None:
    if spec.name in _BUILTIN:
        logger.warning("工具重名,丢弃后注册者 name=%s(先到者保留)", spec.name)
        return
    _BUILTIN[spec.name] = spec


def scan_builtin() -> None:
    """服务启动(lifespan)时调用:导入 builtin/ 包全部模块,模块 import 即注册。幂等。"""
    global _scanned
    if _scanned:
        return
    _scanned = True
    from app.tools import builtin as pkg

    for m in pkgutil.iter_modules(pkg.__path__):
        try:
            importlib.import_module(f"{pkg.__name__}.{m.name}")
        except Exception:  # noqa: BLE001 坏文件跳过告警,不拖垮其余工具注册(丢文件即插即用的稳健性)
            logger.exception("内置工具模块导入失败,跳过 module=%s", m.name)
    logger.info("内置工具注册完成:%s", sorted(_BUILTIN))


def builtin_specs() -> list[ToolSpec]:
    scan_builtin()
    return list(_BUILTIN.values())


def get_builtin_spec(name: str) -> ToolSpec | None:
    scan_builtin()
    return _BUILTIN.get(name)


async def get_all_specs(settings=None) -> dict[str, ToolSpec]:
    """主力 Agent 的工具清单 = 内置(启动登记)+ MCP(现问现拿动态发现),一视同仁合并。

    不可达 Server 告警跳过只上内置;清单内重名 → 内置先到者保留
    (内置工具名不允许被外部 Server 抢注,权限 WRITE_TOOLS 也只认内置)。"""
    specs = {s.name: s for s in builtin_specs()}
    from app.tools import mcp_client  # noqa: PLC0415 延迟导入防 registry↔mcp_client 环

    for spec in await mcp_client.fetch_mcp_specs(settings):
        if spec.name in specs:
            logger.warning("MCP 工具与内置重名,保留内置 name=%s server=%s", spec.name, spec.mcp_server)
            continue
        specs[spec.name] = spec
    return specs
