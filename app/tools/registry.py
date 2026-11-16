"""Explicit allowlist mapping tool names to their LangChain tools."""

from collections.abc import Callable

from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from app.tools.business import build_business_tools

TOOL_NAMES: tuple[str, ...] = (
    "query_order",
    "query_product",
    "query_logistics",
    "query_faq",
    "create_ticket",
)


def build_tool_registry(session_factory: Callable[[], Session]) -> dict[str, BaseTool]:
    """Return the explicit name-to-tool mapping the customer support model may call."""
    registry = {business_tool.name: business_tool for business_tool in build_business_tools(session_factory)}
    if set(registry) != set(TOOL_NAMES):
        raise RuntimeError("tool registry does not match the approved allowlist")
    return registry
