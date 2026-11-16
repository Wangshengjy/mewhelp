"""Run the labeled tool-selection cases against the configured upstream.

Tool selection uses LangChain's default (unforced) ``tool_choice``, which the
thinking-by-default upstream accepts. Pass ``--reasoning-effort none`` only as
an extra diagnostic when the upstream cannot produce tool calls at all;
production keeps the model exactly as configured.

An expected argument value of ``"*"`` means "present and a non-empty string".
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.database import build_engine, build_session_factory
from app.prompts import CHAT_SYSTEM_PROMPT
from app.settings import Settings
from app.tools.registry import build_tool_registry

CASES_PATH = Path(__file__).with_name("tool_call_cases.jsonl")


def build_eval_model(settings: Settings, reasoning_effort: str | None) -> ChatOpenAI:
    """Build the evaluation model without changing any production configuration."""
    parameters: dict[str, Any] = {
        "base_url": settings.chat_base_url,
        "api_key": settings.chat_api_key,
        "model": settings.chat_model,
        "temperature": 0,
    }
    if reasoning_effort is not None:
        parameters["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**parameters)


def build_eval_registry() -> list[Any]:
    """Build the approved tools against an unused offline session factory."""
    session_factory = build_session_factory(build_engine("sqlite+pysqlite:///:memory:"))
    return list(build_tool_registry(session_factory).values())


def arguments_match(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    """Require every expected argument; ``"*"`` only checks a non-empty value."""
    for key, value in expected.items():
        if key not in actual:
            return False
        if value == "*":
            if not isinstance(actual[key], str) or not actual[key].strip():
                return False
        elif actual[key] != value:
            return False
    return True


async def run_evaluation(reasoning_effort: str | None = None) -> int:
    settings = Settings.load()
    model = build_eval_model(settings, reasoning_effort)
    bound_model = model.bind_tools(build_eval_registry())
    cases = [
        json.loads(line)
        for line in CASES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    passed = 0
    for index, case in enumerate(cases, start=1):
        response = await bound_model.ainvoke(
            [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=case["message"]),
            ]
        )
        calls = list(response.tool_calls or [])
        selected = calls[0] if len(calls) == 1 else None
        tool_name = selected["name"] if selected else None
        arguments = selected["args"] if selected else {}
        matched = (
            len(calls) == 1
            and tool_name == case["tool"]
            and arguments_match(case["args"], arguments)
        )
        passed += int(matched)
        print(
            json.dumps(
                {
                    "case": index,
                    "matched": matched,
                    "message": case["message"],
                    "expected": {"tool": case["tool"], "args": case["args"]},
                    "actual_tool_calls": calls,
                    "reasoning_effort": reasoning_effort,
                },
                ensure_ascii=False,
            )
        )

    print(f"Tool selection: {passed}/{len(cases)} cases matched")
    return 0 if passed == len(cases) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the labeled tool-selection cases.")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["none", "minimal", "low", "medium", "high"],
        help="Optional diagnostic override; omit to use the configured model mode.",
    )
    arguments = parser.parse_args()
    return asyncio.run(run_evaluation(arguments.reasoning_effort))


if __name__ == "__main__":
    raise SystemExit(main())