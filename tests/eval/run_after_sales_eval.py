"""Run the labeled after-sales extraction examples against the configured model.

Structured output forces a single named ``tool_choice``. The DeepSeek
``deepseek-flash`` upstream runs in thinking mode by default, and thinking mode
rejects a forced ``tool_choice`` with HTTP 400, so the evaluation disables
reasoning for this call only.
"""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI

from app.extraction import build_extraction_chain
from app.settings import Settings


CASES_PATH = Path(__file__).with_name("after_sales_cases.jsonl")


async def run_evaluation() -> int:
    settings = Settings.load()
    model = ChatOpenAI(
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        model=settings.chat_model,
        temperature=0,
        reasoning_effort="none",
    )
    chain = build_extraction_chain(model)
    cases = [json.loads(line) for line in CASES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    passed = 0
    for index, case in enumerate(cases, start=1):
        result = await chain.ainvoke({"description": case["description"]})
        actual = result.model_dump()
        expected = case["expected"]
        matched = actual == expected
        passed += int(matched)
        print(
            json.dumps(
                {
                    "case": index,
                    "matched": matched,
                    "expected": expected,
                    "actual": actual,
                },
                ensure_ascii=False,
            )
        )

    print(f"Labeled evaluation: {passed}/{len(cases)} cases matched all fields")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_evaluation()))
