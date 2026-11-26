"""Citation bookkeeping for the online chat: parse ``[n]`` markers, stash maps.

``query_faq`` stashes the numbered citation list (number → chunk id / section
path / original text) under its ``tool_call_id`` right after retrieval; the chat
orchestrator pops it after the answer stream completes, resolves the ``[n]``
markers the model actually emitted, and pushes the matching subset to the
frontend as a ``citations`` SSE event.
"""

from __future__ import annotations

import re
from typing import Dict, List

# [1] [2] [10] — square brackets around a positive integer, not adjacent digits
_MARKER_RE = re.compile(r"\[(\d{1,2})\]")

# Small in-process stash; entries are popped in the same request that stashed
# them, so this never grows beyond the concurrent tool calls in flight.
_STASH: Dict[str, dict] = {}


def stash(tool_call_id: str, payload: dict) -> None:
    """Remember the citation payload for one tool call (latest wins)."""
    if not tool_call_id:
        return
    _STASH[tool_call_id] = payload


def pop(tool_call_id: str) -> dict | None:
    """Consume the stashed payload (``citations`` list) for a tool call."""
    return _STASH.pop(tool_call_id, None)


def parse_markers(answer: str) -> List[int]:
    """Return the distinct [n] citation numbers in the answer, in first-seen order."""
    seen: List[int] = []
    for match in _MARKER_RE.finditer(answer or ""):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def resolve_citations(answer: str, citations: List[dict]) -> List[dict]:
    """Filter the citation list down to the numbers the answer actually cites.

    Unrecognized numbers (model hallucinating a marker) are dropped silently.
    """
    by_number = {entry.get("n"): entry for entry in citations or []}
    return [by_number[number] for number in parse_markers(answer) if number in by_number]
