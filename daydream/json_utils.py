"""Shared JSON extraction utilities.

Used by backends (structured-output extraction) and ``run_agent`` (raw-text
fallback) to robustly pull JSON out of model output that may be wrapped in
reasoning prose or markdown code fences — common with GLM and other
OpenAI-compatible models.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json(text: str) -> Any:
    """Extract a JSON object or array from possibly prose-wrapped model text.

    Tries, in priority order:

    1. Strip leading/trailing whitespace.
    2. Strip markdown code fences (```` ```json ... ``` ```` or bare ```` ``` ````).
    3. ``json.loads`` on the cleaned text (fast path for clean JSON).
    4. If that fails, scan for the first balanced ``{...}`` or ``[...]`` block
       (depth-counting, string-aware) and parse that substring. Whichever brace
       type appears earliest in the text is tried first; if a balanced span of
       that type fails to parse, keep scanning forward for a later valid span of
       the same type before falling back to the other brace type.

    Returns the parsed value (dict, list, str, int, …) or ``None`` if no valid
    JSON was found. Never raises.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Strip markdown code fences: ```json\n...\n``` or ```\n...\n```
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Drop the opening fence line (may include language tag like "json")
        lines = lines[1:]
        # Drop the closing fence line
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Fast path — the entire text is valid JSON.
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Slow path — find the first balanced JSON object/array within the text.
    # This handles "Here is my analysis:\n{...json...}" patterns.
    # Try whichever brace type ({ or [) appears earliest in the text first.
    candidates: list[tuple[int, str, str]] = []
    brace_idx = cleaned.find("{")
    bracket_idx = cleaned.find("[")
    if brace_idx != -1:
        candidates.append((brace_idx, "{", "}"))
    if bracket_idx != -1:
        candidates.append((bracket_idx, "[", "]"))
    candidates.sort()  # by position in text

    for _, start_char, end_char in candidates:
        scan_from = 0
        while True:
            start_idx = cleaned.find(start_char, scan_from)
            if start_idx == -1:
                break  # no more spans of this brace type; try the next candidate
            depth = 0
            in_string = False
            escape = False
            end_idx = -1
            for i in range(start_idx, len(cleaned)):
                ch = cleaned[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx == -1:
                break  # unbalanced; try the next brace-type candidate
            candidate = cleaned[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                # This balanced span is unparseable; scan forward for a later
                # valid span of the same brace type before giving up on it.
                scan_from = end_idx + 1

    return None
