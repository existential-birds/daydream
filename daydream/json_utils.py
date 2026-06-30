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
    4. If that fails, scan for every balanced, parseable ``{...}`` or ``[...]``
       span (depth-counting, string-aware) and return the LARGEST one. The real
       structured-output payload is a substantial object/array, so size
       disambiguates it from incidental brackets in the surrounding prose — e.g.
       a ``metadata["sender"]`` code snippet, which parses as the tiny list
       ``["sender"]``. Returning the largest span avoids handing a caller that
       expects ``{"findings": [...]}`` a bogus bare list grabbed from prose.

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
    except ValueError:
        pass

    # Slow path — the text is prose with one or more embedded JSON spans. Find
    # EVERY balanced, parseable {...}/[...] span and return the LARGEST one.
    #
    # The largest span is the model's actual answer: a structured-output
    # response is a substantial object/array, whereas stray brackets in the
    # surrounding prose (e.g. a `metadata["sender"]` code snippet, which parses
    # as the one-element list `["sender"]`) are tiny. An earlier-bracket-wins
    # rule would return that incidental `["sender"]` and hand a bogus bare list
    # to a caller expecting `{"findings": [...]}`. Size disambiguates reliably:
    # the real payload dwarfs prose noise.
    best: Any = None
    best_len = 0
    for start_char, end_char in (("{", "}"), ("[", "]")):
        scan_from = 0
        while True:
            start_idx = cleaned.find(start_char, scan_from)
            if start_idx == -1:
                break
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
                # Unbalanced from here; advance one char and keep scanning for a
                # later valid span of this brace type.
                scan_from = start_idx + 1
                continue
            span_len = end_idx + 1 - start_idx
            try:
                parsed = json.loads(cleaned[start_idx : end_idx + 1])
            except ValueError:
                parsed = None
            if parsed is not None and span_len > best_len:
                best = parsed
                best_len = span_len
            if parsed is not None:
                # A parsed span's nested children can only be smaller, so
                # skipping past it never drops the winner and keeps the scan
                # near-linear on well-formed payloads.
                scan_from = end_idx + 1
            else:
                # Balanced but invalid: a nested {...}/[...] inside may still be
                # valid JSON, so re-enter the span instead of discarding it.
                scan_from = start_idx + 1

    return best
