"""Acquire review-artifact scoring inputs for frozen capture and sealed RL runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.training.reward import ScoringInputs

_VERDICTS_FILE = "recommendation-verdicts.json"
"""Bronze artifact (under ``deep/``) carrying the ``verdicts`` list."""

_RECORDS_GLOB = "stack-*-records.json"
"""Bronze per-stack finding-record artifacts (under ``deep/``)."""

_REVIEW_OUTPUT_FILE = "review-output.md"
"""Length-proxy artifact; at the run root for shallow runs, under ``deep/`` for deep runs."""

def _read_review_output(run_dir: Path) -> str | None:
    """Read review-output.md from the run root, then deep/; return None when absent.

    Propagate filesystem errors other than FileNotFoundError.
    """
    for candidate in (run_dir / _REVIEW_OUTPUT_FILE, run_dir / "deep" / _REVIEW_OUTPUT_FILE):
        try:
            return candidate.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
    return None


def assemble_scoring_inputs(run_dir: Path) -> ScoringInputs:
    """Read verifier verdicts, artifact validity, and output length from a run.

    Missing verifier verdicts leave correctness absent. Malformed structured
    artifacts fail the format gate; missing evidence never earns credit.
    """
    deep_dir = run_dir / "deep"

    verifier_verdicts: list[dict[str, Any]] | None = None
    format_valid = True

    verdicts_path = deep_dir / _VERDICTS_FILE
    try:
        data = json.loads(verdicts_path.read_text(encoding="utf-8"))
        verdicts = data.get("verdicts") if isinstance(data, dict) else None
        if isinstance(verdicts, list):
            verifier_verdicts = verdicts
    except FileNotFoundError:
        # No structured verdicts; nothing failed to parse. Expected for a
        # shallow run and, after the verify relocation, a declined deep run
        # that skipped recommendation verification at the apply-fixes gate.
        pass
    except json.JSONDecodeError:
        # Present but malformed ⇒ format gate floors.
        format_valid = False

    # A present-but-malformed records file also trips the format gate.
    if deep_dir.is_dir():
        for records_path in sorted(deep_dir.glob(_RECORDS_GLOB)):
            try:
                json.loads(records_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            except json.JSONDecodeError:
                format_valid = False

    review_text = _read_review_output(run_dir)
    return ScoringInputs(
        verifier_verdicts=verifier_verdicts,
        format_valid=format_valid,
        length=len(review_text) if review_text is not None else None,
    )
