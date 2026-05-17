"""Training-record and span builders for the JSONL exporter.

This module owns ATIF-v1.6 trajectory → training-record conversion. Higher-level
orchestration (SQLite query, filters, stratification, file emission) lands in
later waves; this file is deliberately limited to the two pure-function
builders so the record shape and span-derivation rule can be reviewed and
tested in isolation.

Both helpers are private (underscore-prefixed): callers outside this package
should depend on ``run_export`` once it lands in Wave 6, not on these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.training.schema import TRAINING_SCHEMA_VERSION


def _build_spans(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an ATIF v1.6 trajectory dict into REASON/ACT span refs.

    Implements plan §5. The output is a list of ``{step_id, kind, content_path}``
    dicts that point at substructures of ``trajectory["steps"]`` rather than
    embedding the content itself (per the "pass refs, not contents" rule).

    Rules:
    - Only ``source == "agent"`` steps contribute spans.
    - Steps marked ``is_copied_context=True`` are skipped (ATIF v1.5 semantics).
    - REASON prefers ``reasoning_content``; if absent, falls back to ``message``
      when ``message`` is a non-empty string. List-typed ``message`` values
      (ContentPart arrays) are not used as REASON sources.
    - ACT is emitted when ``tool_calls`` is a truthy (non-empty) list.
    - Within a single step REASON is appended before ACT, preserving the
      natural reason-then-act ordering required by the schema consumers.

    Args:
        trajectory: ATIF v1.6 trajectory dict (e.g. ``json.loads(open(p))``).

    Returns:
        Spans in insertion order. Empty list when ``trajectory`` has no
        agent-authored steps or when no agent step carries reason/action data.
    """
    spans: list[dict[str, Any]] = []
    for i, step in enumerate(trajectory.get("steps", [])):
        if step.get("source") != "agent":
            continue
        if step.get("is_copied_context") is True:
            continue
        step_id = step.get("step_id", i + 1)
        # REASON: prefer explicit reasoning_content; fall back to text message.
        has_reasoning = bool(step.get("reasoning_content"))
        message = step.get("message")
        has_text_message = isinstance(message, str) and bool(message.strip())
        if has_reasoning:
            spans.append(
                {
                    "step_id": step_id,
                    "kind": "REASON",
                    "content_path": f"steps[{i}].reasoning_content",
                }
            )
        elif has_text_message:
            spans.append(
                {
                    "step_id": step_id,
                    "kind": "REASON",
                    "content_path": f"steps[{i}].message",
                }
            )
        # ACT: any non-empty tool_calls list.
        if step.get("tool_calls"):
            spans.append(
                {
                    "step_id": step_id,
                    "kind": "ACT",
                    "content_path": f"steps[{i}].tool_calls",
                }
            )
    return spans


def _build_record(
    manifest_row: dict[str, Any],
    trajectory: dict[str, Any],
    stack: str | None,
) -> dict[str, Any]:
    """Assemble a training record matching ``schema/v1.json``.

    The returned dict carries refs (``fix_diff_ref``, ``trajectory_ref``,
    ``spans[*].content_path``) instead of embedded content, so records stay
    small and downstream consumers materialize bytes from the archive on
    demand.

    Fields that the R1 manifest does not yet populate (``base_sha``,
    ``changed_files``, ``review_output``) are emitted as ``None`` / ``[]``;
    the schema allows nullable values in those slots.

    Args:
        manifest_row: Dict shaped like ``Manifest.to_dict()["manifest"]`` or a
            flat row from the SQLite index. Must carry ``session_id``,
            ``skill``, ``repo_slug``, ``branch``, ``base_branch``, ``head_sha``,
            ``grounding_rate``, ``outcome_labels`` (JSON-encoded string list),
            and ``archive_path``.
        trajectory: ATIF v1.6 trajectory dict for this run.
        stack: Routing label (e.g. ``"python"``, ``"react"``). The caller
            derives this from deep-stack detection; pass ``None`` when
            unavailable.

    Returns:
        A dict that validates against ``daydream/training/schema/v1.json``.
    """
    archive_path = Path(manifest_row.get("archive_path", ""))
    diff_path = archive_path / "diff.patch"
    fix_diff_ref = {
        "available": diff_path.is_file(),
        "archive_relative_path": "diff.patch",
    }

    # outcome_labels is a JSON-encoded list string on the manifest row.
    raw_labels = manifest_row.get("outcome_labels", "[]")
    try:
        labels = json.loads(raw_labels) if isinstance(raw_labels, str) else []
    except (json.JSONDecodeError, TypeError):
        labels = []
    if not isinstance(labels, list):
        labels = []

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "session_id": manifest_row["session_id"],
        "repo_slug": manifest_row.get("repo_slug"),
        "skill": manifest_row.get("skill"),
        "stack": stack,
        "code_context": {
            "base_sha": None,  # not populated by R1 manifest
            "head_sha": manifest_row.get("head_sha"),
            "base_branch": manifest_row.get("base_branch"),
            "branch": manifest_row.get("branch"),
            "changed_files": [],  # not populated by R1 manifest
        },
        "review_output": None,  # not loaded from disk in R1
        "fix_diff_ref": fix_diff_ref,
        "test_outcome": manifest_row.get("test_outcome"),
        "outcome_label": labels[0] if labels else None,
        "grounding_score": manifest_row.get("grounding_rate"),
        "spans": _build_spans(trajectory),
        "trajectory_ref": {"archive_relative_path": "trajectory.json"},
    }
