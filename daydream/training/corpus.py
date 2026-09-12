"""Shared projection helpers over the bitemporal training archive.

This module owns the small set of reusable pieces the projection packages
import from here: the span builder (``_build_spans``), the skill-to-stack
decoder (:func:`_stack_for_skill`), the temporal-leakage guard
(:func:`_is_posterior_leak`), the gold-admission gate
(:func:`_is_admitted_outcome_gold`), and the trajectory-set content hash
(:func:`_trajectory_set_hash`).

The temporal-leakage guard protects an ``as_of`` pin against posterior label
leakage: an annotation may be *recorded* before ``as_of`` yet describe an
outcome whose valid time (``valid_at``, e.g. a PR merge timestamp) lands
*after* the pin. When ``valid_at > as_of`` the posterior-derived
``outcome_label`` must be dropped (the run is treated as unlabeled); the
intrinsic, capture-time reward fields survive. The comparison is
**chronological** (parsed datetimes), so any ISO-8601 spelling difference —
``Z`` vs ``+00:00``, sub-second precision, a non-UTC offset — can never
mis-order the guard. When ``as_of`` is ``None`` no valid-time exclusion
applies. Callers resolve the pin via
:func:`daydream.archive.index.normalize_as_of` so the lexical ``observed_at
<= as_of`` SQL pin and :func:`_is_posterior_leak` receive the same canonical
string.


All symbols are private (underscore-prefixed) shared infrastructure for the
corpus projection packages; the canonical emitter is
:mod:`daydream.training.corpus_projection`, which imports the span builder,
the leak guard, the trajectory-set hash, and the skill-to-stack decoder;
:mod:`daydream.training.reward_model` imports the gold-admission gate from
here.
"""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime
from typing import Any


def _build_spans(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an ATIF v1.7 trajectory dict into REASON/ACT span refs.

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
        raw_step_id = step.get("step_id", i + 1)
        try:
            step_id = int(raw_step_id)
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid step_id {raw_step_id!r} at steps[{i}] — using fallback {i + 1}.",
                stacklevel=2,
            )
            step_id = i + 1
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


# Legacy skill->stack decode map for historically captured corpus metadata.
# Built-in reviews no longer emit skills (#886): this maps archived manifest
# `skill` fields to stack labels so legacy runs stay stratifiable. It is only
# a decoding table for already-captured runs -- never a source of new skill
# invocations -- and is intentionally local to the training corpus, not exported.
_LEGACY_SKILL_TO_STACK: dict[str, str] = {
    "beagle-python:review-python": "python",
    "beagle-react:review-frontend": "react",
    "beagle-elixir:review-elixir": "elixir",
    "beagle-go:review-go": "go",
    "beagle-rust:review-rust": "rust",
    "beagle-ios:review-ios": "ios",
}
# Dual keys: both the full skill string and the short stack name map to the
# lowercase stack label, so _stack_for_skill is one dict lookup regardless of
# which form the manifest stored.
_SKILL_TO_STACK: dict[str, str] = {}
for _skill, _stack in _LEGACY_SKILL_TO_STACK.items():
    _SKILL_TO_STACK[_skill] = _stack
    _SKILL_TO_STACK[_stack] = _stack
del _skill, _stack


def _stack_for_skill(skill: str | None) -> str | None:
    """Return the stack label (e.g. ``"python"``, ``"react"``) for a skill.

    Args:
        skill: The manifest's skill field, either the full skill string
            (e.g. ``"beagle-python:review-python"``) or the short stack
            name (e.g. ``"python"``); both are accepted. ``None`` is also
            accepted.

    Returns:
        The lowercase stack label, or ``None`` when ``skill`` is ``None``
        or not present in the legacy skill map under either form.
    """
    if skill is None:
        return None
    return _SKILL_TO_STACK.get(skill)


def _is_posterior_leak(annotation: dict[str, Any] | None, as_of: str | None) -> bool:
    """Return ``True`` when the annotation's outcome only became true after ``as_of``.

    Temporal-leakage guard: an annotation may be *recorded* before the ``as_of``
    pin (``observed_at <= as_of``, already enforced by
    ``latest_label_observation``) yet describe an outcome whose valid time —
    e.g. a PR merge timestamp — lands *after* the pin. Such posterior-derived
    fields (the outcome label and posterior reward axes) would leak future
    information into a corpus pinned to ``as_of``, so they are dropped.

    The comparison is **chronological**: both sides are parsed with
    :func:`datetime.fromisoformat` and compared as aware datetimes, so spelling
    differences — ``Z`` vs ``+00:00``, differing sub-second precision, or a
    non-UTC offset — can never mis-order the guard. In the production path both
    strings are already canonical UTC (``as_of`` is normalized once at its
    entry boundary via :func:`daydream.archive.index.normalize_as_of`;
    ``valid_at`` is canonicalized at write time by
    ``append_label_observation``), making the parse a semantic statement
    rather than a compatibility shim.

    Args:
        annotation: The ``as_of``-pinned ``label_observations`` row, or ``None``.
        as_of: The transaction-time pin (aware ISO-8601). When ``None`` no
            valid-time exclusion applies (every recorded annotation is in-time).

    Returns:
        ``True`` when ``valid_at`` is non-null and chronologically after
        ``as_of`` — the outcome is posterior to the pin and must be excluded.
        ``valid_at == as_of`` is in-time, not a leak.
    """
    if annotation is None or as_of is None:
        return False
    valid_at = annotation.get("valid_at")
    if valid_at is None:
        return False
    return datetime.fromisoformat(valid_at) > datetime.fromisoformat(as_of)


_OUTCOME_GOLD_LABELS = frozenset({"accepted"})


def _is_admitted_outcome_gold(
    label: str | None,
    has_posterior: bool,
    labeler_policy_version: str | None,
    decisive_mix: bool,
    decisive_only: bool,
) -> bool:
    """Gold-admission guard for outcome-bearing observations (M16/M22).

    Admitted iff the label is an accepted-class gold label **and** the row
    carries posterior evidence **and** the reply-classifier policy version is
    known **and** the run-level rubric is decisive-only (no contested mix of
    accepted/rejected dispositions, no ambiguous findings). Legacy rows —
    reply-count / merge-presence gold with no ``labeler_policy_version`` — are
    rejected: a missing policy version is the guard working, never a silent
    fallback to admitted.
    """
    return (
        label is not None
        and label in _OUTCOME_GOLD_LABELS
        and has_posterior
        and labeler_policy_version is not None
        and not decisive_mix
        and decisive_only
    )



def _trajectory_set_hash(session_ids: list[str]) -> str:
    """Content-address the set of included sessions (Q3).

    The hash is ``sha256`` of the sorted, newline-joined ``session_id``s, so a
    snapshot's identity is a deterministic function of which runs it contains
    (order-independent). A single-session corpus collapses to
    ``sha256(b"<session_id>")`` — there is no trailing newline or separator for
    one id.
    """
    joined = "\n".join(sorted(session_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()

