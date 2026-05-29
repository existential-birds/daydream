"""Pure build-corpus projection (gold layer) over the bitemporal archive.

This module owns ATIF-v1.6 trajectory → training-record conversion plus the
filter/stratify pipeline that selects which archived runs become training
records. It reads each run's label and reward from the ``as_of``-pinned silver
annotation (``label_observations``) — *not* the denormalized
``runs.outcome_labels`` cache — so a re-projection at a fixed ``as_of``
reproduces the corpus byte-for-byte even after a re-harvest appends newer
annotation generations.

A temporal-leakage guard (:func:`_is_posterior_leak`) protects the ``as_of``
pin against posterior label leakage: an annotation may be *recorded* before
``as_of`` yet describe an outcome whose valid time (``valid_at``, e.g. a PR
merge timestamp) lands *after* the pin. When ``valid_at > as_of`` the
posterior-derived ``outcome_label`` is dropped (the run is treated as
unlabeled); the intrinsic, capture-time reward fields survive and may still
admit the run via the ``min_reward`` path. The comparison is **lexical** on
ISO-8601/UTC strings — no datetime parsing — mirroring the ``observed_at <=
as_of`` SQL pin. When ``as_of`` is ``None`` no valid-time exclusion applies.

Every snapshot writes a lineage manifest (``lineage.json``) beside the JSONL,
pinning the snapshot's provenance: a content-addressed ``trajectory_set_hash``
(``sha256`` of the sorted, newline-joined included ``session_id``s — Q3), the
``labeler_version``/``reward_version`` observed on the included annotations (a
scalar when uniform, the sorted distinct set when the corpus mixes versions),
the ``as_of`` pin (echoed from config, or the resolved write-time when
unpinned), and a wall-clock UTC ``created_at``. The manifest is written
atomically (tempfile + ``os.replace``) and skipped on ``dry_run``; it subsumes
the retired ``daydream snapshot`` verb's reproducibility role.

The projection is pure: no git, no network, no manifest write-back.
``base_sha`` is *read* from the on-disk manifest (harvest materializes it);
build-corpus never resolves it via ``git merge-base``.

Builders (``_build_spans``, ``_build_record``) and the query pipeline
(``CorpusFilters``, ``_build_query``, ``_query_index``) are private
(underscore-prefixed): callers outside this package depend on
``run_build_corpus``, not these helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daydream.archive import get_archive_dir
from daydream.archive.index import bulk_latest_label_observations, count_runs, query_runs
from daydream.config import REVIEW_SKILLS
from daydream.training.exclusion import is_copyleft, load_copyleft_list, load_exclusion_list
from daydream.training.harvest import _read_review_output
from daydream.training.schema import TRAINING_SCHEMA_VERSION
from daydream.ui import create_console, print_info, print_warning

# Path to the on-disk JSON Schema describing emitted training records.
# Resolves relative to ``__file__`` so the package works when installed.
SCHEMA_V1_PATH: Path = Path(__file__).parent / "schema" / "v1.json"


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


def _single_outcome_label(labels: list[str], session_id: str) -> str | None:
    """Return the sole outcome label, warning when multiple are present.

    The training schema (schema/v1.json) defines ``outcome_label`` as a single
    string.  When a run carries more than one label only the first is exported;
    a warning is emitted so the data loss is visible rather than silent.
    """
    if len(labels) > 1:
        warnings.warn(
            f"Session {session_id!r} has {len(labels)} outcome labels "
            f"{labels!r}; exporting only the first ({labels[0]!r}). "
            "Widen schema/v1.json `outcome_label` to an array to preserve all labels.",
            stacklevel=3,
        )
    return labels[0] if labels else None


def _annotation_labels(annotation: dict[str, Any] | None, session_id: Any) -> list[str]:
    """Parse the label list from a pinned silver annotation row.

    The ``labels`` column is a JSON-encoded string list. A ``None`` annotation
    (no in-time observation) yields ``[]`` — the run is unlabeled. Malformed
    JSON is warned about and treated as no labels rather than crashing the
    projection.

    Args:
        annotation: The ``latest_label_observation`` row dict, or ``None``.
        session_id: Session identifier for warning context.

    Returns:
        The decoded label list, or ``[]`` when absent/unparseable.
    """
    if annotation is None:
        return []
    raw_labels = annotation.get("labels", "[]")
    try:
        labels = json.loads(raw_labels) if isinstance(raw_labels, str) else []
    except (json.JSONDecodeError, TypeError) as exc:
        warnings.warn(
            f"Session {session_id!r} has invalid annotation labels {raw_labels!r}: {exc}",
            stacklevel=3,
        )
        return []
    return labels if isinstance(labels, list) else []


def _annotation_reward(
    annotation: dict[str, Any] | None, session_id: Any
) -> tuple[dict[str, Any] | None, float | None]:
    """Extract ``(reward, composite_reward)`` from a pinned annotation row.

    ``reward`` is the full ``RewardBreakdown.to_dict()`` parsed from the
    ``reward_json`` column; ``None`` when the column is missing/empty/non-object
    or unparseable (warned). ``composite_reward`` is the cached scalar on the
    row (already ``float | None``). A ``None`` annotation yields ``(None, None)``.

    Args:
        annotation: The ``latest_label_observation`` row dict, or ``None``.
        session_id: Session identifier for warning context.

    Returns:
        ``(reward_dict_or_none, composite_reward_or_none)``.
    """
    if annotation is None:
        return None, None
    composite = annotation.get("composite_reward")
    raw_reward = annotation.get("reward_json")
    reward: dict[str, Any] | None = None
    if isinstance(raw_reward, str) and raw_reward:
        try:
            parsed = json.loads(raw_reward)
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"Session {session_id!r}: malformed reward_json: {exc}",
                stacklevel=3,
            )
            parsed = None
        if isinstance(parsed, dict):
            reward = parsed
    return reward, composite


def _build_record(
    manifest_row: dict[str, Any],
    trajectory: dict[str, Any],
    stack: str | None,
    manifest: dict[str, Any] | None = None,
    *,
    annotation: dict[str, Any] | None = None,
    drop_label: bool = False,
) -> dict[str, Any]:
    """Assemble a training record matching ``schema/v1.json``.

    The returned dict carries refs (``fix_diff_ref``, ``trajectory_ref``,
    ``spans[*].content_path``) instead of embedded content, so records stay
    small and downstream consumers materialize bytes from the archive on
    demand.

    ``base_sha`` and ``changed_files`` are *read* from the on-disk
    ``manifest.json`` (passed as ``manifest``) via its ``code_context`` block.
    Older archives written before that block existed surface ``base_sha=None``
    and ``changed_files=[]``; harvest (not this projection) materializes a
    missing ``base_sha`` via ``git merge-base``. build-corpus performs no git,
    no network, and no manifest write-back — it is a pure read.
    ``review_output`` is read from ``<archive_path>/review-output.md`` when
    present; deep-mode archives that only emit the file under
    ``<archive_path>/deep/review-output.md`` fall back to that path. Root
    wins when both exist.

    The ``outcome_label``, ``reward`` and ``composite_reward`` fields come from
    the ``as_of``-pinned silver annotation passed as ``annotation`` — never from
    the denormalized ``runs.outcome_labels`` cache. A run with no pinned
    annotation (``annotation is None``) is unlabeled (``outcome_label=None``,
    ``composite_reward=None``, no ``reward`` key).

    The temporal-leakage guard sets ``drop_label=True`` when the pinned
    annotation's ``valid_at`` is posterior to ``as_of`` (see
    :func:`_is_posterior_leak`): ``outcome_label`` is forced to ``None``
    (the posterior-derived label is excluded as future leakage) while the
    intrinsic ``reward``/``composite_reward`` — capture-time fields — are
    retained.

    Optional surfaced fields (additive — omitted when absent, never written
    as ``None`` unless the schema models a nullable scalar):

    - ``reward``: the parsed ``annotation["reward_json"]`` dict, written
      verbatim with no transform. For an unlabeled run this is
      ``RewardBreakdown.to_dict()``; for a labeled (PR-outcome) run it is
      ``PosteriorBreakdown.to_dict()``, which additionally carries
      ``posterior_cost``. Thus ``"posterior_cost" in record["reward"]`` is the
      **population discriminator** — present only on labeled rows (C3). The key
      is omitted entirely when the annotation has no ``reward_json`` or it is
      unparseable.
    - ``composite_reward``: ``annotation["composite_reward"]`` scalar; written
      unconditionally (``null`` when uncomputable / unscored). The schema models
      it as a nullable scalar (``["number", "null"]``), so the ``None``
      placeholder is valid and every record carries the key uniformly.
    - ``rubric``: the full parsed dict from ``manifest_row["rubric_json"]``
      when that column holds a JSON object. Invalid JSON or non-dict values
      are silently treated as missing.
    - ``posterior_source``: copied from ``rubric["posterior_source"]`` when
      that key is present in the parsed rubric. Omitted otherwise.

    Args:
        manifest_row: Dict shaped like ``Manifest.to_dict()["manifest"]`` or a
            flat row from the SQLite index. Must carry ``session_id``,
            ``skill``, ``repo_slug``, ``branch``, ``base_branch``, ``head_sha``,
            ``grounding_rate``, and ``archive_path``.
        trajectory: ATIF v1.6 trajectory dict for this run.
        stack: Routing label (e.g. ``"python"``, ``"react"``). The caller
            derives this from deep-stack detection; pass ``None`` when
            unavailable.
        manifest: Parsed ``manifest.json`` dict for this run. ``None`` (the
            default) is treated as "manifest unavailable" — ``code_context``
            falls back to the row scalars with ``base_sha=None`` and
            ``changed_files=[]``.
        annotation: The ``as_of``-pinned ``label_observations`` row (silver) for
            this run, or ``None`` when the run has no in-time annotation. The
            label, reward breakdown and composite scalar are sourced from here.
        drop_label: When ``True`` (temporal-leakage guard), emit
            ``outcome_label=None`` regardless of the annotation's label; the
            intrinsic reward fields are still sourced from ``annotation``.

    Returns:
        A dict that validates against ``daydream/training/schema/v1.json``.
    """
    archive_path = Path(manifest_row.get("archive_path", ""))
    diff_path = archive_path / "diff.patch"
    fix_diff_ref = {
        "available": diff_path.is_file(),
        "archive_relative_path": "diff.patch",
    }

    # The label comes from the as_of-pinned silver annotation, NOT the
    # denormalized runs.outcome_labels cache. No annotation ⇒ unlabeled.
    # The leakage guard (drop_label) forces unlabeled when the annotation's
    # outcome is posterior to as_of (future leakage); intrinsic reward stays.
    labels = [] if drop_label else _annotation_labels(annotation, manifest_row.get("session_id"))

    manifest_dict = manifest or {}
    code_ctx = manifest_dict.get("code_context") or {}
    git_block = manifest_dict.get("git") or {}
    changed = code_ctx.get("changed_files") or []
    if not isinstance(changed, list):
        changed = []

    # base_sha is READ from the manifest only — harvest materializes it.
    # build-corpus performs no git and no write-back (pure projection).

    # review-output.md lives at the archive root for shallow-loop runs but
    # only under deep/ for deep-mode runs. _read_review_output tries root
    # first (back-compat), then deep/; OSError (non-missing I/O failure) is
    # preserved as a warn-and-continue here.
    review_output: str | None
    try:
        review_output = _read_review_output(archive_path)
    except OSError as exc:
        warnings.warn(
            f"Session {manifest_row.get('session_id')!r}: failed to read review-output.md: {exc}",
            stacklevel=2,
        )
        review_output = None

    record: dict[str, Any] = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "session_id": manifest_row["session_id"],
        "repo_slug": manifest_row.get("repo_slug"),
        "skill": manifest_row.get("skill"),
        "stack": stack,
        "code_context": {
            "base_sha": code_ctx.get("base_sha"),
            "head_sha": git_block.get("head_sha") or manifest_row.get("head_sha"),
            "base_branch": git_block.get("base_branch") or manifest_row.get("base_branch"),
            "branch": git_block.get("branch") or manifest_row.get("branch"),
            "changed_files": [str(p) for p in changed],
        },
        "review_output": review_output,
        "fix_diff_ref": fix_diff_ref,
        "outcome_label": _single_outcome_label(labels, manifest_row["session_id"]),
        "grounding_score": manifest_row.get("grounding_rate"),
        "spans": _build_spans(trajectory),
        "trajectory_ref": {"archive_relative_path": "trajectory.json"},
    }

    # Optional reward fields from the pinned annotation. The schema models
    # ``composite_reward`` as ``["number", "null"]`` (nullable), so it is
    # written unconditionally; ``reward`` is ``type: object`` (not nullable),
    # so the key is omitted entirely when absent rather than written as
    # ``None``. additionalProperties is false, so additive-field discipline —
    # only insert optional keys when present — is what keeps every record valid
    # against schema/v1.json.
    # No transform: ``reward`` is ``reward_json`` parsed verbatim, so a labeled
    # run's ``PosteriorBreakdown.to_dict()`` carries ``posterior_cost`` and an
    # unlabeled run's ``RewardBreakdown.to_dict()`` does not. That presence/
    # absence is the population discriminator (C3). ``composite_reward`` is the
    # pure intrinsic composite either way (C5: posterior is a sibling field).
    reward, composite_reward = _annotation_reward(annotation, manifest_row.get("session_id"))
    record["composite_reward"] = composite_reward
    if reward is not None:
        record["reward"] = reward

    # Rubric + posterior_source are additive optional fields. The schema models
    # ``rubric`` as ``type: object`` and ``posterior_source`` as an enum string
    # — neither nullable — so each key is written only when a value is present.
    raw_rubric = manifest_row.get("rubric_json")
    parsed_rubric: dict | None = None
    if isinstance(raw_rubric, str) and raw_rubric:
        try:
            parsed_rubric = json.loads(raw_rubric)
        except json.JSONDecodeError as exc:
            warnings.warn(
                f"Session {manifest_row.get('session_id')!r}: malformed rubric_json: {exc}",
                stacklevel=2,
            )
            parsed_rubric = None
        if not isinstance(parsed_rubric, dict):
            parsed_rubric = None
    if parsed_rubric is not None:
        record["rubric"] = parsed_rubric
        posterior_source = parsed_rubric.get("posterior_source")
        if posterior_source is not None:
            record["posterior_source"] = posterior_source

    return record


# ---------------------------------------------------------------------------
# Filter + query pipeline (Wave 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusFilters:
    """Post-exclusion filter knobs for the build-corpus projection.

    The C5 exclusion list is appended to every SQL query unconditionally — it
    is not represented here because callers cannot disable it. C8 copyleft
    handling is opt-in via ``allow_copyleft``: any slug in the frozenset is
    re-admitted past the copyleft filter that ``_query_index`` applies
    post-query (the copyleft list itself lives in
    ``schema/copyleft.txt`` and is loaded by ``is_copyleft``).

    The label admission filter is **not** a SQL clause — it runs in Python
    against the ``as_of``-pinned silver annotation (see ``run_build_corpus``),
    never against the denormalized ``runs.outcome_labels`` cache. The
    admission rule is C9 accepted-only **OR** intrinsic-reward ≥ ``min_reward``.

    Attributes:
        skill: Optional exact-match filter on the ``skill`` column.
        repos: Optional whitelist of ``owner/repo`` slugs (IN-clause).
        labels: Outcome labels considered acceptable, matched against the
            pinned annotation's label. Defaults to ``("accepted",)`` per C9.
            Ignored when ``include_all_labels`` is ``True``.
        min_grounding: Optional minimum ``grounding_rate`` (inclusive).
        status: Exact-match filter on the ``status`` column. Defaults to
            ``"complete"`` so partial / failed runs never enter training.
        include_all_labels: When ``True``, suppresses the label admission
            filter so every run (labeled or not) passes.
        allow_copyleft: ``owner/repo`` slugs the caller has explicitly
            opted in via ``--allow-copyleft``; bypasses the C8 skip.
        min_reward: Optional intrinsic ``composite_reward`` threshold
            (inclusive). When set, a run whose pinned annotation has a
            ``composite_reward >= min_reward`` is admitted even if its label
            is not in ``labels`` — an alternative admission path to C9. The
            threshold is intrinsic-only by construction (C5): the stored
            ``composite_reward`` is the pure intrinsic composite, with the
            posterior false-positive axis kept as a sibling field
            (``posterior_cost``), never subtracted into the composite.
    """

    skill: str | None = None
    repos: tuple[str, ...] = ()
    labels: tuple[str, ...] = ("accepted",)
    min_grounding: float | None = None
    status: str = "complete"
    include_all_labels: bool = False
    allow_copyleft: frozenset[str] = frozenset()
    min_reward: float | None = None


# Inverse of REVIEW_SKILLS with dual keys: both the full skill string
# (e.g. "beagle-python:review-python") AND the short stack name
# (e.g. "python") map to the lowercase stack label. Built at import time
# from a single pass over ``REVIEW_SKILLS.items()`` so ``_stack_for_skill``
# remains a single dict lookup regardless of which form the manifest
# stored.
_SKILL_TO_STACK: dict[str, str] = {}
for _choice, _skill in REVIEW_SKILLS.items():
    _short = _choice.name.lower()
    _SKILL_TO_STACK[_skill] = _short
    _SKILL_TO_STACK[_short] = _short
del _choice, _skill, _short


def _stack_for_skill(skill: str | None) -> str | None:
    """Return the stack label (e.g. ``"python"``, ``"react"``) for a skill.

    Args:
        skill: The manifest's skill field, either the full skill string
            (e.g. ``"beagle-python:review-python"``) or the short stack
            name (e.g. ``"python"``); both are accepted. ``None`` is also
            accepted.

    Returns:
        The lowercase stack label, or ``None`` when ``skill`` is ``None``
        or not present in ``REVIEW_SKILLS`` under either form.
    """
    if skill is None:
        return None
    return _SKILL_TO_STACK.get(skill)


def _build_query(
    filters: CorpusFilters,
    *,
    exclusion: frozenset[str] | set[str] | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """Assemble the WHERE clause + bound params for ``query_runs``.

    The returned ``where`` string is suitable for direct hand-off to
    ``daydream.archive.index.query_runs`` (which prepends ``WHERE`` when
    non-empty). ORDER BY is intentionally *not* emitted here — callers
    sort the result list in Python (see ``_query_index``).

    The label admission filter is deliberately **not** a SQL clause: labels
    come from the ``as_of``-pinned silver annotation resolved per row in
    ``run_build_corpus``, not from the denormalized ``runs.outcome_labels``
    column. SQL only narrows on capture-time, label-independent columns.

    Clauses, in fixed order:

    1. ``status = ?`` — always.
    2. C5 exclusion: ``repo_slug IS NULL OR repo_slug NOT IN (...)`` — always
       (when the exclusion list is non-empty; otherwise omitted to keep the
       SQL clean). Placeholder count is derived from the loaded set size.
    3. ``skill = ?`` — when ``filters.skill`` is set.
    4. ``repo_slug IN (...)`` — when ``filters.repos`` is non-empty.
    5. ``grounding_rate >= ?`` — when ``filters.min_grounding`` is set.

    Args:
        filters: Resolved filter knobs.

    Returns:
        ``(where_clause, params)`` where ``where_clause`` has no leading
        ``WHERE`` keyword (per ``query_runs``'s contract) and ``params`` is
        the tuple of bound values in left-to-right order.
    """
    clauses: list[str] = []
    params: list[Any] = []

    # 1. status — always
    clauses.append("status = ?")
    params.append(filters.status)

    # 2. C5 exclusion — always (when list is non-empty)
    if exclusion is None:
        exclusion = load_exclusion_list()
    if exclusion:
        # Stable ordering so the generated SQL is reproducible across runs.
        ordered_exclusion = sorted(exclusion)
        placeholders = ", ".join(["?"] * len(ordered_exclusion))
        clauses.append(f"(repo_slug IS NULL OR repo_slug NOT IN ({placeholders}))")
        params.extend(ordered_exclusion)

    # 3. skill — optional
    if filters.skill is not None:
        clauses.append("skill = ?")
        params.append(filters.skill)

    # 4. repos — optional
    if filters.repos:
        placeholders = ", ".join(["?"] * len(filters.repos))
        clauses.append(f"repo_slug IN ({placeholders})")
        params.extend(filters.repos)

    # 5. min_grounding — optional
    if filters.min_grounding is not None:
        clauses.append("grounding_rate >= ?")
        params.append(filters.min_grounding)

    where = " AND ".join(clauses)
    return where, tuple(params)


def _query_index(archive_dir: Path, filters: CorpusFilters) -> list[dict[str, Any]]:
    """Query the SQLite index, apply C8 copyleft skip, decorate, and sort.

    Goes through the public ``query_runs`` helper (no direct SQLite
    connection management here). The C8 copyleft check runs *after* the
    SQL query because the copyleft list is small and pulling it into the
    WHERE clause would duplicate logic that already lives in ``is_copyleft``.

    Each surviving row is augmented with a derived ``"stack"`` key so
    downstream stratification and record building can route on stack
    without re-deriving from the skill string.

    Args:
        archive_dir: Path to the daydream archive root (e.g.
            ``~/.daydream/archive``).
        filters: Resolved filter knobs.

    Returns:
        Rows in lexicographic ``session_id`` order so emission is
        reproducible across runs.
    """
    where, params = _build_query(filters)
    rows = query_runs(archive_dir, where, params)

    copyleft_list = load_copyleft_list()
    survivors: list[dict[str, Any]] = []
    unknown_skills: set[str] = set()
    for row in rows:
        if is_copyleft(row.get("repo_slug"), filters.allow_copyleft, copyleft_list=copyleft_list):
            continue
        skill = row.get("skill")
        stack = _stack_for_skill(skill)
        if stack is None and isinstance(skill, str) and skill:
            unknown_skills.add(skill)
        row["stack"] = stack
        survivors.append(row)

    if unknown_skills:
        console = create_console()
        for skill in sorted(unknown_skills):
            print_warning(
                console,
                f"Skill {skill!r} is not in REVIEW_SKILLS — records will be "
                f"stratified under stack=None. Add it to ReviewSkillChoice if "
                f"it should route to its own stack.",
            )

    survivors.sort(key=lambda r: r["session_id"])
    return survivors


# ---------------------------------------------------------------------------
# Stratification (Wave 5)
# ---------------------------------------------------------------------------


def _stratify(records: list[dict], max_stack_share: float) -> list[dict]:
    """Apply a rough dominance cap so no single stack floods the corpus.

    Implements plan §6:

    1. Group records by ``record["stack"]`` (preserving input order within
       each group).
    2. ``total = len(records)``.
    3. ``cap_per_stack = max(1, floor(total * max_stack_share))`` — the
       ``max(1, ...)`` guard is the degenerate-corpus rule from §6: when the
       floor rounds to 0 for a small archive, keep at least one record per
       stack so no group is fully erased.
    4. For each stack: keep the first ``min(len(group), cap_per_stack)``
       records, preserving the original within-group order.
    5. Concatenate groups (iterating stack keys in deterministic order),
       then sort the final list by ``session_id`` to restore global ordering
       (AC #7).

    **Important — input-corpus cap, not output-share guarantee.**
    ``cap_per_stack`` is computed from the *input* corpus size, not the
    output size.  When other stacks contribute fewer records than their cap
    allows, the output share of a large stack can exceed ``max_stack_share``.
    Example: 80 ``react`` + 20 ``python`` records with ``max_stack_share=0.6``
    → ``cap=60``, ``react`` keeps 60, ``python`` keeps 20, output is 80
    records, ``react`` share = 75 % > 60 %.  ``max_stack_share`` is therefore
    a rough dominance guard, not a strict per-stack output-fraction cap.

    Records whose ``stack`` is ``None`` form their own group. Mixing ``None``
    and ``str`` in ``sorted()`` would raise ``TypeError`` on Python 3.12, so
    the intermediate group iteration uses a ``key`` that maps ``None`` to the
    empty string (sorts first); the final ``session_id`` sort makes this
    cosmetic but keeps the intermediate trace deterministic.

    Args:
        records: List of records (each with ``"stack"`` and ``"session_id"``
            keys). Records with ``stack=None`` group under the ``None`` key.
            The input list is not mutated.
        max_stack_share: Fraction in ``(0, 1]`` — typically ``0.6`` (the CLI
            default). Validation of the range happens at the CLI layer; this
            helper trusts its caller.

    Returns:
        A new list, sorted by ``session_id``. Empty when ``records`` is
        empty.
    """
    if not records:
        return []

    groups: dict[str | None, list[dict]] = defaultdict(list)
    for record in records:
        groups[record.get("stack")].append(record)

    total = len(records)
    cap_per_stack = max(1, math.floor(total * max_stack_share))

    out: list[dict] = []
    for stack_key in sorted(groups.keys(), key=lambda k: "" if k is None else k):
        group = groups[stack_key]
        out.extend(group[:cap_per_stack])

    return sorted(out, key=lambda r: r["session_id"])


# ---------------------------------------------------------------------------
# End-to-end orchestration (Wave 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildCorpusConfig:
    """Top-level config for :func:`run_build_corpus`.

    ``filters`` is required (no default) — callers must construct a
    :class:`CorpusFilters` explicitly so the C9 ``("accepted",)`` default is
    a deliberate choice, not a silent fall-through.

    Attributes:
        out_path: Destination JSONL file. Written atomically via tempfile +
            ``Path.replace``; ``schema.json`` is emitted next to it.
        filters: Resolved post-exclusion filter knobs.
        archive_dir: Daydream archive root. ``None`` defers to
            :func:`daydream.archive.get_archive_dir`.
        stratify_by: ``"stack"`` to apply :func:`_stratify`; ``None`` to skip.
        max_stack_share: Per-stack cap fraction passed to :func:`_stratify`.
        dry_run: When ``True``, do not write the JSONL output; print a
            summary table to stdout and return ``emitted=0``.
        emit_schema_only: When ``True``, copy ``schema/v1.json`` next to
            ``out_path`` and return immediately without querying the archive.
        as_of: ISO-8601 transaction-time pin. Each run's annotation is resolved
            via ``latest_label_observation(..., as_of=as_of)`` so the corpus is
            reproducible: a re-projection at the same ``as_of`` reproduces the
            prior corpus even after a re-harvest appends newer generations.
            ``None`` resolves the latest annotation per run (no pin).
    """

    out_path: Path
    filters: CorpusFilters
    archive_dir: Path | None = None
    stratify_by: str | None = None
    max_stack_share: float = 0.6
    dry_run: bool = False
    emit_schema_only: bool = False
    as_of: str | None = None


def _is_posterior_leak(annotation: dict[str, Any] | None, as_of: str | None) -> bool:
    """Return ``True`` when the annotation's outcome only became true after ``as_of``.

    Temporal-leakage guard: an annotation may be *recorded* before the ``as_of``
    pin (``observed_at <= as_of``, already enforced by
    ``latest_label_observation``) yet describe an outcome whose valid time —
    e.g. a PR merge timestamp — lands *after* the pin. Such posterior-derived
    fields (the outcome label and posterior reward axes) would leak future
    information into a corpus pinned to ``as_of``, so they are dropped.

    The comparison is **lexical** on ISO-8601 strings. ``valid_at`` is written
    in UTC (the per-run builder uses ``PRMergeSignal.merged_at``; local runs
    collapse ``None`` to ``observed_at`` at write time, also UTC), so the
    lexical ordering of the strings matches chronological order — no datetime
    parsing is required, mirroring the ``observed_at <= as_of`` SQL pin in
    :func:`daydream.archive.index.latest_label_observation`.

    Both strings are normalised to the ``+00:00`` suffix before comparison so
    that the ``Z`` and ``+00:00`` UTC spellings sort identically.

    Args:
        annotation: The ``as_of``-pinned ``label_observations`` row, or ``None``.
        as_of: The transaction-time pin. When ``None`` no valid-time exclusion
            applies (every recorded annotation is in-time).

    Returns:
        ``True`` when ``valid_at`` is non-null and lexically greater than
        ``as_of`` — the outcome is posterior to the pin and must be excluded.
    """
    if annotation is None or as_of is None:
        return False
    valid_at = annotation.get("valid_at")
    if valid_at is None:
        return False
    # Normalise the UTC suffix so "Z" and "+00:00" compare identically.
    def _norm(s: str) -> str:
        return s.replace("Z", "+00:00") if s.endswith("Z") else s

    return _norm(valid_at) > _norm(as_of)


def _is_admitted(label: str | None, composite_reward: float | None, filters: CorpusFilters) -> bool:
    """Decide whether a run is admitted into the corpus.

    Admission rule (C9, with an alternative reward path):

    - ``include_all_labels=True`` ⇒ always admit.
    - Otherwise admit when the pinned ``label`` is in ``filters.labels``
      (C9 accepted-only), **OR** when ``filters.min_reward`` is set and the
      intrinsic ``composite_reward`` is present and ``>= min_reward``.

    The ``min_reward`` comparison is intrinsic-only **by construction** (C5),
    not by stripping a posterior term. Post-C5 the stored ``composite_reward``
    *is* the pure intrinsic composite (correctness + grounding − length
    penalty); the posterior false-positive axis lives as a sibling field
    (``posterior_cost`` on :class:`PosteriorBreakdown`), never folded into the
    composite. So even a labeled row carrying a posterior penalty is admitted
    on its intrinsic score alone — there is no deduction to remove here. This
    invariant is pinned by
    ``test_is_admitted_min_reward_compares_intrinsic_only``.

    Args:
        label: The pinned annotation's outcome label, or ``None`` (unlabeled).
        composite_reward: The pinned annotation's composite reward scalar.
        filters: Resolved filter knobs.

    Returns:
        ``True`` when the run should be emitted.
    """
    if filters.include_all_labels:
        return True
    if label is not None and label in filters.labels:
        return True
    if filters.min_reward is not None and composite_reward is not None and composite_reward >= filters.min_reward:
        return True
    return False


def _trajectory_set_hash(session_ids: list[str]) -> str:
    """Content-address the set of included sessions (Q3).

    The hash is ``sha256`` of the sorted, newline-joined ``session_id``s, so a
    snapshot's identity is a deterministic function of which runs it contains
    (order-independent). A single-session corpus collapses to
    ``sha256(b"<session_id>")`` — there is no trailing newline or separator for
    one id.

    Args:
        session_ids: The ``session_id`` of every record actually emitted.

    Returns:
        The hex SHA-256 digest of the sorted, newline-joined ids.
    """
    joined = "\n".join(sorted(session_ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _collapse_versions(versions: list[str | None]) -> str | list[str] | None:
    """Reduce observed version tags to a scalar, a sorted list, or ``None``.

    A corpus whose included annotations all share one version records that
    scalar; a corpus mixing versions records the sorted distinct set so the
    lineage manifest is honest about heterogeneity. An empty corpus yields
    ``None``.

    Args:
        versions: The version tag observed on each included annotation.

    Returns:
        The single version string when uniform, the sorted distinct list when
        mixed, or ``None`` when no annotations were included.
    """
    distinct = sorted({v for v in versions if v is not None})
    if not distinct:
        return None
    if len(distinct) == 1:
        return distinct[0]
    return distinct


def _atomic_write_json(out_path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``out_path`` atomically (tempfile + replace).

    Mirrors the snapshot writer's pattern: a temp file in the destination
    directory, an ``os.replace`` rename, and best-effort cleanup of the temp
    file if writing fails. JSON is sorted and indented for stable diffs.

    Args:
        out_path: Destination path for the lineage manifest.
        payload: The lineage dict to serialize.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
            fp.write("\n")
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def run_build_corpus(config: BuildCorpusConfig) -> dict[str, int]:
    """Top-level entry point for the build-corpus projection.

    Pure projection — no git, no network, no manifest write-back. Pipeline:

    1. ``emit_schema_only`` short-circuit — copy ``schema/v1.json`` next to
       ``out_path`` and return.
    2. Resolve archive dir (``config.archive_dir`` or :func:`get_archive_dir`).
    3. Count the unfiltered index for the ``total_runs_in_index`` summary.
    4. Apply :func:`_query_index` (label-independent SQL filters only).
    5. For each row: resolve the ``as_of``-pinned silver annotation, apply the
       temporal-leakage guard (:func:`_is_posterior_leak` — drop the
       posterior-derived label when ``valid_at > as_of``), apply the Python
       label/reward admission gate, then load manifest + trajectory and build a
       record via :func:`_build_record`. Skip rows with missing/unreadable
       files (with a warning).
    6. Optionally stratify (when ``stratify_by == "stack"``).
    7. Dry-run path prints a summary and returns ``emitted=0`` (no JSONL,
       no lineage manifest).
    8. Otherwise: write JSONL atomically (tempfile in same dir +
       ``Path.replace``) and copy ``schema/v1.json`` alongside.
    9. Write ``lineage.json`` beside the JSONL pinning the snapshot's
       provenance (``trajectory_set_hash``, labeler/reward versions, ``as_of``,
       ``created_at``) so the snapshot is reproducible from immutable inputs.

    Args:
        config: Resolved build-corpus config.

    Returns:
        Summary dict with keys ``total_runs_in_index``, ``after_filters``,
        ``after_stratify``, ``emitted``. ``after_filters`` counts rows that
        survive the SQL filters **and** the label/reward admission gate.
        ``emitted`` is ``0`` for ``dry_run`` and ``emit_schema_only`` paths.
    """
    console = create_console()

    # 1. Schema-only path — never touches the archive.
    if config.emit_schema_only:
        config.out_path.parent.mkdir(parents=True, exist_ok=True)
        schema_dst = config.out_path.parent / "schema.json"
        shutil.copyfile(SCHEMA_V1_PATH, schema_dst)
        return {
            "total_runs_in_index": 0,
            "after_filters": 0,
            "after_stratify": 0,
            "emitted": 0,
        }

    # 2. Resolve archive dir.
    archive_dir = config.archive_dir or get_archive_dir()

    # 3. Unfiltered count for the summary.
    total_in_index = count_runs(archive_dir)

    # 4. Apply label-independent SQL filters.
    rows = _query_index(archive_dir, config.filters)

    # 5. Resolve the pinned annotation per row, apply the admission gate, and
    #    build records. The label/reward come from the silver annotation, never
    #    the denormalized runs.outcome_labels cache.
    #
    #    C3 population separation: this is a pure per-row projection. There is
    #    NO cross-row aggregate over rewards/breakdowns here — each record's
    #    reward dict is emitted independently and the only aggregate downstream
    #    (``_stratify``) groups by ``stack``, never by reward. Labeled and
    #    unlabeled populations are therefore never mixed into a single mean.
    #    Future SQL consumers that DO aggregate over rewards must split the
    #    populations first; the ``has_posterior`` boolean column on the ``runs``
    #    mirror (archive/index.py) is the guard for that split, and within a
    #    JSONL record ``"posterior_cost" in record["reward"]`` is the equivalent
    #    discriminator.
    records: list[dict[str, Any]] = []
    # Map each emitted session to the labeler/reward versions on its pinned
    # annotation so the lineage manifest reports the versions actually included.
    session_versions: dict[str, tuple[str | None, str | None]] = {}
    after_filters = 0
    # Pre-fetch all annotations in a single query to avoid N+1 round-trips.
    all_session_ids = [row["session_id"] for row in rows]
    annotations_by_session = bulk_latest_label_observations(
        archive_dir, all_session_ids, as_of=config.as_of
    )
    for row in rows:
        session_id = row["session_id"]
        annotation = annotations_by_session.get(session_id)
        # Temporal-leakage guard: when the pinned annotation's outcome only
        # became true after ``as_of`` (``valid_at > as_of``), drop the
        # posterior-derived label — the run is treated as unlabeled. Intrinsic
        # reward/composite_reward are capture-time fields and survive (they may
        # still admit the run via the min_reward path).
        posterior_leak = _is_posterior_leak(annotation, config.as_of)
        if posterior_leak:
            label: str | None = None
        else:
            labels = _annotation_labels(annotation, session_id)
            label = _single_outcome_label(labels, session_id)
        composite_reward = annotation.get("composite_reward") if annotation is not None else None
        if not _is_admitted(label, composite_reward, config.filters):
            continue
        after_filters += 1

        archive_path = Path(row["archive_path"])
        traj_path = archive_path / "trajectory.json"
        manifest_path = archive_path / "manifest.json"
        if not traj_path.exists() or not manifest_path.exists():
            print_warning(
                console,
                f"Skipping {session_id}: missing manifest.json or "
                f"trajectory.json at {archive_path}",
            )
            continue
        try:
            trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print_warning(
                console,
                f"Skipping {session_id}: corrupt or unreadable manifest.json or "
                f"trajectory.json at {archive_path}",
            )
            continue
        records.append(
            _build_record(
                row, trajectory, row.get("stack"), manifest,
                annotation=annotation, drop_label=posterior_leak,
            )
        )
        session_versions[session_id] = (
            annotation.get("labeler_version") if annotation is not None else None,
            annotation.get("reward_version") if annotation is not None else None,
        )

    # 6. Stratification (optional).
    if config.stratify_by == "stack":
        none_stack_count = sum(1 for r in records if r.get("stack") is None)
        if none_stack_count:
            print_warning(
                console,
                f"{none_stack_count} record(s) have stack=None (unmapped skills) "
                "and will be grouped together during stratification.",
            )
        records = _stratify(records, config.max_stack_share)
    after_stratify = len(records)

    # 7. Dry-run path — print summary, write nothing.
    if config.dry_run:
        print_info(console, f"Dry run — total_runs_in_index: {total_in_index}")
        print_info(console, f"Dry run — after_filters: {after_filters}")
        print_info(console, f"Dry run — after_stratify: {after_stratify}")
        print_info(console, f"Dry run — would emit: {len(records)} records")
        return {
            "total_runs_in_index": total_in_index,
            "after_filters": after_filters,
            "after_stratify": after_stratify,
            "emitted": 0,
        }

    # 8. Atomic write — tempfile in same dir + Path.replace.
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config.out_path.parent,
            delete=False,
            suffix=".jsonl.tmp",
        ) as fh:
            tmp_name = fh.name
            for record in records:
                fh.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
        Path(tmp_name).replace(config.out_path)
    except Exception:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise
    shutil.copyfile(SCHEMA_V1_PATH, config.out_path.parent / "schema.json")

    # 9. Lineage manifest — write ``lineage.json`` beside the JSONL pinning the
    #    provenance of this snapshot. The included set drives the content-address
    #    (``trajectory_set_hash``); versions reflect the annotations actually
    #    emitted (post-stratify). ``as_of`` echoes the config pin, falling back to
    #    the resolved write-time when unpinned. Skipped on dry_run (handled above).
    #    Also skipped when the corpus is empty — an empty-set hash is misleading.
    if not records:
        print_info(console, f"Wrote 0 records to {config.out_path} — skipping lineage manifest")
        return {
            "total_runs_in_index": total_in_index,
            "after_filters": after_filters,
            "after_stratify": after_stratify,
            "emitted": 0,
        }
    included_session_ids = [r["session_id"] for r in records]
    included_versions = [session_versions.get(sid, (None, None)) for sid in included_session_ids]
    resolved_as_of = config.as_of or datetime.now(timezone.utc).isoformat()
    lineage = {
        "trajectory_set_hash": _trajectory_set_hash(included_session_ids),
        "labeler_version": _collapse_versions([lv for lv, _ in included_versions]),
        "reward_version": _collapse_versions([rv for _, rv in included_versions]),
        "as_of": resolved_as_of,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(config.out_path.parent / "lineage.json", lineage)
    print_info(console, f"Wrote {len(records)} records to {config.out_path}")

    return {
        "total_runs_in_index": total_in_index,
        "after_filters": after_filters,
        "after_stratify": after_stratify,
        "emitted": len(records),
    }
