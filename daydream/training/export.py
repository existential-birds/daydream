"""Training-record and span builders for the JSONL exporter.

This module owns ATIF-v1.6 trajectory → training-record conversion plus the
filter pipeline that selects which archived runs become training records.
Higher-level orchestration (stratification, file emission, CLI surface) lands
in later waves.

Builders (``_build_spans``, ``_build_record``) and the query pipeline
(``ExportFilters``, ``_build_query``, ``_query_index``) are private
(underscore-prefixed): callers outside this package should depend on
``run_export`` once it lands in Wave 6, not on these helpers.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.archive import get_archive_dir
from daydream.archive.index import query_runs
from daydream.config import REVIEW_SKILLS
from daydream.training.exclusion import is_copyleft, load_exclusion_list
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
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a training record matching ``schema/v1.json``.

    The returned dict carries refs (``fix_diff_ref``, ``trajectory_ref``,
    ``spans[*].content_path``) instead of embedded content, so records stay
    small and downstream consumers materialize bytes from the archive on
    demand.

    ``base_sha`` and ``changed_files`` are sourced from the on-disk
    ``manifest.json`` (passed as ``manifest``) via its ``code_context`` block.
    Older archives written before that block existed surface ``base_sha=None``
    and ``changed_files=[]``. ``review_output`` is read from
    ``<archive_path>/review-output.md`` when present.

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
        manifest: Parsed ``manifest.json`` dict for this run. ``None`` (the
            default) is treated as "manifest unavailable" — ``code_context``
            falls back to the row scalars with ``base_sha=None`` and
            ``changed_files=[]``.

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

    code_ctx = (manifest or {}).get("code_context") or {}
    base_sha = code_ctx.get("base_sha")
    changed = code_ctx.get("changed_files") or []
    if not isinstance(changed, list):
        changed = []

    review_output_path = archive_path / "review-output.md"
    try:
        review_output: str | None = review_output_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        review_output = None

    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "session_id": manifest_row["session_id"],
        "repo_slug": manifest_row.get("repo_slug"),
        "skill": manifest_row.get("skill"),
        "stack": stack,
        "code_context": {
            "base_sha": base_sha,
            "head_sha": manifest_row.get("head_sha"),
            "base_branch": manifest_row.get("base_branch"),
            "branch": manifest_row.get("branch"),
            "changed_files": [str(p) for p in changed],
        },
        "review_output": review_output,
        "fix_diff_ref": fix_diff_ref,
        "test_outcome": manifest_row.get("test_outcome"),
        "outcome_label": labels[0] if labels else None,
        "grounding_score": manifest_row.get("grounding_rate"),
        "spans": _build_spans(trajectory),
        "trajectory_ref": {"archive_relative_path": "trajectory.json"},
    }


# ---------------------------------------------------------------------------
# Filter + query pipeline (Wave 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportFilters:
    """Post-exclusion filter knobs for ``_build_query`` / ``_query_index``.

    The C5 exclusion list is appended to every query unconditionally — it is
    not represented here because callers cannot disable it. C8 copyleft
    handling is opt-in via ``allow_copyleft``: any slug in the frozenset is
    re-admitted past the copyleft filter that ``_query_index`` applies
    post-query (the copyleft list itself lives in
    ``schema/copyleft.txt`` and is loaded by ``is_copyleft``).

    Attributes:
        skill: Optional exact-match filter on the ``skill`` column.
        repos: Optional whitelist of ``owner/repo`` slugs (IN-clause).
        labels: Outcome labels considered acceptable. Defaults to
            ``("accepted",)`` per C9. Ignored when ``include_all_labels``
            is ``True``.
        min_grounding: Optional minimum ``grounding_rate`` (inclusive).
        status: Exact-match filter on the ``status`` column. Defaults to
            ``"complete"`` so partial / failed runs never enter training.
        include_all_labels: When ``True``, suppresses the label IN-clause
            so every label value passes the filter.
        allow_copyleft: ``owner/repo`` slugs the caller has explicitly
            opted in via ``--allow-copyleft``; bypasses the C8 skip.
    """

    skill: str | None = None
    repos: tuple[str, ...] = ()
    labels: tuple[str, ...] = ("accepted",)
    min_grounding: float | None = None
    status: str = "complete"
    include_all_labels: bool = False
    allow_copyleft: frozenset[str] = frozenset()


# Inverse of REVIEW_SKILLS: full skill string → short stack label
# (e.g. "beagle-python:review-python" → "python"). Built at import time
# so ``_stack_for_skill`` is a single dict lookup.
_SKILL_TO_STACK: dict[str, str] = {skill: choice.name.lower() for choice, skill in REVIEW_SKILLS.items()}


def _stack_for_skill(skill: str | None) -> str | None:
    """Return the stack label (e.g. ``"python"``, ``"react"``) for a skill.

    Args:
        skill: The full skill string from a manifest row
            (e.g. ``"beagle-python:review-python"``) or ``None``.

    Returns:
        The lowercase stack label, or ``None`` when ``skill`` is ``None``
        or not present in ``REVIEW_SKILLS``.
    """
    if skill is None:
        return None
    return _SKILL_TO_STACK.get(skill)


def _build_query(filters: ExportFilters) -> tuple[str, tuple[Any, ...]]:
    """Assemble the WHERE clause + bound params for ``query_runs``.

    The returned ``where`` string is suitable for direct hand-off to
    ``daydream.archive.index.query_runs`` (which prepends ``WHERE`` when
    non-empty). ORDER BY is intentionally *not* emitted here — callers
    sort the result list in Python (see ``_query_index``).

    Clauses, in fixed order:

    1. ``status = ?`` — always.
    2. C5 exclusion: ``repo_slug IS NULL OR repo_slug NOT IN (...)`` — always
       (when the exclusion list is non-empty; otherwise omitted to keep the
       SQL clean). Placeholder count is derived from the loaded set size.
    3. ``skill = ?`` — when ``filters.skill`` is set.
    4. ``repo_slug IN (...)`` — when ``filters.repos`` is non-empty.
    5. ``grounding_rate >= ?`` — when ``filters.min_grounding`` is set.
    6. SQLite JSON1 label match against ``outcome_labels`` — unless
       ``filters.include_all_labels`` is ``True``.

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

    # 6. labels — unless include_all_labels
    if not filters.include_all_labels and filters.labels:
        placeholders = ", ".join(["?"] * len(filters.labels))
        # outcome_labels is a JSON-encoded string list; json_each unpacks it
        # so we can run a normal IN match against the raw values.
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(outcome_labels) WHERE value IN (" + placeholders + "))"
        )
        params.extend(filters.labels)

    where = " AND ".join(clauses)
    return where, tuple(params)


def _query_index(archive_dir: Path, filters: ExportFilters) -> list[dict[str, Any]]:
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

    survivors: list[dict[str, Any]] = []
    unknown_skills: set[str] = set()
    for row in rows:
        if is_copyleft(row.get("repo_slug"), filters.allow_copyleft):
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
    """Cap any single stack's share of the corpus at ``max_stack_share``.

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
class ExportConfig:
    """Top-level config for :func:`run_export`.

    ``filters`` is required (no default) — callers must construct an
    :class:`ExportFilters` explicitly so the C9 ``("accepted",)`` default is
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
    """

    out_path: Path
    filters: ExportFilters
    archive_dir: Path | None = None
    stratify_by: str | None = None
    max_stack_share: float = 0.6
    dry_run: bool = False
    emit_schema_only: bool = False


def run_export(config: ExportConfig) -> dict[str, int]:
    """Top-level entry point for the JSONL exporter.

    Pipeline (see plan §10 step 6):

    1. ``emit_schema_only`` short-circuit — copy ``schema/v1.json`` next to
       ``out_path`` and return.
    2. Resolve archive dir (``config.archive_dir`` or :func:`get_archive_dir`).
    3. Count the unfiltered index for the ``total_runs_in_index`` summary.
    4. Apply :func:`_query_index` → ``after_filters`` rows.
    5. For each row: load manifest + trajectory from disk, build a record via
       :func:`_build_record`. Skip rows with missing/unreadable files (with a
       warning).
    6. Optionally stratify (when ``stratify_by == "stack"``).
    7. Dry-run path prints a summary and returns ``emitted=0``.
    8. Otherwise: write JSONL atomically (tempfile in same dir +
       ``Path.replace``) and copy ``schema/v1.json`` alongside.

    Args:
        config: Resolved exporter config.

    Returns:
        Summary dict with keys ``total_runs_in_index``, ``after_filters``,
        ``after_stratify``, ``emitted``. ``emitted`` is ``0`` for ``dry_run``
        and ``emit_schema_only`` paths.
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
    total_in_index = len(query_runs(archive_dir, "", ()))

    # 4. Apply filters.
    rows = _query_index(archive_dir, config.filters)
    after_filters = len(rows)

    # 5. Build records, skipping rows whose on-disk files are missing.
    records: list[dict[str, Any]] = []
    for row in rows:
        archive_path = Path(row["archive_path"])
        traj_path = archive_path / "trajectory.json"
        manifest_path = archive_path / "manifest.json"
        if not traj_path.exists() or not manifest_path.exists():
            print_warning(
                console,
                f"Skipping {row['session_id']}: missing manifest.json or "
                f"trajectory.json at {archive_path}",
            )
            continue
        try:
            trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            print_warning(
                console,
                f"Skipping {row['session_id']}: missing manifest.json or "
                f"trajectory.json at {archive_path}",
            )
            continue
        records.append(_build_record(row, trajectory, row.get("stack"), manifest))

    # 6. Stratification (optional).
    if config.stratify_by == "stack":
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
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=config.out_path.parent,
        delete=False,
        suffix=".jsonl.tmp",
    ) as fh:
        tmp_name = fh.name
        for record in records:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    Path(tmp_name).replace(config.out_path)
    shutil.copyfile(SCHEMA_V1_PATH, config.out_path.parent / "schema.json")
    print_info(console, f"Wrote {len(records)} records to {config.out_path}")

    return {
        "total_runs_in_index": total_in_index,
        "after_filters": after_filters,
        "after_stratify": after_stratify,
        "emitted": len(records),
    }
