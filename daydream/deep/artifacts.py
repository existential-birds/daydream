"""Deep-review artifact path helpers + predecessor-check guard.

All deep-mode artifacts live under `target / ".daydream" / "deep"` per D-41.
The final merged report writes to `target / REVIEW_OUTPUT_FILE` per D-24/D-42.

The check_deep_artifacts() helper mirrors check_review_file_exists()
(daydream/phases.py:611-629) -- same exception type, same actionable message format.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Stage prerequisites -- single source of truth.
# Value is a list of file names (relative to deep_dir) that must exist before the
# given stage can run. Special handling for "merge" (needs at least one glob match)
# and "fix" (checks merged-items.json in deep_dir -- the canonical source of truth).
_DEEP_STAGE_PREREQS: dict[str, list[str]] = {
    "ttt": [],
    "per-stack": ["intent.md", "alternatives.json"],
    "merge": ["intent.md", "alternatives.json"],  # + at least one stack-*-records.json
    "fix": [],  # special-cased: needs merged-items.json in deep_dir
}

# Which --start-at to suggest when a stage's prerequisites are missing.
_EARLIER_STAGE: dict[str, str] = {
    "per-stack": "ttt",
    "merge": "per-stack",
    "fix": "merge",
}


def deep_dir(target: Path) -> Path:
    """Return the `.daydream/deep/` directory for `target`, creating it if absent."""
    d = target / ".daydream" / "deep"
    d.mkdir(parents=True, exist_ok=True)
    return d


def intent_path(deep_dir_path: Path) -> Path:
    """Path to the TTT intent summary artifact (D-19 context bus)."""
    return deep_dir_path / "intent.md"


def alternatives_path(deep_dir_path: Path) -> Path:
    """Path to the TTT alternative-review findings artifact (D-19 context bus)."""
    return deep_dir_path / "alternatives.json"


def per_stack_review_path(deep_dir_path: Path, stack_name: str) -> Path:
    """Per-stack review markdown output (D-18 deterministic, unique per stack)."""
    return deep_dir_path / f"stack-{stack_name}-review.md"


def per_stack_records_path(deep_dir_path: Path, stack_name: str) -> Path:
    """Per-stack parsed-records JSON (output of pre-merge parse stage, D-21/D-22)."""
    return deep_dir_path / f"stack-{stack_name}-records.json"


def arbiter_input_path(deep_dir_path: Path) -> Path:
    """Scoped-arbiter input findings JSON (issue #168).

    The high-severity / contested per-stack records selected for the Opus
    arbiter, each tagged with an ``arb_id`` the arbiter echoes back.
    """
    return deep_dir_path / "arbiter-input.json"


def suppression_input_path(deep_dir_path: Path) -> Path:
    """Precision-mode suppression input findings JSON (issue #232).

    The borderline (LOW-confidence / low-severity uncontested) per-stack records
    selected for the skeptical suppression pass, each tagged with a ``sup_id`` the
    suppression agent echoes back. Distinct from ``arbiter-input.json`` so a run's
    arbiter and suppression inputs are separately auditable.
    """
    return deep_dir_path / "suppression-input.json"


def adjudication_complete_path(deep_dir_path: Path) -> Path:
    """Marker proving the WHOLE adjudication block finalised the per-stack records.

    Covers BOTH adjudication passes that rewrite ``stack-*-records.json`` before
    the cross-stack merge: the scoped arbiter (#168) AND, when precision mode is
    on, the suppression pass (#232). Written only once the on-disk records are
    known-final for a fresh run -- after ``_rewrite_stack_records`` persists the
    final pass's verdicts, or when nothing qualified for either pass. Its presence
    lets a ``--start-at merge`` resume trust the records; its absence forces the
    whole block to re-run from disk so an interrupted arbiter OR suppression pass
    cannot leak partly-adjudicated findings into the merge.

    The on-disk filename is kept as ``arbiter-complete.marker`` (not renamed when
    suppression was added) so an in-flight run dir from before that change still
    satisfies ``--start-at merge`` resume; do not rename the file without a
    migration. The symbol was renamed from ``arbiter_complete_path`` so the scope
    it actually proves (both passes) is not silently under-read as arbiter-only.
    """
    return deep_dir_path / "arbiter-complete.marker"


def dedup_candidates_path(deep_dir_path: Path) -> Path:
    """Dedup pre-filter candidate-pairs output (D-27)."""
    return deep_dir_path / "dedup-candidates.json"


def merged_report_path(deep_dir_path: Path) -> Path:
    """Rendered human review report inside the deep artifact directory.

    ``phase_cross_stack_merge`` renders this markdown *from* the canonical
    ``merged-items.json`` (the merge agent no longer emits markdown) into the
    deep dir -- which avoids sandbox write restrictions on repo-root dotfiles --
    then copies it to ``target / REVIEW_OUTPUT_FILE`` for downstream consumers.
    """
    return deep_dir_path / "review-output.md"


def merged_items_path(deep_dir_path: Path) -> Path:
    """Canonical merged finding items (JSON) inside the deep artifact directory.

    This is the single source of truth produced by the cross-stack merge: a
    schema-validated item list (``{"items": [...]}``) carrying per-stack,
    cross-stack, and structural findings, each tagged with ``lens`` and
    ``severity``. The human ``review-output.md`` is rendered *from* this file;
    downstream consumers (fix gate, PR posting, verifier) read it rather than
    re-parsing prose.
    """
    return deep_dir_path / "merged-items.json"


def per_stack_failures_path(deep_dir_path: Path) -> Path:
    """Per-stack agent failure summary ({stack_name: reason} JSON).

    Persisted so a resume at `merge` can still surface uncovered stacks in the
    final report -- otherwise the failure info lives only in-memory inside the
    per-stack fan-out call.
    """
    return deep_dir_path / "per-stack-failures.json"


def fix_failures_path(deep_dir_path: Path) -> Path:
    """Fix-phase agent failure summary ({file_group: reason} JSON).

    Persisted whenever ``phase_fix_parallel`` drops one or more file-groups so a
    user inspecting the run -- or the archive manifest builder -- can see that
    fixes were left unapplied. Mirrors :func:`per_stack_failures_path`; the
    archive reads this file to mark the run ``partial`` instead of ``complete``.
    """
    return deep_dir_path / "fix-failures.json"


def fix_outcomes_path(deep_dir_path: Path) -> Path:
    """Post-fix verifier outcomes ({finding_id: verdict} JSON, issue #744).

    Sidecar adjacent to :func:`fix_failures_path`: every finding the fix phase
    dispatched has exactly one recorded terminal verdict here (``resolved`` /
    ``unresolved`` / ``wrong_target`` / ``regressed``), so an attempted-but-
    unconfirmed finding cannot silently pass as fixed. Accumulates across
    rounds; deleted when empty.
    """
    return deep_dir_path / "fix-outcomes.json"


def recommended_capture_path(deep_dir_path: Path) -> Path:
    """Capture-point sidecar recording which tree produced recommended.patch.

    Mirrors :func:`fix_quality_gate_path`: written session-bound by the deep
    orchestrator's post-test re-capture (``"post_test"``) or, when absent,
    defaulted to ``"pre_test"`` by the archive manifest builder. The archive
    reads this file to record which capture produced the archived patch.
    """
    return deep_dir_path / "recommended-capture.json"


def fix_quality_gate_path(deep_dir_path: Path) -> Path:
    """Fix-phase anti-degradation quality gate verdict (issue #315).

    Per-round before/after erosion + verbosity deltas over the files the fix
    phase edited, computed from :func:`daydream.eval.analyzer.analyze_quality`.
    Fail-open: written whenever the gate runs (``{"enabled": false}`` when
    disabled, a per-round ``per_file`` map when enabled), never aborting the
    run. The archive manifest reads this file to surface flagged files.
    """
    return deep_dir_path / "fix-quality-gate.json"


def generated_file_violations_path(deep_dir_path: Path) -> Path:
    """Generated-file edits rejected by the fix-phase runtime guard."""
    return deep_dir_path / "generated-file-violations.json"


def fix_leftover_untracked_path(deep_dir_path: Path) -> Path:
    """Untracked paths that newly appeared during a failed fix pass (JSON list).

    Parallel fix groups share one working tree, so an untracked file left behind
    cannot be attributed to a specific group. Rather than risk deleting a
    successful group's legitimate new file, the orchestrator records every path
    that appeared during the fix pass and survived tree-protection here, so the
    partial run is fully auditable. Written only alongside ``fix-failures.json``.
    """
    return deep_dir_path / "fix-leftover-untracked.json"


def verdicts_path(deep_dir_path: Path) -> Path:
    """Path to the recommendation-verifier verdicts artifact."""
    return deep_dir_path / "recommendation-verdicts.json"


def test_verdict_path(deep_dir_path: Path) -> Path:
    """Post-fix test-suite verdict (``{"passed": bool, "retries": int}``).

    The verdict itself comes from ``detect_test_success``, a regex over the
    agent's prose, so it is a *claim* rather than ground truth. It is persisted
    anyway because otherwise the only durable trace of the test phase is the
    process exit code, which conflates "tests failed" with every other fatal
    exit. Written for BOTH outcomes -- a run that stops at a red suite is exactly
    the run whose verdict a consumer needs.
    """
    return deep_dir_path / "test-verdict.json"


def diff_key_path(deep_dir_path: Path) -> Path:
    """Sibling file recording which diff the deep artifacts were produced from."""
    return deep_dir_path / "diff-key"


def diff_key(diff: str) -> str:
    """Content key for a diff: sha256 hex of its UTF-8 bytes."""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def check_deep_artifacts(
    stage: str, deep_dir_path: Path, *, current_diff_sha: str | None = None
) -> None:
    """Validate predecessor artifacts exist, and are fresh, for a resume stage.

    Args:
        stage: The ``--start-at`` stage being resumed into.
        deep_dir_path: The run's ``.daydream/deep`` directory.
        current_diff_sha: When given, the artifacts must have been produced from
            this diff. A missing key file (a pre-upgrade artifact directory) or a
            mismatched one refuses the resume: an unverifiable artifact set is
            treated as stale, because resuming onto artifacts from a different
            diff silently reviews the wrong code. Safety over back-compat — the
            message says how to regenerate.

    Raises:
        ValueError: If stage is not a known deep-mode stage.
        FileNotFoundError: With an actionable multi-line message naming missing
            files and the --start-at value that would produce them, or naming the
            staleness when the diff key does not match.
    """
    if stage not in _DEEP_STAGE_PREREQS:
        raise ValueError(f"Unknown deep stage: {stage!r}")

    missing: list[Path] = []

    # Regular file prerequisites.
    # Use is_file() (not exists()) so a directory sharing the prereq name doesn't
    # pass the gate and fail later in less actionable places.
    for name in _DEEP_STAGE_PREREQS[stage]:
        p = deep_dir_path / name
        if not p.is_file():
            missing.append(p)

    # Merge stage additionally needs at least one stack-*-records.json.
    if stage == "merge":
        records = [p for p in deep_dir_path.glob("stack-*-records.json") if p.is_file()]
        if not records:
            missing.append(deep_dir_path / "stack-*-records.json")

    # Fix stage needs the canonical merged items (merged-items.json) -- the
    # single source of truth the fix gate reads. The markdown review-output.md
    # is render-only (the fix gate, verifier, and PR posting all read the JSON),
    # so its absence must NOT block a --start-at fix resume when the JSON is
    # present. Only the JSON's absence is fatal here.
    if stage == "fix":
        items_file = merged_items_path(deep_dir_path)
        if not items_file.is_file():
            missing.append(items_file)

    if missing:
        expected_block = "\n".join(f"  - {p}" for p in missing)
        earlier = _EARLIER_STAGE.get(stage, "ttt")
        msg = (
            f"Cannot resume at stage '{stage}' -- missing artifacts:\n\n"
            f"{expected_block}\n\n"
            f"Re-run from an earlier stage:\n"
            f"  daydream --start-at {earlier}"
        )
        raise FileNotFoundError(msg)

    if current_diff_sha is not None:
        key_file = diff_key_path(deep_dir_path)
        try:
            stored = key_file.read_text(encoding="utf-8").strip()
        except OSError:
            stored = ""
        prerequisites = [deep_dir_path / name for name in _DEEP_STAGE_PREREQS[stage]]
        if stage == "merge":
            prerequisites.extend(records)
        if stage == "fix":
            prerequisites.append(merged_items_path(deep_dir_path))

        try:
            key_mtime = key_file.stat().st_mtime_ns
            has_stale_prerequisite = any(
                artifact.stat().st_mtime_ns < key_mtime for artifact in prerequisites
            )
        except OSError:
            has_stale_prerequisite = True

        if stored != current_diff_sha or has_stale_prerequisite:
            detail = (
                f"  - {key_file} is missing (produced before diff tracking)"
                if not stored
                else (
                    f"  - prerequisite artifacts predate {key_file}"
                    if has_stale_prerequisite
                    else f"  - {key_file} records a different diff"
                )
            )
            raise FileNotFoundError(
                f"Cannot resume at stage '{stage}' -- the artifacts in\n"
                f"  {deep_dir_path}\n"
                f"were produced from a different diff than the current one:\n\n"
                f"{detail}\n\n"
                f"Resuming would review stale findings against changed code.\n"
                f"Re-run without --start-at to regenerate them."
            )
