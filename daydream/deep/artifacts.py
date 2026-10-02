"""Session-private deep-review artifact paths and resume prerequisites.

Standalone callers explicitly use target/.daydream/deep; sessions publish the
merged markdown report to target/REVIEW_OUTPUT_FILE.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from daydream.artifact_visibility import ArtifactSession, artifact_dir_for
from daydream.json_utils import read_json_object

# Reserved entry; resume loaders must not interpret it as a stack name.
MERGE_FAILURE_KEY = "__merge__"

_DEEP_STAGE_PREREQS: dict[str, list[str]] = {
    "ttt": [],
    "per-stack": ["intent.md", "alternatives.json"],
    "merge": ["intent.md", "alternatives.json"],  # + at least one current reviewer output
    "fix": ["merged-items.json"],  # Markdown is only a derived view.
}

# Which --start-at to suggest when a stage's prerequisites are missing.
_EARLIER_STAGE: dict[str, str] = {
    "per-stack": "ttt",
    "merge": "per-stack",
    "fix": "merge",
}


def deep_dir(
    target: Path, *, session: ArtifactSession | None = None, allow_standalone: bool = False,
) -> Path:
    """Return the `.daydream/deep/` directory for `target`, creating it if absent."""
    d = artifact_dir_for(target, session=session, allow_standalone=allow_standalone) / "deep"
    d.mkdir(parents=True, exist_ok=True)
    return d


def intent_path(deep_dir_path: Path) -> Path:
    """Path to the TTT intent summary artifact (D-19 context bus)."""
    return deep_dir_path / "intent.md"


def alternatives_path(deep_dir_path: Path) -> Path:
    """Path to the TTT alternative-review findings artifact (D-19 context bus)."""
    return deep_dir_path / "alternatives.json"


def write_review_markdown(path: Path, issues: list[dict[str, Any]]) -> None:
    """Persist the host-owned review sidecar from authoritative structured issues."""
    path.write_text("# Review\n\n" + "\n".join(
        f"- {issue.get('file', '')}:{issue.get('line', '')} {issue.get('description', '')}"
        for issue in issues
    ))


def per_stack_review_path(deep_dir_path: Path, stack_name: str) -> Path:
    """Per-stack review markdown output (D-18 deterministic, unique per stack)."""
    return deep_dir_path / f"stack-{stack_name}-review.md"


def per_stack_records_path(deep_dir_path: Path, stack_name: str) -> Path:
    """Per-stack parsed-records JSON (output of pre-merge parse stage, D-21/D-22)."""
    return deep_dir_path / f"stack-{stack_name}-records.json"


def arbiter_input_path(deep_dir_path: Path) -> Path:
    """High-severity/contested records tagged with arb_id for the arbiter to echo."""
    return deep_dir_path / "arbiter-input.json"


def suppression_input_path(deep_dir_path: Path) -> Path:
    """Borderline records tagged with sup_id, audited separately from arbiter inputs."""
    return deep_dir_path / "suppression-input.json"


def adjudication_complete_path(deep_dir_path: Path) -> Path:
    """Marker proving final records were persisted after arbiter and suppression.

    Absence forces the whole adjudication block to rerun before merge. Group
    markers cannot replace it. Keep the legacy arbiter-complete.marker filename
    for resume compatibility.
    """
    return deep_dir_path / "arbiter-complete.marker"


def arbiter_group_input_path(deep_dir_path: Path, group_id: str) -> Path:
    """One sharded group's resume input; the unsharded input path remains separate."""
    return deep_dir_path / f"{group_id}-input.json"


def arbiter_group_verdicts_path(deep_dir_path: Path, group_id: str) -> Path:
    """One group's verdicts, merged into records after every group completes."""
    return deep_dir_path / f"{group_id}-verdicts.json"


def arbiter_group_complete_path(deep_dir_path: Path, group_id: str) -> Path:
    """Persisted group verdict marker; does not replace whole-block completion."""
    return deep_dir_path / f"{group_id}-complete.marker"


def dedup_candidates_path(deep_dir_path: Path) -> Path:
    """Dedup pre-filter candidate-pairs output (D-27)."""
    return deep_dir_path / "dedup-candidates.json"


def diagram_path(deep_dir_path: Path) -> Path:
    """Eligibility and per-kind proposed/final specs, grounding verdicts and mermaid.

    Written even when all kinds are skipped so the decision remains auditable.
    """
    return deep_dir_path / "diagram.json"


def diagram_markdown_path(deep_dir_path: Path) -> Path:
    """Folded blocks rendered from diagram.json; empty/absent when no kind rendered."""
    return deep_dir_path / "diagram.md"


def merged_report_path(deep_dir_path: Path) -> Path:
    """Markdown rendered from canonical merged-items.json and copied to the public report."""
    return deep_dir_path / "review-output.md"


def merged_items_path(deep_dir_path: Path) -> Path:
    """Canonical {items: [...]} findings for fixes, verification and PR posting.

    Schema-validated items carry lens and severity; markdown is a derived view.
    """
    return deep_dir_path / "merged-items.json"


def per_stack_failures_path(deep_dir_path: Path) -> Path:
    """Persisted {stack_name: reason} failures, retained across merge resumes."""
    return deep_dir_path / "per-stack-failures.json"


def fix_failures_path(deep_dir_path: Path) -> Path:
    """Persisted {file_group: reason} failures; the archive marks such runs partial."""
    return deep_dir_path / "fix-failures.json"


def fix_outcomes_path(deep_dir_path: Path) -> Path:
    """Full-canonical verification keyed by durable item UID and bound to session/evidence.

    Each round replaces the preceding envelope.
    """
    return deep_dir_path / "fix-outcomes.json"


def fix_footprint_path(deep_dir_path: Path) -> Path:
    """Session-bound authorization and enforcement audit for the fix cycle."""
    return deep_dir_path / "fix-footprint.json"


def stabilization_failed_path(deep_dir_path: Path) -> Path:
    """Terminal reason emitted when the bounded retained tree cannot stabilize."""
    return deep_dir_path / "stabilization-failed.json"


def recommended_capture_path(deep_dir_path: Path) -> Path:
    """Identify the tree producing recommended.patch: post_test, or pre_test fallback.

    The orchestrator writes the session-bound capture; the archive supplies fallback.
    """
    return deep_dir_path / "recommended-capture.json"


def fix_quality_gate_path(deep_dir_path: Path) -> Path:
    """Fail-open per-round quality deltas for edited files, consumed by the archive.

    Disabled gates write {enabled: false}; enabled gates include per_file results.
    """
    return deep_dir_path / "fix-quality-gate.json"


def generated_file_violations_path(deep_dir_path: Path) -> Path:
    """Generated-file edits rejected by the fix-phase runtime guard."""
    return deep_dir_path / "generated-file-violations.json"


def fix_leftover_untracked_path(deep_dir_path: Path) -> Path:
    """New surviving paths from a failed fix pass, written alongside fix failures.

    Parallel groups share a tree, so unattributable paths are preserved and audited.
    """
    return deep_dir_path / "fix-leftover-untracked.json"


def verdicts_path(deep_dir_path: Path) -> Path:
    """Path to the recommendation-verifier verdicts artifact."""
    return deep_dir_path / "recommendation-verdicts.json"


def adjudication_provenance_path(deep_dir_path: Path) -> Path:
    """Host-stamped targeting, verdict binding, survival and rewrite provenance.

    Verify selection reads this alongside the separate recommendation verdicts.
    """
    return deep_dir_path / "adjudication-provenance.json"


def test_verdict_path(deep_dir_path: Path) -> Path:
    """Persist {passed: bool, retries: int} for both successful and failed test runs.

    The verdict is inferred from agent prose, not independent proof of test success.
    """
    return deep_dir_path / "test-verdict.json"


def evidence_reuse_path(deep_dir_path: Path) -> Path:
    """Gate-keyed audit retaining both declined-commit and pre-push decisions.

    Contains identity facts only: no command, config digest, secret or prompt.
    """
    return deep_dir_path / "evidence-reuse.json"


def push_verdict_path(deep_dir_path: Path) -> Path:
    """Session-bound outcome of the current run's ordinary push attempt."""
    return deep_dir_path / "push-verdict.json"


def remote_ci_verdict_path(deep_dir_path: Path) -> Path:
    """Session- and pushed-SHA-bound remote CI state."""
    return deep_dir_path / "remote-ci-verdict.json"


def remote_ci_handoff_path(deep_dir_path: Path) -> Path:
    """Operator handoff for a failed or incomplete remote CI state."""
    return deep_dir_path / "remote-ci-handoff.json"


def latency_routing_path(deep_dir_path: Path) -> Path:
    """Append-only profile/risk/wonder/arbiter routing evidence; never a decision input."""
    return deep_dir_path / "latency-routing.json"


def diff_key_path(deep_dir_path: Path) -> Path:
    """Sibling file recording which diff the deep artifacts were produced from."""
    return deep_dir_path / "diff-key"


def diff_key(diff: str) -> str:
    """Content key for a diff: sha256 hex of its UTF-8 bytes."""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()


def check_deep_artifacts(
    stage: str, deep_dir_path: Path, *, current_diff_sha: str | None = None,
    record_paths: Sequence[Path] = (),
) -> None:
    """Require current predecessor files and, when supplied, a matching fresh diff key.

    Only record_paths from current assignments satisfy merge. Missing/mismatched
    keys or prerequisites older than the key reject resume with regeneration advice.
    Unknown stages raise ValueError; absent/stale artifacts raise FileNotFoundError.
    """
    if stage not in _DEEP_STAGE_PREREQS:
        raise ValueError(f"Unknown deep stage: {stage!r}")

    prerequisites = [deep_dir_path / name for name in _DEEP_STAGE_PREREQS[stage]]
    # is_file rejects directories shadowing prerequisite filenames.
    missing = [path for path in prerequisites if not path.is_file()]
    if stage == "merge":
        records = [path for path in record_paths if path.is_file()]
        if not records:
            missing.append(deep_dir_path / "stack-*-records.json")
        prerequisites.extend(records)

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


def _load_failures(path: Path) -> dict[str, Any]:
    """Load failures verbatim, including the reserved merge entry.

    Missing files, malformed JSON and non-dict roots yield {}; other I/O errors propagate.
    """
    return read_json_object(path)
