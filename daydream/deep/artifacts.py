"""Session-private deep-review artifact paths and resume prerequisites.

Standalone callers explicitly use target/.daydream/deep; sessions publish the
merged markdown report to target/REVIEW_OUTPUT_FILE.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import ValidationError

from daydream.artifact_visibility import ArtifactSession, artifact_dir_for
from daydream.diagnostics import exception_text
from daydream.json_utils import atomic_write_json

if TYPE_CHECKING:
    from daydream.deep.state import DeepState
    from daydream.review_result import ReasonCode, ReviewCoverage


class DeepArtifact(StrEnum):
    """Fixed session-private files; dynamic stack and group paths remain separate."""

    # Intent and alternatives form the review context bus.
    INTENT = "intent.md"
    ALTERNATIVES = "alternatives.json"
    # Whole adjudication completion requires persisted final records.
    ARBITER_INPUT = "arbiter-input.json"
    SUPPRESSION_INPUT = "suppression-input.json"
    ADJUDICATION_COMPLETE = "adjudication-complete.marker"
    DEDUP_CANDIDATES = "dedup-candidates.json"
    # Diagram decisions are auditable even when no kind renders.
    DIAGRAM = "diagram.json"
    DIAGRAM_MARKDOWN = "diagram.md"
    # Canonical merged JSON drives fixes and posting; markdown is a derived view.
    MERGED_REPORT = "review-output.md"
    MERGED_ITEMS = "merged-items.json"
    # Resume evidence binds positive scope coverage to the analyzed revision.
    REVIEW_COVERAGE = "review-coverage.json"
    FIX_FAILURES = "fix-failures.json"
    # Fix outcomes and authorization audit are session/evidence bound.
    FIX_OUTCOMES = "fix-outcomes.json"
    FIX_FOOTPRINT = "fix-footprint.json"
    STABILIZATION_FAILED = "stabilization-failed.json"
    # Recommended capture identifies the post-test or pre-test patch tree.
    RECOMMENDED_CAPTURE = "recommended-capture.json"
    FIX_QUALITY_GATE = "fix-quality-gate.json"
    GENERATED_FILE_VIOLATIONS = "generated-file-violations.json"
    FIX_LEFTOVER_UNTRACKED = "fix-leftover-untracked.json"
    VERDICTS = "recommendation-verdicts.json"
    ADJUDICATION_PROVENANCE = "adjudication-provenance.json"
    # Agent-inferred test verdicts are separate from pushed-SHA-bound remote CI.
    TEST_VERDICT = "test-verdict.json"
    EVIDENCE_REUSE = "evidence-reuse.json"
    PUSH_VERDICT = "push-verdict.json"
    REMOTE_CI_VERDICT = "remote-ci-verdict.json"
    REMOTE_CI_HANDOFF = "remote-ci-handoff.json"
    # Routing evidence is append-only and never a decision input.
    LATENCY_ROUTING = "latency-routing.json"
    DIFF_KEY = "diff-key"

    def at(self, root: Path) -> Path:
        return root / self.value


_DEEP_STAGE_PREREQS: dict[str, list[DeepArtifact]] = {
    "ttt": [],
    "per-stack": [DeepArtifact.INTENT, DeepArtifact.ALTERNATIVES],
    "merge": [DeepArtifact.INTENT, DeepArtifact.ALTERNATIVES],  # + at least one current reviewer output
    "fix": [DeepArtifact.MERGED_ITEMS],  # Markdown is only a derived view.
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


def arbiter_group_input_path(deep_dir_path: Path, group_id: str) -> Path:
    """One sharded group's resume input; the unsharded input path remains separate."""
    return deep_dir_path / f"{group_id}-input.json"


def arbiter_group_verdicts_path(deep_dir_path: Path, group_id: str) -> Path:
    """One group's verdicts, merged into records after every group completes."""
    return deep_dir_path / f"{group_id}-verdicts.json"


def arbiter_group_complete_path(deep_dir_path: Path, group_id: str) -> Path:
    """Persisted group verdict marker; does not replace whole-block completion."""
    return deep_dir_path / f"{group_id}-complete.marker"


def persist_review_coverage(deep_dir_path: Path, coverage: ReviewCoverage, *, error: Exception | None = None) -> None:
    """Persist checked evidence atomically, retaining any earlier failure as primary."""
    from daydream.review_result import ReviewCoverage

    try:
        payload = coverage.to_dict()
        ReviewCoverage.from_dict(payload)
        atomic_write_json(DeepArtifact.REVIEW_COVERAGE.at(deep_dir_path), payload)
    except Exception as exc:
        if error is None:
            raise
        error.add_note(f"Review coverage persistence failed: {type(exc).__name__}")


@contextmanager
def review_stage(state: DeepState, phase: str | Callable[[], str], *, persist: bool = False,
                 reasons: tuple[ReasonCode, ...] = ()) -> Iterator[None]:
    """Record a stage failure without replacing it with a secondary evidence-write error."""
    from daydream.review_result import reason_for_exception

    coverage = state.review_coverage
    if isinstance(phase, str):
        coverage.require_phase(phase)
    try:
        yield
    except Exception as exc:
        coverage.record_phase(phase if isinstance(phase, str) else phase(), "failed",
                              reasons=(*reasons, reason_for_exception(exc)),
                              diagnostic=f"{type(exc).__name__}: {exception_text(exc) or '(unavailable)'}")
        if persist:
            persist_review_coverage(state.dd, coverage, error=exc)
        raise
    else:
        if persist:
            persist_review_coverage(state.dd, coverage)


def restore_review_coverage(deep_dir_path: Path, current: ReviewCoverage) -> ReviewCoverage:
    """Restore matching evidence into this run; corrupt or foreign evidence fails closed."""
    from daydream.review_result import ReviewCoverage

    restart = "Re-run without --start-at to restart review and regenerate coverage evidence."
    try:
        stored = ReviewCoverage.from_dict(json.loads(DeepArtifact.REVIEW_COVERAGE.at(deep_dir_path).read_text()))
    except (OSError, ValueError, TypeError, KeyError, ValidationError) as exc:
        raise ValueError(f"Cannot resume review: missing or corrupt versioned coverage evidence. {restart}") from exc
    stored_inventory = stored.to_dict()["planned_scopes"]
    current_inventory = current.to_dict()["planned_scopes"]
    if stored.revision != current.revision or stored_inventory != current_inventory:
        raise ValueError(f"Cannot resume review: analyzed revision or planned scope inventory changed. {restart}")
    extra_phases = stored.required_phases - current.required_phases
    extra_phases = {phase for phase in extra_phases
                    if phase not in {"arbiter", "suppression", "findings", "pipeline"}
                    and re.fullmatch(r"arbiter-group-\d+", phase) is None}
    if not current.required_phases <= stored.required_phases or extra_phases:
        raise ValueError(f"Cannot resume review: required review stages changed. {restart}")
    payload = stored.to_dict()
    payload["run_id"] = current.run_id
    return ReviewCoverage.from_dict(payload)


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

    prerequisites = [artifact.at(deep_dir_path) for artifact in _DEEP_STAGE_PREREQS[stage]]
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
        key_file = DeepArtifact.DIFF_KEY.at(deep_dir_path)
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
