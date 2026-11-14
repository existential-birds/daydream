"""Session-private deep-review artifact paths.

Standalone callers explicitly use target/.daydream/deep; sessions publish the
merged markdown report to target/REVIEW_OUTPUT_FILE.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

from daydream.artifact_visibility import ArtifactSession, artifact_dir_for


class DeepArtifact(StrEnum):
    """Fixed session-private files."""

    # Diagram decisions are auditable even when no kind renders.
    DIAGRAM = "diagram.json"
    DIAGRAM_MARKDOWN = "diagram.md"
    # Canonical merged JSON drives fixes and posting; markdown is a derived view.
    MERGED_REPORT = "review-output.md"
    MERGED_ITEMS = "merged-items.json"
    # Coverage binds positive scope evidence to the analyzed revision.
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
    # Agent-inferred test verdicts are separate from pushed-SHA-bound remote CI.
    TEST_VERDICT = "test-verdict.json"
    EVIDENCE_REUSE = "evidence-reuse.json"
    PUSH_VERDICT = "push-verdict.json"
    REMOTE_CI_VERDICT = "remote-ci-verdict.json"
    REMOTE_CI_HANDOFF = "remote-ci-handoff.json"
    # Repair-job state and its per-turn checkpoint survive the process; the
    # checkpoint carries captured work, the record carries the job's own state.
    REPAIR_CHECKPOINT = "repair-checkpoint.json"
    REPAIR_JOB = "repair-job.json"
    # Written by earlier builds only; still read for the archive evaluation.
    LATENCY_ROUTING = "latency-routing.json"
    DIFF_KEY = "diff-key"

    def at(self, root: Path) -> Path:
        return root / self.value


def deep_dir(
    target: Path, *, session: ArtifactSession | None = None, allow_standalone: bool = False,
) -> Path:
    """Return the `.daydream/deep/` directory for `target`, creating it if absent."""
    d = artifact_dir_for(target, session=session, allow_standalone=allow_standalone) / "deep"
    d.mkdir(parents=True, exist_ok=True)
    return d


def diff_key(diff: str) -> str:
    """Content key for a diff: sha256 hex of its UTF-8 bytes."""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()
