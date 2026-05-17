"""Deep-review artifact path helpers + predecessor-check guard.

All deep-mode artifacts live under `target / ".daydream" / "deep"` per D-41.
The final merged report writes to `target / REVIEW_OUTPUT_FILE` per D-24/D-42.

The check_deep_artifacts() helper mirrors check_review_file_exists()
(daydream/phases.py:611-629) -- same exception type, same actionable message format.
"""

from __future__ import annotations

from pathlib import Path

from daydream.config import REVIEW_OUTPUT_FILE

# Stage prerequisites -- single source of truth.
# Value is a list of file names (relative to deep_dir) that must exist before the
# given stage can run. Special handling for "merge" (needs at least one glob match)
# and "fix" (checks REVIEW_OUTPUT_FILE in target_dir, not deep_dir).
_DEEP_STAGE_PREREQS: dict[str, list[str]] = {
    "ttt": [],
    "per-stack": ["intent.md", "alternatives.json"],
    "merge": ["intent.md", "alternatives.json"],  # + at least one stack-*-records.json
    "fix": [],  # special-cased: needs REVIEW_OUTPUT_FILE in target_dir
}

# Which --start-at to suggest when a stage's prerequisites are missing.
_EARLIER_STAGE: dict[str, str] = {
    "per-stack": "ttt",
    "merge": "per-stack",
    "fix": "merge",
}


def deep_dir(target: Path) -> Path:
    """Return the `.daydream/deep/` directory for `target`, creating it if absent.

    Args:
        target: Resolved target directory path.

    Returns:
        Path to `target / ".daydream" / "deep"`.
    """
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


def dedup_candidates_path(deep_dir_path: Path) -> Path:
    """Dedup pre-filter candidate-pairs output (D-27)."""
    return deep_dir_path / "dedup-candidates.json"


def merged_report_path(deep_dir_path: Path) -> Path:
    """Merged review report inside the deep artifact directory.

    The merge agent writes here first (same directory as per-stack artifacts,
    which avoids sandbox write restrictions). The orchestrator then copies the
    result to ``target / REVIEW_OUTPUT_FILE`` for downstream consumers.
    """
    return deep_dir_path / "review-output.md"


def per_stack_failures_path(deep_dir_path: Path) -> Path:
    """Per-stack agent failure summary ({stack_name: reason} JSON).

    Persisted so a resume at `merge` can still surface uncovered stacks in the
    final report -- otherwise the failure info lives only in-memory inside the
    per-stack fan-out call.
    """
    return deep_dir_path / "per-stack-failures.json"


def verdicts_path(deep_dir_path: Path) -> Path:
    """Path to the recommendation-verifier verdicts artifact."""
    return deep_dir_path / "recommendation-verdicts.json"


def check_deep_artifacts(stage: str, deep_dir_path: Path) -> None:
    """Validate predecessor artifacts exist for the given resume stage.

    Args:
        stage: One of "ttt", "per-stack", "merge", "fix".
        deep_dir_path: Path to the `.daydream/deep/` directory.

    Raises:
        ValueError: If stage is not a known deep-mode stage.
        FileNotFoundError: With an actionable multi-line message naming missing
            files and the --start-at value that would produce them.
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

    # Fix stage needs the merged report. Check both the canonical location
    # (target_dir / .review-output.md) and the deep artifact copy
    # (deep_dir / review-output.md) so resume works even when the agent
    # could only write to the deep directory.
    if stage == "fix":
        target_dir = deep_dir_path.parent.parent
        canonical = target_dir / REVIEW_OUTPUT_FILE
        deep_copy = merged_report_path(deep_dir_path)
        if not canonical.is_file() and not deep_copy.is_file():
            missing.append(canonical)

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
