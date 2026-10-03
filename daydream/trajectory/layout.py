"""Canonical paths shared by live, archived, and hydrated trajectories."""

from pathlib import Path

# Run-document layout: this module is the sole declaration of
# <root>/runs/<session_id>/trajectory.json and the sibling <root>/runs/<session_id>/
# trajectories/<descriptor>.json set. The same shape is carried by the live
# <target>/.daydream root, the public live root, the archive root and the hydrated
# index root; every reader composes it through the helpers below rather than
# retyping the names. The archive *bundle* layout (manifest, diff, deep/, handoff)
# is deliberately not unified with this shape beyond the shared names.
RUNS_DIRNAME = "runs"


RUN_DOCUMENT_NAME = "trajectory.json"


SIBLINGS_DIRNAME = "trajectories"


PARTIAL_SUFFIX = ".partial"


_DAYDREAM_DIRNAME = ".daydream"


def run_directory(root: Path, session_id: str) -> Path:
    """Compose runs/<session_id> under any layout root; callers validate identity."""
    return root / RUNS_DIRNAME / session_id


def run_document_path(run_dir: Path) -> Path:
    """Compose the root document path from an existing run directory; no validation."""
    return run_dir / RUN_DOCUMENT_NAME


def siblings_directory(run_dir: Path) -> Path:
    """Return the directory holding a run's sibling trajectory documents."""
    return run_dir / SIBLINGS_DIRNAME


def sibling_document_path(run_dir: Path, name: str) -> Path:
    """Compose a sibling path from its complete filename; no identity validation."""
    return siblings_directory(run_dir) / name


def partial_document_path(document: Path) -> Path:
    """Append .partial after the existing suffix, preserving dotted filenames."""
    return document.with_suffix(document.suffix + PARTIAL_SUFFIX)


def default_trajectory_path(target_dir: Path, session_id: str) -> Path:
    """Compose <target>/.daydream/runs/<session_id>/trajectory.json."""
    return run_document_path(run_directory(target_dir / _DAYDREAM_DIRNAME, session_id))
