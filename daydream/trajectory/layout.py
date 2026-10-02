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
    """Return the per-run directory under the layout *root*.

    *root* is any layout root (the live ``<target>/.daydream``, the public live
    root, the archive root, or the hydrated index root); this helper composes no
    ``.daydream`` segment and validates no session identity -- callers own both.
    """
    return root / RUNS_DIRNAME / session_id


def run_document_path(run_dir: Path) -> Path:
    """Return the run's root trajectory document within *run_dir*.

    Takes the run directory, not ``(root, session_id)``: three callers hold one
    and no root. It performs no existence or identity validation.
    """
    return run_dir / RUN_DOCUMENT_NAME


def siblings_directory(run_dir: Path) -> Path:
    """Return the directory holding a run's sibling trajectory documents."""
    return run_dir / SIBLINGS_DIRNAME


def sibling_document_path(run_dir: Path, name: str) -> Path:
    """Return a named sibling document path within *run_dir*.

    *name* is the full file name (e.g. ``"deep-python.json"``); the helper adds
    the siblings directory and nothing else. It validates no duplicate or
    descriptor identity.
    """
    return siblings_directory(run_dir) / name


def partial_document_path(document: Path) -> Path:
    """Return the partial-write variant of *document*.

    The ``.partial`` suffix is appended after the existing suffix, so dotted names
    are preserved (``a.b.json`` -> ``a.b.json.partial``). The helper only composes
    the path; the caller owns deciding when a document is partial.
    """
    return document.with_suffix(document.suffix + PARTIAL_SUFFIX)


def default_trajectory_path(target_dir: Path, session_id: str) -> Path:
    """Return the default trajectory path under ``<target>/.daydream/runs/<session_id>/``.

    The session_id segment guarantees uniqueness per run; the recorder
    creates the directory before its first write.
    """
    return run_document_path(run_directory(target_dir / _DAYDREAM_DIRNAME, session_id))
