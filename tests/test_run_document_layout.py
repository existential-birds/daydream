"""#1220: the run-document layout has exactly one declaration."""
from __future__ import annotations

from pathlib import Path

from daydream.trajectory import (
    PARTIAL_SUFFIX,
    RUN_DOCUMENT_NAME,
    RUNS_DIRNAME,
    SIBLINGS_DIRNAME,
    default_trajectory_path,
    partial_document_path,
    run_directory,
    run_document_path,
    sibling_document_path,
    siblings_directory,
)

SESSION = "11111111-2222-3333-4444-555555555555"


def test_the_five_capabilities_compose_one_shape_for_every_root(tmp_path: Path) -> None:
    live = tmp_path / "proj" / ".daydream"
    for root in (live, tmp_path / "public" / ".daydream", tmp_path / "archive", tmp_path / "index"):
        run_dir = run_directory(root, SESSION)
        assert run_dir == root / RUNS_DIRNAME / SESSION
        assert run_document_path(run_dir) == run_dir / RUN_DOCUMENT_NAME
        assert run_document_path(run_dir).name == "trajectory.json"
        assert siblings_directory(run_dir) == run_dir / SIBLINGS_DIRNAME
        assert sibling_document_path(run_dir, "deep-python.json") == (
            siblings_directory(run_dir) / "deep-python.json"
        )


def test_partial_variant_is_appended_without_mangling_dotted_names(tmp_path: Path) -> None:
    document = run_document_path(run_directory(tmp_path / ".daydream", SESSION))
    assert partial_document_path(document) == document.with_suffix(document.suffix + PARTIAL_SUFFIX)
    assert partial_document_path(document).name == "trajectory.json.partial"
    dotted = siblings_directory(document.parent) / "a.b.json"
    assert partial_document_path(dotted).name == "a.b.json.partial"


def test_default_trajectory_path_keeps_its_exact_output(tmp_path: Path) -> None:
    assert default_trajectory_path(tmp_path, SESSION) == run_document_path(
        run_directory(tmp_path / ".daydream", SESSION)
    )
