from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from daydream.benchmark.manifest import load_benchmark_manifest
from daydream.benchmark.storage import WorkspaceCorrupt
from daydream.benchmark.workspace import init_workspace
from tests.harness.table_cases import isolated_case


def test_manifest_loader_bounds_every_invalid_shape(tmp_path: Path, subtests: pytest.Subtests) -> None:
    case_root = tmp_path
    cases: Sequence[str] = [
        "cases: null\n",
        "cases: {}\n",
        "cases: 7\n",
        "cases:\n  - nope\n",
        "cases:\n  - case_id: broken\n",
        "x: 1\nx: 2\n",
        "x: !!python/object/apply:os.system ['false']\n",
        "",
        "- not\n- a\n- mapping\n",
    ]
    for case_index, payload in enumerate(cases):
        with subtests.test(payload=payload):
            with isolated_case(case_root, str(case_index)) as sandbox:
                tmp_path = sandbox.tmp_path
                (tmp_path / "benchmark.yaml").write_text(payload)
                with pytest.raises(WorkspaceCorrupt) as excinfo:
                    load_benchmark_manifest(tmp_path, canonicalize_case_order=True)
                assert str(excinfo.value) == f"{tmp_path}: invalid benchmark.yaml"


def test_manifest_loader_preserves_absent_cases_default(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["review.example"], ["judge.example"])
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    raw.pop("cases")
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    loaded = load_benchmark_manifest(root, canonicalize_case_order=True)

    assert loaded.cases == []

def test_manifest_loader_canonicalizes_copy_without_mutating_manifest(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["review.example"], ["judge.example"])
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    first = {"case_id": "pr-000002-" + "b" * 12, "case_file": "cases/b.yaml", "pr_number": 2}
    second = {"case_id": "pr-000001-" + "a" * 12, "case_file": "cases/a.yaml", "pr_number": 1}
    raw["cases"] = [first, second]
    manifest_path = root / "benchmark.yaml"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    original_bytes = manifest_path.read_bytes()

    with pytest.raises(WorkspaceCorrupt, match="invalid benchmark.yaml"):
        load_benchmark_manifest(root)

    loaded = load_benchmark_manifest(root, canonicalize_case_order=True)

    assert manifest_path.read_bytes() == original_bytes
    assert yaml.safe_load(manifest_path.read_text())["cases"] == [first, second]
    assert [row.case_id for row in loaded.cases] == [second["case_id"], first["case_id"]]

def test_manifest_loader_bounds_real_read_and_decode_errors(tmp_path: Path, subtests: pytest.Subtests) -> None:
    case_root = tmp_path
    cases: Sequence[str] = ["missing", "directory", "invalid-encoding", "unhashable-key"]
    for case_index, kind in enumerate(cases):
        with subtests.test(kind=kind):
            with isolated_case(case_root, str(case_index)) as sandbox:
                tmp_path = sandbox.tmp_path
                path = tmp_path / "benchmark.yaml"
                if kind == "directory":
                    path.mkdir()
                elif kind == "invalid-encoding":
                    path.write_bytes(b"\xff\xff\x00")
                elif kind == "unhashable-key":
                    path.write_text("? [one, two]\n: value\n")

                with pytest.raises(WorkspaceCorrupt) as excinfo:
                    load_benchmark_manifest(tmp_path)

                assert str(excinfo.value) == f"{tmp_path}: invalid benchmark.yaml"
                assert excinfo.value.__cause__ is not None
