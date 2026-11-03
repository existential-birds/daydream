from pathlib import Path

import pytest
import yaml

from daydream.benchmark.manifest import load_benchmark_manifest
from daydream.benchmark.storage import WorkspaceCorrupt
from daydream.benchmark.workspace import init_workspace


def test_manifest_loader_bounds_every_invalid_shape_and_case_index(tmp_path: Path) -> None:
    """Missing root metadata must not hide acceptance of an invalid case index either."""
    payloads = ["x: 1\nx: 2\n", "x: !!python/object/apply:os.system ['false']\n", "", "- not\n- a\n- mapping\n",
                "? [one, two]\n: value\n"]
    for index, payload in enumerate(payloads):
        root = tmp_path / f"shape-{index}"
        root.mkdir()
        (root / "benchmark.yaml").write_text(payload)
        with pytest.raises(WorkspaceCorrupt) as excinfo:
            load_benchmark_manifest(root, canonicalize_case_order=True)
        assert str(excinfo.value) == f"{root}: invalid benchmark.yaml"
    assert excinfo.value.__cause__ is not None  # the unhashable-key decode error is chained
    for index, payload in enumerate(["cases: 7\n", "cases:\n  - nope\n"]):
        root = tmp_path / f"cases-{index}"
        init_workspace(root, "O/R", ["review.example"], ["judge.example"])
        path = root / "benchmark.yaml"
        raw = yaml.safe_load(path.read_text())
        raw.pop("cases")
        path.write_text(yaml.safe_dump(raw, sort_keys=False) + payload)
        with pytest.raises(WorkspaceCorrupt) as rejected:
            load_benchmark_manifest(root, canonicalize_case_order=True)
        assert str(rejected.value) == f"{root}: invalid benchmark.yaml"


def test_manifest_loader_canonicalizes_copy_without_mutating_manifest(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    init_workspace(root, "O/R", ["review.example"], ["judge.example"])
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    raw.pop("cases")
    manifest_path = root / "benchmark.yaml"
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert load_benchmark_manifest(root, canonicalize_case_order=True).cases == []
    first = {"case_id": "pr-000002-" + "b" * 12, "case_file": "cases/b.yaml", "pr_number": 2}
    second = {"case_id": "pr-000001-" + "a" * 12, "case_file": "cases/a.yaml", "pr_number": 1}
    raw["cases"] = [first, second]
    manifest_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    original_bytes = manifest_path.read_bytes()

    with pytest.raises(WorkspaceCorrupt, match="invalid benchmark.yaml"):
        load_benchmark_manifest(root)

    loaded = load_benchmark_manifest(root, canonicalize_case_order=True)

    assert manifest_path.read_bytes() == original_bytes
    assert yaml.safe_load(manifest_path.read_text())["cases"] == [first, second]
    assert [row.case_id for row in loaded.cases] == [second["case_id"], first["case_id"]]
