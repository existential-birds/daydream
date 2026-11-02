import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream import cli as top_cli
from daydream.benchmark import curation, snapshot as sn
from daydream.benchmark.harbor.build import task_spec_digest
from daydream.benchmark.schema import (
    CaseDocument,
    CaseSource,
    Curation,
    ImportDocument,
    PullRequestEntry,
    PullRequestMeta,
    SnapshotReady,
    _EvidenceAuthor,
    _FetchInfo,
    _ImportRepository,
    _PrRef,
    derive_finding_id,
)
from daydream.benchmark.storage import load_yaml_strict
from daydream.benchmark.workspace import InitError, init_workspace, validate_workspace, workspace_status
from tests.harness.git_helpers import seed_pr_origin


def test_init_refuses_existing_nonempty_dir_and_empty_reviewer_hosts(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "existing.txt").write_text("x")
    with pytest.raises(InitError):
        init_workspace(root, "O/R", ["h1.example.com"], ["h2.example.com"])
    with pytest.raises(InitError):
        init_workspace(tmp_path / "ws", "O/R", [], ["h2.example.com"])


def _seed_local_origin(root: Path) -> tuple[str, str, str]:
    """Real local bare origin: main base1->base2->base3 + feature head off base2.

    The feature head adds ``feature.py`` — the ready fixture's finding
    location — so the frozen head tree contains the location the curated
    finding references. Returns ``(origin_url, base_sha, head_sha)``.
    """
    return seed_pr_origin(root.parent, feature_body="LINE 1\n")


def _assert_corrupt_workspace(root: Path) -> None:
    """Assert ``validate_workspace`` reports a corrupt workspace."""
    code, label = validate_workspace(root)
    assert code == 1 and "corrupt" in label.lower()


def _write_case_docs(root: Path, curation_state: str) -> Any:
    """Add a valid ledger, import, case, and real bundle to an initialized workspace.

    Original SHAs, tree IDs, and digests all describe the same local Git origin;
    case documents validate directly without compatibility preprocessing.
    """

    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    raw["source"]["repository_id"] = "R_kgDOABC123"
    raw["source"]["visibility"] = "private"
    repo_slug = raw["source"]["repository"]

    # The PR and snapshot metadata name the real source commits so the local
    # mirror provenance check and the portable bundle fidelity check agree.
    origin_url, base_sha, head_sha = _seed_local_origin(root)
    case_id = f"pr-000101-{head_sha[:12]}"
    case_file = f"cases/{case_id}.yaml"
    import_file = "imports/pr-000101.json"
    bundle_rel = f"snapshots/{case_id}.bundle"

    sn.ensure_mirror(root)
    sn.fetch_base_tip(root, repo_slug, base_sha, origin_url)
    sn.fetch_head_refs(root, repo_slug, 101, [head_sha], origin_url)
    m = sn.mirror(root)
    bundle_path = root / bundle_rel
    sn.build_bundle(m, base_sha, head_sha, bundle_path)
    base_tree_sha = sn.rev_parse(m, f"{base_sha}^{{tree}}")
    head_tree_sha = sn.rev_parse(m, f"{head_sha}^{{tree}}")
    bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    pr_meta = PullRequestMeta(
        number=101, url=f"https://github.com/{repo_slug}/pull/101", title="Fix cache", state="open",
        base=_PrRef(ref="main", sha=base_sha), head=_PrRef(ref="feature/cache", sha=head_sha),
        created_at=datetime.fromisoformat("2026-01-01T00:00:00Z"),
        updated_at=datetime.fromisoformat("2026-01-01T00:00:00Z"), author=_EvidenceAuthor(login="alice", type="User"),
        body="Fix the cache invalidation on write.",
    )
    import_doc = ImportDocument(schema_version=1,
        repository=_ImportRepository(id="R_kgDOABC123", name_with_owner=repo_slug, visibility="private"),
        pull_request=pr_meta, evidence=[],
        fetch=_FetchInfo(fetched_at="2026-01-01T00:00:00Z", etag=None, payload_sha256="0" * 64),
    )
    import_bytes = import_doc.model_dump_json(indent=2).encode("utf-8")
    import_sha256 = hashlib.sha256(import_bytes).hexdigest()

    curation: dict[str, Any]
    if curation_state == "ready":
        finding = {"finding_id": "f" * 64, "title": "Cache is never invalidated",
            "body": "The cache key is stable across writes, so stale data is served.", "severity": "high",
            "location": {"path": "feature.py", "start_line": 1, "end_line": 1},
            "provenance": {"kind": "authored", "source_ids": []},
        }
        finding["finding_id"] = derive_finding_id(finding, case_id=case_id)
        curation = {"state": "ready", "snapshot_attested": True, "clean_attested": False, "gold_status": "findings",
            "findings": [finding], "exclusions": [], "case_exclusion": None, "task_spec_sha256": "d" * 64,
        }
    else:  # draft
        curation = {
            "state": "draft", "snapshot_attested": False, "clean_attested": False, "gold_status": None, "findings": [],
            "exclusions": [], "case_exclusion": None,
        }

    case_doc = CaseDocument(schema_version=2, case_id=case_id, pull_request=pr_meta,
        snapshot=SnapshotReady(
            status="ready", base_resolution="merge_base_v1", policy="final_pr_head", requested_head="final",
            original_base_sha=base_sha, requested_base_sha=base_sha, original_head_sha=head_sha,
            base_tree_sha=base_tree_sha, head_tree_sha=head_tree_sha, bundle_file=bundle_rel,
            bundle_sha256=bundle_sha256, error=None,
        ), source=CaseSource(import_file=import_file, import_sha256=import_sha256),
        curation=Curation.model_validate(curation), candidates=[],
    )

    raw["pull_requests"] = [PullRequestEntry(
            number=101, import_state="fetched", import_file=import_file, import_sha256=import_sha256,
            case_ids=[case_id],
        ).model_dump(mode="json")
    ]
    raw["cases"] = [{"case_id": case_id, "pr_number": 101, "case_file": case_file}]
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))

    (root / "imports").mkdir(parents=True, exist_ok=True)
    (root / import_file).write_bytes(import_bytes)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / "cases").mkdir(parents=True, exist_ok=True)
    case_raw = case_doc.model_dump(mode="json")
    if case_raw["curation"]["state"] == "ready":

        case_raw["curation"]["task_spec_sha256"] = task_spec_digest(case_raw)
    (root / case_file).write_text(yaml.safe_dump(case_raw, sort_keys=False))
    return root


_CURATED: dict[str, Path] = {}


def _write_curated_workspace(tmp_path: Path, curation_state: Any, *, resolved: Any=True) -> Any:
    """Copy a valid one-case workspace with real Git content and requested curation state.

    One real build per state and worker is copied (nothing in it names its own location).
    Optionally leave source identity unresolved to test readiness independently.
    """

    if curation_state not in _CURATED:
        template = tmp_path.parent / f"{tmp_path.name}-curated-template-{curation_state}" / "ws"
        init_workspace(template, "O/R", ["h1.example.com"], ["h2.example.com"])
        _CURATED[curation_state] = _write_case_docs(template, curation_state)
    root = tmp_path / "ws"
    if root.exists():
        shutil.rmtree(root)   # the fixture is re-invocable (b-valid case)
    shutil.copytree(_CURATED[curation_state], root, symlinks=True)
    if not resolved:
        raw = yaml.safe_load((root / "benchmark.yaml").read_text())
        raw["source"]["repository_id"] = None
        raw["source"]["visibility"] = "unresolved"
        (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return root


def _ready_bundle_checksum_mismatch_is_validate_corruption(tmp_path: Path) -> None:
    ws = _write_curated_workspace(tmp_path, "ready")   # one ready case: bundle YAML + snapshots/<case>.bundle
    code, _ = validate_workspace(ws)
    assert code == 0
    # Corrupt the bundle bytes (keeps the case document and ledger intact).
    case = load_yaml_strict(next((ws / "cases").glob("*.yaml")))
    (ws / case["snapshot"]["bundle_file"]).write_bytes(b"tampered")
    code2, label = validate_workspace(ws)
    assert code2 == 1 and "corrupt" in label
    case_after = load_yaml_strict(next((ws / "cases").glob("*.yaml")))
    assert case_after["snapshot"]["status"] == "ready"

def _make_case_base_drift(root: Path) -> str:
    """Convert the fixture's ready case into a valid typed drift refusal."""

    case_path = next((root / "cases").glob("*.yaml"))
    raw = load_yaml_strict(case_path)
    snapshot = raw["snapshot"]
    (root / snapshot["bundle_file"]).unlink()
    raw["snapshot"] = {
        "status": "unreplayable", "policy": "explicit_head", "requested_head": snapshot["original_head_sha"],
        "original_base_sha": snapshot["original_base_sha"], "requested_base_sha": snapshot["requested_base_sha"],
        "original_head_sha": snapshot["original_head_sha"], "base_tree_sha": None, "head_tree_sha": None,
        "diff_sha256": None, "bundle_file": None, "bundle_sha256": None,
        "error": {"reason": "base_drift", "detail": "private path omitted"},
    }
    raw["curation"].update(state="unreplayable", snapshot_attested=False, clean_attested=False, gold_status="findings")
    raw["curation"].pop("task_spec_sha256", None)
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return str(raw["case_id"])


def test_status_and_validate_surface_typed_unreplayable_reason(tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Public status/validate expose the reason code, never private detail."""
    root = _write_curated_workspace(tmp_path, "ready")
    with pytest.raises(SystemExit) as ready_exit:
        top_cli.main(["benchmark", "status", str(root)])
    assert ready_exit.value.code == 0
    assert "snapshot ready @ " in capsys.readouterr().out
    case_id = _make_case_base_drift(root)
    assert curation.list_cases(root)[0]["changed_files"] == 0  # unreplayable snapshots have no frozen stats
    status = workspace_status(root)
    assert status.case_snapshots == [{
            "case_id": case_id, "snapshot_status": "unreplayable", "head_prefix": case_id.rsplit("-", 1)[-1],
            "error_reason": "base_drift", "task_spec_approval": "not-required",
        }
    ]
    assert validate_workspace(root) == (
        2, "incomplete: workspace state curating; unreplayable snapshot reasons: base_drift",
    )

    with pytest.raises(SystemExit) as status_exit:
        top_cli.main(["benchmark", "status", str(root)])
    assert status_exit.value.code == 0
    status_out = capsys.readouterr().out
    assert "snapshot unreplayable (base_drift)" in status_out
    assert "private path omitted" not in status_out

    with pytest.raises(SystemExit) as validate_exit:
        top_cli.main(["benchmark", "validate", str(root)])
    assert validate_exit.value.code == 2
    validate_out = capsys.readouterr().out
    assert "unreplayable snapshot reasons: base_drift" in validate_out
    assert "private path omitted" not in validate_out

def _validate_rechecks_marked_snapshot_source_when_mirror_is_present(tmp_path: Path) -> None:
    root = _write_curated_workspace(tmp_path, "ready")
    assert validate_workspace(root) == (0, "ready")
    case_path = next((root / "cases").glob("*.yaml"))
    raw = load_yaml_strict(case_path)
    raw["snapshot"]["original_base_sha"] = "a" * 40
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    code, label = validate_workspace(root)
    assert code == 1
    assert "source provenance" in label
    assert "feature.py" not in label

def test_validate_partial_mirror_reports_restore_guidance_without_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = tmp_path / "parent with spaces"
    parent.mkdir()
    root = _write_curated_workspace(parent, "ready")
    mirror = root / "cache" / "repository.git"
    shutil.rmtree(mirror / "objects")
    (mirror / "objects").mkdir()
    sentinel = mirror / "restore-me"
    sentinel.write_text("operator recovery marker\n")

    with pytest.raises(SystemExit) as exc:
        top_cli.main(["benchmark", "validate", str(root)])
    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert str(root) in output
    assert "cache/repository.git" in output
    assert "restore" in output.lower()
    assert sentinel.read_text() == "operator recovery marker\n"

def _portable_validation_binds_case_to_checksummed_import(tmp_path: Path, tamper: str) -> None:
    root = _write_curated_workspace(tmp_path, "ready")
    shutil.rmtree(root / "cache" / "repository.git")
    case_path = next((root / "cases").glob("*.yaml"))
    raw = load_yaml_strict(case_path)
    if tamper == "source_file":
        raw["source"]["import_file"] = "imports/other.json"
    else:
        raw["pull_request"]["changed_files"] = []
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    code, label = validate_workspace(root)
    assert code == 1
    assert "import provenance" in label

def _mutate_manifest_case(tmp_path: Path, pr_number: Any=None, case_file: Any=None) -> None:
    """Rewrite the manifest cases[] entry (pr_number and/or case_file)."""

    root = tmp_path / "ws"
    raw = yaml.safe_load((root / "benchmark.yaml").read_text())
    row = raw["cases"][0]
    if pr_number is not None:
        row["pr_number"] = pr_number
    if case_file is not None:
        row["case_file"] = case_file
    (root / "benchmark.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))


def _case_pr_number_mismatch_manifest_is_corruption(tmp_path: Path) -> None:
    root = _write_curated_workspace(tmp_path, "ready")
    _mutate_manifest_case(tmp_path, pr_number=999)
    _assert_corrupt_workspace(root)

def _duplicate_inode_indexed_files_is_corruption(tmp_path: Path) -> None:
    root = _write_curated_workspace(tmp_path, "ready")
    # hard-link the import to the bundle path (one inode, two indexed names).
    # the import bytes are a valid ImportDocument whose ledger sha is unchanged;
    # only the bundle sha is re-stamped to the import bytes, so every checksum /
    # model gate stays green and only the duplicate-inode cross-check can flag it
    imp = next((root / "imports").glob("pr-*.json"))
    bundle = next((root / "snapshots").glob("*.bundle"))
    imp_bytes = imp.read_bytes()
    bundle.unlink()
    os.link(imp, bundle)   # duplicate-inode surprise
    case_raw = yaml.safe_load(next((root / "cases").glob("*.yaml")).read_text())
    case_raw["snapshot"]["bundle_sha256"] = hashlib.sha256(imp_bytes).hexdigest()
    next((root / "cases").glob("*.yaml")).write_text(yaml.safe_dump(case_raw, sort_keys=False))
    _assert_corrupt_workspace(root)   # gated on Task 0 spike 4 verdict


def _unresolved_identity_and_stale_approval_are_incomplete(tmp_path: Path) -> None:
    unresolved = _write_curated_workspace(tmp_path / "unresolved", "ready", resolved=False)
    assert validate_workspace(unresolved) == (2, "incomplete: repository identity unresolved")
    root = _write_curated_workspace(tmp_path / "stale", "ready")
    case_path = next((root / "cases").glob("*.yaml"))
    raw = load_yaml_strict(case_path)
    raw["curation"]["task_spec_sha256"] = "0" * 64   # approval of a different Task Spec
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert validate_workspace(root) == (2, "incomplete: workspace state stale")


def test_validate_reports_tampered_bundles_provenance_and_index_as_corruption(tmp_path: Path) -> None:
    for check in (_ready_bundle_checksum_mismatch_is_validate_corruption,
                  _validate_rechecks_marked_snapshot_source_when_mirror_is_present,
                  _case_pr_number_mismatch_manifest_is_corruption, _duplicate_inode_indexed_files_is_corruption,
                  _unresolved_identity_and_stale_approval_are_incomplete):
        root = tmp_path / check.__name__
        root.mkdir()
        check(root)
    for tamper in ("source_file", "inventory"):
        root = tmp_path / tamper
        root.mkdir()
        _portable_validation_binds_case_to_checksummed_import(root, tamper)
