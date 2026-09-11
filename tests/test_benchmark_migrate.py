import hashlib
import os
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream.benchmark import migrate, schema, storage

_BASE = "0123456789abcdef0123456789abcdef01234567"
_HEAD_HEX = "0123456789abcdef0123456789abcdef01234567"
_CASE_ID = "pr-000101-0123456789ab"
_TITLE = "Cache misses"


def _legacy_finding_id(title: Any, body: Any, severity: Any, path: Any, start_line: Any, end_line: Any) -> Any:
    payload = "\x1f".join([str(title or ""), str(body or ""), str(severity or ""),
                           str(path or ""), str(start_line or ""), str(end_line or "")])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark_id": str(uuid.uuid4()),
        "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com",
                   "repository": "OWNER/REPO", "repository_id": None,
                   "visibility": "unresolved"},
        "privacy": {
            "classification": "confidential",
            "reviewer_data": "source_snapshot",
            "reviewer_allowed_hosts": ["api.anthropic.com"],
            "judge_data": "finding_text_and_location_only",
            "judge_allowed_hosts": ["api.anthropic.com"],
            "archive": "disabled",
            "uploads": "disabled",
        },
        "pull_requests": [],
        "cases": [{"case_id": _CASE_ID, "pr_number": 101, "case_file": f"cases/{_CASE_ID}.yaml"}],
    }


def _seed_v1_case() -> dict[str, Any]:
    finding = {
        "title": _TITLE,
        "body": "The cache layers never populate.",
        "severity": "high",
        "location": {"path": "src/cache.py", "start_line": 2, "end_line": 2},
        "provenance": {"kind": "authored", "source_ids": []},
        "finding_id": _legacy_finding_id(_TITLE, "The cache layers never populate.", "high",
                                         "src/cache.py", 2, 2),
    }
    return {
        "schema_version": 1,
        "case_id": _CASE_ID,
        "pull_request": {
            "number": 101,
            "url": "https://github.com/o/r/pull/101",
            "title": "Fix cache",
            "state": "open",
            "base": {"ref": "main", "sha": "b" * 40},
            "head": {"ref": "feature/cache", "sha": "h" * 40},
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "author": {"login": "alice", "type": "User"},
        },
        "snapshot": {
            "status": "ready", "policy": "final_pr_head", "requested_head": "final",
            "original_base_sha": _BASE, "requested_base_sha": _BASE,
            "original_head_sha": _HEAD_HEX,
            "base_tree_sha": "0" * 40, "head_tree_sha": "0" * 40,
            "diff_sha256": "a" * 64, "bundle_file": "snapshots/x.bundle",
            "bundle_sha256": "b" * 64, "error": None,
        },
        "source": {"import_file": "imports/pr-101.json", "import_sha256": "c" * 64},
        "curation": {
            "state": "draft", "snapshot_attested": False, "clean_attested": False,
            "gold_status": None, "findings": [finding], "exclusions": [],
            "case_exclusion": None,
        },
    }


def _seed_v1_workspace(tmp_path: Path) -> tuple[Any, ...]:
    ws = tmp_path / "ws"
    storage.ensure_private_dir(ws)
    repo = tmp_path / "source"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Tester",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Tester",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }

    def commit(name: str, content: str, message: str) -> str:
        (repo / name).write_text(content)
        subprocess.run(["git", "add", name], cwd=repo, env=env, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", message], cwd=repo, env=env, check=True, capture_output=True)
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    merge_base = commit("base.py", "BASE = 1\n", "base")
    subprocess.run(["git", "checkout", "-b", "feature"], cwd=repo, check=True, capture_output=True)
    head_sha = commit("feature.py", "FEATURE = 1\n", "feature")
    subprocess.run(["git", "checkout", "main"], cwd=repo, check=True, capture_output=True)
    requested_tip = commit("upstream.py", "UPSTREAM = 1\n", "advanced base")
    cache = ws / "cache"
    storage.ensure_private_dir(cache)
    subprocess.run(
        ["git", "clone", "--mirror", str(repo), str(cache / "repository.git")],
        check=True,
        capture_output=True,
    )
    base_tree = subprocess.run(
        ["git", "rev-parse", f"{merge_base}^{{tree}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head_tree = subprocess.run(
        ["git", "rev-parse", f"{head_sha}^{{tree}}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    case_id = f"pr-000101-{head_sha[:12]}"
    manifest = _seed_manifest()
    manifest["cases"] = [
        {"case_id": case_id, "pr_number": 101, "case_file": f"cases/{case_id}.yaml"}
    ]
    storage.atomic_write_yaml(ws / "benchmark.yaml", manifest)
    case_dir = ws / "cases"
    storage.ensure_private_dir(case_dir)
    case = _seed_v1_case()
    case["case_id"] = case_id
    case["pull_request"]["base"]["sha"] = requested_tip
    case["pull_request"]["head"]["sha"] = head_sha
    case["snapshot"].update(
        original_base_sha=merge_base,
        requested_base_sha=requested_tip,
        original_head_sha=head_sha,
        base_tree_sha=base_tree,
        head_tree_sha=head_tree,
    )
    storage.atomic_write_yaml(case_dir / f"{case_id}.yaml", case)
    return ws, case_id, _TITLE


def _write_legacy_unreplayable_case(
    ws: Path,
    case_id: str,
    *,
    schema_version: int,
    curation_state: str = "draft",
) -> Path:
    case_path = ws / "cases" / f"{case_id}.yaml"
    raw = storage.load_yaml_strict(case_path)
    raw["schema_version"] = schema_version
    if schema_version == 2:
        finding = raw["curation"]["findings"][0]
        finding["finding_id"] = schema.derive_finding_id(finding, case_id=case_id)
    raw["snapshot"] = {
        "status": "unreplayable",
        "policy": "final_pr_head",
        "requested_head": "final",
        "original_base_sha": None,
        "requested_base_sha": raw["pull_request"]["base"]["sha"],
        "original_head_sha": raw["pull_request"]["head"]["sha"],
        "base_tree_sha": None,
        "head_tree_sha": None,
        "diff_sha256": None,
        "bundle_file": None,
        "bundle_sha256": None,
        "error": {
            "reason": "head_not_on_pr",
            "detail": "legacy producer detail must be preserved",
        },
    }
    curation = raw["curation"]
    curation["state"] = curation_state
    curation["snapshot_attested"] = curation_state == "ready"
    curation["clean_attested"] = False
    curation["gold_status"] = None if curation_state == "draft" else "findings"
    curation["exclusions"] = [
        {
            "source_id": "github:review:7",
            "reason": "other",
            "note": "authored exclusion note must be preserved",
        }
    ]
    curation["case_exclusion"] = None
    if curation_state == "ready":
        curation["task_spec_sha256"] = "d" * 64
        curation["task_spec_approved_at"] = "2026-08-20T12:00:00Z"
    storage.atomic_write_yaml(case_path, raw)
    return case_path


def test_migrate_recomputes_finding_ids_and_bumps_version(tmp_path: Path) -> None:
    ws, case_id, title = _seed_v1_workspace(tmp_path)
    report = migrate.migrate_workspace(ws)
    assert [c.case_id for c in report.cases] == [case_id]
    assert report.cases[0].finding_ids_recomputed == 1
    assert report.cases[0].changed is True
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["schema_version"] == 2
    f = raw["curation"]["findings"][0]
    assert f["finding_id"] == schema.derive_finding_id(f, case_id=case_id)  # now case-scoped
    assert f["title"] == title and f["provenance"]["kind"] == "authored"    # authored content preserved
    # migrated doc fully validates
    from daydream.benchmark.schema import _schema_ready
    schema.CaseDocument.model_validate(_schema_ready(raw))


def test_migrate_backfills_requested_base_sha_on_v1_ready_snapshot(tmp_path: Path) -> None:
    """The sole legacy base candidate is verified before splitting provenance."""
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    requested_tip = raw["snapshot"]["requested_base_sha"]
    merge_base = raw["snapshot"]["original_base_sha"]
    raw["snapshot"]["original_base_sha"] = requested_tip
    del raw["snapshot"]["requested_base_sha"]
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)

    report = migrate.migrate_workspace(ws)
    assert report.errors == []
    assert [c.case_id for c in report.cases] == [case_id]
    assert report.cases[0].changed is True
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["schema_version"] == 2
    assert raw["snapshot"]["requested_base_sha"] == requested_tip
    assert raw["snapshot"]["original_base_sha"] == merge_base
    assert raw["snapshot"]["base_resolution"] == "merge_base_v1"
    from daydream.benchmark.schema import _schema_ready
    schema.CaseDocument.model_validate(_schema_ready(raw))  # no longer corrupt


def test_migrate_backfills_requested_base_sha_on_v2_ready_snapshot(tmp_path: Path) -> None:
    """An old v2 blind backfill is verified and repaired without touching ids."""
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    migrate.migrate_workspace(ws)                       # v1 -> v2 (field present)
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["schema_version"] == 2
    finding_ids = [f["finding_id"] for f in raw["curation"]["findings"]]
    requested_tip = raw["snapshot"]["requested_base_sha"]
    merge_base = raw["snapshot"]["original_base_sha"]
    raw["snapshot"]["original_base_sha"] = requested_tip
    del raw["snapshot"]["requested_base_sha"]          # simulate pre-break v2
    del raw["snapshot"]["base_resolution"]
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)

    report = migrate.migrate_workspace(ws)
    assert report.errors == []
    assert report.cases[0].finding_ids_recomputed == 0  # ids untouched
    assert report.cases[0].changed is True
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["schema_version"] == 2                  # no bump
    assert raw["snapshot"]["requested_base_sha"] == requested_tip
    assert raw["snapshot"]["original_base_sha"] == merge_base
    assert raw["snapshot"]["base_resolution"] == "merge_base_v1"
    assert [f["finding_id"] for f in raw["curation"]["findings"]] == finding_ids
    from daydream.benchmark.schema import _schema_ready
    schema.CaseDocument.model_validate(_schema_ready(raw))

    second = migrate.migrate_workspace(ws)              # idempotent
    assert second.cases == [] and second.errors == []


def test_migrate_leaves_unreplayable_snapshot_without_backfill(tmp_path: Path) -> None:
    """Unreplayable snapshots carry requested_base_sha as nullable, so a v2
    case that omits it is left byte-unchanged (no repair needed, no rewrite)."""
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    migrate.migrate_workspace(ws)                       # v1 -> v2 (field present)
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    raw["snapshot"] = {
        "status": "unreplayable", "policy": "final_pr_head", "requested_head": "final",
        "original_base_sha": None,
        "original_head_sha": raw["pull_request"]["head"]["sha"],
        "base_tree_sha": None, "head_tree_sha": None, "diff_sha256": None,
        "bundle_file": None, "bundle_sha256": None,
        "error": {"reason": "head_not_on_pr", "detail": "head sha not on PR"},
    }
    raw["curation"]["state"] = "unreplayable"
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)
    case_path = ws / "cases" / f"{case_id}.yaml"
    before = case_path.read_bytes()

    report = migrate.migrate_workspace(ws)
    after = storage.load_yaml_strict(case_path)
    assert report.cases == [] and report.errors == []
    assert case_path.read_bytes() == before
    assert "requested_base_sha" not in after["snapshot"]


@pytest.mark.parametrize("legacy_version", [1, 2])
def test_migrate_repairs_legacy_draft_unreplayable_curation(
    tmp_path: Path, legacy_version: int
) -> None:
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    case_path = _write_legacy_unreplayable_case(
        ws, case_id, schema_version=legacy_version
    )
    before = storage.load_yaml_strict(case_path)
    immutable_before = {
        key: deepcopy(before[key])
        for key in ("pull_request", "snapshot", "source", "candidates", "prioritization")
        if key in before
    }
    findings_before = deepcopy(before["curation"]["findings"])
    exclusions_before = deepcopy(before["curation"]["exclusions"])

    report = migrate.migrate_workspace(ws)

    assert report.errors == []
    assert [case.case_id for case in report.cases] == [case_id]
    migrated = storage.load_yaml_strict(case_path)
    assert migrated["schema_version"] == 2
    assert migrated["curation"]["state"] == "unreplayable"
    assert migrated["curation"]["snapshot_attested"] is False
    assert migrated["curation"]["clean_attested"] is False
    assert migrated["curation"]["gold_status"] == "findings"
    assert migrated["curation"]["exclusions"] == exclusions_before
    if legacy_version == 1:
        expected_findings = deepcopy(findings_before)
        expected_findings[0]["finding_id"] = schema.derive_finding_id(
            expected_findings[0], case_id=case_id
        )
        assert migrated["curation"]["findings"] == expected_findings
    else:
        assert migrated["curation"]["findings"] == findings_before
    for key, expected in immutable_before.items():
        assert migrated[key] == expected
    schema.CaseDocument.model_validate(schema._schema_ready(migrated))


@pytest.mark.parametrize("curation_state", ["ready", "stale", "excluded"])
def test_migrate_rejects_other_v2_unreplayable_curation_mismatches(
    tmp_path: Path, curation_state: str
) -> None:
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    case_path = _write_legacy_unreplayable_case(
        ws, case_id, schema_version=2, curation_state=curation_state
    )
    before = case_path.read_bytes()

    report = migrate.migrate_workspace(ws)

    assert report.cases == []
    assert len(report.errors) == 1
    assert "unreplayable snapshot and curation states must match" in report.errors[0]
    assert case_path.read_bytes() == before


def test_upgrade_cli_repairs_legacy_unreplayable_draft_atomically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from daydream import cli as top_cli

    parent = tmp_path / "workspace parent with spaces"
    parent.mkdir()
    ws, case_id, _ = _seed_v1_workspace(parent)
    case_path = _write_legacy_unreplayable_case(ws, case_id, schema_version=2)
    before = case_path.read_bytes()

    with pytest.raises(SystemExit) as dry_exit:
        top_cli.main(["benchmark", "upgrade", str(ws), "--dry-run"])
    assert dry_exit.value.code == 0
    assert "changed=True" in capsys.readouterr().out
    assert case_path.read_bytes() == before

    with pytest.raises(SystemExit) as write_exit:
        top_cli.main(["benchmark", "upgrade", str(ws)])
    assert write_exit.value.code == 0
    assert "changed=True" in capsys.readouterr().out
    migrated = storage.load_yaml_strict(case_path)
    assert migrated["curation"]["state"] == "unreplayable"
    assert migrated["curation"]["exclusions"][0]["note"] == (
        "authored exclusion note must be preserved"
    )
    after = case_path.read_bytes()

    with pytest.raises(SystemExit) as second_exit:
        top_cli.main(["benchmark", "upgrade", str(ws)])
    assert second_exit.value.code == 0
    assert case_path.read_bytes() == after


def test_upgrade_cli_reports_invalid_unchanged_v2_without_rewriting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from daydream import cli as top_cli

    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    case_path = _write_legacy_unreplayable_case(
        ws, case_id, schema_version=2, curation_state="stale"
    )
    before = case_path.read_bytes()

    with pytest.raises(SystemExit) as exc_info:
        top_cli.main(["benchmark", "upgrade", str(ws)])

    assert exc_info.value.code == 1
    assert "unreplayable snapshot and curation states must match" in capsys.readouterr().err
    assert case_path.read_bytes() == before


def test_migrate_dry_run_writes_nothing_and_is_idempotent(tmp_path: Path) -> None:
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    migrate.migrate_workspace(ws, dry_run=True)
    assert storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["schema_version"] == 1
    migrate.migrate_workspace(ws)
    second = migrate.migrate_workspace(ws)
    assert all(c.changed is False for c in second.cases)   # no-op second run


def test_migrate_surfaces_invalid_case_without_rewriting(tmp_path: Path) -> None:
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    # corrupt the case: duplicate finding_id (uniqueness violated)
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    raw["curation"]["findings"].append(dict(raw["curation"]["findings"][0]))
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)
    report = migrate.migrate_workspace(ws)
    assert report.cases == []                        # no case rewritten
    assert any("duplicate" in e for e in report.errors)
    assert storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["schema_version"] == 1  # untouched


@pytest.mark.parametrize("failure", ["missing_mirror", "tree_mismatch"])
def test_migrate_ready_provenance_failure_is_atomic(
    tmp_path: Path, failure: str
) -> None:
    """A failed proof reports the case and leaves its v1 bytes unchanged."""
    import shutil

    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    case_path = ws / "cases" / f"{case_id}.yaml"
    if failure == "missing_mirror":
        shutil.rmtree(ws / "cache" / "repository.git")
    else:
        raw = storage.load_yaml_strict(case_path)
        raw["snapshot"]["base_tree_sha"] = "f" * 40
        storage.atomic_write_yaml(case_path, raw)
    before = case_path.read_bytes()

    report = migrate.migrate_workspace(ws)

    assert report.cases == []
    assert len(report.errors) == 1
    assert "provenance" in report.errors[0] or "source trees" in report.errors[0]
    assert case_path.read_bytes() == before


def test_migrate_imported_snapshot_preserves_sole_base_without_mirror(tmp_path: Path) -> None:
    """Imported legacy snapshots have no trees to verify and keep their candidate."""
    import shutil

    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    case_path = ws / "cases" / f"{case_id}.yaml"
    raw = storage.load_yaml_strict(case_path)
    base_tip = raw["pull_request"]["base"]["sha"]
    head_sha = raw["snapshot"]["original_head_sha"]
    raw["snapshot"] = {
        "status": "imported",
        "policy": "final_pr_head",
        "requested_head": "final",
        "original_base_sha": base_tip,
        "original_head_sha": head_sha,
        "error": None,
    }
    storage.atomic_write_yaml(case_path, raw)
    shutil.rmtree(ws / "cache" / "repository.git")

    report = migrate.migrate_workspace(ws)

    assert report.errors == []
    migrated = storage.load_yaml_strict(case_path)
    assert migrated["snapshot"]["requested_base_sha"] == base_tip
    assert "base_resolution" not in migrated["snapshot"]

def test_upgrade_cli_wiring_dry_run_and_real_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The ``upgrade`` verb drives migrate_workspace through the CLI seam (exit 0)."""
    from daydream.benchmark.cli import _handle_benchmark_command

    ws, case_id, _ = _seed_v1_workspace(tmp_path)

    # --dry-run reports the upgrade (finding recomputed, would change) without writing.
    rc = _handle_benchmark_command(["upgrade", str(ws), "--dry-run"])
    assert rc == 0
    assert storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["schema_version"] == 1
    out = capsys.readouterr().out
    assert "changed=True" in out and "finding_ids_recomputed=1" in out

    # A real run rewrites the v1 case and exits 0.
    rc = _handle_benchmark_command(["upgrade", str(ws)])
    assert rc == 0
    assert storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")["schema_version"] == 2

    # The no-op second run still exits 0.
    rc = _handle_benchmark_command(["upgrade", str(ws)])
    assert rc == 0


def test_upgrade_cli_error_returns_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An errored case surfaces on stderr and yields exit code 1."""
    from daydream.benchmark.cli import _handle_benchmark_command

    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    raw["schema_version"] = "bogus"
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)
    rc = _handle_benchmark_command(["upgrade", str(ws)])
    assert rc == 1
    assert "error" in capsys.readouterr().err


def test_migrate_heals_interrupted_journal_under_lock(tmp_path: Path) -> None:
    """migrate_workspace must recover_startup under the workspace lock (like every
    other locked writer) so a crashed curator journal is healed before it reads;
    and its transaction op_id must be flat so no residue is left that bricks a
    later recover_startup with WorkspaceCorrupt."""
    from tests.harness.transaction_faults import TransactionFaultDriver

    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = storage.load_yaml_strict(path)
    mutated = dict(raw)
    mutated["curation"] = dict(raw["curation"])
    mutated["curation"]["state"] = "excluded"    # an interrupted mutation left in flight
    faults = TransactionFaultDriver(ws, op_id=f"migrate-{case_id}", kind="migrate")
    with faults.transaction as tx:
        tx.stage(f"cases/{case_id}.yaml", yaml.safe_dump(mutated, sort_keys=False).encode("utf-8"))
        faults.halt_at("target-1")               # target applied under 'committing', then halt
    assert storage.load_yaml_strict(path)["curation"]["state"] == "excluded"

    migrate.migrate_workspace(ws)               # recover_startup under lock rolls the crash back

    assert not list((ws / "transactions").iterdir())    # healed AND no residue left behind
    final = storage.load_yaml_strict(path)
    assert final["curation"]["state"] == "draft"        # interrupted 'excluded' write rolled back
    assert final["schema_version"] == 2                 # the migration still ran
    storage.recover_startup(ws)                         # a follow-up recovery must not brick
