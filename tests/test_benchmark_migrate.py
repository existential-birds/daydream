import hashlib
import uuid
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
    storage.atomic_write_yaml(ws / "benchmark.yaml", _seed_manifest())
    case_dir = ws / "cases"
    storage.ensure_private_dir(case_dir)
    storage.atomic_write_yaml(case_dir / f"{_CASE_ID}.yaml", _seed_v1_case())
    return ws, _CASE_ID, _TITLE


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
    """A pre-provenance-split v1 workspace (ready snapshot without
    requested_base_sha) is repaired: the backfill copies the recorded
    original_base_sha so the migrated doc validates."""
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    del raw["snapshot"]["requested_base_sha"]
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)

    report = migrate.migrate_workspace(ws)
    assert report.errors == []
    assert [c.case_id for c in report.cases] == [case_id]
    assert report.cases[0].changed is True
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["schema_version"] == 2
    assert raw["snapshot"]["requested_base_sha"] == raw["snapshot"]["original_base_sha"]
    from daydream.benchmark.schema import _schema_ready
    schema.CaseDocument.model_validate(_schema_ready(raw))  # no longer corrupt


def test_migrate_backfills_requested_base_sha_on_v2_ready_snapshot(tmp_path: Path) -> None:
    """A v2 workspace persisted before requested_base_sha became required is
    backfilled without touching finding ids (no recompute, no version bump),
    and a second run is a no-op."""
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    migrate.migrate_workspace(ws)                       # v1 -> v2 (field present)
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["schema_version"] == 2
    finding_ids = [f["finding_id"] for f in raw["curation"]["findings"]]
    del raw["snapshot"]["requested_base_sha"]          # simulate pre-break v2
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)

    report = migrate.migrate_workspace(ws)
    assert report.errors == []
    assert report.cases[0].finding_ids_recomputed == 0  # ids untouched
    assert report.cases[0].changed is True
    raw = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert raw["schema_version"] == 2                  # no bump
    assert raw["snapshot"]["requested_base_sha"] == raw["snapshot"]["original_base_sha"]
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
        "original_head_sha": "0123456789abcdef0123456789abcdef01234567",
        "base_tree_sha": None, "head_tree_sha": None, "diff_sha256": None,
        "bundle_file": None, "bundle_sha256": None,
        "error": {"reason": "head_not_on_pr", "detail": "head sha not on PR"},
    }
    raw["curation"]["state"] = "unreplayable"
    storage.atomic_write_yaml(ws / "cases" / f"{case_id}.yaml", raw)
    before = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")

    migrate.migrate_workspace(ws)
    after = storage.load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert after == before                            # nothing rewritten
    assert "requested_base_sha" not in after["snapshot"]


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
    ws, case_id, _ = _seed_v1_workspace(tmp_path)
    path = ws / "cases" / f"{case_id}.yaml"
    raw = storage.load_yaml_strict(path)
    mutated = dict(raw)
    mutated["curation"] = dict(raw["curation"])
    mutated["curation"]["state"] = "excluded"    # an interrupted mutation left in flight
    with storage.Transaction(ws, op_id=f"migrate-{case_id}", kind="migrate") as tx:
        tx.stage(f"cases/{case_id}.yaml", yaml.safe_dump(mutated, sort_keys=False).encode("utf-8"))
        tx.inject_crash("target-1")              # target applied under 'committing', then halt
    assert storage.load_yaml_strict(path)["curation"]["state"] == "excluded"

    migrate.migrate_workspace(ws)               # recover_startup under lock rolls the crash back

    assert not list((ws / "transactions").iterdir())    # healed AND no residue left behind
    final = storage.load_yaml_strict(path)
    assert final["curation"]["state"] == "draft"        # interrupted 'excluded' write rolled back
    assert final["schema_version"] == 2                 # the migration still ran
    storage.recover_startup(ws)                         # a follow-up recovery must not brick
