import hashlib
import os
import shutil
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream import cli as top_cli
from daydream.benchmark import migrate, schema, storage
from daydream.benchmark.schema import _schema_ready
from tests.harness.git_helpers import commit as _commit, git as _git, write_and_stage
from tests.harness.transaction_faults import TransactionFaultDriver

_BASE = "0123456789abcdef0123456789abcdef01234567"
_HEAD_HEX = "0123456789abcdef0123456789abcdef01234567"
_CASE_ID = "pr-000101-0123456789ab"
_TITLE = "Cache misses"


def _legacy_finding_id(title: Any, body: Any, severity: Any, path: Any, start_line: Any, end_line: Any) -> Any:
    payload = "\x1f".join([str(title or ""), str(body or ""), str(severity or ""),
                           str(path or ""), str(start_line or ""), str(end_line or "")])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _seed_manifest() -> dict[str, Any]:
    return {"schema_version": 1, "benchmark_id": str(uuid.uuid4()), "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com",
                   "repository": "OWNER/REPO", "repository_id": None, "visibility": "unresolved"},
        "privacy": {"classification": "confidential", "reviewer_data": "source_snapshot",
            "reviewer_allowed_hosts": ["api.anthropic.com"], "judge_data": "finding_text_and_location_only",
            "judge_allowed_hosts": ["api.anthropic.com"], "archive": "disabled", "uploads": "disabled",
        }, "pull_requests": [],
        "cases": [{"case_id": _CASE_ID, "pr_number": 101, "case_file": f"cases/{_CASE_ID}.yaml"}],
    }


def _seed_v1_case() -> dict[str, Any]:
    finding = {"title": _TITLE, "body": "The cache layers never populate.", "severity": "high",
        "location": {"path": "src/cache.py", "start_line": 2, "end_line": 2},
        "provenance": {"kind": "authored", "source_ids": []},
        "finding_id": _legacy_finding_id(_TITLE, "The cache layers never populate.", "high", "src/cache.py", 2, 2),
    }
    return {"schema_version": 1, "case_id": _CASE_ID,
        "pull_request": {"number": 101, "url": "https://github.com/o/r/pull/101", "title": "Fix cache", "state": "open",
            "base": {"ref": "main", "sha": "b" * 40}, "head": {"ref": "feature/cache", "sha": "h" * 40},
            "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z",
            "author": {"login": "alice", "type": "User"},
        }, "snapshot": {"status": "ready", "policy": "final_pr_head", "requested_head": "final",
            "original_base_sha": _BASE, "requested_base_sha": _BASE, "original_head_sha": _HEAD_HEX,
            "base_tree_sha": "0" * 40, "head_tree_sha": "0" * 40,
            "diff_sha256": "a" * 64, "bundle_file": "snapshots/x.bundle", "bundle_sha256": "b" * 64, "error": None,
        }, "source": {"import_file": "imports/pr-101.json", "import_sha256": "c" * 64},
        "curation": {"state": "draft", "snapshot_attested": False, "clean_attested": False,
            "gold_status": None, "findings": [finding], "exclusions": [], "case_exclusion": None,
        },
    }


def _seed_v1_workspace(tmp_path: Path) -> tuple[Any, ...]:
    ws = tmp_path / "ws"
    storage.ensure_private_dir(ws)
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    env = {**os.environ, "GIT_AUTHOR_NAME": "Tester", "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Tester", "GIT_COMMITTER_EMAIL": "test@example.com",
    }

    def seed_file(name: str, content: str, message: str) -> str:
        write_and_stage(repo, name, content)
        return _commit(repo, message, env=env)

    merge_base = seed_file("base.py", "BASE = 1\n", "base")
    _git(repo, "checkout", "-b", "feature")
    head_sha = seed_file("feature.py", "FEATURE = 1\n", "feature")
    _git(repo, "checkout", "main")
    requested_tip = seed_file("upstream.py", "UPSTREAM = 1\n", "advanced base")
    cache = ws / "cache"
    storage.ensure_private_dir(cache)
    _git(repo, "clone", "--mirror", str(repo), str(cache / "repository.git"))
    base_tree = _git(repo, "rev-parse", f"{merge_base}^{{tree}}")
    head_tree = _git(repo, "rev-parse", f"{head_sha}^{{tree}}")
    case_id = f"pr-000101-{head_sha[:12]}"
    manifest = _seed_manifest()
    manifest["cases"] = [{"case_id": case_id, "pr_number": 101, "case_file": f"cases/{case_id}.yaml"}]
    storage.atomic_write_yaml(ws / "benchmark.yaml", manifest)
    case_dir = ws / "cases"
    storage.ensure_private_dir(case_dir)
    case = _seed_v1_case()
    case["case_id"] = case_id
    case["pull_request"]["base"]["sha"] = requested_tip
    case["pull_request"]["head"]["sha"] = head_sha
    case["snapshot"].update(original_base_sha=merge_base, requested_base_sha=requested_tip, original_head_sha=head_sha,
        base_tree_sha=base_tree, head_tree_sha=head_tree,
    )
    storage.atomic_write_yaml(case_dir / f"{case_id}.yaml", case)
    return ws, case_id, _TITLE


def _write_legacy_unreplayable_case(ws: Path, case_id: str, *, schema_version: int, curation_state: str = "draft",
) -> Path:
    case_path = ws / "cases" / f"{case_id}.yaml"
    raw = storage.load_yaml_strict(case_path)
    raw["schema_version"] = schema_version
    if schema_version == 2:
        finding = raw["curation"]["findings"][0]
        finding["finding_id"] = schema.derive_finding_id(finding, case_id=case_id)
    raw["snapshot"] = {
        "status": "unreplayable", "policy": "final_pr_head", "requested_head": "final", "original_base_sha": None,
        "requested_base_sha": raw["pull_request"]["base"]["sha"],
        "original_head_sha": raw["pull_request"]["head"]["sha"], "base_tree_sha": None, "head_tree_sha": None,
        "diff_sha256": None, "bundle_file": None, "bundle_sha256": None,
        "error": {"reason": "head_not_on_pr", "detail": "legacy producer detail must be preserved"},
    }
    curation = raw["curation"]
    curation["state"] = curation_state
    curation["snapshot_attested"] = curation_state == "ready"
    curation["clean_attested"] = False
    curation["gold_status"] = None if curation_state == "draft" else "findings"
    curation["exclusions"] = [
        {"source_id": "github:review:7", "reason": "other", "note": "authored exclusion note must be preserved"}
    ]
    curation["case_exclusion"] = None
    if curation_state == "ready":
        curation["task_spec_sha256"] = "d" * 64
        curation["task_spec_approved_at"] = "2026-08-20T12:00:00Z"
    storage.atomic_write_yaml(case_path, raw)
    return case_path


def test_upgrade_cli_migrates_native_population_and_preserves_noops(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """One real workspace upgrades v1/v2 records while leaving a valid no-op byte-identical."""
    parent = tmp_path / "workspace parent with spaces"
    parent.mkdir()
    ws, original_id, title = _seed_v1_workspace(parent)
    template = storage.load_yaml_strict(ws / "cases" / f"{original_id}.yaml")
    requested = template["snapshot"]["requested_base_sha"]
    merge_base = template["snapshot"]["original_base_sha"]
    head = template["snapshot"]["original_head_sha"]
    cases = {number: deepcopy(template) for number in (101, 102, 103, 104, 105, 106)}
    cases[102]["snapshot"]["original_base_sha"] = requested
    del cases[102]["snapshot"]["requested_base_sha"]
    cases[103]["schema_version"] = 2
    cases[103]["snapshot"]["original_base_sha"] = requested
    del cases[103]["snapshot"]["requested_base_sha"]
    cases[104]["schema_version"] = 2
    cases[104]["snapshot"] = {
        "status": "unreplayable", "policy": "final_pr_head", "requested_head": "final", "original_base_sha": None,
        "original_head_sha": head, "base_tree_sha": None, "head_tree_sha": None, "diff_sha256": None,
        "bundle_file": None, "bundle_sha256": None,
        "error": {"reason": "head_not_on_pr", "detail": "head sha not on PR"},
    }
    cases[104]["curation"]["state"] = "unreplayable"
    ids = {number: schema.case_id_for(number, head) for number in cases}
    for number, doc in cases.items():
        doc["case_id"] = ids[number]
        doc["pull_request"].update(number=number, url=f"https://github.com/o/r/pull/{number}")
        if doc["schema_version"] == 2:
            finding = doc["curation"]["findings"][0]
            finding["finding_id"] = schema.derive_finding_id(finding, case_id=ids[number])
        storage.atomic_write_yaml(ws / "cases" / f"{ids[number]}.yaml", doc)
    manifest_path = ws / "benchmark.yaml"
    manifest = storage.load_yaml_strict(manifest_path)
    manifest["cases"] = [{"case_id": ids[n], "pr_number": n, "case_file": f"cases/{ids[n]}.yaml"} for n in cases]
    storage.atomic_write_yaml(manifest_path, manifest)
    _write_legacy_unreplayable_case(ws, ids[105], schema_version=1)
    _write_legacy_unreplayable_case(ws, ids[106], schema_version=2)
    paths = {number: ws / "cases" / f"{case_id}.yaml" for number, case_id in ids.items()}
    before = {number: path.read_bytes() for number, path in paths.items()}
    originals = {number: storage.load_yaml_strict(path) for number, path in paths.items()}
    manifest_before = manifest_path.read_bytes()
    expected = [
        f"case {ids[101]}: finding_ids_recomputed=1 changed=True",
        f"case {ids[102]}: finding_ids_recomputed=1 changed=True",
        f"case {ids[103]}: finding_ids_recomputed=0 changed=True",
        f"case {ids[105]}: finding_ids_recomputed=1 changed=True",
        f"case {ids[106]}: finding_ids_recomputed=0 changed=True",
    ]

    with pytest.raises(SystemExit) as dry_exit:
        top_cli.main(["benchmark", "upgrade", str(ws), "--dry-run"])
    assert dry_exit.value.code == 0
    assert capsys.readouterr().out.splitlines() == expected
    assert {n: path.read_bytes() for n, path in paths.items()} == before
    assert manifest_path.read_bytes() == manifest_before
    assert not (ws / ".benchmark.lock").exists()

    with pytest.raises(SystemExit) as write_exit:
        top_cli.main(["benchmark", "upgrade", str(ws)])
    assert write_exit.value.code == 0
    assert capsys.readouterr().out.splitlines() == expected
    migrated = {number: storage.load_yaml_strict(path) for number, path in paths.items()}
    assert {n: doc["schema_version"] for n, doc in migrated.items()} == {101: 2, 102: 2, 103: 2, 104: 2, 105: 2, 106: 2}
    finding = migrated[101]["curation"]["findings"][0]
    assert finding["finding_id"] == schema.derive_finding_id(finding, case_id=ids[101])
    assert finding["title"] == title and finding["provenance"]["kind"] == "authored"
    assert {n: (migrated[n]["snapshot"]["requested_base_sha"], migrated[n]["snapshot"]["original_base_sha"],
                migrated[n]["snapshot"]["base_resolution"]) for n in (102, 103)} == {
        102: (requested, merge_base, "merge_base_v1"), 103: (requested, merge_base, "merge_base_v1"),
    }
    assert migrated[103]["curation"]["findings"] == originals[103]["curation"]["findings"]
    assert paths[104].read_bytes() == before[104]
    assert "requested_base_sha" not in migrated[104]["snapshot"]
    assert {n: (migrated[n]["curation"]["state"], migrated[n]["curation"]["snapshot_attested"],
                migrated[n]["curation"]["clean_attested"], migrated[n]["curation"]["gold_status"])
            for n in (105, 106)} == {
        105: ("unreplayable", False, False, "findings"), 106: ("unreplayable", False, False, "findings"),
    }
    v1_findings = deepcopy(originals[105]["curation"]["findings"])
    v1_findings[0]["finding_id"] = schema.derive_finding_id(v1_findings[0], case_id=ids[105])
    assert migrated[105]["curation"]["findings"] == v1_findings
    assert migrated[106]["curation"]["findings"] == originals[106]["curation"]["findings"]
    for number in (105, 106):
        assert migrated[number]["curation"]["exclusions"] == originals[number]["curation"]["exclusions"]
        assert migrated[number]["curation"]["exclusions"][0]["note"] == "authored exclusion note must be preserved"
        assert {k: v for k, v in migrated[number].items() if k not in ("schema_version", "curation")} == {
            k: v for k, v in originals[number].items() if k not in ("schema_version", "curation")
        }
    for doc in migrated.values():
        schema.CaseDocument.model_validate(_schema_ready(doc))
    assert manifest_path.read_bytes() == manifest_before
    after = {number: path.read_bytes() for number, path in paths.items()}

    with pytest.raises(SystemExit) as second_exit:
        top_cli.main(["benchmark", "upgrade", str(ws)])
    assert second_exit.value.code == 0
    assert capsys.readouterr().out == ""
    assert {n: path.read_bytes() for n, path in paths.items()} == after


def test_upgrade_cli_retains_independent_errors_while_migrating_valid_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A native manifest reports every bad case without losing valid migration progress."""
    ws, original_id, _ = _seed_v1_workspace(tmp_path)
    template = storage.load_yaml_strict(ws / "cases" / f"{original_id}.yaml")
    head = template["snapshot"]["original_head_sha"]
    cases = {number: deepcopy(template) for number in (101, 102, 103, 104, 105, 106, 107)}
    cases[102]["curation"]["findings"].append(deepcopy(cases[102]["curation"]["findings"][0]))
    cases[103]["schema_version"] = "bogus"
    cases[101]["snapshot"]["base_tree_sha"] = "f" * 40
    ids = {number: schema.case_id_for(number, head) for number in cases}
    for number, doc in cases.items():
        doc["case_id"] = ids[number]
        doc["pull_request"].update(number=number, url=f"https://github.com/o/r/pull/{number}")
        storage.atomic_write_yaml(ws / "cases" / f"{ids[number]}.yaml", doc)
    manifest = storage.load_yaml_strict(ws / "benchmark.yaml")
    manifest["cases"] = [{"case_id": ids[n], "pr_number": n, "case_file": f"cases/{ids[n]}.yaml"} for n in cases]
    storage.atomic_write_yaml(ws / "benchmark.yaml", manifest)
    _write_legacy_unreplayable_case(ws, ids[104], schema_version=2, curation_state="ready")
    _write_legacy_unreplayable_case(ws, ids[105], schema_version=2, curation_state="stale")
    _write_legacy_unreplayable_case(ws, ids[106], schema_version=2, curation_state="excluded")
    invalid = {number: ws / "cases" / f"{ids[number]}.yaml" for number in (101, 102, 103, 104, 105, 106)}
    before = {number: path.read_bytes() for number, path in invalid.items()}

    with pytest.raises(SystemExit) as outcome:
        top_cli.main(["benchmark", "upgrade", str(ws)])
    assert outcome.value.code == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [f"case {ids[107]}: finding_ids_recomputed=1 changed=True"]
    errors = dict(part.split(": ", 1) for part in captured.err.split("error: ")[1:])
    assert set(errors) == {ids[n] for n in invalid}
    assert "duplicate" in errors[ids[102]]
    assert "unsupported schema_version" in errors[ids[103]]
    assert "unreplayable snapshot and curation states must match" in errors[ids[104]]
    assert "unreplayable snapshot and curation states must match" in errors[ids[105]]
    assert "unreplayable snapshot and curation states must match" in errors[ids[106]]
    assert "source trees" in errors[ids[101]]
    assert {number: path.read_bytes() for number, path in invalid.items()} == before
    valid = storage.load_yaml_strict(ws / "cases" / f"{ids[107]}.yaml")
    assert valid["schema_version"] == 2
    schema.CaseDocument.model_validate(_schema_ready(valid))


def test_migrate_imported_snapshot_preserves_sole_base_without_mirror(tmp_path: Path) -> None:
    """Missing mirror rejects a ready case but cannot block an imported record in the same batch."""
    ws, ready_id, _ = _seed_v1_workspace(tmp_path)
    ready_path = ws / "cases" / f"{ready_id}.yaml"
    raw = storage.load_yaml_strict(ready_path)
    base_tip = raw["pull_request"]["base"]["sha"]
    head = raw["snapshot"]["original_head_sha"]
    imported_id = schema.case_id_for(102, head)
    raw["case_id"] = imported_id
    raw["pull_request"].update(number=102, url="https://github.com/o/r/pull/102")
    raw["snapshot"] = {
        "status": "imported", "policy": "final_pr_head", "requested_head": "final", "original_base_sha": base_tip,
        "original_head_sha": head, "error": None,
    }
    imported_path = ws / "cases" / f"{imported_id}.yaml"
    storage.atomic_write_yaml(imported_path, raw)
    manifest = storage.load_yaml_strict(ws / "benchmark.yaml")
    manifest["cases"].append({"case_id": imported_id, "pr_number": 102, "case_file": f"cases/{imported_id}.yaml"})
    storage.atomic_write_yaml(ws / "benchmark.yaml", manifest)
    before = ready_path.read_bytes()
    shutil.rmtree(ws / "cache" / "repository.git")

    report = migrate.migrate_workspace(ws)

    assert [(case.case_id, case.finding_ids_recomputed) for case in report.cases] == [(imported_id, 1)]
    assert len(report.errors) == 1
    assert report.errors[0].startswith(f"{ready_id}:") and "provenance" in report.errors[0]
    assert ready_path.read_bytes() == before
    migrated = storage.load_yaml_strict(imported_path)
    assert migrated["snapshot"]["requested_base_sha"] == base_tip
    assert "base_resolution" not in migrated["snapshot"]


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
