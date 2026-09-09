import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _write_curated_workspace_with_sensitive_evidence(tmp_path: Path) -> Any:
    """The Task-2 curated fixture + evidence/finding bodies carrying a secret.

    Plants ``SUPER_SECRET_EVIDENCE`` in the import's ``evidence[].body`` and in
    the case doc's finding body, then corrupts the workspace (import file
    rewritten so the ledger ``import_sha256`` no longer matches) so ``validate``
    reports a failure whose diagnostics must never disclose the sentinel.
    """
    import json

    from tests.test_benchmark_workspace import _write_curated_workspace

    root = _write_curated_workspace(tmp_path, "ready")
    imp = next((root / "imports").glob("pr-*.json"))
    doc = json.loads(imp.read_text())
    doc["evidence"] = [
        {
            "source_id": "github:inline_comment:4242",
            "kind": "inline_comment",
            "database_id": 4242,
            "node_id": "DIFF_4242",
            "author": {"login": "alice", "type": "User"},
            "body": "SUPER_SECRET_EVIDENCE",
            "body_sha256": hashlib.sha256(b"SUPER_SECRET_EVIDENCE").hexdigest(),
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "submitted_at": None,
            "commit_id": None,
            "original_commit_id": None,
            "path": "feature.py",
            "line": 2,
            "original_line": 2,
            "review_id": None,
            "thread_id": None,
            "reply_to_id": None,
            "subject_type": "line",
            "side": "RIGHT",
            "start_side": None,
            "resolved": False,
            "outdated": False,
            "dismissed": False,
            "state": None,
            "is_bot": False,
            "url": "https://github.com/o/r/pull/101#discussion_r4242",
        }
    ]
    imp.write_bytes(json.dumps(doc).encode())   # ledger sha no longer matches -> corrupt
    case = next((root / "cases").glob("*.yaml"))
    text = case.read_text()
    assert "The cache key is stable across writes" in text
    case.write_text(
        text.replace(
            "The cache key is stable across writes, so stale data is served.",
            "SUPER_SECRET_EVIDENCE",
        )
    )
    return root


def _tree_sha(root: Path) -> str:
    """Deterministic sha256 over a workspace tree's file bytes (read-only check)."""
    import hashlib as _h

    digest = _h.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            digest.update(p.read_bytes())
    return digest.hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _git_states(root: Path) -> dict[str, tuple[str, bytes | None]]:
    states: dict[str, tuple[str, bytes | None]] = {}
    for git_dir in sorted(root.rglob(".git")):
        if not git_dir.is_dir():
            continue
        repo = git_dir.parent
        head_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=repo, check=False, capture_output=True, text=True
        )
        if head_result.returncode != 0:
            continue
        head = head_result.stdout
        index_path = git_dir / "index"
        index = index_path.read_bytes() if index_path.is_file() else None
        states[repo.relative_to(root).as_posix()] = (head, index)
    return states


def test_benchmark_help_lists_subcommands() -> None:
    r = subprocess.run(  # noqa: S603 - args are not user-controlled
        [sys.executable, "-m", "daydream", "benchmark", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 0 and "init" in r.stdout and "status" in r.stdout and "validate" in r.stdout


def test_benchmark_init_status_validate_roundtrip(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    r = subprocess.run(  # noqa: S603
        [
            sys.executable, "-m", "daydream", "benchmark", "init", str(ws),
            "--repo", "OWNER/REPO",
            "--reviewer-host", "api.anthropic.com",
            "--judge-host", "api.anthropic.com",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "confidential" in r.stdout  # prints privacy classification
    assert "api.anthropic.com" in r.stdout  # prints egress boundary
    assert (ws / "benchmark.yaml").exists()

    r2 = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "daydream", "benchmark", "status", str(ws)],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0 and "empty" in r2.stdout and "unresolved" in r2.stdout

    r3 = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "daydream", "benchmark", "validate", str(ws)],
        capture_output=True, text=True,
    )
    assert r3.returncode == 2  # fresh workspace: structurally valid but incomplete


def test_legacy_bench_is_rejected_not_routed() -> None:
    # The old `bench` verb is removed; it must exit non-zero with a clear error
    # instead of falling through to the review path (issue-785). Assert the
    # rejection rather than the former coexistence.
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "daydream", "bench", "--help"], capture_output=True, text=True
    )
    assert r.returncode == 2
    assert "no longer a command" in r.stderr
    assert "daydream benchmark" in r.stderr


def test_validate_diagnostics_never_disclose_evidence_bodies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from daydream.benchmark.cli import _handle_benchmark_command

    ws = _write_curated_workspace_with_sensitive_evidence(tmp_path)
    before = _tree_sha(ws)
    rc = _handle_benchmark_command(["validate", str(ws)])
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "SUPER_SECRET_EVIDENCE" not in out       # evidence bodies never printed
    assert "corrupt" in out.lower() or "ready" in out.lower()   # only labels/short strings
    assert rc == 1
    assert _tree_sha(ws) == before                  # validate is read-only: nothing written


def test_validate_exit_code_contract_preserved(tmp_path: Path) -> None:
    from daydream.benchmark.workspace import validate_workspace
    from tests.test_benchmark_workspace import (  # noqa: F401
        _write_curated_workspace,
        _write_minimal_invalid_workspace,
    )

    assert validate_workspace(_write_curated_workspace(tmp_path / "r1", "ready"))[0] == 0
    assert validate_workspace(_write_curated_workspace(tmp_path / "r2", "draft"))[0] == 2
    assert validate_workspace(_write_minimal_invalid_workspace(tmp_path / "r3"))[0] == 1
    assert validate_workspace(_write_curated_workspace(tmp_path / "r4", "ready", resolved=False))[0] == 2


def test_real_cli_reports_and_rejects_stale_task_spec_without_mutation(
    tmp_path: Path,
    fake_gh: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.metadata

    import yaml

    from daydream import cli as top_cli
    from daydream.benchmark import cli as benchmark_cli
    from daydream.benchmark.harbor.build import task_spec_digest
    from daydream.benchmark.schema import derive_finding_id
    from daydream.benchmark.storage import load_yaml_strict
    from tests.test_benchmark_curate_tui import _scripted
    from tests.test_benchmark_curation import _seed_ready_case

    parent = tmp_path / "workspace parent with spaces"
    parent.mkdir()
    ws, case_id, _head_sha = _seed_ready_case(parent, fake_gh, candidate=True)
    monkeypatch.setattr(benchmark_cli, "_is_interactive_tty", lambda: True)
    monkeypatch.setattr("builtins.input", _scripted("a", "1", "r", "y", "q"))
    with pytest.raises(SystemExit) as curated:
        top_cli.main(["benchmark", "curate", str(ws), "--case", case_id])
    assert curated.value.code == 0
    approved = load_yaml_strict(ws / "cases" / f"{case_id}.yaml")
    assert approved["curation"]["state"] == "ready"
    assert approved["curation"]["task_spec_sha256"] == task_spec_digest(approved)
    wheel = tmp_path / f"daydream-{importlib.metadata.version('daydream')}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    with pytest.raises(SystemExit) as built:
        top_cli.main(["benchmark", "build-harbor", str(ws), "--daydream-wheel", str(wheel)])
    assert built.value.code == 0
    harbor_before = _tree_bytes(ws / "harbor")
    assert harbor_before
    git_before = _git_states(parent)
    assert git_before
    manifest = (ws / "benchmark.yaml").read_bytes()
    case_path = next((ws / "cases").glob("*.yaml"))
    import_path = next((ws / "imports").glob("*.json"))
    bundle_path = next((ws / "snapshots").glob("*.bundle"))

    raw = load_yaml_strict(case_path)
    approved_digest = raw["curation"]["task_spec_sha256"]
    raw["curation"]["findings"][0]["title"] = "Approval is now stale"
    raw["curation"]["findings"][0]["severity"] = "medium"
    raw["curation"]["findings"][0]["finding_id"] = derive_finding_id(
        raw["curation"]["findings"][0], case_id=raw["case_id"]
    )
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    authored = {
        "manifest": manifest,
        "case": case_path.read_bytes(),
        "import": import_path.read_bytes(),
        "bundle": bundle_path.read_bytes(),
    }

    with pytest.raises(SystemExit) as status:
        top_cli.main(["benchmark", "status", str(ws)])
    assert status.value.code == 0
    assert "task-spec approval stale" in capsys.readouterr().out
    with pytest.raises(SystemExit) as validated:
        top_cli.main(["benchmark", "validate", str(ws)])
    assert validated.value.code == 2
    with pytest.raises(SystemExit) as rejected:
        top_cli.main(["benchmark", "build-harbor", str(ws), "--daydream-wheel", str(wheel)])
    assert rejected.value.code == 1

    assert (ws / "benchmark.yaml").read_bytes() == authored["manifest"]
    assert case_path.read_bytes() == authored["case"]
    assert import_path.read_bytes() == authored["import"]
    assert bundle_path.read_bytes() == authored["bundle"]
    retained = load_yaml_strict(case_path)
    assert retained["curation"]["state"] == "ready"
    assert retained["curation"]["task_spec_sha256"] == approved_digest
    assert _tree_bytes(ws / "harbor") == harbor_before
    assert _git_states(parent) == git_before
    assert not (ws / "cache" / "harbor-build-stage").exists()


@pytest.mark.parametrize("digest", [None, "PRIVATE_DIGEST_SENTINEL"])
def test_real_cli_rejects_corrupt_approval_digest_without_disclosure_or_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    digest: str | None,
) -> None:
    import importlib.metadata

    import yaml

    from daydream import cli as top_cli
    from daydream.benchmark.storage import load_yaml_strict
    from tests.test_benchmark_workspace import _write_curated_workspace

    ws = _write_curated_workspace(tmp_path / "malformed approval workspace", "ready")
    wheel = tmp_path / f"daydream-{importlib.metadata.version('daydream')}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    with pytest.raises(SystemExit) as built:
        top_cli.main(["benchmark", "build-harbor", str(ws), "--daydream-wheel", str(wheel)])
    assert built.value.code == 0
    assert _tree_bytes(ws / "harbor")
    capsys.readouterr()

    case_path = next((ws / "cases").glob("*.yaml"))
    raw = load_yaml_strict(case_path)
    raw["curation"]["task_spec_sha256"] = digest
    case_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    before = _tree_bytes(ws)
    git_before = _git_states(tmp_path)

    commands = [
        ["benchmark", "status", str(ws)],
        ["benchmark", "validate", str(ws)],
        ["benchmark", "build-harbor", str(ws), "--daydream-wheel", str(wheel)],
    ]
    for command in commands:
        with pytest.raises(SystemExit) as result:
            top_cli.main(command)
        assert result.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "PRIVATE_DIGEST_SENTINEL" not in output
        assert "Traceback" not in output
        assert _tree_bytes(ws) == before
        assert _git_states(tmp_path) == git_before


def test_real_cli_validate_distinguishes_recovery_corruption_without_disclosure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from daydream import cli as top_cli
    from tests.test_benchmark_workspace import _write_curated_workspace

    ws = _write_curated_workspace(tmp_path / "corrupt journal workspace", "ready")
    residue = ws / "transactions" / "PRIVATE_JOURNAL_SENTINEL"
    residue.write_bytes(b"private unknown transaction residue\n")
    before = _tree_bytes(ws)

    with pytest.raises(SystemExit) as result:
        top_cli.main(["benchmark", "validate", str(ws)])

    assert result.value.code == 1
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "workspace recovery failed" in output
    assert "invalid benchmark.yaml" not in output
    assert "PRIVATE_JOURNAL_SENTINEL" not in output
    assert "private unknown transaction residue" not in output
    assert _tree_bytes(ws) == before


def test_real_cli_calibration_treats_current_and_stale_approval_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml

    from daydream import cli as top_cli
    from daydream.benchmark.harbor import calibrate
    from daydream.benchmark.schema import derive_finding_id
    from daydream.benchmark.storage import load_yaml_strict
    from tests.test_benchmark_calibrate import _env, _scripted_http, _scripted_responses
    from tests.test_benchmark_workspace import _write_curated_workspace

    workspaces = [
        _write_curated_workspace(tmp_path / "current", "ready"),
        _write_curated_workspace(tmp_path / "stale", "ready"),
    ]
    for ws in workspaces:
        manifest = load_yaml_strict(ws / "benchmark.yaml")
        manifest["privacy"]["judge_allowed_hosts"] = ["127.0.0.1"]
        (ws / "benchmark.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    stale_case = next((workspaces[1] / "cases").glob("*.yaml"))
    stale = load_yaml_strict(stale_case)
    stale["curation"]["findings"][0]["title"] = "Changed after approval"
    stale["curation"]["findings"][0]["finding_id"] = derive_finding_id(
        stale["curation"]["findings"][0], case_id=stale["case_id"]
    )
    stale_case.write_text(yaml.safe_dump(stale, sort_keys=False))

    import httpx

    responses = _scripted_responses(calibrate._load_fixture())
    fake_http, request_counter = _scripted_http(responses)
    construction_counter = 0

    class FakeAsyncClient:
        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> Any:
            return await fake_http.post(*args, **kwargs)

    def external_http_client() -> FakeAsyncClient:
        nonlocal construction_counter
        construction_counter += 1
        return FakeAsyncClient()

    monkeypatch.setattr(httpx, "AsyncClient", external_http_client)
    for name, value in _env().items():
        monkeypatch.setenv(name, value)

    from daydream.benchmark.harbor.build import task_spec_approval

    assert task_spec_approval(load_yaml_strict(next((workspaces[0] / "cases").glob("*.yaml")))).state == "current"
    assert task_spec_approval(load_yaml_strict(stale_case)).state == "stale"
    request_totals = []
    construction_totals = []
    for ws in workspaces:
        with pytest.raises(SystemExit) as result:
            top_cli.main(["benchmark", "calibrate-judge", str(ws), "--yes"])
        assert result.value.code == 0
        assert (ws / "runtime" / "calibration-receipt.json").is_file()
        request_totals.append(request_counter[0])
        construction_totals.append(construction_counter)
    assert request_totals == [72, 144]
    assert construction_totals == [72, 144]


def test_real_cli_malformed_manifest_matrix_is_bounded_and_preserves_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import importlib.metadata

    import httpx
    import yaml

    from daydream import cli as top_cli
    from tests.test_benchmark_workspace import _write_curated_workspace

    ws = _write_curated_workspace(tmp_path / "malformed workspace", "ready")
    wheel = tmp_path / f"daydream-{importlib.metadata.version('daydream')}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    with pytest.raises(SystemExit) as built:
        top_cli.main(["benchmark", "build-harbor", str(ws), "--daydream-wheel", str(wheel)])
    assert built.value.code == 0
    harbor_before = _tree_bytes(ws / "harbor")
    assert harbor_before
    manifest_path = ws / "benchmark.yaml"
    malformed = yaml.safe_load(manifest_path.read_text())
    malformed["schema_version"] = "PRIVATE_SENTINEL"
    manifest_path.write_text(yaml.safe_dump(malformed, sort_keys=False))
    receipt = ws / "runtime" / "calibration-receipt.json"
    receipt.write_bytes(b"prior-receipt\n")
    receipt_before = receipt.read_bytes()
    http_constructions = 0

    def forbidden_http_client(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal http_constructions
        http_constructions += 1
        raise AssertionError("malformed manifest reached external HTTP client")

    monkeypatch.setattr(httpx, "AsyncClient", forbidden_http_client)

    commands = [
        ["benchmark", "status", str(ws)],
        ["benchmark", "validate", str(ws)],
        ["benchmark", "build-harbor", str(ws), "--daydream-wheel", str(wheel)],
        ["benchmark", "calibrate-judge", str(ws), "--yes"],
    ]
    for command in commands:
        with pytest.raises(SystemExit) as result:
            top_cli.main(command)
        assert result.value.code == 1
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "invalid benchmark.yaml" in output
        assert "PRIVATE_SENTINEL" not in output
    assert http_constructions == 0
    assert receipt.read_bytes() == receipt_before
    assert _tree_bytes(ws / "harbor") == harbor_before
    assert not (ws / "cache" / "harbor-build-stage").exists()


def test_stale_approval_keeps_collecting_and_curating_priority_contract() -> None:
    from daydream.benchmark.schema import derive_workspace_state

    assert derive_workspace_state(
        pull_requests=[{"import_state": "pending"}],
        cases=[{"curation_state": "stale"}],
    ) == "collecting"
    assert derive_workspace_state(
        pull_requests=[{"import_state": "fetched"}],
        cases=[{"curation_state": "stale"}, {"curation_state": "draft"}],
    ) == "curating"


