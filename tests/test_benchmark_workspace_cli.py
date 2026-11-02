import importlib.metadata
import stat
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from daydream import cli as top_cli
from daydream.benchmark.schema import BenchmarkManifest, derive_workspace_state
from daydream.benchmark.storage import load_yaml_strict
from daydream.benchmark.workspace import workspace_status
from tests.test_benchmark_workspace import (
    _write_curated_workspace,
)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


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


def test_benchmark_init_status_validate_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ws = tmp_path / "ws"
    with pytest.raises(SystemExit) as initialized:
        top_cli.main(["benchmark", "init", str(ws), "--repo", "OWNER/REPO",
            "--reviewer-host", "https://user:pass@API.Anthropic.COM:443/path", "--judge-host", "api.anthropic.com",
        ])
    assert initialized.value.code == 0
    assert capsys.readouterr().out.splitlines() == [
        "classification: confidential",
        "egress boundary: api.anthropic.com api.anthropic.com",
    ]
    manifest = BenchmarkManifest.model_validate(load_yaml_strict(ws / "benchmark.yaml"))
    assert manifest.source.model_dump() == {
        "provider": "github", "hostname": "github.com", "repository": "OWNER/REPO",
        "repository_id": None, "visibility": "unresolved",
    }
    assert manifest.privacy.reviewer_allowed_hosts == manifest.privacy.judge_allowed_hosts == ["api.anthropic.com"]
    assert manifest.pull_requests == [] and manifest.cases == []
    assert stat.S_IMODE(ws.stat().st_mode) == 0o700
    for name in ("imports", "cases", "snapshots", "transactions", "runtime", "cache"):
        assert (ws / name).is_dir() and stat.S_IMODE((ws / name).stat().st_mode) == 0o700
    assert not (ws / "harbor").exists()
    assert (ws / ".gitignore").read_bytes() == b"*\n!.gitignore\n"
    assert stat.S_IMODE((ws / "benchmark.yaml").stat().st_mode) == 0o600
    assert stat.S_IMODE((ws / ".gitignore").stat().st_mode) == 0o600

    assert _cli_exit(["benchmark", "status", str(ws)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "workspace state: empty", "repository identity: unresolved",
        "repository identity/access verification: not yet run", "ledger entries: 0",
    ]
    status = workspace_status(ws)
    assert status.workspace_state == "empty" and not status.repository_identity_resolved
    assert status.ledger.pull_requests == [] and status.case_snapshots == []
    assert status.source == manifest.source

    assert _cli_exit(["benchmark", "validate", str(ws)]) == 2  # fresh workspace: structurally valid but incomplete
    capsys.readouterr()
    assert _cli_exit(["benchmark", "--help"]) == 0
    usage = capsys.readouterr().out
    assert all(name in usage for name in ("init", "status", "validate", "import-prs")), usage
    assert _cli_exit(["bench", "--help"]) == 2  # the legacy name is rejected, never routed
    legacy = capsys.readouterr().err
    assert "no longer a command" in legacy and "daydream benchmark" in legacy
    assert derive_workspace_state(pull_requests=[{"import_state": "pending"}], cases=[{"curation_state": "stale"}],
    ) == "collecting"
    assert derive_workspace_state(
        pull_requests=[{"import_state": "fetched"}], cases=[{"curation_state": "stale"}, {"curation_state": "draft"}],
    ) == "curating"
    assert derive_workspace_state(pull_requests=[{"import_state": "fetched"}],
                                  cases=[{"curation_state": "stale"}, {"curation_state": "ready"}]) == "stale"


def _cli_exit(argv: list[str]) -> Any:
    with pytest.raises(SystemExit) as result:
        top_cli.main(argv)
    return result.value.code

def _rejects_corrupt_approval_digest_without_disclosure_or_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], digest: str | None = None,
) -> None:
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

    commands = [["benchmark", "status", str(ws)], ["benchmark", "validate", str(ws)],
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

def _validate_distinguishes_recovery_corruption_without_disclosure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
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

def _malformed_manifest_matrix_is_bounded_and_preserves_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    commands = [["benchmark", "status", str(ws)], ["benchmark", "validate", str(ws)],
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


def test_real_cli_rejects_corrupt_workspaces_without_disclosure_or_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("approval", "recovery"):
        (tmp_path / name).mkdir()
    _rejects_corrupt_approval_digest_without_disclosure_or_mutation(tmp_path / "approval", capsys)
    _validate_distinguishes_recovery_corruption_without_disclosure(tmp_path / "recovery", capsys)
    (tmp_path / "manifest").mkdir()
    _malformed_manifest_matrix_is_bounded_and_preserves_outputs(tmp_path / "manifest", capsys, monkeypatch)
