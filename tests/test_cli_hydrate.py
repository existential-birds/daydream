"""CLI wiring tests for `daydream corpus hydrate-hub` (#982 M1/M2/M17)."""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import pytest

from daydream import cli
from daydream.archive import hydrate
from daydream.archive.hydrate_client import FakeHub
from tests.fixtures.training.build_hub_snapshot import SNAPSHOT_REVISION, build_snapshot


def test_hydrate_hub_requires_explicit_args(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli._handle_hydrate_hub_command([])
    assert rc == 1
    assert "--source-repo" in capsys.readouterr().out


def test_hydrate_hub_rejects_moving_branch_without_optin(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "main",
         "--destination-repo", "org/ds", "--stage-dir", "/tmp/x"])
    assert rc == 1
    assert "exploratory" in capsys.readouterr().out


def test_hydrate_hub_missing_token_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", "/tmp/x"])
    assert rc == 1


def test_help_lists_hydrate_hub(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["corpus", "hydrate-hub", "--help"])
    assert "hydrate-hub" in capsys.readouterr().out


def test_success_path_drives_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    calls: list[Any] = []

    class FakeSummary:
        curation_id = "cur-" + "0" * 16
        output_commit_sha = "b" * 40
        verified = True
        dry_run_discovered = 1
        dry_run_admitted = 1
        dry_run_rejected = 0
        dry_run_incomplete_manifests: tuple[str, ...] = ()
        verify_admitted = 1
        license_admission: dict[str, int] = {}

    def fake_run(config: Any) -> FakeSummary:
        calls.append(config)
        return FakeSummary()

    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr(cli, "_run_hydrate_hub", fake_run, raising=False)
    policy = tmp_path / "license-policy.json"
    policy.write_text('{"policy_version": "1", "spdx_decisions": {}}')
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", str(tmp_path),
         "--license-policy", str(policy)])
    assert rc == 0
    assert calls and calls[0].source_revision == "a" * 40


def test_cli_wires_license_policy_into_hydration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--license-policy/--allow-copyleft reach HydrateHubConfig, and the
    previously unreachable license admission summary prints (issue #1080)."""
    calls: list[Any] = []

    class FakeSummary:
        curation_id = "cur-" + "0" * 16
        output_commit_sha = "b" * 40
        verified = True
        dry_run_discovered = 2
        dry_run_admitted = 1
        dry_run_rejected = 1
        dry_run_incomplete_manifests: tuple[str, ...] = ()
        verify_admitted = 1
        license_admission = {
            "admitted": 1, "c5_excluded": 1,
            "c8_copyleft_unopted": 0, "license_evidence_missing": 0,
        }

    def fake_run(config: Any) -> FakeSummary:
        calls.append(config)
        return FakeSummary()

    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr(cli, "_run_hydrate_hub", fake_run, raising=False)
    policy = tmp_path / "license-policy.json"
    policy.write_text('{"policy_version": "1", "spdx_decisions": {}}')
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", str(tmp_path),
         "--license-policy", str(policy),
         "--allow-copyleft", "Owner/Gpl-Repo"])
    assert rc == 0
    assert calls and calls[0].license_policy_path == str(policy)
    assert calls[0].allow_copyleft == frozenset({"owner/gpl-repo"})
    output = " ".join(capsys.readouterr().out.split())
    assert "license admission: admitted 1; c5-excluded 1" in output


def test_success_path_surfaces_incomplete_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeSummary:
        curation_id = "cur-" + "0" * 16
        output_commit_sha = "b" * 40
        verified = True
        dry_run_discovered = 2
        dry_run_admitted = 1
        dry_run_rejected = 0
        dry_run_incomplete_manifests = ("sess-a (missing trajectory.json)",)
        verify_admitted = 1
        license_admission: dict[str, int] = {}

    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr(cli, "_run_hydrate_hub", lambda _config: FakeSummary())
    policy = tmp_path / "license-policy.json"
    policy.write_text('{"policy_version": "1", "spdx_decisions": {}}')
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", str(tmp_path),
         "--license-policy", str(policy)])
    assert rc == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "hydration yield reduced" in output
    assert "sess-a (missing trajectory.json)" in output


def test_dry_run_reports_discovery_accounting_without_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = build_snapshot()
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)

    rc = cli._handle_hydrate_hub_command(
        [
            "--source-repo",
            "org/private-ds",
            "--source-revision",
            SNAPSHOT_REVISION,
            "--destination-repo",
            "org/private-ds",
            "--stage-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "discovered 3 candidate(s) of 3 run-shaped manifest(s)" in output
    assert "admitted 3" in output
    assert "rejected 0 batch(es)" in output
    assert "accounted 3 candidate(s)" in output
    assert "yield reduced" not in output
    assert hub.uploaded_paths == []


def test_dry_run_surfaces_incomplete_manifests_with_reduced_yield(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    complete = {
        "manifest.json": b'{"session_id": "complete"}',
        "trajectory.json": b'{"messages": []}',
    }
    incomplete = {"manifest.json": b'{"session_id": "incomplete"}'}
    hub = FakeHub(
        repo_id="org/private-ds",
        private=True,
        files={
            "sess-complete/manifest.json": complete["manifest.json"],
            "sess-complete/trajectory.json": complete["trajectory.json"],
            "sess-incomplete/manifest.json": incomplete["manifest.json"],
        },
    )
    hub.commit_revision(SNAPSHOT_REVISION)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)

    rc = cli._handle_hydrate_hub_command(
        [
            "--source-repo",
            "org/private-ds",
            "--source-revision",
            SNAPSHOT_REVISION,
            "--destination-repo",
            "org/private-ds",
            "--stage-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "discovered 1 candidate(s) of 2 run-shaped manifest(s)" in output
    assert "accounted 1 candidate(s)" in output
    assert "dry-run yield reduced" in output
    assert "sess-incomplete (missing trajectory.json)" in output
    assert hub.uploaded_paths == []


def test_dry_run_fails_closed_on_run_shaped_zero_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = FakeHub(
        repo_id="org/private-ds",
        private=True,
        files={"incomplete/manifest.json": b'{"session_id": "incomplete"}'},
    )
    hub.commit_revision(SNAPSHOT_REVISION)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)

    rc = cli._handle_hydrate_hub_command(
        [
            "--source-repo",
            "org/private-ds",
            "--source-revision",
            SNAPSHOT_REVISION,
            "--destination-repo",
            "org/private-ds",
            "--stage-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 1
    output = capsys.readouterr().out
    assert "zero candidates" in output
    assert "trajectory.json" in output
    assert hub.uploaded_paths == []


def test_hydrate_hub_refuses_non_dry_run_without_policy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """A non-dry hydrate-hub publication requires --license-policy and must
    refuse before any Hub access; the dry-run path still works without one."""
    monkeypatch.setenv("HF_TOKEN", "t")
    argv = [
        "--source-repo",
        "org/ds",
        "--source-revision",
        "a" * 40,
        "--destination-repo",
        "org/ds",
        "--stage-dir",
        str(tmp_path / "stage"),
    ]
    rc = cli._handle_hydrate_hub_command(argv)
    assert rc == 1
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "license policy" in out.lower()

    # The dry-run path still works without a policy (planning affordance): it
    # must not refuse with the policy message (it fails later, at Hub access).
    rc = cli._handle_hydrate_hub_command([*argv, "--dry-run"])
    output = capsys.readouterr().out + capsys.readouterr().err
    assert rc != 1 or "license policy" not in output.lower()


def test_production_policy_file_loads_and_rejects_copyleft() -> None:
    """The checked-in production SPDX policy validates (M10 discipline) and
    fail-closes: missing evidence -> reject, GPL without opt-in -> reject,
    MIT -> admit, exact opt-in -> admit the named repo only."""
    from daydream.training.corpus_v2.license import load_license_policy, resolve_repo_decision

    policy_path = pathlib.Path("daydream/training/schema/license-policy-production.json")
    policy, digest = load_license_policy(policy_path)
    assert len(digest) == 64
    assert resolve_repo_decision("acme/widget", None, policy, frozenset()).reason_code == (
        "license_evidence_missing"
    )
    gpl = resolve_repo_decision("acme/widget", {"spdx_id": "GPL-3.0-only"}, policy, frozenset())
    assert gpl.reason_code == "c8_copyleft_unopted"
    assert (
        resolve_repo_decision("acme/widget", {"spdx_id": "MIT"}, policy, frozenset()).status
        == "admitted"
    )
    opted = resolve_repo_decision(
        "acme/widget", {"spdx_id": "GPL-3.0-only"}, policy, frozenset({"acme/widget"})
    )
    assert opted.status == "admitted"


# --- Issue #1094 Task 8: per-repo auditable dry-run report -------------------

SEED_THREE_REPO: tuple[str, str, str] = ("acme/widget", "acme/widget", "ghost/nope")


class _FakeLicenseResolver:
    """Minimal RepoLicenseResolver: MIT for opted-in slugs, None otherwise."""

    def __init__(self, mit_repos: tuple[str, ...]) -> None:
        self._mit = {slug.casefold() for slug in mit_repos}

    def resolve(self, repo_slug: str, repo_commit: str | None) -> Any:
        from daydream.archive.license_enrich import EnrichedEvidence

        if repo_slug.casefold() not in self._mit:
            return None
        commit = "c" * 40
        return EnrichedEvidence(
            spdx_id="MIT", source=f"github:{repo_slug}@{commit}", repo_commit=commit,
        )


def _seed_three_repo_hub() -> FakeHub:
    """Three-session hub over three repo outcomes: 2x MIT repo, 1x unresolvable."""
    from tests.fixtures.training.build_hub_snapshot import (
        _snapshot_manifest,
        _snapshot_trajectory,
    )

    files: dict[str, bytes] = {}
    for session_id, repo_slug in zip(
        ("acme-run-1", "acme-run-2", "ghost-run-1"), SEED_THREE_REPO, strict=True
    ):
        manifest = _snapshot_manifest(session_id, repo_slug, "beagle-python:review-python", ("accepted",))
        files[f"{session_id}/manifest.json"] = json.dumps(manifest.to_dict(), indent=2).encode()
        files[f"{session_id}/trajectory.json"] = json.dumps(_snapshot_trajectory(session_id), indent=2).encode()
    hub = FakeHub(repo_id="org/private-ds", private=True, files=files)
    hub.commit_revision(SNAPSHOT_REVISION)
    return hub


def run_dry_run_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    *,
    hub: FakeHub,
    resolver: Any,
) -> tuple[int, dict[str, Any]]:
    """Run the real CLI dry-run path and parse the printed report.

    Returns ``(rc, report)`` where ``report["per_repository"]`` maps repo slug
    to the four license-admission buckets parsed from the printed per-repo
    lines, and ``report["discovered"]`` is the discovered-candidate count.
    """
    from daydream.archive import license_enrich

    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)
    monkeypatch.setattr(license_enrich, "_make_license_resolver", lambda: resolver)
    policy = tmp_path / "license-policy.json"
    policy.write_text('{"policy_version": "test", "spdx_decisions": {"MIT": "accepted"}}')
    rc = cli._handle_hydrate_hub_command(
        [
            "--source-repo", "org/private-ds",
            "--source-revision", SNAPSHOT_REVISION,
            "--destination-repo", "org/private-ds",
            "--stage-dir", str(tmp_path / "stage"),
            "--license-policy", str(policy),
            "--dry-run",
        ]
    )
    out = " ".join(capsys.readouterr().out.split())
    per_repo: dict[str, dict[str, int]] = {}
    pattern = (
        r"by repo: ([\w./\-]+) -> admitted (\d+), c5-excluded (\d+), "
        r"copyleft-unopted (\d+), evidence-missing (\d+)"
    )
    for match in re.finditer(pattern, out):
        per_repo[match.group(1)] = {
            "admitted": int(match.group(2)),
            "c5_excluded": int(match.group(3)),
            "c8_copyleft_unopted": int(match.group(4)),
            "license_evidence_missing": int(match.group(5)),
        }
    discovered_match = re.search(r"discovered (\d+) candidate", out)
    discovered = int(discovered_match.group(1)) if discovered_match else 0
    return rc, {"per_repository": per_repo, "discovered": discovered}


def test_dry_run_reports_per_repo_decision_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Dry run over a stage with three repos and mixed outcomes.
    rc, report = run_dry_run_capture(
        monkeypatch, tmp_path, capsys,
        hub=_seed_three_repo_hub(),
        resolver=_FakeLicenseResolver(mit_repos=("acme/widget",)),
    )
    assert rc == 0
    per_repo = report["per_repository"]
    assert per_repo["acme/widget"]["admitted"] == 2
    assert per_repo["ghost/nope"]["license_evidence_missing"] == 1
    # Full accounting: every discovered record lands in exactly one code bucket.
    assert sum(sum(v.values()) for v in per_repo.values()) == report["discovered"]


def test_runbook_hydrate_hub_invocation_carries_policy_and_dry_run_gate() -> None:
    text = pathlib.Path("docs/runbooks/annotation-final-publish.md").read_text()
    # The production hydrate-hub line carries the policy + opt-in args...
    hydrate_line = next(
        line
        for line in text.splitlines()
        if "corpus hydrate-hub" in line and "--source-repo org/run-bundles" in line
    )
    assert "--license-policy" in hydrate_line and "license-policy-production.json" in hydrate_line
    assert "--allow-copyleft" in text  # opt-in argument documented
    # ...and the non-dry publication is gated behind a completed real dry-run:
    # a real --dry-run hydrate-hub step appears before the unpinned publication
    # command, and states that its record accounting gates the next step.
    lines = text.splitlines()
    dry_pos = min(
        i for i, line in enumerate(lines) if "hydrate-hub" in line and "--dry-run" in line
    )
    publish_pos = min(
        i
        for i, line in enumerate(lines)
        if "hydrate-hub" in line and "--dry-run" not in line and "--source-repo org/run-bundles" in line
    )
    assert dry_pos < publish_pos
    gate_text = "\n".join(lines[dry_pos:publish_pos])
    assert "gate" in gate_text.lower() and "discovered" in gate_text.lower()
