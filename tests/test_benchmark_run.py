"""Hermetic suite for the `daydream benchmark run` supervisor (issue #781).

Task 1: parser + dispatch routing.
Task 2: fail-closed preflight checks.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.cli import _build_benchmark_parser, _handle_benchmark_command
from daydream.benchmark.harbor import package as _pkg


def _docker_ok() -> _pkg.DockerNetworkPolicyCapability:
    """Preflight ``docker_ok`` probe reporting Docker networking as supported."""
    return _pkg.DockerNetworkPolicyCapability(supported=True)


@pytest.fixture(autouse=True)
def _stub_harbor_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the run-supervisor suite hermetic without a Harbor install.

    The supervisor's production path only reads ``importlib.metadata.version
    ("harbor")`` / ``package.resolve_harbor()`` after ``_preflight`` has already
    confirmed Harbor is present. These unit tests exercise the receipt and gate
    logic directly, so they must not require the optional ``[benchmark]`` extra
    (Harbor is only installed when that extra is enabled — e.g. the review VMs
    — but the CI ``check`` job installs base deps only). Stub the Harbor
    environment so the suite stays hermetic.
    """
    import importlib.metadata
    import sys

    from daydream.benchmark.harbor import package as _pkg

    real_version = importlib.metadata.version

    def _version(dist: str) -> Any:
        return "0.22.0" if dist == "harbor" else real_version(dist)

    monkeypatch.setattr(importlib.metadata, "version", _version)
    monkeypatch.setattr(
        _pkg, "resolve_harbor", lambda: str(Path(sys.executable).parent / "harbor")
    )


# ---------------------------------------------------------------------------
# shared hermetic fixtures (Tasks 2-8)
# ---------------------------------------------------------------------------


def _ws(tmp_path: Path, **privacy: Any) -> Any:
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    (ws / "harbor").mkdir()
    _seed_compiled_lock(ws)
    (ws / "harbor" / "harbor-job.yaml").write_text("jobs_dir: jobs\n")
    (ws / "harbor" / "harbor-oracle.yaml").write_text("jobs_dir: jobs\n")
    p = {"classification": "confidential", "reviewer_data": "source_snapshot",
         "reviewer_allowed_hosts": ["review.example"], "judge_data": "finding_text_and_location_only",
         "judge_allowed_hosts": ["127.0.0.1"], "archive": "disabled", "uploads": "disabled"}
    p.update(privacy)
    _seed_compiled_task(
        ws, reviewer=p["reviewer_allowed_hosts"], judge=p["judge_allowed_hosts"]
    )
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "6c38dc0a-5f5a-4b73-bf36-9a2eb390f63b",
        "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com", "repository": "OWNER/REPO",
                   "repository_id": None, "visibility": "unresolved"},
        "privacy": p, "pull_requests": [], "cases": []}))
    return ws


def _env(**over: Any) -> Any:
    base = {"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "m",
            "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9", "DAYDREAM_JUDGE_API_KEY": "k",
            "DAYDREAM_REVIEW_MODEL": "rm", "DAYDREAM_REVIEW_BASE_URL": "http://review.example"}
    base.update(over)
    return base


_WHEEL = {"distribution": "daydream", "version": "0.1.0", "sha256": "c" * 64}


def _seed_compiled_lock(ws: Path, wheel: Any=_WHEEL) -> None:
    """Write a compiled lock with the ``daydream`` wheel block logged."""
    lock = {"schema_version": 1, "cases": {"case-a": {"key": "case-a"}}, "files": {},
            "daydream": wheel}
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(lock))


def _seed_compiled_task(ws: Path, *, reviewer: Any, judge: Any) -> None:
    """Write the compiled task.toml egress policy Harbor will enforce.

    Mirrors what ``compile_workspace`` threads into ``harbor/case-*/task.toml``
    (reviewer -> ``[agent].allowed_hosts``, judge -> ``[verifier.environment].allowed_hosts``)
    so the preflight, which now reads the compiled tree (not raw benchmark.yaml),
    has the same artifact Harbor executes.
    """
    case = ws / "harbor" / "case-a"
    case.mkdir(parents=True, exist_ok=True)
    (case / "task.toml").write_text(
        "[agent]\n"
        f"allowed_hosts = {json.dumps(reviewer)}\n"
        "\n"
        "[verifier.environment]\n"
        f"allowed_hosts = {json.dumps(judge)}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Task 1: parser + dispatch
# ---------------------------------------------------------------------------


def test_benchmark_parser_has_run_subcommand() -> None:
    parser = _build_benchmark_parser()
    args = parser.parse_args(["run", "/ws", "--oracle", "--yes"])
    assert args.subcommand == "run"
    assert args.dir == Path("/ws")
    assert args.oracle is True and args.yes is True
    plain = parser.parse_args(["run", "/ws"])
    assert plain.oracle is False and plain.yes is False


def test_handle_benchmark_run_routes_to_supervisor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import daydream.benchmark.harbor.run as run_mod

    captured: dict[str, Any] = {}

    def fake_run_run(ws: Any, *, oracle: Any, yes: Any, env: Any) -> int:
        captured["ws"] = Path(ws)
        captured["oracle"] = oracle
        captured["yes"] = yes
        captured["env"] = env
        return 0

    monkeypatch.setattr(run_mod, "run_run", fake_run_run)
    code = _handle_benchmark_command(["run", str(tmp_path), "--yes"])
    assert code == 0
    assert captured["ws"] == tmp_path
    assert captured["yes"] is True and captured["oracle"] is False
    assert "DAYDREAM_REVIEW_MODEL" in captured["env"]  # env threaded through


# ---------------------------------------------------------------------------
# Task 2: fail-closed preflight checks
# ---------------------------------------------------------------------------


def test_preflight_ok_when_all_checks_pass(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    errs = run_mod._preflight(ws, oracle=True, env=_env(), docker_ok=_docker_ok)
    assert errs == []


def test_preflight_requires_explicit_judge_endpoint(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(
        _ws(tmp_path),
        oracle=True,
        env=_env(DAYDREAM_JUDGE_API_KEY="sk-or-abc", DAYDREAM_JUDGE_BASE_URL=None),
        docker_ok=_docker_ok,
    )

    assert any("missing DAYDREAM_JUDGE_BASE_URL" in error for error in errs)


def test_preflight_blocks_judge_host_outside_allowlist(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True,
                              env=_env(DAYDREAM_JUDGE_BASE_URL="http://evil.example"), docker_ok=_docker_ok)
    assert any("judge host" in e and "evil.example" in e for e in errs)


def test_preflight_blocks_reviewer_host_outside_allowlist(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True,
                              env=_env(DAYDREAM_REVIEW_BASE_URL="http://other.example"), docker_ok=_docker_ok)
    assert any("reviewer host" in e and "other.example" in e for e in errs)


def test_preflight_enforces_compiled_task_policy_not_raw_manifest(tmp_path: Path) -> None:
    """Preflight reads the compiled task.toml Harbor enforces, not raw yaml.

    A host ``benchmark.yaml`` permits but the compiled tree rejects (the tree
    predates the manifest edit) must still block the run, because the stale
    compiled task.toml is the artifact Harbor actually executes.
    """
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path, judge_allowed_hosts=["127.0.0.1", "stale.example"])
    # simulate a compiled tree produced before the manifest added stale.example
    _seed_compiled_task(ws, reviewer=["review.example"], judge=["127.0.0.1"])
    errs = run_mod._preflight(ws, oracle=True,
                              env=_env(DAYDREAM_JUDGE_BASE_URL="http://stale.example"), docker_ok=_docker_ok)
    assert any("judge host" in e and "stale.example" in e for e in errs)


def test_preflight_blocks_uploads_enabled(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path, uploads="enabled"), oracle=True,
                              env=_env(), docker_ok=_docker_ok)
    assert any("upload" in e for e in errs)


def test_preflight_blocks_unsupported_docker_allowlist(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True, env=_env(), docker_ok=lambda: False)
    assert any("Docker allowlist" in e for e in errs)


def test_preflight_uses_live_docker_capability_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production preflight must fail with the actual sidecar probe reason."""
    from types import SimpleNamespace

    import daydream.benchmark.harbor.run as run_mod

    monkeypatch.setattr(
        "daydream.benchmark.harbor.package.docker_network_policy_capability",
        lambda: SimpleNamespace(
            supported=False,
            reason="Harbor egress sidecar rejected nftables fib rules",
        ),
        raising=False,
    )

    errs = run_mod._preflight(_ws(tmp_path), oracle=True, env=_env())

    assert any("Docker allowlist" in e for e in errs)
    assert any("rejected nftables fib rules" in e for e in errs)


def test_preflight_ok_for_oracle_without_calibration_receipt(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    errs = run_mod._preflight(_ws(tmp_path), oracle=True, env=_env(), docker_ok=_docker_ok)
    assert errs == []          # Oracle preflight no longer requires a calibration receipt


# ---------------------------------------------------------------------------
# Task 3: pre-run spend summary
# ---------------------------------------------------------------------------


def test_pre_run_summary_lists_all_required_fields(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    text = run_mod._pre_run_summary(ws, env=_env())
    for needle in (
        "task count", "reviewer model", "judge provider", "judge model",
        "judge host", "attempts", "concurrency", "timeouts",
        "oracle pair", "benchmark judge pair", "time-bounded",
    ):
        assert needle.lower() in text.lower(), f"summary missing {needle!r}"
    assert "rm" in text        # reviewer model threaded from env
    assert "127.0.0.1" in text # judge host threaded from env


# ---------------------------------------------------------------------------
# Task 4: runtime/harbor.json cleanup ledger
# ---------------------------------------------------------------------------


def test_ledger_append_running_and_mark_complete(tmp_path: Path) -> None:
    import stat

    import daydream.benchmark.harbor.run as run_mod

    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    run_id = "00000000-0000-0000-0000-000000000001"
    job_dir = str((ws / "harbor" / "jobs" / run_id).resolve())
    run_mod.ledger_append_running(ws, run_id=run_id, compiled_lock_sha256="a" * 64,
                                  job_dir=job_dir)
    path = ws / "runtime" / "harbor.json"
    doc = json.loads(path.read_text())
    assert doc["schema_version"] == 1
    assert doc["runs"][0]["run_id"] == run_id
    assert doc["runs"][0]["state"] == "running"
    assert doc["runs"][0]["job_dir"] == job_dir
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    run_mod.ledger_mark(ws, run_id, state="complete", environments=[{
        "trial_name": "case-abc__1", "environment_id": "env-1", "backend": "docker",
        "image_id": "sha256:abc", "image_tags": ["tag"], "removed": False}])
    doc = json.loads(path.read_text())
    assert doc["runs"][0]["state"] == "complete"
    assert doc["runs"][0]["environments"][0]["image_id"] == "sha256:abc"


def test_ledger_rejects_non_contained_job_dir(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    with pytest.raises(run_mod.RunError):
        run_mod.ledger_append_running(ws, run_id="x", compiled_lock_sha256="a" * 64,
                                      job_dir=str(tmp_path / "outside"))


# ---------------------------------------------------------------------------
# Task 5: Harbor result parsing + Oracle receipt
# ---------------------------------------------------------------------------


def _score(reward: Any) -> dict[str, Any]:
    return {"reward": reward, "verifier_error": 0, "gold_count": 1, "candidate_count": 1}


def test_oracle_parse_success_writes_receipt(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    (ws / "runtime" / "calibration-receipt.json").write_text(json.dumps({"inputs": {"cal": 1}}))
    job_dir = ws / "harbor" / "jobs" / "run-1"
    verifier = job_dir / "case-abc" / "verifier"
    verifier.mkdir(parents=True)
    # spike-confirmed layout: reward.json lives under <trial>/verifier/
    (verifier / "reward.json").write_text(json.dumps(_score(1.0)))
    ok, _ = run_mod._parse_job_results(job_dir)
    assert ok is True
    code = run_mod._write_oracle_receipt(
        ws, job_dir=job_dir, compiled_lock_sha256="a" * 64, env=_env(),
    )
    assert code == 0
    receipt = json.loads((ws / "harbor" / "oracle-receipt.json").read_text())
    for key in ("compiled_lock_sha256", "harbor_version", "judge_provider", "judge_model",
                "judge_host", "verifier_template_sha256", "threshold", "attempts",
                "result_dir", "timestamp"):
        assert key in receipt, f"receipt missing {key}"
    assert receipt["compiled_lock_sha256"] == "a" * 64


def test_oracle_no_receipt_on_reward_below_one(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    job_dir = ws / "harbor" / "jobs" / "run-1"
    verifier = job_dir / "case-abc" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(json.dumps(_score(0.8)))
    ok, _ = run_mod._parse_job_results(job_dir)
    assert ok is False
    code = run_mod._write_oracle_receipt(
        ws, job_dir=job_dir, compiled_lock_sha256="a" * 64, env=_env(),
    )
    assert code == 1
    assert not (ws / "harbor" / "oracle-receipt.json").exists()


def test_oracle_no_receipt_on_unscored_task(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    job_dir = ws / "harbor" / "jobs" / "run-1"
    verifier = job_dir / "case-abc" / "verifier"
    verifier.mkdir(parents=True)
    # infra error path writes reward-details.json only -> unscored, blocks
    (verifier / "reward-details.json").write_text("{}")
    ok, _ = run_mod._parse_job_results(job_dir)
    assert ok is False


# ---------------------------------------------------------------------------
# Task 6: default-run gate
# ---------------------------------------------------------------------------


def test_gate_blocks_on_compiled_lock_mismatch(tmp_path: Path) -> None:
    import hashlib

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    lock = {"schema_version": 1, "cases": {}, "daydream": _WHEEL}
    lock_sha = hashlib.sha256(json.dumps(lock).encode()).hexdigest()
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(lock))
    job_dir = ws / "harbor" / "jobs" / "run-1"
    verifier = job_dir / "case-abc" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(json.dumps(_score(1.0)))
    assert run_mod._write_oracle_receipt(
        ws, job_dir=job_dir, compiled_lock_sha256=lock_sha, env=_env(),
    ) == 0
    # now the current compiled lock digest differs from the receipt's (wheel
    # block kept so the daydream provenance read stays authoritative)
    changed = {"schema_version": 1, "cases": {}, "touched": True, "daydream": _WHEEL}
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(changed))
    reason = run_mod._default_run_gate(
        ws, env=_env(), compiled_lock_sha256=hashlib.sha256(
            json.dumps(changed).encode()
        ).hexdigest(),
    )
    assert reason is not None
    assert "compiled lock" in reason


def test_gate_blocks_when_oracle_receipt_missing(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    reason = run_mod._default_run_gate(
        _ws(tmp_path), env=_env(), compiled_lock_sha256="a" * 64,
    )
    assert reason is not None
    assert "no matching oracle receipt" in reason


def test_gate_passes_when_inputs_match(tmp_path: Path) -> None:
    import hashlib

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    lock = {"schema_version": 1, "cases": {}, "daydream": _WHEEL}
    lock_sha = hashlib.sha256(json.dumps(lock).encode()).hexdigest()
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(lock))
    job_dir = ws / "harbor" / "jobs" / "run-1"
    verifier = job_dir / "case-abc" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(json.dumps(_score(1.0)))
    assert run_mod._write_oracle_receipt(
        ws, job_dir=job_dir, compiled_lock_sha256=lock_sha, env=_env(),
    ) == 0
    reason = run_mod._default_run_gate(
        ws, env=_env(), compiled_lock_sha256=lock_sha,
    )
    assert reason is None


# ---------------------------------------------------------------------------
# Task 7: run_run orchestrator + unrelated-CWD acceptance
# ---------------------------------------------------------------------------


def test_run_oracle_writes_receipt_and_running_to_complete(tmp_path: Path) -> None:
    import stat

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    captures: dict[str, Any] = {}

    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        captures["cwd"] = str(cwd)
        captures["args"] = cmd
        captures["env"] = env
        # run_run assigns a fresh uuid4 job dir and records it in the ledger
        # before spawning; write reward evidence into that recorded dir.
        ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
        job_dir = Path(ledger["runs"][0]["job_dir"])
        verifier = job_dir / "case-abc" / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.json").write_text(json.dumps(
            {"reward": 1.0, "verifier_error": 0, "gold_count": 1, "candidate_count": 1}))
        return {"returncode": 0}

    code = run_mod.run_run(
        ws, oracle=True, yes=True, env=_env(), spawn=spawn, docker_ok=_docker_ok,
    )
    assert code == 0
    assert (ws / "harbor" / "oracle-receipt.json").exists()
    assert stat.S_IMODE((ws / "harbor" / "oracle-receipt.json").stat().st_mode) == 0o600
    assert str(captures["cwd"]) == str((ws / "harbor").resolve())
    assert "HARBOR_TELEMETRY" in captures["env"] and captures["env"]["HARBOR_TELEMETRY"] == "off"
    assert "--upload" not in captures["args"] and "--publish" not in captures["args"]
    assert any("harbor-oracle.yaml" in str(a) for a in captures["args"])  # selects oracle config
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert captures["args"][-2:] == ["--job-name", ledger["runs"][0]["run_id"]]
    assert Path(ledger["runs"][0]["job_dir"]).name == ledger["runs"][0]["run_id"]
    assert ledger["runs"][0]["state"] == "complete"


def test_run_oracle_from_unrelated_cwd_resolves_harbor_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    captured: dict[str, Any] = {}

    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        captured["cwd"] = str(cwd)
        ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
        job_dir = Path(ledger["runs"][0]["job_dir"])
        verifier = job_dir / "case-abc" / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.json").write_text(json.dumps(
            {"reward": 1.0, "verifier_error": 0, "gold_count": 1, "candidate_count": 1}))
        return {"returncode": 0}

    code = run_mod.run_run(
        ws, oracle=True, yes=True, env=_env(), spawn=spawn, docker_ok=_docker_ok,
    )
    assert code == 0
    assert captured["cwd"] == str((ws / "harbor").resolve())


def test_run_refuses_without_yes_and_no_confirm(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    code = run_mod.run_run(
        ws, oracle=True, yes=False, env=_env(), spawn=None, docker_ok=_docker_ok,
        confirm=lambda _: False,
    )
    assert code == 1
    assert not (ws / "runtime" / "harbor.json").exists()  # no running entry on block


# ---------------------------------------------------------------------------
# Task 8: full acceptance matrix
# ---------------------------------------------------------------------------


def test_oracle_fails_writes_no_receipt_and_ledger_cleanup_pending(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
        job_dir = Path(ledger["runs"][0]["job_dir"])
        verifier = job_dir / "case-abc" / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.json").write_text(json.dumps(
            {"reward": 0.5, "verifier_error": 0, "gold_count": 1, "candidate_count": 2}))
        return {"returncode": 0}

    code = run_mod.run_run(
        ws, oracle=True, yes=True, env=_env(), spawn=spawn, docker_ok=_docker_ok,
    )
    assert code == 1
    assert not (ws / "harbor" / "oracle-receipt.json").exists()
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["state"] == "cleanup_pending"


def test_default_run_propagates_harbor_exit_code(tmp_path: Path) -> None:
    import hashlib

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    lock = {"schema_version": 1, "cases": {}, "daydream": _WHEEL}
    lock_sha = hashlib.sha256(json.dumps(lock).encode()).hexdigest()
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(lock))
    job_dir = ws / "harbor" / "jobs" / "x"
    verifier = job_dir / "case-abc" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(json.dumps(_score(1.0)))
    # seed a matching oracle receipt the gate will accept
    assert run_mod._write_oracle_receipt(
        ws, job_dir=job_dir, compiled_lock_sha256=lock_sha, env=_env(),
    ) == 0

    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        return {"returncode": 3}

    code = run_mod.run_run(
        ws, oracle=False, yes=True, env=_env(), spawn=spawn, docker_ok=_docker_ok,
    )
    assert code == 3  # Harbor's own exit code preserved


def test_default_gate_blocks_before_any_harbor_call(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    called = []

    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        called.append(cmd)
        return {"returncode": 0}

    code = run_mod.run_run(
        ws, oracle=False, yes=True, env=_env(), spawn=spawn, docker_ok=_docker_ok,
    )
    assert code == 1            # no matching receipt -> gate blocks
    assert called == []         # Harbor never spawned, no reviewer call
    assert not (ws / "runtime" / "harbor.json").exists()  # blocked run leaves no running entry

# ---------------------------------------------------------------------------
# Task 10: run supervisor persists trial environments into the cleanup ledger
# ---------------------------------------------------------------------------


def test_run_persists_trial_environments_to_ledger(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
        job = Path(ledger["runs"][0]["job_dir"])
        (job / "case-abc" / "verifier").mkdir(parents=True)
        (job / "case-abc" / "verifier" / "reward.json").write_text(
            json.dumps({"reward": 1.0, "verifier_error": 0,
                        "gold_count": 1, "candidate_count": 1}))
        return {"returncode": 0}

    code = run_mod.run_run(ws, oracle=True, yes=True, env=_env(),
                           spawn=spawn, docker_ok=_docker_ok)
    assert code == 0
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    envs = ledger["runs"][0]["environments"]
    assert len(envs) == 1                                  # was [] before the fix
    assert envs[0]["trial_name"] == "case-abc"
    assert envs[0]["backend"] == "docker" and envs[0]["image_id"]  # exact ref present
    assert envs[0]["removed"] is False


def test_run_failed_path_persists_environments_cleanup_pending(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
        job = Path(ledger["runs"][0]["job_dir"])
        (job / "case-abc" / "verifier").mkdir(parents=True)
        (job / "case-abc" / "verifier" / "reward.json").write_text(
            json.dumps({"reward": 0.5, "verifier_error": 0,
                        "gold_count": 1, "candidate_count": 2}))
        return {"returncode": 0}

    code = run_mod.run_run(ws, oracle=True, yes=True, env=_env(),
                           spawn=spawn, docker_ok=_docker_ok)
    assert code == 1
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["state"] == "cleanup_pending"
    assert ledger["runs"][0]["environments"][0]["image_id"]   # not [] on failure either


def test_parse_job_results_records_env_when_reward_missing(tmp_path: Path) -> None:
    """An aborted run (trial present but no score evidence) must still record
    the environment it spawned, so ``clean --jobs`` can address the Docker
    image it left behind instead of orphaning it by deleting the job dir.
    """
    import daydream.benchmark.harbor.run as run_mod

    job_dir = tmp_path / "jobs" / "r"
    trial = job_dir / "case-abc"
    env_root = trial / "environment"
    env_root.mkdir(parents=True)
    (env_root / "env.txt").write_text("x", encoding="utf-8")
    (trial / "task.toml").write_text("[environment]\n", encoding="utf-8")
    ok, envs = run_mod._parse_job_results(job_dir)
    assert ok is False
    assert envs and envs[0]["image_id"].startswith("hb__")   # was [] before the fix
    assert envs[0]["backend"] == "docker"


# ---------------------------------------------------------------------------
# Task 10: provenance — reviewer effort and Daydream wheel digest
# ---------------------------------------------------------------------------


def test_current_state_mapping_includes_effort_and_wheel_digest(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    _seed_compiled_lock(ws)   # benchmark.lock.json with daydream block
    m = run_mod._current_state_mapping(
        workspace=ws, compiled_lock_sha256="a" * 64, env=_env())
    assert m["daydream_wheel_sha256"] == "c" * 64
    assert m["daydream_version"] == "0.1.0"
    assert "reviewer_effort" in m


def test_ledger_records_reviewer_effort_when_present(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    run_id = "run-1"
    run_mod.ledger_append_running(ws, run_id=run_id, compiled_lock_sha256="a" * 64,
                                  job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()),
                                  profile_digest="d" * 64, reviewer_effort="high")
    run_mod.ledger_mark(ws, run_id, state="complete")
    doc = run_mod._load_ledger(ws)
    assert doc["runs"][0]["profile_digest"] == "d" * 64
    assert doc["runs"][0]["reviewer_effort"] == "high"


# ---------------------------------------------------------------------------
# Task 2 (issue #909): Oracle mapping decoupled from unverified calibration
# ---------------------------------------------------------------------------


def test_default_run_accepts_old_receipt_with_legacy_calibration_field(tmp_path: Path) -> None:
    import hashlib

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    lock_sha = hashlib.sha256(
        (ws / "harbor" / "benchmark.lock.json").read_bytes()).hexdigest()
    receipt = run_mod._current_state_mapping(
        ws, compiled_lock_sha256=lock_sha, env=_env())
    receipt["calibration_receipt_sha256"] = "0" * 64  # legacy extra field
    (ws / "harbor" / "oracle-receipt.json").write_text(json.dumps(receipt))
    reason = run_mod._default_run_gate(
        ws, env=_env(), compiled_lock_sha256=lock_sha)
    assert reason is None  # legacy field is ignored, not compared


def test_oracle_receipt_has_no_calibration_state(tmp_path: Path) -> None:
    import hashlib

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    lock_sha = hashlib.sha256(
        (ws / "harbor" / "benchmark.lock.json").read_bytes()).hexdigest()
    mapping = run_mod._current_state_mapping(
        ws, compiled_lock_sha256=lock_sha, env=_env())
    assert "calibration_receipt_sha256" not in mapping


def test_oracle_writes_receipt_without_calibration_file(tmp_path: Path) -> None:
    import hashlib

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    job_dir = ws / "harbor" / "jobs" / "j1"
    trial = job_dir / "t1" / "verifier"
    trial.mkdir(parents=True)
    (trial / "reward.json").write_text(json.dumps(_score(1.0)))  # gold reproduced
    lock_sha = hashlib.sha256(
        (ws / "harbor" / "benchmark.lock.json").read_bytes()).hexdigest()
    code = run_mod._write_oracle_receipt(ws, job_dir=job_dir,
                                         compiled_lock_sha256=lock_sha, env=_env())
    assert code == 0
    receipt = json.loads((ws / "harbor" / "oracle-receipt.json").read_text())
    assert "calibration_receipt_sha256" not in receipt
    assert receipt["compiled_lock_sha256"] == lock_sha


def test_default_run_still_blocks_without_oracle_receipt(tmp_path: Path) -> None:
    import hashlib

    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    lock_sha = hashlib.sha256(
        (ws / "harbor" / "benchmark.lock.json").read_bytes()).hexdigest()
    reason = run_mod._default_run_gate(ws, env=_env(), compiled_lock_sha256=lock_sha)
    assert reason is not None and "no matching oracle receipt" in reason


def test_ledger_reviewer_effort_defaults_none_when_omitted(tmp_path: Path) -> None:
    import daydream.benchmark.harbor.run as run_mod

    ws = _ws(tmp_path)
    run_id = "run-1"
    run_mod.ledger_append_running(ws, run_id=run_id, compiled_lock_sha256="a" * 64,
                                  job_dir=str((ws / "harbor" / "jobs" / run_id).resolve()))
    run_mod.ledger_mark(ws, run_id, state="complete")
    doc = run_mod._load_ledger(ws)
    # Issue #888: an omitted effort stays None on the entry (byte-stable for
    # legacy callers); the objective reader surfaces it as None, never 0/"".
    assert doc["runs"][0]["reviewer_effort"] is None
