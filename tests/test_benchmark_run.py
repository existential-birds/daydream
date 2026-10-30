"""Hermetic suite for the `daydream benchmark run` supervisor (issue #781).

"""
import hashlib
import importlib.metadata
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream import review_profile as rp
from daydream.benchmark.cli import _handle_benchmark_command
from daydream.benchmark.harbor import package as _pkg, run as run_mod


def _docker_ok() -> _pkg.DockerNetworkPolicyCapability:
    """Preflight ``docker_ok`` probe reporting Docker networking as supported."""
    return _pkg.DockerNetworkPolicyCapability(supported=True)


@pytest.fixture(autouse=True)
def _stub_harbor_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Supply Harbor discovery for receipt/gate tests without requiring the optional extra."""
    real_version = importlib.metadata.version

    def _version(dist: str) -> Any:
        return "0.22.0" if dist == "harbor" else real_version(dist)

    monkeypatch.setattr(importlib.metadata, "version", _version)
    monkeypatch.setattr(_pkg, "resolve_harbor", lambda: str(Path(sys.executable).parent / "harbor"))

# shared hermetic fixtures (Tasks 2-8)


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
    _seed_compiled_task(ws, reviewer=p["reviewer_allowed_hosts"], judge=p["judge_allowed_hosts"])
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
    lock = {"schema_version": 1, "cases": {"case-a": {"key": "case-a"}}, "files": {}, "daydream": wheel}
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(lock))


def _seed_compiled_task(ws: Path, *, reviewer: Any, judge: Any) -> None:
    """Write reviewer and verifier egress rules into the task.toml that Harbor executes."""
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


def test_preflight_requires_explicit_judge_endpoint(tmp_path: Path) -> None:
    errs = run_mod._preflight(
        _ws(tmp_path), oracle=True, env=_env(DAYDREAM_JUDGE_API_KEY="sk-or-abc", DAYDREAM_JUDGE_BASE_URL=None),
        docker_ok=_docker_ok,
    )

    assert any("missing DAYDREAM_JUDGE_BASE_URL" in error for error in errs)

def test_preflight_claude_cli_judge_needs_no_base_url(tmp_path: Path) -> None:
    ws = _ws(tmp_path, judge_allowed_hosts=["api.anthropic.com"])
    errs = run_mod._preflight(
        ws, oracle=True, env=_env(DAYDREAM_JUDGE_PROVIDER="claude-cli", DAYDREAM_JUDGE_BASE_URL=None),
        docker_ok=_docker_ok,
    )
    assert not any("cannot resolve judge host" in e for e in errs)
    assert errs == []

def test_spend_summary_prints_claude_cli_judge_host(tmp_path: Path) -> None:
    text = run_mod._pre_run_summary(_ws(tmp_path), env=_env(DAYDREAM_JUDGE_PROVIDER="claude-cli"))
    # exact-host match avoids CodeQL py/incomplete-url-substring-sanitization
    host_lines = [ln for ln in text.splitlines() if ln.strip().startswith("judge host:")]
    assert len(host_lines) == 1
    assert host_lines[0].split(":", 1)[1].strip() == "api.anthropic.com"

def test_preflight_reports_each_unsafe_host_privacy_and_runtime_setting(tmp_path: Path) -> None:
    errs = run_mod._preflight(
        _ws(tmp_path, uploads="enabled"), oracle=True,
        env=_env(DAYDREAM_JUDGE_BASE_URL="http://evil.example", DAYDREAM_REVIEW_BASE_URL="http://other.example"),
        docker_ok=lambda: _pkg.DockerNetworkPolicyCapability(supported=False),
    )
    assert set(errs) == {
        "judge host 'evil.example' is not in the compiled judge allowed_hosts policy (['127.0.0.1'])",
        "reviewer host 'other.example' is not in the compiled reviewer allowed_hosts policy (['review.example'])",
        "privacy uploads must be disabled before a Harbor run (got 'enabled')",
        "Docker allowlist is unsupported on the selected runtime; refusing to fall back to public networking",
    }


def test_preflight_enforces_compiled_task_policy_not_raw_manifest(tmp_path: Path) -> None:
    """Preflight reads the compiled task.toml Harbor enforces, not raw yaml.

    A host ``benchmark.yaml`` permits but the compiled tree rejects (the tree
    predates the manifest edit) must still block the run, because the stale
    compiled task.toml is the artifact Harbor actually executes.
    """

    ws = _ws(tmp_path, judge_allowed_hosts=["127.0.0.1", "stale.example"])
    # simulate a compiled tree produced before the manifest added stale.example
    _seed_compiled_task(ws, reviewer=["review.example"], judge=["127.0.0.1"])
    errs = run_mod._preflight(ws, oracle=True,
                              env=_env(DAYDREAM_JUDGE_BASE_URL="http://stale.example"), docker_ok=_docker_ok)
    assert any("judge host" in e and "stale.example" in e for e in errs)


def test_preflight_uses_live_docker_capability_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Production preflight must fail with the actual sidecar probe reason."""

    monkeypatch.setattr("daydream.benchmark.harbor.package.docker_network_policy_capability",
        lambda: SimpleNamespace(supported=False, reason="Harbor egress sidecar rejected nftables fib rules"),
        raising=False,
    )

    errs = run_mod._preflight(_ws(tmp_path), oracle=True, env=_env())

    assert any("Docker allowlist" in e for e in errs)
    assert any("rejected nftables fib rules" in e for e in errs)


def test_ledger_rejects_non_contained_job_dir(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "runtime").mkdir(parents=True)
    with pytest.raises(run_mod.RunError):
        run_mod.ledger_append_running(ws, run_id="x", compiled_lock_sha256="a" * 64, job_dir=str(tmp_path / "outside"))


def _score(reward: Any, *, candidate_count: int = 1) -> dict[str, Any]:
    return {"reward": reward, "verifier_error": 0, "gold_count": 1, "candidate_count": candidate_count}


def _compiled_lock_sha(ws: Path) -> str:
    """Digest of the compiled ``benchmark.lock.json`` a run was gated against."""
    return hashlib.sha256((ws / "harbor" / "benchmark.lock.json").read_bytes()).hexdigest()


def _reward_spawn(ws: Path, *, reward: float = 1.0, candidate_count: int = 1, capture: dict[str, Any] | None = None,
    returncode: int = 0,
) -> Any:
    """Write fake verifier scores under the job directory recorded before spawn.

    Optionally capture the supervisor's command, environment, and working directory.
    """
    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        if capture is not None:
            capture["cwd"] = str(cwd)
            capture["args"] = cmd
            capture["env"] = env
        ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
        running = next(row for row in ledger["runs"] if row["run_id"] == cmd[-1])
        assert running["state"] == "running"
        assert stat.S_IMODE((ws / "runtime/harbor.json").stat().st_mode) == 0o600
        trial = Path(running["job_dir"]) / "case-abc"
        (trial / "environment").mkdir(parents=True)
        (trial / "environment/env.txt").write_text("trial environment\n")
        (trial / "task.toml").write_text("[environment]\n")
        verifier = trial / "verifier"
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.json").write_text(json.dumps(_score(reward, candidate_count=candidate_count)))
        return {"returncode": returncode}

    return spawn


def _seed_verifier_reward(ws: Path, *, job_dir_name: str = "run-1", reward: float = 1.0, trial_name: str = "case-abc",
) -> Path:
    """Write the scored ``reward.json`` a real trial leaves under its verifier."""
    job_dir = ws / "harbor" / "jobs" / job_dir_name
    verifier = job_dir / trial_name / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "reward.json").write_text(json.dumps(_score(reward)))
    return job_dir


def _seed_passing_oracle_receipt(
    ws: Path, *, job_dir_name: str = "run-1", env: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    """Seed a matching compiled lock plus the receipt the gate accepts."""

    lock = {"schema_version": 1, "cases": {}, "daydream": _WHEEL}
    lock_sha = hashlib.sha256(json.dumps(lock).encode()).hexdigest()
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(lock))
    job_dir = _seed_verifier_reward(ws, job_dir_name=job_dir_name)
    assert run_mod._write_oracle_receipt(
        ws, job_dir=job_dir, compiled_lock_sha256=lock_sha, env=_env() if env is None else env,
    ) == 0
    return job_dir, lock_sha


def test_oracle_no_receipt_on_reward_below_one(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    job_dir = _seed_verifier_reward(ws, reward=0.8)
    ok, _ = run_mod._parse_job_results(job_dir)
    assert ok is False
    code = run_mod._write_oracle_receipt(ws, job_dir=job_dir, compiled_lock_sha256="a" * 64, env=_env())
    assert code == 1
    assert not (ws / "harbor" / "oracle-receipt.json").exists()

def test_oracle_no_receipt_on_unscored_task(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    job_dir = ws / "harbor" / "jobs" / "run-1"
    verifier = job_dir / "case-abc" / "verifier"
    verifier.mkdir(parents=True)
    # infra error path writes reward-details.json only -> unscored, blocks
    (verifier / "reward-details.json").write_text("{}")
    ok, _ = run_mod._parse_job_results(job_dir)
    assert ok is False

def test_gate_blocks_on_compiled_lock_mismatch(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    _seed_passing_oracle_receipt(ws)
    # now the current compiled lock digest differs from the receipt's (wheel
    # block kept so the daydream provenance read stays authoritative)
    changed = {"schema_version": 1, "cases": {}, "touched": True, "daydream": _WHEEL}
    (ws / "harbor" / "benchmark.lock.json").write_text(json.dumps(changed))
    reason = run_mod._default_run_gate(
        ws, env=_env(), compiled_lock_sha256=hashlib.sha256(json.dumps(changed).encode()).hexdigest(),
    )
    assert reason is not None
    assert "compiled lock" in reason


def test_cli_oracle_then_confirmed_benchmark_persists_trial_and_receipt_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A real CLI Oracle receipt gates a later paid run and retains the exact tested profile."""
    ws = _ws(tmp_path)
    candidate = tmp_path / "candidate.toml"
    candidate.write_text(
        'schema_version = 1\nname = "candidate"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"'
    )
    digest = rp.resolve_harbor_profile(env={"DAYDREAM_REVIEW_PROFILE_CANDIDATE": str(candidate)}).digest
    assert digest is not None
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(candidate))
    for name, value in _env(DAYDREAM_REVIEW_EFFORT="high").items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DAYDREAM_REVIEW_BACKEND", raising=False)
    captures: dict[str, Any] = {}
    monkeypatch.setattr(run_mod, "_default_spawn", _reward_spawn(ws, capture=captures))
    monkeypatch.setattr(run_mod, "_default_docker_ok", _docker_ok)
    confirmations: list[str] = []

    def decline(message: str) -> bool:
        confirmations.append(message)
        return False

    monkeypatch.setattr(run_mod, "_default_confirm", decline)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    assert not (ws / "runtime/calibration-receipt.json").exists()
    assert _handle_benchmark_command(["run", str(ws), "--oracle", "--yes"]) == 0
    assert confirmations == []
    receipt_path = ws / "harbor/oracle-receipt.json"
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    receipt = json.loads(receipt_path.read_text())
    assert receipt["compiled_lock_sha256"] == _compiled_lock_sha(ws)
    assert receipt["harbor_version"] == "0.22" and receipt["judge_provider"] == "openai-compatible"
    assert receipt["judge_model"] == "m" and receipt["judge_host"] == "127.0.0.1"
    assert receipt["daydream_version"] == "0.1.0" and receipt["daydream_wheel_sha256"] == "c" * 64
    assert receipt["profile_digest"] == digest and receipt["reviewer_effort"] == "high"
    assert {"verifier_template_sha256", "threshold", "attempts", "result_dir", "timestamp"} <= receipt.keys()
    assert "calibration_receipt_sha256" not in receipt
    assert captures["cwd"] == str((ws / "harbor").resolve())
    assert captures["args"][3] == str((ws / "harbor/harbor-oracle.yaml").resolve())
    assert captures["env"]["HARBOR_TELEMETRY"] == "off"
    assert "--upload" not in captures["args"] and "--publish" not in captures["args"]
    ledger_path = ws / "runtime/harbor.json"
    oracle = json.loads(ledger_path.read_text())["runs"][0]
    assert captures["args"][-2:] == ["--job-name", oracle["run_id"]]
    assert Path(oracle["job_dir"]).name == oracle["run_id"]
    assert (oracle["mode"], oracle["state"], oracle["profile_digest"], oracle["reviewer_effort"]) == (
        "oracle", "complete", digest, "high",
    )
    assert len(oracle["environments"]) == 1
    assert oracle["environments"][0]["trial_name"] == "case-abc"
    assert oracle["environments"][0]["backend"] == "docker" and oracle["environments"][0]["removed"] is False
    image_id = "hb__" + hashlib.sha256(b"task.toml[environment]\nenv.txttrial environment\n").hexdigest()
    assert oracle["environments"][0]["image_id"] == image_id
    before = ledger_path.read_bytes()
    assert _handle_benchmark_command(["run", str(ws)]) == 1
    assert confirmations == ["Refusing unconfirmed the paid benchmark Harbor run"]
    assert ledger_path.read_bytes() == before
    receipt["calibration_receipt_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    assert _handle_benchmark_command(["run", str(ws), "--yes"]) == 0
    paid = json.loads(ledger_path.read_text())["runs"][1]
    assert (paid["mode"], paid["state"], paid["profile_digest"], paid["reviewer_effort"]) == (
        "benchmark", "complete", digest, "high",
    )
    assert paid["environments"][0]["image_id"] == image_id
    assert captures["args"][3] == str((ws / "harbor/harbor-job.yaml").resolve())
    out = capsys.readouterr().out.lower()
    assert all(field in out for field in (
        "task count", "reviewer model", "judge provider", "judge model", "judge host", "attempts",
        "concurrency", "timeouts", "oracle pair", "benchmark judge pair", "time-bounded", "rm", "127.0.0.1",
    ))


def test_oracle_fails_writes_no_receipt_and_ledger_cleanup_pending(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    spawn = _reward_spawn(ws, reward=0.5, candidate_count=2)

    code = run_mod.run_run(
        ws, oracle=True, yes=True, env=_env(DAYDREAM_REVIEW_MODEL=None), spawn=spawn, docker_ok=_docker_ok,
    )
    assert code == 1
    assert not (ws / "harbor" / "oracle-receipt.json").exists()
    ledger = json.loads((ws / "runtime" / "harbor.json").read_text())
    assert ledger["runs"][0]["state"] == "cleanup_pending"
    assert ledger["runs"][0]["profile_digest"] is None and ledger["runs"][0]["reviewer_effort"] is None
    assert ledger["runs"][0]["environments"][0]["trial_name"] == "case-abc"
    assert ledger["runs"][0]["environments"][0]["image_id"].startswith("hb__")

@pytest.mark.parametrize("model", [None, "", " \t ", " rm "])
def test_default_run_requires_model_before_spend_and_preserves_harbor_exit_code(
    tmp_path: Path, model: str | None, capsys: pytest.CaptureFixture[str],
) -> None:
    """A valid receipt cannot allow a blank reviewer selection to start Harbor."""
    ws = _ws(tmp_path)
    environment = _env(DAYDREAM_REVIEW_MODEL=model)
    if model is None:
        environment.pop("DAYDREAM_REVIEW_MODEL")
    _seed_passing_oracle_receipt(ws, job_dir_name="x", env=environment)
    calls: list[dict[str, Any]] = []

    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        calls.append(env)
        return {"returncode": 3}

    code = run_mod.run_run(ws, oracle=False, yes=True, env=environment, spawn=spawn, docker_ok=_docker_ok)
    ledger_path = ws / "runtime/harbor.json"
    if model is None or not model.strip():
        assert code == 1
        assert "DAYDREAM_REVIEW_MODEL" in capsys.readouterr().err
        assert calls == []
        assert not ledger_path.exists()
    else:
        assert code == 3  # Harbor's own exit code and exact operator selection survive.
        assert calls[0]["DAYDREAM_REVIEW_MODEL"] == " rm "
        assert json.loads(ledger_path.read_text())["runs"][0]["reviewer_model"] == " rm "

def test_default_gate_blocks_before_any_harbor_call(tmp_path: Path) -> None:
    ws = _ws(tmp_path)
    called = []

    def spawn(cmd: Any, *, cwd: Any, env: Any) -> dict[str, Any]:
        called.append(cmd)
        return {"returncode": 0}

    code = run_mod.run_run(ws, oracle=False, yes=True, env=_env(), spawn=spawn, docker_ok=_docker_ok)
    assert code == 1            # no matching receipt -> gate blocks
    assert called == []         # Harbor never spawned, no reviewer call
    assert not (ws / "runtime" / "harbor.json").exists()  # blocked run leaves no running entry


def test_parse_job_results_records_env_when_reward_missing(tmp_path: Path) -> None:
    """An aborted run (trial present but no score evidence) must still record
    the environment it spawned, so ``clean --jobs`` can address the Docker
    image it left behind instead of orphaning it by deleting the job dir.
    """

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


# Oracle mapping accepts the legacy calibration field without using it.
