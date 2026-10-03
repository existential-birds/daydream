"""Harbor accepts only the control plane's explicit profile candidate.

Ignore normal-run environment, operator config, and target-repo profiles;
without a candidate, use the packaged default.
"""
import asyncio
from pathlib import Path
from typing import Any

import pytest

from daydream import review_profile as rp
from daydream.benchmark import cli as bc
from daydream.benchmark.harbor import calibrate, entrypoint, run, run as run_mod
from daydream.config_file import DaydreamFileConfig
from daydream.review_profile import ProfileError

_resolver_fixture = (
    'schema_version = 1\nname = "candidate"\n[strategies.intent]\n'
    'content = "C"\nsource = "copied: a"'
)


def _judge_env(**extra: str) -> dict[str, str]:
    return {"DAYDREAM_JUDGE_PROVIDER": "openai-compatible", "DAYDREAM_JUDGE_MODEL": "openrouter/test-model",
        "DAYDREAM_JUDGE_BASE_URL": "https://openrouter.ai/api/v1", **extra,
    }


def test_harbor_resolver_ignores_user_env_and_repo_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", "/tmp/user-evil.toml")
    malicious = DaydreamFileConfig(review_profile=Path("/tmp/repo-evil.toml"))
    resolved = rp.resolve_harbor_profile(file_config=malicious)  # no candidate requested
    assert resolved.source_kind == "default"  # falls to packaged default, ignores env+repo
    assert resolved.profile.name  # the packaged default, not user/repo

def test_harbor_resolver_accepts_only_explicit_control_plane_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    p = tmp_path / "control-plane-candidate.toml"
    p.write_text(_resolver_fixture)
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(p))
    resolved = rp.resolve_harbor_profile(env={"DAYDREAM_REVIEW_PROFILE_CANDIDATE": str(p)})
    assert resolved.profile.name == "candidate"

def test_entrypoint_parses_and_validates_candidate_before_runconfig(tmp_path: Path, ) -> None:
    good = tmp_path / "good.toml"
    good.write_text('schema_version = 1\nname = "g"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"')
    cfg = entrypoint.build_run_config(repo_dir=str(tmp_path), trajectory_path=str(tmp_path / "t.json"),
        backend="pi", model="deepseek/deepseek-v4-flash-0731", profile_candidate=str(good),
    )
    assert cfg.review_profile is not None
    assert cfg.review_profile.name == "g"  # candidate parsed+validated into RunConfig

def test_entrypoint_invalid_candidate_fails_and_writes_no_review(tmp_path: Path,) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 99\nname = "bad"')
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    artifact.parent.mkdir(parents=True)
    rc = asyncio.run(entrypoint.main({
        "DAYDREAM_REVIEW_CASE_ID": "case-x", "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
        "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path), "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api", "DAYDREAM_REVIEW_PROFILE_CANDIDATE": str(bad),
    }))
    assert rc == 1  # agent/config error, non-zero exit
    assert not artifact.exists()  # no candidate review artifact written

def test_malicious_target_config_cannot_change_harbor_candidate(tmp_path: Path,) -> None:
    evil = tmp_path / ".daydream.toml"
    evil.write_text('review_profile = "/tmp/evil.toml"')
    good = tmp_path / "good.toml"
    good.write_text('schema_version = 1\nname = "g"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"')
    cfg = entrypoint.build_run_config(repo_dir=str(tmp_path), trajectory_path=str(tmp_path / "t.json"),
        backend="pi", model="deepseek/deepseek-v4-flash-0731", profile_candidate=str(good),
    )
    assert cfg.review_profile is not None
    assert cfg.review_profile.name == "g"  # candidate wins; target config ignored

def test_ledger_entry_records_candidate_digest(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "harbor" / "jobs").mkdir(parents=True)
    job_dir = str((ws / "harbor" / "jobs" / "job-1").resolve())
    run.ledger_append_running(
        ws, run_id="run-1", compiled_lock_sha256="lock", job_dir=job_dir, mode="benchmark", profile_digest="abc123",
    )
    led = run._load_ledger(ws)
    entry = led["runs"][0]
    assert entry["profile_digest"] == "abc123"



# Candidate-scoped receipts must satisfy oracle preflight; defaults retain the legacy shape.
def test_calibrate_invalidation_inputs_folds_candidate_digest() -> None:
    sr = calibrate._load_judge_template()
    inputs = calibrate._invalidation_inputs(_judge_env(DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST="abc"), pairs=[], sr=sr)
    assert inputs["profile_digest"] == "abc"
    # Without a candidate digest -> legacy contract stays byte-stable.
    legacy = calibrate._invalidation_inputs(_judge_env(), pairs=[], sr=sr)
    assert "profile_digest" not in legacy

# The supervisor records provenance before the separate container entrypoint starts.
def test_benchmark_run_threads_candidate_digest_to_supervisor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", raising=False)
    assert bc._candidate_profile_digest() is None

    cand = tmp_path / "candidate.toml"
    cand.write_text(
        'schema_version = 1\nname = "candidate"\n[strategies.intent]\n'
        'content = "C"\nsource = "copied: a"'
    )
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(cand))
    digest = bc._candidate_profile_digest()
    assert digest and isinstance(digest, str) and len(digest) == 64  # sha256


    captured = {}

    def fake_run_run(workspace: Any, *, oracle: Any=False, yes: Any=False, env: Any=None, **kw: Any) -> int:
        captured["env"] = env
        return 0

    monkeypatch.setattr(run_mod, "run_run", fake_run_run)
    rc = bc._handle_benchmark_run(type("Args", (), {"dir": str(tmp_path), "oracle": False, "yes": True})())
    assert rc == 0
    assert captured["env"]["DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST"] == digest

def test_benchmark_run_invalid_candidate_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 99\nname = "bad"')
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(bad))
    try:
        bc._candidate_profile_digest()
        raise AssertionError("expected ProfileError for invalid candidate")
    except ProfileError:
        pass
