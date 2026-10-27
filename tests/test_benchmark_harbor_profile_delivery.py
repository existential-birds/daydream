"""Harbor accepts only the control plane's explicit profile candidate.

Ignore normal-run environment, operator config, and target-repo profiles;
without a candidate, use the packaged default.
"""
import asyncio
from pathlib import Path

import pytest

from daydream import review_profile as rp
from daydream.benchmark import cli as bc
from daydream.benchmark.harbor import calibrate, entrypoint
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

# Candidate-scoped receipts must satisfy oracle preflight; defaults retain the legacy shape.
def test_calibrate_invalidation_inputs_folds_candidate_digest() -> None:
    sr = calibrate._load_judge_template()
    inputs = calibrate._invalidation_inputs(_judge_env(DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST="abc"), pairs=[], sr=sr)
    assert inputs["profile_digest"] == "abc"
    # Without a candidate digest -> legacy contract stays byte-stable.
    legacy = calibrate._invalidation_inputs(_judge_env(), pairs=[], sr=sr)
    assert "profile_digest" not in legacy

# The supervisor records provenance before the separate container entrypoint starts.

def test_benchmark_run_invalid_candidate_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 99\nname = "bad"')
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(bad))
    try:
        bc._candidate_profile_digest()
        raise AssertionError("expected ProfileError for invalid candidate")
    except ProfileError:
        pass
