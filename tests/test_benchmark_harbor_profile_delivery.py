"""Task 10 (R10): Harbor explicit-only profile resolver mode.

The Harbor control plane supplies the candidate through a dedicated
``DAYDREAM_REVIEW_PROFILE_CANDIDATE`` var; the resolver must NEVER read the
normal-run ``DAYDREAM_REVIEW_PROFILE`` env, the operator's file config, or any
target-repo profile. No candidate -> packaged default.
"""
import asyncio

from daydream import review_profile as rp
from daydream.config_file import DaydreamFileConfig

_resolver_fixture = (
    'schema_version = 1\nname = "candidate"\n[strategies.intent]\n'
    'content = "C"\nsource = "copied: a"'
)


def test_harbor_resolver_ignores_user_env_and_repo_config(monkeypatch):
    # Malicious target config / ambient env must NOT change the Harbor candidate.
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", "/tmp/user-evil.toml")
    malicious = DaydreamFileConfig(review_profile="/tmp/repo-evil.toml")
    resolved = rp.resolve_harbor_profile(file_config=malicious)  # no candidate requested
    assert resolved.source_kind == "default"  # falls to packaged default, ignores env+repo
    assert resolved.profile.name  # the packaged default, not user/repo


def test_harbor_resolver_accepts_only_explicit_control_plane_candidate(monkeypatch, tmp_path):
    p = tmp_path / "control-plane-candidate.toml"
    p.write_text(_resolver_fixture)
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(p))
    resolved = rp.resolve_harbor_profile(env=None)  # env passed explicitly as the trusted control plane
    assert resolved.profile.name == "candidate"

# Task 11 (R11): controlled Harbor delivery -- entrypoint validation + no
# artifact on failure.
def test_entrypoint_parses_and_validates_candidate_before_runconfig(tmp_path, monkeypatch):
    from daydream.benchmark.harbor import entrypoint

    good = tmp_path / "good.toml"
    good.write_text('schema_version = 1\nname = "g"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"')
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(good))
    cfg = entrypoint.build_run_config(
        repo_dir=str(tmp_path), trajectory_path=str(tmp_path / "t.json"),
        backend="claude", model="sonnet",
    )
    assert cfg.review_profile.name == "g"  # candidate parsed+validated into RunConfig


def test_entrypoint_invalid_candidate_fails_and_writes_no_review(tmp_path, monkeypatch):
    from daydream.benchmark.harbor import entrypoint

    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 99\nname = "bad"')
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(bad))
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    artifact.parent.mkdir(parents=True)
    # main() is async and RETURNS exit code 1 on EntrypointError (it does not raise).
    rc = asyncio.run(entrypoint.main(
        monkeypatch_env={"DAYDREAM_REVIEW_CASE_ID": "case-x",
                         "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
                         "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path)}
    ))
    assert rc == 1  # agent/config error, non-zero exit
    assert not artifact.exists()  # no candidate review artifact written


def test_malicious_target_config_cannot_change_harbor_candidate(tmp_path, monkeypatch):
    from daydream.benchmark.harbor import entrypoint

    # target repo .daydream.toml tries to point at its own profile
    evil = tmp_path / ".daydream.toml"
    evil.write_text('review_profile = "/tmp/evil.toml"')
    good = tmp_path / "good.toml"
    good.write_text('schema_version = 1\nname = "g"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"')
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", str(good))
    cfg = entrypoint.build_run_config(
        repo_dir=str(tmp_path), trajectory_path=str(tmp_path / "t.json"),
        backend="claude", model="sonnet",
    )
    assert cfg.review_profile.name == "g"  # candidate wins; target config ignored


# Task 12 (R12): Harbor ledger/receipt provenance.
def test_ledger_entry_records_candidate_digest(tmp_path):
    from daydream.benchmark.harbor import run

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "harbor" / "jobs").mkdir(parents=True)
    job_dir = str((ws / "harbor" / "jobs" / "job-1").resolve())
    run.ledger_append_running(
        ws, run_id="run-1", compiled_lock_sha256="lock",
        job_dir=job_dir, mode="benchmark",
        profile_digest="abc123",
    )
    led = run._load_ledger(ws)
    entry = led["runs"][0]
    assert entry["profile_digest"] == "abc123"


def test_receipt_invalidation_inputs_include_candidate_digest():
    from daydream.benchmark.harbor import run

    inputs = run._calibration_invalidation_inputs(
        {"DAYDREAM_REVIEW_PROFILE_CANDIDATE_DIGEST": "xyz"}
    )
    assert "profile_digest" in inputs and inputs["profile_digest"] == "xyz"
