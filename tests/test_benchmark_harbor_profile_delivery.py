"""Task 10 (R10): Harbor explicit-only profile resolver mode.

The Harbor control plane supplies the candidate through a dedicated
``DAYDREAM_REVIEW_PROFILE_CANDIDATE`` var; the resolver must NEVER read the
normal-run ``DAYDREAM_REVIEW_PROFILE`` env, the operator's file config, or any
target-repo profile. No candidate -> packaged default.
"""
import pytest

from daydream import review_profile as rp
from daydream.config_file import DaydreamFileConfig


def test_harbor_resolver_ignores_user_env_and_repo_config(monkeypatch):
    # Malicious target config / ambient env must NOT change the Harbor candidate.
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", "/tmp/user-evil.toml")
    malicious = DaydreamFileConfig(review_profile="/tmp/repo-evil.toml")
    resolved = rp.resolve_harbor_profile(file_config=malicious)  # no candidate requested
    assert resolved.source_kind == "default"  # falls to packaged default, ignores env+repo
    assert resolved.profile.name  # the packaged default, not user/repo


def test_harbor_resolver_accepts_only_explicit_control_plane_candidate(monkeypatch):
    p = "/tmp/control-plane-candidate.toml"
    open(p, "w").write('schema_version = 1\nname = "candidate"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"')
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE_CANDIDATE", p)
    resolved = rp.resolve_harbor_profile(env=None)  # env passed explicitly as the trusted control plane
    assert resolved.profile.name == "candidate"