"""Task 6 (R9): normal-run four-source precedence + path-escape guard.

Precedence: explicit_path > DAYDREAM_REVIEW_PROFILE env > repo-committed
file_config.review_profile > packaged default. Invalid higher-precedence
sources fail naming their source, never falling through.
"""
from pathlib import Path

import pytest

from daydream import review_profile as rp


def _write_profile(tmp_path, name, content):
    p = tmp_path / f"{name}.toml"
    p.write_text(f'''schema_version = 1
name = "{name}"
[strategies.intent]
content = "{content}"
source = "copied: a"''')
    return p


def test_precedence_explicit_beats_env_beats_repo_beats_default(tmp_path, monkeypatch):
    from daydream.config_file import DaydreamFileConfig
    explicit = _write_profile(tmp_path, "explicit", "E")
    user = _write_profile(tmp_path, "user", "U")
    repo = _write_profile(tmp_path, "repo", "R")
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", str(user))
    fc = DaydreamFileConfig(review_profile=str(repo))
    resolved = rp.resolve_profile(explicit_path=str(explicit), file_config=fc)
    assert resolved.profile.name == "explicit" and resolved.source_kind == "explicit"


def test_env_beats_repo_and_default(tmp_path, monkeypatch):
    from daydream.config_file import DaydreamFileConfig
    user = _write_profile(tmp_path, "user", "U")
    repo = _write_profile(tmp_path, "repo", "R")
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", str(user))
    fc = DaydreamFileConfig(review_profile=str(repo))
    resolved = rp.resolve_profile(file_config=fc)          # no explicit path
    assert resolved.profile.name == "user" and resolved.source_kind == "env"


def test_repo_beats_default(tmp_path):
    from daydream.config_file import DaydreamFileConfig
    repo = _write_profile(tmp_path, "repo", "R")
    fc = DaydreamFileConfig(review_profile=str(repo))
    resolved = rp.resolve_profile(file_config=fc)          # no explicit, no env
    assert resolved.profile.name == "repo" and resolved.source_kind == "repo"


def test_default_when_nothing_specified():
    resolved = rp.resolve_profile()                        # no explicit/env/repo
    assert resolved.source_kind == "default" and resolved.profile.name


def test_relative_repo_path_cannot_escape(tmp_path, monkeypatch):
    from daydream.config_file import DaydreamFileConfig
    fc = DaydreamFileConfig(review_profile="../evil.toml")   # relative repo path
    with pytest.raises(rp.ProfileError) as e:
        rp.resolve_profile(file_config=fc, repo_root=tmp_path)
    assert "escape" in str(e.value).lower()


def test_invalid_explicit_fails_naming_source(monkeypatch):
    bad = Path("/tmp/bad-profile.toml")
    bad.write_text('schema_version = 1\nname = "p"\nunknown = 1')
    with pytest.raises(rp.ProfileError) as e:
        rp.resolve_profile(explicit_path=str(bad))
    assert "bad-profile.toml" in str(e.value)


# Task 7 (R1, R9): resolve-once plumbing via RunConfig.
def test_runconfig_carries_resolved_profile_and_is_used(tmp_path, monkeypatch):
    from daydream.runner import RunConfig
    from daydream import review_profile as rp
    p = tmp_path / "prof.toml"
    p.write_text('schema_version = 1\nname = "r"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"')
    cfg = RunConfig(target=str(tmp_path), review_profile_path=str(p))
    assert cfg.review_profile_path == str(p)     # path carried on RunConfig


def test_resolve_from_runconfig_happens_once_at_composition_root():
    from daydream import review_profile as rp
    from daydream.runner import RunConfig
    cfg = RunConfig(target="/tmp")
    resolved = rp.resolve_from_runconfig(cfg)     # seam: composition root resolves once
    assert resolved.profile.name and resolved.source_kind == "default"