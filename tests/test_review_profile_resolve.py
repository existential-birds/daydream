"""Tests for review-profile precedence, confinement, and CLI resolution.

Precedence: explicit_path > DAYDREAM_REVIEW_PROFILE env > repo-committed
file_config.review_profile > packaged default. Invalid higher-precedence
sources fail naming their source, never falling through.
"""
from pathlib import Path
from typing import Any

import pytest

from daydream import review_profile as rp
from daydream.config_file import DaydreamFileConfig


def _write_profile(tmp_path: Path, name: Any, content: Any) -> Any:
    p = tmp_path / f"{name}.toml"
    p.write_text(f'''schema_version = 1
name = "{name}"
[strategies.intent]
content = "{content}"
source = "copied: a"''')
    return p

def test_precedence_and_repo_confinement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = _write_profile(tmp_path, "user", "U")
    repo = _write_profile(tmp_path, "repo", "R")
    # Supply the repository root to admit confined absolute paths.
    resolved = rp.resolve_profile(file_config=DaydreamFileConfig(review_profile=repo), repo_root=tmp_path)
    assert resolved.profile.name == "repo" and resolved.source_kind == "repo"
    # An escaping repository value would expose host profile text to the model.
    for escaping in (Path("../evil.toml"), Path("/etc/host-marker.toml")):
        with pytest.raises(rp.ProfileError) as e:
            rp.resolve_profile(file_config=DaydreamFileConfig(review_profile=escaping), repo_root=tmp_path)
        assert "escape" in str(e.value).lower()
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", str(user))
    resolved = rp.resolve_profile(file_config=DaydreamFileConfig(review_profile=repo))  # no explicit path
    assert resolved.profile.name == "user" and resolved.source_kind == "env"
    # An explicit path beats the environment, and an invalid one fails at its source without falling back.
    explicit = rp.resolve_profile(explicit_path=str(repo), file_config=DaydreamFileConfig(review_profile=repo))
    assert explicit.profile.name == "repo"
    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 1\nname = "bad"\nunknown = 1')
    with pytest.raises(rp.ProfileError) as e:
        rp.resolve_profile(explicit_path=str(bad), file_config=DaydreamFileConfig(review_profile=repo))
    assert "bad.toml" in str(e.value)
