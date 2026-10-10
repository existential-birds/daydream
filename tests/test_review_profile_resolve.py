"""Tests for review-profile precedence, confinement, and CLI resolution.

Precedence: explicit_path > DAYDREAM_REVIEW_PROFILE env > repo-committed
file_config.review_profile > packaged default. Invalid higher-precedence
sources fail naming their source, never falling through.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream import review_profile as rp
from daydream.config_file import DaydreamFileConfig
from tests.harness.git_helpers import commit, init_repo, write_and_stage


def _write_profile(tmp_path: Path, name: Any, content: Any) -> Any:
    p = tmp_path / f"{name}.toml"
    p.write_text(f'''schema_version = 1
name = "{name}"
[strategies.intent]
content = "{content}"
source = "copied: a"''')
    return p

def test_precedence_explicit_beats_env_beats_repo_beats_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = _write_profile(tmp_path, "explicit", "E")
    user = _write_profile(tmp_path, "user", "U")
    repo = _write_profile(tmp_path, "repo", "R")
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", str(user))
    fc = DaydreamFileConfig(review_profile=repo)
    resolved = rp.resolve_profile(explicit_path=str(explicit), file_config=fc)
    assert resolved.profile.name == "explicit" and resolved.source_kind == "explicit"

def test_env_beats_repo_and_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = _write_profile(tmp_path, "user", "U")
    repo = _write_profile(tmp_path, "repo", "R")
    monkeypatch.setenv("DAYDREAM_REVIEW_PROFILE", str(user))
    fc = DaydreamFileConfig(review_profile=repo)
    resolved = rp.resolve_profile(file_config=fc)          # no explicit path
    assert resolved.profile.name == "user" and resolved.source_kind == "env"

def test_repo_beats_default(tmp_path: Path) -> None:
    repo = _write_profile(tmp_path, "repo", "R")
    # Supply the repository root to admit confined absolute paths.
    fc = DaydreamFileConfig(review_profile=repo)
    resolved = rp.resolve_profile(file_config=fc, repo_root=tmp_path)
    assert resolved.profile.name == "repo" and resolved.source_kind == "repo"

def test_absolute_repo_path_cannot_escape(tmp_path: Path) -> None:
    # An escaping repository value would expose host profile text to the model.
    fc = DaydreamFileConfig(review_profile=Path("/etc/host-marker.toml"))
    with pytest.raises(rp.ProfileError) as e:
        rp.resolve_profile(file_config=fc, repo_root=tmp_path)
    assert "escape" in str(e.value).lower()

def test_relative_repo_path_cannot_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fc = DaydreamFileConfig(review_profile=Path("../evil.toml"))   # relative repo path
    with pytest.raises(rp.ProfileError) as e:
        rp.resolve_profile(file_config=fc, repo_root=tmp_path)
    assert "escape" in str(e.value).lower()

def test_real_cli_entry_resolves_profile_and_inspects(tmp_path: Path) -> None:

    # A real Git target reaches profile resolution after workspace opening.
    init_repo(tmp_path)
    write_and_stage(tmp_path, "seed.txt", "x\n")
    commit(tmp_path, "init")

    p = tmp_path / "prof.toml"
    p.write_text('schema_version = 1\nname = "cli-p"\n[strategies.intent]\ncontent = "C"\nsource = "copied: a"')
    env = {**os.environ, "DAYDREAM_REVIEW_PROFILE": str(p)}
    repo_root = Path(__file__).resolve().parents[1]
    # Exercise the profile flag through the real CLI.
    out = subprocess.run([sys.executable, "-m", "daydream", "--review", "--review-profile", str(p), str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=repo_root, timeout=120,
    )
    # An empty diff with a valid explicit profile exits successfully.
    assert "ProfileError" not in out.stderr
    assert out.returncode == 0  # clean review of an empty diff: resolved + dispatched

    # An invalid explicit profile fails at its source without falling back.
    bad = tmp_path / "bad.toml"
    bad.write_text('schema_version = 1\nname = "bad"\nunknown = 1')
    out_bad = subprocess.run(
        [sys.executable, "-m", "daydream", "--review", "--review-profile", str(bad), str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=repo_root, timeout=120,
    )
    assert out_bad.returncode != 0
    # Rich can wrap a long path inside panel borders; remove rendering characters to recover its basename.
    cleaned = re.sub(r"[\s║═╔╗╚╝]", "", out_bad.stdout + out_bad.stderr)
    assert "bad.toml" in cleaned
