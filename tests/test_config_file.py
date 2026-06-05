"""Tests for daydream.config_file module (config-file loader)."""

from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig, load_file_config


def test_dotfile_wins_over_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.daydream]\nmodel = "from-pyproject"\n')
    (tmp_path / ".daydream.toml").write_text('model = "from-dotfile"\n[phases.fix]\nbackend = "codex"\n')
    cfg = load_file_config(tmp_path)
    assert cfg.model == "from-dotfile"  # .daydream.toml overrides pyproject, per-key
    assert cfg.phase_backend("fix") == "codex"
    assert cfg.phase_model("review") is None


def test_absent_config_is_empty(tmp_path: Path) -> None:
    cfg = load_file_config(tmp_path)
    assert cfg.model is None and cfg.backend is None and cfg.phase_model("fix") is None


def test_malformed_toml_raises_valueerror(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text("model = =bad")
    with pytest.raises(ValueError, match=r"\.daydream\.toml"):
        load_file_config(tmp_path)


def test_per_key_merge_preserves_pyproject_phase(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream]\nbackend = "claude"\n[tool.daydream.phases.review]\nmodel = "pyproject-review"\n'
    )
    (tmp_path / ".daydream.toml").write_text('[phases.fix]\nbackend = "codex"\n')
    cfg = load_file_config(tmp_path)
    # dotfile's [phases.fix] must not wipe pyproject's [phases.review]
    assert cfg.phase_model("review") == "pyproject-review"
    assert cfg.phase_backend("fix") == "codex"
    assert cfg.backend == "claude"


def test_empty_config_helper() -> None:
    cfg = DaydreamFileConfig()
    assert cfg.model is None and cfg.backend is None
    assert cfg.phase_model("fix") is None and cfg.phase_backend("review") is None
