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
    assert cfg.reasoning_effort is None and cfg.phase_reasoning_effort("fix") is None


def test_reasoning_effort_global_and_phase_override(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text(
        'reasoning_effort = "medium"\n[phases.fix]\nreasoning_effort = "high"\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.reasoning_effort == "medium"
    assert cfg.phase_reasoning_effort("fix") == "high"
    assert cfg.phase_reasoning_effort("review") is None


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


def test_load_file_config_reads_bench_table(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream.bench]\nbenchmark-repo = "/b"\nmodel = "anthropic/claude-opus-4-5-20251101"\n'
        '[tool.daydream.bench.reviewers.glm]\nbackend = "pi"\nmodel = "z-ai/glm-5.2"\nprovider = "openrouter"\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.bench["benchmark-repo"] == "/b"
    assert cfg.bench["reviewers"]["glm"] == {"backend": "pi", "model": "z-ai/glm-5.2", "provider": "openrouter"}


def test_precision_mode_true_parses_as_bool(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text("precision_mode = true\n")
    cfg = load_file_config(tmp_path)
    assert cfg.precision_mode is True


def test_precision_mode_non_bool_degrades_to_none(tmp_path: Path) -> None:
    # bool-only coercion (mirrors raw_precision): a truthy int is NOT enabled, it
    # degrades to None (unset) rather than crashing or coercing to True.
    (tmp_path / ".daydream.toml").write_text("precision_mode = 1\n")
    cfg = load_file_config(tmp_path)
    assert cfg.precision_mode is None


def test_supervision_config_parses_all_keys(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text(
        'supervisor = "rules"\n'
        'supervisor_deny_globs = ["vendor/**"]\n'
        'tool_supervisor = "rules"\n'
        'tool_bash_deny = ["rm -rf"]\n'
    )

    cfg = load_file_config(tmp_path)

    assert cfg.supervisor == "rules"
    assert cfg.supervisor_deny_globs == ["vendor/**"]
    assert cfg.tool_supervisor == "rules"
    assert cfg.tool_bash_deny == ["rm -rf"]


def test_supervision_config_bad_values_degrade_to_unset(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text(
        'supervisor = "unknown"\n'
        "supervisor_deny_globs = [1]\n"
        'tool_supervisor = "unknown"\n'
        "tool_bash_deny = [1]\n"
    )

    cfg = load_file_config(tmp_path)

    assert cfg.supervisor is None
    assert cfg.supervisor_deny_globs == []
    assert cfg.tool_supervisor is None
    assert cfg.tool_bash_deny == []


def test_empty_config_helper() -> None:
    cfg = DaydreamFileConfig()
    assert cfg.model is None and cfg.backend is None
    assert cfg.phase_model("fix") is None and cfg.phase_backend("review") is None
