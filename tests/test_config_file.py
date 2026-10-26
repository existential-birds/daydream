from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig, load_file_config


def test_config_reload_merges_independent_keys_and_degrades_invalid_edits(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A reload must admit each field independently and preserve sibling file settings."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream]\nbackend = "claude"\n'
        '[tool.daydream.phases.review]\nmodel = "pyproject-review"\n'
        '[tool.daydream.improve]\nservice_roots = ["apps/*"]\n'
        '[tool.daydream.improve.service_groups]\ncore = ["apps/billing", "apps/catalog"]\n'
        "[tool.daydream.improve.github]\npublish-issues = true\n"
        '[tool.daydream.diagram]\nmin_branch_points = 6\nmode = "auto"\n'
        '[tool.daydream.bench]\nmodel = "claude-opus-4-5-20251101"\n'
    )
    dotfile = tmp_path / ".daydream.toml"
    dotfile.write_text(
        "precision_mode = true\napprove_on_clean = true\nscope_issue_filing = true\n"
        'supervisor = "rules"\nsupervisor_deny_globs = ["vendor/**"]\n'
        'tool_supervisor = "rules"\ntool_bash_deny = ["rm -rf"]\n'
        "quality_gate_erosion_delta = 0\nquality_gate_verbosity_delta = 0.05\n"
        "quality_gate_erosion_absolute = 100\nquality_gate_verbosity_absolute = 0.25\n"
        "review_cache_enabled = false\nreview_cache_max_entries = 7\n"
        "review_cache_max_bytes = 2048\nreview_cache_max_age_days = 3\n"
        'verify_all = true\nextra_risk_categories = ["security", "migration"]\n'
        '[phases.fix]\nbackend = "codex"\n[diagram]\nmode = "off"\n'
    )

    config = load_file_config(tmp_path)

    assert not hasattr(config, "bench")
    assert any("no longer a supported daydream config section" in rec.message for rec in caplog.records)
    assert config.backend == "claude"
    assert config.phases == {"review": {"model": "pyproject-review"}, "fix": {"backend": "codex"}}
    assert (config.precision_mode, config.approve_on_clean, config.scope_issue_filing) == (True, True, True)
    assert (config.supervisor, config.supervisor_deny_globs) == ("rules", ["vendor/**"])
    assert (config.tool_supervisor, config.tool_bash_deny) == ("rules", ["rm -rf"])
    assert (
        config.quality_gate_erosion_delta,
        config.quality_gate_verbosity_delta,
        config.quality_gate_erosion_absolute,
        config.quality_gate_verbosity_absolute,
    ) == (0.0, 0.05, 100.0, 0.25)
    assert (
        config.review_cache_enabled,
        config.review_cache_max_entries,
        config.review_cache_max_bytes,
        config.review_cache_max_age_days,
    ) == (False, 7, 2048, 3)
    assert (config.verify_all, config.extra_risk_categories) == (True, ["security", "migration"])
    assert config.improve_service_roots == ["apps/*"]
    assert config.improve_service_groups == {"core": ["apps/billing", "apps/catalog"]}
    assert config.improve_github_publish_issues is True
    assert (config.diagram_mode, config.diagram_min_branch_points) == ("off", 6)

    dotfile.write_text(
        "retry_recovery_allowance_s = -5\n"
        "precision_mode = 1\napprove_on_clean = 1\nscope_issue_filing = 1\n"
        'supervisor = "unknown"\nsupervisor_deny_globs = [1]\n'
        'tool_supervisor = "unknown"\ntool_bash_deny = [1]\n'
        'review_cache_enabled = "yes"\nreview_cache_max_entries = -1\n'
        "review_cache_max_bytes = 1.5\nreview_cache_max_age_days = -1\n"
        'verify_all = 1\nextra_risk_categories = "security"\n'
        '[improve]\npartition-max-files = 0\nmax-partition-groups = "many"\n'
        "[improve.github]\npublish_issues = 1\n"
        '[diagram]\nmode = "both"\nmin_code_files = 0\nmin_modules = "two"\n'
        'min_branch_points = -1\nservice_roots = "apps/*"\n'
    )

    degraded = load_file_config(tmp_path)

    assert degraded.retry_recovery_allowance_s is None
    assert any("retry_recovery_allowance_s=-5" in rec.message for rec in caplog.records)
    assert (degraded.precision_mode, degraded.approve_on_clean, degraded.scope_issue_filing) == (None, None, None)
    assert (degraded.supervisor, degraded.supervisor_deny_globs) == (None, [])
    assert (degraded.tool_supervisor, degraded.tool_bash_deny) == (None, [])
    assert (
        degraded.review_cache_enabled,
        degraded.review_cache_max_entries,
        degraded.review_cache_max_bytes,
        degraded.review_cache_max_age_days,
    ) == (None, None, None, None)
    assert (degraded.verify_all, degraded.extra_risk_categories) == (None, [])
    assert (degraded.improve_partition_max_files, degraded.improve_max_partition_groups) == (None, None)
    assert degraded.improve_github_publish_issues is False
    assert DaydreamFileConfig().improve_github_publish_issues is False
    assert (
        degraded.diagram_mode,
        degraded.diagram_min_code_files,
        degraded.diagram_min_modules,
        degraded.diagram_min_branch_points,
    ) == (None, None, None, None)
    assert degraded.diagram_service_roots == []
    assert degraded.improve_service_roots == ["apps/*"]
    assert degraded.phases == {"review": {"model": "pyproject-review"}}


def test_malformed_toml_raises_valueerror(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text("model = =bad")
    with pytest.raises(ValueError, match=r"\.daydream\.toml"):
        load_file_config(tmp_path)


@pytest.mark.parametrize("value",
    [pytest.param("-0.1", id="negative"), pytest.param("nan", id="nan"), pytest.param("inf", id="inf"),
        pytest.param("true", id="bool"), pytest.param('"0.25"', id="string"),
    ],
)
def test_quality_gate_thresholds_in_file_config_degrade_to_none(tmp_path: Path, value: str) -> None:
    (tmp_path / ".daydream.toml").write_text(
        f"quality_gate_erosion_delta = {value}\n"
        f"quality_gate_verbosity_delta = {value}\n"
        f"quality_gate_erosion_absolute = {value}\n"
        f"quality_gate_verbosity_absolute = {value}\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.quality_gate_erosion_delta is None
    assert cfg.quality_gate_verbosity_delta is None
    assert cfg.quality_gate_erosion_absolute is None
    assert cfg.quality_gate_verbosity_absolute is None


def test_diagram_junk_table_degrades_to_unset(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.daydream]\ndiagram = "on"\n')
    cfg = load_file_config(tmp_path)
    assert cfg.diagram_mode is None
    assert cfg.diagram_min_branch_points is None
