from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig, load_file_config


def test_config_reload_merges_independent_keys_and_degrades_invalid_edits(tmp_path: Path) -> None:
    """A reload must admit each field independently, preserve sibling file settings and refuse unread keys."""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream]\nbackend = "claude"\n'
        '[tool.daydream.phases.review]\nmodel = "pyproject-review"\n'
        '[tool.daydream.improve]\nservice_roots = ["apps/*"]\n'
        '[tool.daydream.improve.service_groups]\ncore = ["apps/billing", "apps/catalog"]\n'
        "[tool.daydream.improve.github]\npublish-issues = true\n"
        '[tool.daydream.diagram]\nmin_branch_points = 6\nmode = "auto"\n'
        '[tool.daydream.workspace]\ncopy = [".env"]\n'
    )
    dotfile = tmp_path / ".daydream.toml"
    dotfile.write_text(
        "scope_issue_filing = true\nretained_runs = 3\n"
        'supervisor_deny_globs = ["vendor/**"]\n'
        'tool_supervisor = "rules"\ntool_bash_deny = ["rm -rf"]\n'
        "quality_gate_erosion_delta = 0\nquality_gate_verbosity_delta = 0.05\n"
        "quality_gate_erosion_absolute = 100\nquality_gate_verbosity_absolute = 0.25\n"
        '[phases.fix]\nbackend = "codex"\n[diagram]\nmode = "off"\n'
    )

    config = load_file_config(tmp_path)

    assert config.backend == "claude"
    assert config.phases == {"review": {"model": "pyproject-review"}, "fix": {"backend": "codex"}}
    assert (config.scope_issue_filing, config.retained_runs) == (True, 3)
    assert config.supervisor_deny_globs == ["vendor/**"]
    assert (config.tool_supervisor, config.tool_bash_deny) == ("rules", ["rm -rf"])
    assert (
        config.quality_gate_erosion_delta,
        config.quality_gate_verbosity_delta,
        config.quality_gate_erosion_absolute,
        config.quality_gate_verbosity_absolute,
    ) == (0.0, 0.05, 100.0, 0.25)
    assert config.improve_service_roots == ["apps/*"]
    assert config.improve_service_groups == {"core": ["apps/billing", "apps/catalog"]}
    assert config.improve_github_publish_issues is True
    assert (config.diagram_mode, config.diagram_min_branch_points) == ("off", 6)

    dotfile.write_text(
        "scope_issue_filing = 1\nretained_runs = 0\n"
        "supervisor_deny_globs = [1]\n"
        'tool_supervisor = "unknown"\ntool_bash_deny = [1]\n'
        '[improve]\npartition-max-files = 0\nmax-partition-groups = "many"\n'
        "[improve.github]\npublish_issues = 1\n"
        '[diagram]\nmode = "both"\nmin_code_files = 0\nmin_modules = "two"\n'
        'min_branch_points = -1\nservice_roots = "apps/*"\n'
    )

    degraded = load_file_config(tmp_path)

    assert (degraded.scope_issue_filing, degraded.retained_runs) == (None, None)
    assert degraded.supervisor_deny_globs == []
    assert (degraded.tool_supervisor, degraded.tool_bash_deny) == (None, [])
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

    # A retired setting or the removed bench section fails the load, naming each key, instead of being ignored.
    with (tmp_path / "pyproject.toml").open("a") as pyproject:
        pyproject.write('[tool.daydream.bench]\nmodel = "bench"\n')
    dotfile.write_text("precision_mode = true\n")
    with pytest.raises(ValueError, match=r"\[tool\.daydream\]: bench, precision_mode$"):
        load_file_config(tmp_path)
