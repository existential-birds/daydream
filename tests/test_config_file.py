
from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig, _coerce_non_negative_float, load_file_config
from tests.harness.config import write_daydream_pyproject


def test_improve_config_table_parses_service_roots(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream.improve]\nservice_roots = ["apps/*"]\n'
        '[tool.daydream.improve.service_groups]\ncore = ["apps/billing", "apps/catalog"]\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.improve_service_roots == ["apps/*"]
    assert cfg.improve_service_groups == {"core": ["apps/billing", "apps/catalog"]}

def test_improve_config_absent_defaults_empty(tmp_path: Path) -> None:
    assert load_file_config(tmp_path).improve_service_roots == []

def test_improve_github_issue_publishing_is_explicitly_configurable(tmp_path: Path,) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.daydream.improve.github]\npublish_issues = true\n")

    config = load_file_config(tmp_path)

    assert config.improve_github_publish_issues is True

def test_improve_github_issue_publishing_accepts_kebab_case_fallback(tmp_path: Path,) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.daydream.improve.github]\npublish-issues = true\n")

    config = load_file_config(tmp_path)

    assert config.improve_github_publish_issues is True

@pytest.mark.parametrize("raw", ['"yes"', "1", "[]"])
def test_improve_github_issue_publishing_rejects_non_boolean_values(tmp_path: Path, raw: str,) -> None:
    (tmp_path / ".daydream.toml").write_text(f"[improve.github]\npublish_issues = {raw}\n")

    assert load_file_config(tmp_path).improve_github_publish_issues is False
    assert DaydreamFileConfig().improve_github_publish_issues is False

def test_improve_partition_bounds_parse_from_tool_daydream_improve(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream.improve]\npartition-max-files = 7\nmax-partition-groups = 2\n"
    )
    config = load_file_config(tmp_path)
    assert config.improve_partition_max_files == 7
    assert config.improve_max_partition_groups == 2

def test_improve_partition_bounds_parse_from_dotfile_snake_case(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text("[improve]\npartition_max_files = 30\nmax_partition_groups = 4\n")
    config = load_file_config(tmp_path)
    assert config.improve_partition_max_files == 30
    assert config.improve_max_partition_groups == 4

def test_improve_partition_bounds_default_to_none(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.daydream.improve]\nservice_roots = ["apps/*"]\n')
    config = load_file_config(tmp_path)
    assert config.improve_partition_max_files is None
    assert config.improve_max_partition_groups is None
    assert DaydreamFileConfig().improve_partition_max_files is None
    assert DaydreamFileConfig().improve_max_partition_groups is None

def test_improve_partition_bounds_reject_non_positive(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream.improve]\npartition-max-files = 0\nmax-partition-groups = "many"\n'
    )
    config = load_file_config(tmp_path)
    assert config.improve_partition_max_files is None
    assert config.improve_max_partition_groups is None

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
    (tmp_path / ".daydream.toml").write_text('reasoning_effort = "medium"\n[phases.fix]\nreasoning_effort = "high"\n')
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
    assert cfg.phase_model("review") == "pyproject-review"
    assert cfg.phase_backend("fix") == "codex"
    assert cfg.backend == "claude"

def test_config_has_no_bench_field(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.daydream.bench]\nmodel = "claude-opus-4-5-20251101"\n')
    cfg = load_file_config(tmp_path)
    assert not hasattr(cfg, "bench")
    # A stale bench table must produce a warning, matching the removed CLI verb's explicit
    # rejection.
    assert any("no longer a supported daydream config section" in rec.message for rec in caplog.records)

@pytest.mark.parametrize("key", ["precision_mode", "approve_on_clean", "scope_issue_filing"])
def test_bool_key_true_parses_as_bool(tmp_path: Path, key: str) -> None:
    (tmp_path / ".daydream.toml").write_text(f"{key} = true\n")
    cfg = load_file_config(tmp_path)
    assert getattr(cfg, key) is True

@pytest.mark.parametrize("key", ["precision_mode", "approve_on_clean", "scope_issue_filing"])
def test_bool_key_non_bool_degrades_to_none(tmp_path: Path, key: str) -> None:
    # Truthy integers stay unset under bool-only coercion.
    (tmp_path / ".daydream.toml").write_text(f"{key} = 1\n")
    cfg = load_file_config(tmp_path)
    assert getattr(cfg, key) is None

def test_target_trajectory_hub_repo_key_is_ignored(tmp_path: Path) -> None:
    write_daydream_pyproject(tmp_path, trajectory_hub_repo="evil/repo")
    cfg = load_file_config(tmp_path)
    assert not hasattr(cfg, "trajectory_hub_repo")  # field removed from the model

@pytest.mark.parametrize(("content", "expected"),
    [pytest.param(
            'supervisor = "rules"\n'
            'supervisor_deny_globs = ["vendor/**"]\n'
            'tool_supervisor = "rules"\n'
            'tool_bash_deny = ["rm -rf"]\n',
            ("rules", ["vendor/**"], "rules", ["rm -rf"]), id="valid",
        ),
        pytest.param(
            'supervisor = "unknown"\nsupervisor_deny_globs = [1]\ntool_supervisor = "unknown"\ntool_bash_deny = [1]\n',
            (None, [], None, []), id="invalid-degrades-to-unset",
        ),
    ],
)
def test_supervision_config(tmp_path: Path, content: str, expected: tuple[str | None, list[str], str | None, list[str]],
) -> None:
    (tmp_path / ".daydream.toml").write_text(content)
    cfg = load_file_config(tmp_path)

    assert (cfg.supervisor, cfg.supervisor_deny_globs, cfg.tool_supervisor, cfg.tool_bash_deny,) == expected

def test_empty_config_helper() -> None:
    cfg = DaydreamFileConfig()
    assert cfg.model is None and cfg.backend is None
    assert cfg.phase_model("fix") is None and cfg.phase_backend("review") is None

@pytest.mark.parametrize(("raw", "expected"),
    [pytest.param(-0.1, None, id="negative"), pytest.param(float("nan"), None, id="nan"),
        pytest.param(float("inf"), None, id="inf"), pytest.param(float("-inf"), None, id="negative-inf"),
        pytest.param(True, None, id="bool"), pytest.param("0.05", None, id="string"),
        pytest.param([0.05], None, id="list"), pytest.param(None, None, id="absent"), pytest.param(0, 0.0, id="zero"),
        pytest.param(0.05, 0.05, id="valid"), pytest.param(100, 100.0, id="int-coerced"),
    ],
)
def test_quality_gate_threshold_coercion(raw: object, expected: float | None) -> None:
    """Thresholds require finite nonnegative numbers; invalid values defer to defaults.

    Negative floors flag unchanged files, while NaN/inf disable comparisons. Reject
    booleans, strings, and lists as well as invalid numeric values.
    """
    value = _coerce_non_negative_float(raw)
    if expected is None:
        assert value is None
    else:
        assert value == expected

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

def test_quality_gate_thresholds_accept_finite_non_negative(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text(
        "quality_gate_erosion_delta = 0.0\n"
        "quality_gate_verbosity_delta = 0.25\n"
        "quality_gate_erosion_absolute = 0.5\n"
        "quality_gate_verbosity_absolute = 0.75\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.quality_gate_erosion_delta == 0.0
    assert cfg.quality_gate_verbosity_delta == 0.25
    assert cfg.quality_gate_erosion_absolute == 0.5
    assert cfg.quality_gate_verbosity_absolute == 0.75

def test_diagram_table_parses_from_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream.diagram]\n"
        'mode = "off"\n'
        "min_code_files = 5\n"
        "min_modules = 3\n"
        "min_branch_points = 6\n"
        'service_roots = ["apps/*", "services/*"]\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.diagram_mode == "off"
    assert cfg.diagram_min_code_files == 5
    assert cfg.diagram_min_modules == 3
    assert cfg.diagram_min_branch_points == 6
    assert cfg.diagram_service_roots == ["apps/*", "services/*"]

def test_diagram_table_parses_from_dotfile_and_merges_per_key(tmp_path: Path) -> None:
    """Dotfile diagram keys override individually without discarding pyproject siblings."""
    (tmp_path / "pyproject.toml").write_text("[tool.daydream.diagram]\nmin_branch_points = 6\nmode = \"auto\"\n")
    (tmp_path / ".daydream.toml").write_text('[diagram]\nmode = "off"\n')
    cfg = load_file_config(tmp_path)
    assert cfg.diagram_mode == "off"
    assert cfg.diagram_min_branch_points == 6

def test_diagram_config_absent_defaults_to_unset(tmp_path: Path) -> None:
    """Absent diagram settings stay unset so the orchestrator can apply CLI > file > default."""
    cfg = load_file_config(tmp_path)
    assert cfg.diagram_mode is None
    assert cfg.diagram_min_code_files is None
    assert cfg.diagram_min_modules is None
    assert cfg.diagram_min_branch_points is None
    assert cfg.diagram_service_roots == []

def test_diagram_junk_values_degrade_to_unset(tmp_path: Path) -> None:
    """Malformed diagram keys stay unset; file config cannot force a diagram kind."""
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream.diagram]\n"
        'mode = "both"\n'
        "min_code_files = 0\n"
        'min_modules = "two"\n'
        "min_branch_points = -1\n"
        'service_roots = "apps/*"\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.diagram_mode is None
    assert cfg.diagram_min_code_files is None
    assert cfg.diagram_min_modules is None
    assert cfg.diagram_min_branch_points is None
    assert cfg.diagram_service_roots == []

def test_diagram_junk_table_degrades_to_unset(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.daydream]\ndiagram = "on"\n')
    cfg = load_file_config(tmp_path)
    assert cfg.diagram_mode is None
    assert cfg.diagram_min_branch_points is None

def test_diagram_threshold_keys_accept_hyphenated_spellings(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream.diagram]\n"
        "min-code-files = 4\n"
        "min-modules = 5\n"
        "min-branch-points = 7\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.diagram_min_code_files == 4
    assert cfg.diagram_min_modules == 5
    assert cfg.diagram_min_branch_points == 7

def test_retry_recovery_allowance_round_trips_from_the_file_config(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text("retry_recovery_allowance_s = 120\n", encoding="utf-8")

    config = load_file_config(tmp_path)

    assert config.retry_recovery_allowance_s == 120.0

def test_an_invalid_retry_recovery_allowance_degrades_observably(tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / ".daydream.toml").write_text('retry_recovery_allowance_s = -5\n', encoding="utf-8")

    config = load_file_config(tmp_path)

    assert config.retry_recovery_allowance_s is None            # default applies
    assert any("retry_recovery_allowance_s" in r.message for r in caplog.records)

def test_an_absent_retry_recovery_allowance_stays_silent(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = load_file_config(tmp_path)

    assert config.retry_recovery_allowance_s is None
    assert not any("retry_recovery_allowance_s" in r.message for r in caplog.records)

def test_review_cache_keys_round_trip_and_ill_typed_values_degrade(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text(
        "review_cache_enabled = false\n"
        "review_cache_max_entries = 7\n"
        "review_cache_max_bytes = 2048\n"
        "review_cache_max_age_days = 3\n",
        encoding="utf-8",
    )

    config = load_file_config(tmp_path)

    assert config.review_cache_enabled is False
    assert config.review_cache_max_entries == 7
    assert config.review_cache_max_bytes == 2048
    assert config.review_cache_max_age_days == 3

    (tmp_path / ".daydream.toml").write_text(
        'review_cache_enabled = "yes"\n'
        "review_cache_max_entries = -1\n"
        "review_cache_max_bytes = 1.5\n",
        encoding="utf-8",
    )

    degraded = load_file_config(tmp_path)

    assert degraded.review_cache_enabled is None
    assert degraded.review_cache_max_entries is None
    assert degraded.review_cache_max_bytes is None

def test_verify_selection_keys_round_trip_and_ill_typed_values_degrade(tmp_path: Path) -> None:
    write_daydream_pyproject(tmp_path, verify_all=True, extra_risk_categories=["security", "migration"])
    config = load_file_config(tmp_path)
    assert config.verify_all is True
    assert config.extra_risk_categories == ["security", "migration"]

    write_daydream_pyproject(tmp_path, verify_all=1, extra_risk_categories="security")
    degraded = load_file_config(tmp_path)
    assert degraded.verify_all is None            # real bool only
    assert degraded.extra_risk_categories == []   # non-list degrades to unset, never to a guess

def test_test_required_suites_is_additive_to_test_command(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream]\ntest_command = 'uv run pytest'\ntest_required_suites = ['python', 'rl']\n"
    )
    config = load_file_config(tmp_path)

    assert config.test_command == "uv run pytest"
    assert config.test_required_suites == ["python", "rl"]
    assert load_file_config(tmp_path / "empty").test_required_suites == []
