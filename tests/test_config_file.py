"""Tests for daydream.config_file module (config-file loader)."""

from pathlib import Path

import pytest

from daydream.config_file import DaydreamFileConfig, load_file_config
from tests.harness.config import write_target_hub_key


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


def test_external_review_bots_parses_from_tool_daydream(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.daydream]\nexternal_review_bots = ["greptile-apps[bot]", "coderabbitai[bot]"]\n'
    )
    cfg = load_file_config(tmp_path)
    assert cfg.external_review_bots == ["greptile-apps[bot]", "coderabbitai[bot]"]


def test_external_review_bots_absent_defaults_empty(tmp_path: Path) -> None:
    assert load_file_config(tmp_path).external_review_bots == []


def test_improve_github_issue_publishing_is_explicitly_configurable(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream.improve.github]\npublish_issues = true\n"
    )

    config = load_file_config(tmp_path)

    assert config.improve_github_publish_issues is True


def test_improve_github_issue_publishing_accepts_kebab_case_fallback(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        "[tool.daydream.improve.github]\npublish-issues = true\n"
    )

    config = load_file_config(tmp_path)

    assert config.improve_github_publish_issues is True


@pytest.mark.parametrize("raw", ['"yes"', "1", "[]"])
def test_improve_github_issue_publishing_rejects_non_boolean_values(
    tmp_path: Path,
    raw: str,
) -> None:
    (tmp_path / ".daydream.toml").write_text(
        f"[improve.github]\npublish_issues = {raw}\n"
    )

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


def test_approve_on_clean_true_parses_as_bool(tmp_path: Path) -> None:
    (tmp_path / ".daydream.toml").write_text("approve_on_clean = true\n")
    cfg = load_file_config(tmp_path)
    assert cfg.approve_on_clean is True


def test_approve_on_clean_non_bool_degrades_to_none(tmp_path: Path) -> None:
    # bool-only coercion: a truthy int is NOT enabled, it degrades to None (unset).
    (tmp_path / ".daydream.toml").write_text("approve_on_clean = 1\n")
    cfg = load_file_config(tmp_path)
    assert cfg.approve_on_clean is None


def test_target_trajectory_hub_repo_key_is_ignored(tmp_path: Path) -> None:
    """A target file setting trajectory_hub_repo loads cleanly and contributes
    nothing — the field is removed from the model entirely."""
    write_target_hub_key(tmp_path)
    cfg = load_file_config(tmp_path)
    assert not hasattr(cfg, "trajectory_hub_repo")  # field removed from the model


def test_uncovered_sweep_toggle_is_bool_only(tmp_path: Path) -> None:
    """``uncovered_sweep`` is bool-only: an accidental int degrades to unset."""
    (tmp_path / ".daydream.toml").write_text("uncovered_sweep = false\n")
    assert load_file_config(tmp_path).uncovered_sweep is False
    (tmp_path / ".daydream.toml").write_text("uncovered_sweep = 1\n")
    assert load_file_config(tmp_path).uncovered_sweep is None


def test_uncovered_sweep_numerics_accept_zero_and_keep_non_negative(tmp_path: Path) -> None:
    """Non-negative-only coercion: ``0`` is meaningful, negatives degrade to None."""
    (tmp_path / ".daydream.toml").write_text(
        "uncovered_sweep_max_files = 0\nuncovered_sweep_min_hunk_lines = 0\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.uncovered_sweep_max_files == 0
    assert cfg.uncovered_sweep_min_hunk_lines == 0


def test_uncovered_sweep_numerics_negative_degrade_to_none(tmp_path: Path) -> None:
    """A negative capacity cap or hunk floor is never meaningful: degrade to None."""
    (tmp_path / ".daydream.toml").write_text(
        "uncovered_sweep_max_files = -1\nuncovered_sweep_min_hunk_lines = -5\n"
    )
    cfg = load_file_config(tmp_path)
    assert cfg.uncovered_sweep_max_files is None
    assert cfg.uncovered_sweep_min_hunk_lines is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            'supervisor = "rules"\n'
            'supervisor_deny_globs = ["vendor/**"]\n'
            'tool_supervisor = "rules"\n'
            'tool_bash_deny = ["rm -rf"]\n',
            ("rules", ["vendor/**"], "rules", ["rm -rf"]),
            id="valid",
        ),
        pytest.param(
            'supervisor = "unknown"\nsupervisor_deny_globs = [1]\ntool_supervisor = "unknown"\ntool_bash_deny = [1]\n',
            (None, [], None, []),
            id="invalid-degrades-to-unset",
        ),
    ],
)
def test_supervision_config(
    tmp_path: Path,
    content: str,
    expected: tuple[str | None, list[str], str | None, list[str]],
) -> None:
    """Load supervisor identities and deny lists from supported config spellings."""
    (tmp_path / ".daydream.toml").write_text(content)
    cfg = load_file_config(tmp_path)

    assert (
        cfg.supervisor,
        cfg.supervisor_deny_globs,
        cfg.tool_supervisor,
        cfg.tool_bash_deny,
    ) == expected


def test_empty_config_helper() -> None:
    cfg = DaydreamFileConfig()
    assert cfg.model is None and cfg.backend is None
    assert cfg.phase_model("fix") is None and cfg.phase_backend("review") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(-0.1, None, id="negative"),
        pytest.param(float("nan"), None, id="nan"),
        pytest.param(float("inf"), None, id="inf"),
        pytest.param(float("-inf"), None, id="negative-inf"),
        pytest.param(True, None, id="bool"),
        pytest.param("0.05", None, id="string"),
        pytest.param([0.05], None, id="list"),
        pytest.param(None, None, id="absent"),
        pytest.param(0, 0.0, id="zero"),
        pytest.param(0.05, 0.05, id="valid"),
        pytest.param(100, 100.0, id="int-coerced"),
    ],
)
def test_quality_gate_threshold_coercion(raw: object, expected: float | None) -> None:
    """#329/Finding 7: quality-gate thresholds accept only finite non-negative values.

    A negative threshold would flag every unchanged file (a zero delta exceeds
    it); NaN/inf would silently disable the metric (every comparison is False)
    and write non-standard ``NaN`` into JSON; bool/string/list are never
    meaningful floors. Each invalid input degrades to ``None`` so the
    ``config.py`` default applies.
    """
    from daydream.config_file import _coerce_quality_threshold

    value = _coerce_quality_threshold(raw)
    if expected is None:
        assert value is None
    else:
        assert value == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("-0.1", id="negative"),
        pytest.param("nan", id="nan"),
        pytest.param("inf", id="inf"),
        pytest.param("true", id="bool"),
        pytest.param('"0.25"', id="string"),
    ],
)
def test_quality_gate_thresholds_in_file_config_degrade_to_none(tmp_path: Path, value: str) -> None:
    """#329/Finding 7: invalid thresholds in ``.daydream.toml`` degrade to None.

    Exercises the real TOML parse path: negative, NaN, inf, bool, and string
    values for any of the four threshold keys (the delta and the absolute
    undefined-baseline knobs) land as ``None`` in the loaded config, so
    ``_step_fix`` resolves the named default rather than a gate-breaking floor.
    """
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
    """#329/Finding 7: valid thresholds parse through unchanged."""
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
