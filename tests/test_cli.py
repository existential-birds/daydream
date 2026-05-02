# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys
import warnings
from pathlib import Path

import pytest

from daydream.cli import _parse_args
from daydream.config import SKILL_MAP


def test_default_backend_is_claude(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.backend == "claude"


def test_backend_flag_codex(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--backend", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_backend_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "-b", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_model_default_is_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.model is None


def test_model_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--model", "sonnet"])
    config = _parse_args()
    assert config.model == "sonnet"


def test_invalid_backend_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_review_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--backend", "claude", "--review-backend", "codex",
    ])
    config = _parse_args()
    assert config.backend == "claude"
    assert config.review_backend == "codex"


def test_fix_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--fix-backend", "codex",
    ])
    config = _parse_args()
    assert config.fix_backend == "codex"


def test_test_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--test-backend", "codex",
    ])
    config = _parse_args()
    assert config.test_backend == "codex"


def test_loop_flag_default_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.loop is False
    assert config.max_iterations == 5


def test_loop_flag_enabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--loop"])
    config = _parse_args()
    assert config.loop is True


def test_max_iterations_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--loop", "--max-iterations", "10",
    ])
    config = _parse_args()
    assert config.max_iterations == 10


def test_loop_review_only_conflict(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--loop", "--review-only",
    ])
    with pytest.raises(SystemExit):
        _parse_args()


def test_loop_start_at_conflict(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--loop", "--start-at", "fix",
    ])
    with pytest.raises(SystemExit):
        _parse_args()


def test_max_iterations_without_loop_accepted(monkeypatch, capsys):
    """--max-iterations without --loop is accepted but prints a warning."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--max-iterations", "3",
    ])
    config = _parse_args()
    assert config.max_iterations == 3
    assert config.loop is False


def test_skill_map_includes_go():
    assert SKILL_MAP["go"] == "beagle-go:review-go"


def test_go_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--go"])
    config = _parse_args()
    assert config.skill == "go"


def test_skill_choice_go(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--skill", "go"])
    config = _parse_args()
    assert config.skill == "go"


def test_skill_map_includes_rust():
    assert SKILL_MAP["rust"] == "beagle-rust:review-rust"


def test_rust_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--rust"])
    config = _parse_args()
    assert config.skill == "rust"


def test_skill_choice_rust(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--skill", "rust"])
    config = _parse_args()
    assert config.skill == "rust"


def test_skill_map_includes_ios():
    assert SKILL_MAP["ios"] == "beagle-ios:review-ios"


def test_ios_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ios"])
    config = _parse_args()
    assert config.skill == "ios"


def test_skill_choice_ios(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--skill", "ios"])
    config = _parse_args()
    assert config.skill == "ios"


def test_trust_the_technology_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--trust-the-technology"])
    config = _parse_args()
    assert config.trust_the_technology is True


def test_ttt_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt"])
    config = _parse_args()
    assert config.trust_the_technology is True


def test_ttt_excludes_skill_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--python"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_excludes_review_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--review-only"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_excludes_loop(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--loop"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_excludes_pr(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--pr", "1", "--bot", "x"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_ttt_compatible_with_backend(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--backend", "codex"])
    config = _parse_args()
    assert config.trust_the_technology is True
    assert config.backend == "codex"


def test_ttt_compatible_with_model(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--ttt", "--model", "sonnet"])
    config = _parse_args()
    assert config.trust_the_technology is True
    assert config.model == "sonnet"


def test_ttt_default_is_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.trust_the_technology is False


def test_ignore_paths_default_empty(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.ignore_paths == []


def test_ignore_paths_single(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python", "--ignore-path", ".planning",
    ])
    config = _parse_args()
    assert config.ignore_paths == [".planning"]


def test_ignore_paths_repeatable(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--ignore-path", ".planning",
        "--ignore-path", "vendor",
    ])
    config = _parse_args()
    assert config.ignore_paths == [".planning", "vendor"]


# ---------------------------------------------------------------------------
# Stage 4.1a — consolidated CLI surface (worktree-isolation refactor)
# ---------------------------------------------------------------------------


def test_parse_args_branch_and_base(monkeypatch):
    """--branch and --base populate the new RunConfig fields; output_mode defaults to loop."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--branch", "feat/x", "--base", "develop", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.branch == "feat/x"
    assert config.base == "develop"
    assert config.output_mode == "loop"


def test_parse_args_comment_mode_excludes_review(monkeypatch):
    """--comment and --review are mutually exclusive (argparse output group)."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--comment", "--review", "/tmp/repo"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_parse_args_comment_mode_sets_output_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--comment", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "comment"


def test_parse_args_review_mode_sets_output_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "review"


def test_parse_args_worktree_modifier(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--worktree", "/tmp/repo"])
    config = _parse_args()
    assert config.force_worktree is True


def test_parse_args_shallow_modifier(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--shallow", "/tmp/repo"])
    config = _parse_args()
    assert config.shallow is True


def test_parse_args_copy_repeatable(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--copy", "a.env", "--copy", "b.env", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.extra_copy == [Path("a.env"), Path("b.env")]


def test_parse_args_copy_default_empty(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/repo"])
    config = _parse_args()
    assert config.extra_copy == []


def test_parse_args_integer_target_errors(monkeypatch, capsys):
    """Pure-numeric TARGET errors with the suggested feedback message."""
    monkeypatch.setattr(sys, "argv", ["daydream", "42"])
    with pytest.raises(SystemExit):
        _parse_args()
    err = capsys.readouterr().err
    assert "did you mean: daydream feedback 42" in err


def test_parse_args_feedback_subcommand(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "feedback", "7", "--bot", "copilot"])
    config = _parse_args()
    assert config.pr_number == 7
    assert config.bot == "copilot"


def test_parse_args_feedback_subcommand_with_target(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "feedback", "7", "--bot", "copilot", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.pr_number == 7
    assert config.bot == "copilot"
    assert config.target == "/tmp/repo"


def test_parse_args_feedback_requires_bot(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "feedback", "7"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_parse_args_deprecated_python_warns(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--python", "/tmp/repo"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = _parse_args()
    assert any(
        issubclass(w.category, DeprecationWarning) and "--python" in str(w.message)
        for w in caught
    )
    assert config.forced_skill == "python"
    assert config.shallow is True


def test_parse_args_deprecated_ttt_maps_to_comment(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--ttt", "/tmp/repo"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = _parse_args()
    assert config.output_mode == "comment"
    assert any(
        issubclass(w.category, DeprecationWarning) and "--ttt" in str(w.message)
        for w in caught
    )


def test_parse_args_deprecated_review_only_maps_to_review(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--review-only", "/tmp/repo"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = _parse_args()
    assert config.output_mode == "review"
    assert any(
        issubclass(w.category, DeprecationWarning) and "--review-only" in str(w.message)
        for w in caught
    )


def test_parse_args_deprecated_deep_is_noop(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--deep", "/tmp/repo"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = _parse_args()
    # --deep still sets the legacy field for now (Stage 4.1b removes that),
    # but output_mode stays "loop" — the new surface treats deep as default.
    assert config.output_mode == "loop"
    assert any(
        issubclass(w.category, DeprecationWarning) and "--deep" in str(w.message)
        for w in caught
    )


def test_phase_subtitles_include_wonder_and_envision():
    from daydream.ui import PHASE_SUBTITLES
    assert "WONDER" in PHASE_SUBTITLES
    assert "ENVISION" in PHASE_SUBTITLES
    assert len(PHASE_SUBTITLES["WONDER"]) >= 2
    assert len(PHASE_SUBTITLES["ENVISION"]) >= 2


def test_print_issues_table_renders(capsys):
    from io import StringIO

    from rich.console import Console

    from daydream.ui import NEON_THEME, print_issues_table

    test_console = Console(file=StringIO(), theme=NEON_THEME, force_terminal=True)
    issues = [
        {"id": 1, "title": "Bad pattern", "severity": "high", "description": "Uses antipattern",
         "recommendation": "Refactor", "files": ["src/main.py"]},
        {"id": 2, "title": "Missing test", "severity": "low", "description": "No test coverage",
         "recommendation": "Add tests", "files": ["src/utils.py"]},
    ]
    print_issues_table(test_console, issues)
    output = test_console.file.getvalue()
    assert "Bad pattern" in output
    assert "Missing test" in output
