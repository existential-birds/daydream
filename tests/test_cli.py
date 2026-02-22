# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys

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
