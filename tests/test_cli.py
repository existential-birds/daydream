# tests/test_cli.py
"""Tests for CLI argument parsing."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from daydream.cli import _parse_args
from daydream.config import SKILL_MAP
from daydream.runner import RunConfig, _resolved_backend_name


def test_default_backend_is_none_and_resolves_to_claude(monkeypatch):
    # --backend default is now None so env/config-file can supply it; the
    # terminal fallback in _resolved_backend_name is "claude".
    monkeypatch.delenv("DAYDREAM_BACKEND", raising=False)
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project"])
    config = _parse_args()
    assert config.backend is None
    assert _resolved_backend_name(config, "review") == "claude"


def test_backend_flag_codex(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--backend", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_backend_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "-b", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_invalid_backend_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_review_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project",
        "--backend", "claude", "--review-backend", "codex",
    ])
    config = _parse_args()
    assert config.backend == "claude"
    assert config.review_backend == "codex"


def test_fix_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--fix-backend", "codex",
    ])
    config = _parse_args()
    assert config.fix_backend == "codex"


def test_test_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--test-backend", "codex",
    ])
    config = _parse_args()
    assert config.test_backend == "codex"


def test_loop_flag_default_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project"])
    config = _parse_args()
    assert config.loop is False
    assert config.max_iterations == 5


def test_loop_flag_enabled(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--loop"])
    config = _parse_args()
    assert config.loop is True


def test_max_iterations_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--loop", "--max-iterations", "10",
    ])
    config = _parse_args()
    assert config.max_iterations == 10


def test_loop_start_at_conflict(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--shallow", "--loop", "--start-at", "fix",
    ])
    with pytest.raises(SystemExit):
        _parse_args()


def test_max_iterations_without_loop_accepted(monkeypatch):
    """--max-iterations without --loop is accepted but prints a warning."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--max-iterations", "3",
    ])
    with pytest.warns(UserWarning, match="no effect without --loop"):
        config = _parse_args()
    assert config.max_iterations == 3
    assert config.loop is False


def test_skill_map_includes_go():
    assert SKILL_MAP["go"] == "beagle-go:review-go"


def test_skill_choice_go(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--skill", "go"])
    config = _parse_args()
    assert config.skill == "go"


def test_skill_map_includes_rust():
    assert SKILL_MAP["rust"] == "beagle-rust:review-rust"


def test_skill_choice_rust(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--skill", "rust"])
    config = _parse_args()
    assert config.skill == "rust"


def test_skill_map_includes_ios():
    assert SKILL_MAP["ios"] == "beagle-ios:review-ios"


def test_skill_choice_ios(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--skill", "ios"])
    config = _parse_args()
    assert config.skill == "ios"


def test_skill_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "-s", "python"])
    config = _parse_args()
    assert config.skill == "python"


def test_ignore_paths_default_empty(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project"])
    config = _parse_args()
    assert config.ignore_paths == []


def test_ignore_paths_single(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--ignore-path", ".planning",
    ])
    config = _parse_args()
    assert config.ignore_paths == [".planning"]


def test_ignore_paths_repeatable(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project",
        "--ignore-path", ".planning",
        "--ignore-path", "vendor",
    ])
    config = _parse_args()
    assert config.ignore_paths == [".planning", "vendor"]


# ---------------------------------------------------------------------------
# Consolidated CLI surface (worktree-isolation refactor)
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


def test_parse_args_default_is_loop(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "loop"


def test_parse_args_worktree_modifier(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--worktree", "/tmp/repo"])
    config = _parse_args()
    assert config.force_worktree is True


def test_parse_args_shallow_modifier(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--shallow", "/tmp/repo"])
    config = _parse_args()
    assert config.shallow is True


def test_parse_args_non_interactive_sets_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--non-interactive", "/some/target"])
    config = _parse_args()
    assert config.non_interactive is True


def test_parse_args_non_interactive_defaults_false(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/some/target"])
    config = _parse_args()
    assert config.non_interactive is False


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


def test_phase_subtitles_include_wonder_and_envision():
    from daydream.ui import PHASE_SUBTITLES
    assert "WONDER" in PHASE_SUBTITLES
    assert "ENVISION" in PHASE_SUBTITLES
    assert len(PHASE_SUBTITLES["WONDER"]) >= 2
    assert len(PHASE_SUBTITLES["ENVISION"]) >= 2


def test_print_issues_table_renders():
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


# ---------------------------------------------------------------------------
# Per-phase model override flags (Task 3 of per-phase-model-overrides)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,attr,value",
    [
        ("--review-model", "review_model", "claude-haiku-4-5"),
        ("--parse-model", "parse_model", "claude-haiku-4-5"),
        ("--fix-model", "fix_model", "claude-opus-4-6"),
        ("--test-model", "test_model", "gpt-5.5"),
    ],
)
def test_per_phase_model_flags_set_runconfig_field(flag, attr, value, tmp_path):
    config = _parse_args([flag, value, str(tmp_path)])
    assert getattr(config, attr) == value


def test_no_per_phase_model_flag_leaves_field_none(tmp_path):
    config = _parse_args([str(tmp_path)])
    assert config.review_model is None
    assert config.parse_model is None
    assert config.fix_model is None
    assert config.test_model is None


def test_existing_exploration_model_flag_unchanged(tmp_path):
    config = _parse_args(["--exploration-model", "claude-haiku-4-5", str(tmp_path)])
    assert config.exploration_model == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Global --model flag (cli-verb-redesign Task 2 — re-added as a global override)
# ---------------------------------------------------------------------------


def test_global_model_flag_populates_runconfig(tmp_path):
    config = _parse_args(["--model", "claude-opus-4-8", str(tmp_path)])
    assert config.model == "claude-opus-4-8"


def test_runconfig_has_model_field():
    config = RunConfig(backend="claude")
    assert hasattr(config, "model"), "RunConfig.model is the global model override source"
    assert config.model is None


# ---------------------------------------------------------------------------
# build-corpus exit-code regression guard (Task 11 / corpus-pipeline-architecture)
# ---------------------------------------------------------------------------
# Tier-3 subprocess test: drives the real CLI entry point through `uv run`
# against an empty archive directory and asserts a clean exit 0. This catches
# regressions where post-projection cleanup paths (signal handlers, atexit
# hooks, warnings escalation) leak a non-zero exit even though
# `_handle_build_corpus_command` itself returned 0.


def test_build_corpus_exits_0_on_dry_run(tmp_path: Path) -> None:
    """Production entrypoint must exit 0 on successful dry-run."""
    out = tmp_path / "out.jsonl"
    result = subprocess.run(  # noqa: S603 - args are not user-controlled
        [  # noqa: S607 - hardcoded uv/daydream entrypoint
            "uv", "run", "daydream", "build-corpus",
            "--out", str(out), "--include-all-labels", "--dry-run",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "DAYDREAM_ARCHIVE_DIR": str(tmp_path / "empty-archive")},
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )


# ---------------------------------------------------------------------------
# harvest / build-corpus subcommand wiring (Task 11 / corpus-pipeline-architecture)
# ---------------------------------------------------------------------------


def test_harvest_and_build_corpus_dispatch(monkeypatch, tmp_path):
    from daydream import cli
    called = {}
    monkeypatch.setattr("daydream.training.harvest.run_harvest",
                        lambda cfg: called.setdefault("harvest", cfg) or {"annotated": 0})
    assert cli._handle_harvest_command(["--archive-dir", str(tmp_path)]) == 0
    assert "harvest" in called


def test_harvest_parser_accepts_repo_clone_root():
    """--repo-clone-root is parsed and forwarded to HarvestConfig."""
    from daydream.cli import _build_harvest_parser

    parser = _build_harvest_parser()
    args = parser.parse_args(["--repo-clone-root", "/tmp/clones"])
    assert args.repo_clone_root == Path("/tmp/clones")


def test_harvest_parser_repo_clone_root_defaults_to_none():
    """--repo-clone-root defaults to None (derived from cache_dir at runtime)."""
    from daydream.cli import _build_harvest_parser

    parser = _build_harvest_parser()
    args = parser.parse_args([])
    assert args.repo_clone_root is None


def test_removed_verbs_no_longer_dispatch():
    from daydream import cli
    # export-jsonl / snapshot handlers are gone. ``label`` was reintroduced as
    # the human-override surface (daydream label <prefix> --outcome ...).
    assert not hasattr(cli, "_handle_export_command")
    assert not hasattr(cli, "_handle_snapshot_command")
    assert hasattr(cli, "_handle_label_command")


def test_pr_repo_detected_from_target_not_cwd(monkeypatch, tmp_path):
    """pr_repo records the target checkout's slug, not the invoking cwd (#128).

    Drives the production ``_parse_args`` path with a target distinct from cwd
    and a stubbed ``gh repo view`` seam that returns a different slug per path.
    Asserts the resulting ``RunConfig.pr_repo`` (which flows verbatim into the
    trajectory's ``extra.pr_repo``) reflects the target — the benchmark-harness
    pattern of running daydream from one repo against a checkout of another.
    """
    target = tmp_path / "target-checkout"
    target.mkdir()

    def fake_gh_repo_view(repo):
        # Slug keyed on the inspected path so the assertion proves which dir
        # was passed: the target, not Path.cwd().
        if Path(repo) == target:
            return ("grafana", "grafana")
        return ("existential-birds", "daydream")

    monkeypatch.setattr("daydream.git_ops.gh_repo_view", fake_gh_repo_view)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, _branch: None)
    monkeypatch.setattr(sys, "argv", ["daydream", str(target)])

    config = _parse_args()

    assert config.pr_repo == "grafana/grafana"


def test_pr_repo_falls_back_to_cwd_without_target(monkeypatch, tmp_path):
    """With no target positional, slug detection falls back to the cwd (#128)."""
    invoking_repo = tmp_path / "invoking-repo"
    invoking_repo.mkdir()
    monkeypatch.chdir(invoking_repo)

    def fake_gh_repo_view(repo):
        assert Path(repo) == invoking_repo
        return ("existential-birds", "daydream")

    monkeypatch.setattr("daydream.git_ops.gh_repo_view", fake_gh_repo_view)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, _branch: None)
    monkeypatch.setattr(sys, "argv", ["daydream"])

    config = _parse_args()

    assert config.target is None
    assert config.pr_repo == "existential-birds/daydream"


def test_feedback_pr_repo_detected_from_target_not_cwd(monkeypatch, tmp_path):
    """The feedback subcommand also attributes pr_repo to the target (#128)."""
    target = tmp_path / "target-checkout"
    target.mkdir()

    def fake_gh_repo_view(repo):
        if Path(repo) == target:
            return ("grafana", "grafana")
        return ("existential-birds", "daydream")

    monkeypatch.setattr("daydream.git_ops.gh_repo_view", fake_gh_repo_view)
    monkeypatch.setattr(sys, "argv", ["daydream", "feedback", "42", "--bot", "x[bot]", str(target)])

    config = _parse_args()

    assert config.pr_number == 42
    assert config.pr_repo == "grafana/grafana"
