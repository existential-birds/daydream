# tests/test_cli.py
"""Tests for CLI argument parsing."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from daydream.cli import _parse_args
from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig, _resolved_backend_name, _resolved_model


def test_approved_head_sha_flag_populates_config(monkeypatch):
    """--approved-head-sha pins config.approved_head_sha (no normalization)."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "--approved-head-sha", "a" * 40, "/tmp/repo"])
    config = _parse_args()
    assert config.approved_head_sha == "a" * 40


def test_approved_head_sha_defaults_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "/tmp/repo"])
    assert _parse_args().approved_head_sha is None


def test_default_backend_is_none_and_resolves_to_claude(monkeypatch):
    # --backend default is now None so the config file can supply it; the
    # terminal fallback in _resolved_backend_name is "claude".
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project"])
    config = _parse_args()
    assert config.backend is None
    assert _resolved_backend_name(config, "review") == "claude"


# Backend names selectable via --backend/-b.
BACKEND_NAMES = ["codex", "osprey"]


@pytest.mark.parametrize("flag", ["--backend", "-b"], ids=["long", "short"])
@pytest.mark.parametrize("backend", BACKEND_NAMES, ids=lambda name: name)
def test_backend_flag(monkeypatch, flag, backend):
    """Accept each backend flag spelling and select the named backend."""
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", flag, backend])
    config = _parse_args()
    assert config.backend == backend


def test_invalid_backend_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.parametrize(
    ("global_backend", "overrides", "phase", "expected"),
    [
        pytest.param("claude", {"review": {"backend": "codex"}}, "review", "codex", id="review-override"),
        pytest.param("claude", {"review": {"backend": "codex"}}, "fix", "claude", id="review-fallback"),
        pytest.param(None, {"fix": {"backend": "codex"}}, "fix", "codex", id="fix-override"),
        pytest.param(None, {"test": {"backend": "codex"}}, "test", "codex", id="test-override"),
    ],
)
def test_phase_backend_override_via_config_file(global_backend, overrides, phase, expected):
    """Resolve phase-specific backends ahead of the global configured backend."""
    # Per-phase backend overrides moved to the config file (Task 8); resolver still honours them.
    fc = DaydreamFileConfig(backend=global_backend, phases=overrides)
    config = RunConfig(target="/tmp/project", backend=None, file_config=fc)
    assert _resolved_backend_name(config, phase) == expected


def _cfg(monkeypatch, args: list[str]) -> RunConfig:
    """Parse ``daydream <args>`` into a RunConfig via the real CLI parser."""
    monkeypatch.setattr(sys, "argv", ["daydream", *args])
    return _parse_args()


def test_run_config_flow_name_defaults_none():
    assert RunConfig(target="/tmp/p").flow_name is None


def test_run_config_flow_name_settable():
    assert RunConfig(target="/tmp/p", flow_name="ro-audit").flow_name == "ro-audit"


def test_flow_flag_sets_flow_name(monkeypatch):
    cfg = _cfg(monkeypatch, ["--flow", "ro-audit", "/tmp/project"])
    assert cfg.flow_name == "ro-audit"


def test_flow_default_none(monkeypatch):
    assert _cfg(monkeypatch, ["/tmp/project"]).flow_name is None


@pytest.mark.parametrize("conflict", [["--review"], ["--comment"], ["--shallow"]])
def test_flow_conflicts_rejected(monkeypatch, conflict):
    monkeypatch.setattr(sys, "argv", ["daydream", "--flow", "x", *conflict, "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_loop_flag_rejected(monkeypatch, capsys):
    """``--loop`` was removed entirely (#330); the CLI rejects it as unknown."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--loop", "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()
    assert "unrecognized arguments" in capsys.readouterr().err


def test_runconfig_has_no_loop_fields():
    """RunConfig carries no loop mode after the collapse (#330)."""
    assert not hasattr(RunConfig(), "loop")
    assert not hasattr(RunConfig(), "max_iterations")


@pytest.mark.parametrize("output_flag", ["--review", "--comment"], ids=["review", "comment"])
def test_yes_with_review_only_output_errors(monkeypatch, capsys, output_flag):
    """--yes has no effect in review-only output modes and must be rejected."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--yes", output_flag, "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()
    assert "--yes" in capsys.readouterr().err


@pytest.mark.parametrize("stack", ["go", "rust", "ios"])
def test_stack_choice_routes_to_stack_field(monkeypatch, stack):
    """Every CLI stack selector routes into ``RunConfig.stack``."""
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--stack", stack])
    config = _parse_args()
    assert config.stack == stack


def test_stack_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "-s", "python"])
    config = _parse_args()
    assert config.stack == "python"


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


# Consolidated CLI surface (worktree-isolation refactor)


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


def test_findings_out_with_review_populates_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--review", "--findings-out", "findings/findings.json", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.findings_out == "findings/findings.json"
    assert config.output_mode == "review"


@pytest.mark.parametrize("extra", [["--comment"], ["--shallow"]])
def test_findings_out_rejects_flows_without_pipeline_errors(monkeypatch, capsys, extra):
    """--findings-out is rejected for flows with no findings pipeline (--comment, --shallow)."""
    monkeypatch.setattr(sys, "argv", ["daydream", *extra, "--findings-out", "f.json", "/tmp/repo"])
    with pytest.raises(SystemExit) as exc_info:
        _parse_args()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--findings-out" in err


def test_findings_out_with_deep_flow_populates_config(monkeypatch):
    """The default deep loop flow (no --review/--comment/--shallow) permits --findings-out."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--findings-out", "findings/findings.json", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.findings_out == "findings/findings.json"
    assert config.output_mode == "loop"
    assert config.shallow is False


def test_findings_out_defaults_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "/tmp/repo"])
    assert _parse_args().findings_out is None


def test_pr_number_flag_populates_config(monkeypatch):
    """--pr-number pins config.pr_number, bypassing branch auto-detection."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "--pr-number", "42", "/tmp/repo"])
    config = _parse_args()
    assert config.pr_number == 42


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


def test_parse_args_feedback_requires_bot(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "feedback", "7"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_phase_subtitles_include_wonder():
    from daydream.ui import PHASE_SUBTITLES
    assert "WONDER" in PHASE_SUBTITLES
    assert len(PHASE_SUBTITLES["WONDER"]) >= 2


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


# Per-phase model overrides — config-file path (cli-verb-redesign Task 8)


@pytest.mark.parametrize(
    "phase,value",
    [
        ("review", "claude-haiku-4-5"),
        ("parse", "claude-haiku-4-5"),
        ("fix", "claude-opus-4-6"),
        ("test", "gpt-5.5"),
    ],
)
def test_per_phase_model_set_via_config_file(phase, value):
    # Per-phase model overrides moved to the config file; resolver still honours them.
    fc = DaydreamFileConfig(phases={phase: {"model": value}})
    config = RunConfig(target="/tmp/project", backend=None, model=None, file_config=fc)
    assert _resolved_model(config, phase) == value


def test_no_per_phase_model_flag_leaves_field_none(tmp_path):
    config = _parse_args([str(tmp_path)])
    assert config.review_model is None
    assert config.parse_model is None
    assert config.fix_model is None
    assert config.test_model is None
    assert config.exploration_model is None


# Per-phase model/backend flags removed (cli-verb-redesign Task 8 — config-only)


@pytest.mark.parametrize(
    "flag,phase",
    [
        ("--review-backend", "review"),
        ("--fix-backend", "fix"),
        ("--test-backend", "test"),
        ("--exploration-model", "exploration"),
        ("--review-model", "review"),
        ("--parse-model", "parse"),
        ("--fix-model", "fix"),
        ("--test-model", "test"),
    ],
)
def test_per_phase_flag_rejected_with_config_pointer(flag, phase, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse_args([flag, "claude-opus-5", str(tmp_path)])
    err = capsys.readouterr().err
    assert flag in err
    assert f"[tool.daydream.phases.{phase}]" in err


@pytest.mark.parametrize(
    "flag,phase",
    [
        ("--fix-model", "fix"),
        ("--review-backend", "review"),
    ],
)
def test_per_phase_flag_rejected_equals_form(flag, phase, tmp_path, capsys):
    with pytest.raises(SystemExit):
        _parse_args([f"{flag}=claude-opus-5", str(tmp_path)])
    err = capsys.readouterr().err
    assert flag in err
    assert f"[tool.daydream.phases.{phase}]" in err


# Global --model flag (cli-verb-redesign Task 2 — re-added as a global override)


def test_global_model_flag_populates_runconfig(tmp_path):
    config = _parse_args(["--model", "claude-opus-5", str(tmp_path)])
    assert config.model == "claude-opus-5"


def test_runconfig_has_model_field():
    config = RunConfig(backend="claude")
    assert hasattr(config, "model"), "RunConfig.model is the global model override source"
    assert config.model is None


# corpus build exit-code regression guard (Task 11 / corpus-pipeline-architecture).
# Tier-3 subprocess test driving the real CLI through `uv run` against an empty
# archive: catches cleanup paths (signal handlers, atexit, warnings) leaking a
# non-zero exit even when _handle_build_corpus_command returned 0.


def test_build_corpus_exits_0_on_dry_run(tmp_path: Path) -> None:
    """Production entrypoint must exit 0 on successful dry-run."""
    out = tmp_path / "out.jsonl"
    result = subprocess.run(  # noqa: S603 - args are not user-controlled
        [  # noqa: S607 - hardcoded uv/daydream entrypoint
            "uv", "run", "daydream", "corpus", "build",
            "--out", str(out), "--include-all-labels", "--dry-run",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "DAYDREAM_ARCHIVE_DIR": str(tmp_path / "empty-archive")},
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )


# corpus harvest / build subcommand wiring (Task 11 / corpus-pipeline-architecture)


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


def test_cli_stack_selector_and_skill_rejected():
    from daydream.cli import _build_main_parser

    p = _build_main_parser()
    args = p.parse_args(["--stack", "python", "/tmp"])
    assert args.stack == "python"
    # --skill is rejected as an unknown option (no alias).
    with pytest.raises(SystemExit) as e:
        p.parse_args(["--skill", "python", "/tmp"])
    assert e.value.code == 2   # argparse unknown-option exit code


def test_runconfig_uses_stack_terminology():
    from daydream.runner import RunConfig

    cfg = RunConfig(target="/tmp", stack="go")
    assert cfg.stack == "go"
    assert not hasattr(cfg, "skill")   # old name removed


def test_real_cli_stack_entry(tmp_path):
    """Real entrypoint: --stack python reaches dispatch; --skill is rejected."""
    import os
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parent.parent
    (tmp_path / "a.py").write_text("x = 1\n")
    # skill-free env: no plugin/skill registry directory.
    env = {**os.environ, "DAYDREAM_SKILLS_DIR": ""}
    r = subprocess.run(
        [sys.executable, "-m", "daydream", "--review", "--stack", "python", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=repo_root, timeout=90,
    )
    # The run should get past arg parsing + profile resolution (exit 0 or the
    # review's own later failure, never an unknown-option / ProfileError).
    assert "--skill" not in r.stderr and "ProfileError" not in r.stderr
    r2 = subprocess.run(
        [sys.executable, "-m", "daydream", "--review", "--skill", "python", str(tmp_path)],
        capture_output=True, text=True, env=env, cwd=repo_root, timeout=30,
    )
    assert r2.returncode == 2 or "unrecognized arguments" in r2.stderr or "invalid choice" in r2.stderr
