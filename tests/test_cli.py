"""Tests for CLI argument parsing."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from daydream.atif import validate as atif_validate
from daydream.cli import _parse_args
from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig, _resolved_backend_name, _resolved_model
from tests.harness.git_helpers import bare_remote, commit, git, init_repo


def test_approved_head_sha_flag_populates_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """--approved-head-sha pins config.approved_head_sha (no normalization)."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "--approved-head-sha", "a" * 40, "/tmp/repo"])
    config = _parse_args()
    assert config.approved_head_sha == "a" * 40


def test_approved_head_sha_defaults_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "/tmp/repo"])
    assert _parse_args().approved_head_sha is None


def test_default_backend_is_none_and_resolves_to_claude(monkeypatch: pytest.MonkeyPatch) -> None:
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
def test_backend_flag(monkeypatch: pytest.MonkeyPatch, flag: Any, backend: Any) -> None:
    """Accept each backend flag spelling and select the named backend."""
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", flag, backend])
    config = _parse_args()
    assert config.backend == backend


def test_invalid_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
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
def test_phase_backend_override_via_config_file(global_backend: Any, overrides: Any, phase: Any, expected: Any) -> None:
    """Resolve phase-specific backends ahead of the global configured backend."""
    # Per-phase backend overrides moved to the config file (Task 8); resolver still honours them.
    fc = DaydreamFileConfig(backend=global_backend, phases=overrides)
    config = RunConfig(target="/tmp/project", backend=None, file_config=fc)
    assert _resolved_backend_name(config, phase) == expected


def _cfg(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> RunConfig:
    """Parse ``daydream <args>`` into a RunConfig via the real CLI parser."""
    monkeypatch.setattr(sys, "argv", ["daydream", *args])
    return _parse_args()


def test_run_config_flow_name_defaults_none() -> None:
    assert RunConfig(target="/tmp/p").flow_name is None


def test_run_config_flow_name_settable() -> None:
    assert RunConfig(target="/tmp/p", flow_name="ro-audit").flow_name == "ro-audit"


def test_file_scope_issues_flag_reaches_runconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(monkeypatch, ["--file-scope-issues", "/tmp/project"])
    assert cfg.scope_issue_filing is True


def test_file_scope_issues_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(monkeypatch, ["/tmp/project"])
    assert cfg.scope_issue_filing is False


def test_runconfig_scope_issue_filing_defaults_false() -> None:
    assert RunConfig(target="/t").scope_issue_filing is False


def test_flow_flag_sets_flow_name(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg(monkeypatch, ["--flow", "ro-audit", "/tmp/project"])
    assert cfg.flow_name == "ro-audit"


def test_flow_default_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _cfg(monkeypatch, ["/tmp/project"]).flow_name is None


@pytest.mark.parametrize("conflict", [["--review"], ["--comment"], ["--shallow"]])
def test_flow_conflicts_rejected(monkeypatch: pytest.MonkeyPatch, conflict: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--flow", "x", *conflict, "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_loop_flag_rejected(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """``--loop`` was removed entirely (#330); the CLI rejects it as unknown."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--loop", "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()
    assert "unrecognized arguments" in capsys.readouterr().err


def test_runconfig_has_no_loop_fields() -> None:
    """RunConfig carries no loop mode after the collapse (#330)."""
    assert not hasattr(RunConfig(), "loop")
    assert not hasattr(RunConfig(), "max_iterations")


@pytest.mark.parametrize("output_flag", ["--review", "--comment"], ids=["review", "comment"])
def test_yes_with_review_only_output_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output_flag: Any,
) -> None:
    """--yes has no effect in review-only output modes and must be rejected."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--yes", output_flag, "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()
    assert "--yes" in capsys.readouterr().err


@pytest.mark.parametrize("stack", ["go", "rust", "ios"])
def test_stack_choice_routes_to_stack_field(monkeypatch: pytest.MonkeyPatch, stack: Any) -> None:
    """Every CLI stack selector routes into ``RunConfig.stack``."""
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--stack", stack])
    config = _parse_args()
    assert config.stack == stack


def test_stack_short_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "-s", "python"])
    config = _parse_args()
    assert config.stack == "python"


def test_ignore_paths_default_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project"])
    config = _parse_args()
    assert config.ignore_paths == []


def test_ignore_paths_single(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--ignore-path", ".planning",
    ])
    config = _parse_args()
    assert config.ignore_paths == [".planning"]


def test_ignore_paths_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project",
        "--ignore-path", ".planning",
        "--ignore-path", "vendor",
    ])
    config = _parse_args()
    assert config.ignore_paths == [".planning", "vendor"]


# Consolidated CLI surface (worktree-isolation refactor)


def test_parse_args_branch_and_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """--branch and --base populate the new RunConfig fields; output_mode defaults to loop."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--branch", "feat/x", "--base", "develop", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.branch == "feat/x"
    assert config.base == "develop"
    assert config.output_mode == "loop"


def test_parse_args_comment_mode_excludes_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """--comment and --review are mutually exclusive (argparse output group)."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--comment", "--review", "/tmp/repo"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_parse_args_comment_mode_sets_output_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--comment", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "comment"


def test_parse_args_review_mode_sets_output_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "review"


def test_parse_args_default_is_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "loop"


def test_findings_out_with_review_populates_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--review", "--findings-out", "findings/findings.json", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.findings_out == "findings/findings.json"
    assert config.output_mode == "review"


@pytest.mark.parametrize("extra", [["--comment"], ["--shallow"]])
def test_findings_out_rejects_flows_without_pipeline_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    extra: Any,
) -> None:
    """--findings-out is rejected for flows with no findings pipeline (--comment, --shallow)."""
    monkeypatch.setattr(sys, "argv", ["daydream", *extra, "--findings-out", "f.json", "/tmp/repo"])
    with pytest.raises(SystemExit) as exc_info:
        _parse_args()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--findings-out" in err


def test_diagram_only_sets_output_mode_and_diagram_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #1113: ``--diagram-only KIND`` selects the mode AND the kind."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--diagram-only", "flowchart", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "diagram"
    assert config.diagram == "flowchart"


def test_diagram_flag_leaves_output_mode_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--diagram`` modifies the review paths; it is not an output mode."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--diagram", "off", "/tmp/repo"])
    config = _parse_args()
    assert config.output_mode == "loop"
    assert config.diagram == "off"


def test_diagram_defaults_to_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset (not ``"auto"``) so a repo file's ``mode = "off"`` can still win."""
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/repo"])
    assert _parse_args().diagram is None


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--diagram", "both", "--diagram-only", "sequence"], "--diagram-only"),
        (["--diagram-only", "sequence", "--comment"], "not allowed with argument"),
        (["--diagram-only", "sequence", "--review"], "not allowed with argument"),
        (["--diagram-only", "sequence", "--flow", "improve"], "--flow"),
        (["--diagram-only", "sequence", "--start-at", "merge"], "--start-at merge"),
        (["--diagram-only", "sequence", "--yes"], "--yes"),
    ],
    ids=[
        "diagram_with_diagram_only",
        "diagram_only_with_comment",
        "diagram_only_with_review",
        "diagram_only_with_flow",
        "diagram_only_with_start_at",
        "diagram_only_with_yes",
    ],
)
def test_diagram_only_conflicts_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    expected: str,
) -> None:
    """Every incompatible combination fails loudly instead of silently picking one."""
    monkeypatch.setattr(sys, "argv", ["daydream", *argv, "/tmp/repo"])
    with pytest.raises(SystemExit) as exc_info:
        _parse_args()
    assert exc_info.value.code == 2
    assert expected in capsys.readouterr().err


def test_findings_out_with_diagram_only_populates_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--findings-out`` is honored in diagram-only mode (Phase A of #1113)."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--diagram-only", "sequence", "--findings-out", "f.json", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.findings_out == "f.json"
    assert config.output_mode == "diagram"


def test_findings_out_with_deep_flow_populates_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default deep loop flow (no --review/--comment/--shallow) permits --findings-out."""
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--findings-out", "findings/findings.json", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.findings_out == "findings/findings.json"
    assert config.output_mode == "loop"
    assert config.shallow is False


def test_findings_out_defaults_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "/tmp/repo"])
    assert _parse_args().findings_out is None


def test_pr_number_flag_populates_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """--pr-number pins config.pr_number, bypassing branch auto-detection."""
    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "--pr-number", "42", "/tmp/repo"])
    config = _parse_args()
    assert config.pr_number == 42


def test_parse_args_worktree_modifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--worktree", "/tmp/repo"])
    config = _parse_args()
    assert config.force_worktree is True


def test_parse_args_shallow_modifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--shallow", "/tmp/repo"])
    config = _parse_args()
    assert config.shallow is True


def test_parse_args_non_interactive_sets_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--non-interactive", "/some/target"])
    config = _parse_args()
    assert config.non_interactive is True


def test_parse_args_non_interactive_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "/some/target"])
    config = _parse_args()
    assert config.non_interactive is False


def test_parse_args_copy_repeatable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [
        "daydream", "--copy", "a.env", "--copy", "b.env", "/tmp/repo",
    ])
    config = _parse_args()
    assert config.extra_copy == [Path("a.env"), Path("b.env")]


def test_parse_args_copy_default_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/repo"])
    config = _parse_args()
    assert config.extra_copy == []


def test_feedback_subcommand_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The removed feedback command is rejected before review dispatch."""
    from daydream import cli

    called = False

    async def fake_run(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["daydream", "feedback", "7", "--bot", "x[bot]", "/tmp/repo"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert not called
    assert "unrecognized arguments: 7 --bot x[bot] /tmp/repo" in capsys.readouterr().err


def test_phase_subtitles_include_wonder() -> None:
    from daydream.ui import PHASE_SUBTITLES
    assert "WONDER" in PHASE_SUBTITLES
    assert len(PHASE_SUBTITLES["WONDER"]) >= 2


def test_print_issues_table_renders() -> None:
    from io import StringIO
    from typing import cast

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
    output = cast("StringIO", test_console.file).getvalue()
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
def test_per_phase_model_set_via_config_file(phase: Any, value: Any) -> None:
    # Per-phase model overrides moved to the config file; resolver still honours them.
    fc = DaydreamFileConfig(phases={phase: {"model": value}})
    config = RunConfig(target="/tmp/project", backend=None, model=None, file_config=fc)
    assert _resolved_model(config, phase) == value


def test_no_per_phase_model_flag_leaves_field_none(tmp_path: Path) -> None:
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
def test_per_phase_flag_rejected_with_config_pointer(
    flag: Any,
    phase: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
def test_per_phase_flag_rejected_equals_form(
    flag: Any,
    phase: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        _parse_args([f"{flag}=claude-opus-5", str(tmp_path)])
    err = capsys.readouterr().err
    assert flag in err
    assert f"[tool.daydream.phases.{phase}]" in err


# Global --model flag (cli-verb-redesign Task 2 — re-added as a global override)


def test_global_model_flag_populates_runconfig(tmp_path: Path) -> None:
    config = _parse_args(["--model", "claude-opus-5", str(tmp_path)])
    assert config.model == "claude-opus-5"


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


@pytest.mark.parametrize("backend_name", ["codex", "pi", "osprey"])
def test_improve_audit_isolation_rejects_unsupported_cli_before_spawn(
    tmp_path: Path,
    backend_name: str,
) -> None:
    """The real parser/runner path refuses unsupported improve executables."""
    repo = tmp_path / "repo"
    init_repo(repo)
    (repo / ".gitignore").write_text(".daydream/\ndaydream_plans/\n", encoding="utf-8")
    (repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    git(repo, "add", ".gitignore", "app.py")
    commit(repo, "initial")
    origin = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-u", "origin", "main")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    backend_marker = tmp_path / "backend-launched"
    backend_script = "#!/bin/sh\nprintf '%s\\n' \"$0 $*\" >> \"$BACKEND_MARKER\"\nexit 97\n"
    for executable in ("codex", "pi", "osprey"):
        path = fake_bin / executable
        path.write_text(backend_script, encoding="utf-8")
        path.chmod(0o755)

    gh_log = tmp_path / "gh.log"
    gh = fake_bin / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$GH_LOG\"\n"
        "case \"$*\" in\n"
        "  'repo view --json nameWithOwner -q .nameWithOwner') "
        "printf '%s\\n' 'acme/widgets' ;;\n"
        "  'api /user') printf '%s\\n' '{\"login\":\"fixture-user\"}' ;;\n"
        "  *) printf '%s\\n' 'unexpected gh invocation' >&2; exit 91 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    process_tmp = tmp_path / "process-tmp"
    process_tmp.mkdir()

    before_head = git(repo, "rev-parse", "HEAD")
    before_refs = git(repo, "show-ref")
    before_status = git(repo, "status", "--porcelain=v1")
    before_origin_refs = git(origin, "show-ref")
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "DAYDREAM_GITHUB_APP_ID",
            "DAYDREAM_GITHUB_APP_PRIVATE_KEY",
        }
    }
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}",
            "BACKEND_MARKER": str(backend_marker),
            "GH_LOG": str(gh_log),
            "TMPDIR": str(process_tmp),
            "CI": "1",
        }
    )
    result = subprocess.run(  # noqa: S603 - fixed module and enum parameter
        [
            sys.executable,
            "-m",
            "daydream",
            "improve",
            "--backend",
            backend_name,
            "--no-archive",
            "--no-eval",
            str(repo),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert f"backend '{backend_name}'" in combined
    assert "Use backend 'claude'" in combined
    assert not backend_marker.exists()
    assert gh_log.read_text(encoding="utf-8").splitlines() == [
        "repo view --json nameWithOwner -q .nameWithOwner",
        "api /user",
    ]
    assert git(repo, "rev-parse", "HEAD") == before_head
    assert git(repo, "show-ref") == before_refs
    assert git(repo, "status", "--porcelain=v1") == before_status
    assert git(origin, "show-ref") == before_origin_refs
    assert not list(process_tmp.glob("daydream-audit-*"))


_SIGNAL_FIXTURE_EXTENSION = """
import os
from pathlib import Path

import anyio

from daydream.backends import ResultEvent, TextEvent
from daydream.extensions import FlowStep
from daydream.trajectory import DaydreamPhase, get_current_recorder


async def _hold_child(root, descriptor, marker, entered):
    async with root.fork(descriptor) as child:
        async with child.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text=marker))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
            entered.set()
            await anyio.sleep_forever()


async def _signal_fixture(ctx):
    root = get_current_recorder()
    assert root is not None
    async with root.invocation(phase=DaydreamPhase.REVIEW) as inv:
        inv.observe(TextEvent(text="ROOT_SIGNAL_ONLY"))
        inv.observe(ResultEvent(structured_output=None, continuation=None))

    entered_a = anyio.Event()
    entered_b = anyio.Event()
    async with anyio.create_task_group() as tg:
        tg.start_soon(_hold_child, root, "signal-a", "SIGNAL_A_ONLY", entered_a)
        await entered_a.wait()
        tg.start_soon(_hold_child, root, "signal-b", "SIGNAL_B_ONLY", entered_b)
        await entered_b.wait()
        Path(os.environ["DAYDREAM_SIGNAL_READY"]).write_text("ready", encoding="utf-8")
        await anyio.sleep_forever()


def register(registry):
    registry.register_phase(FlowStep(name="signal-fixture-step", run=_signal_fixture))
    registry.set_flow("signal-fixture", ["signal-fixture-step"])
"""


def _write_signal_fake_gh(tmp_path: Path) -> tuple[Path, Path]:
    """Create the subprocess's only external API boundary: two read-only gh calls."""
    bin_dir = tmp_path / "signal-bin"
    bin_dir.mkdir()
    log_path = tmp_path / "signal-gh.log"
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DAYDREAM_SIGNAL_GH_LOG\"\n"
        "if [ \"$*\" = \"repo view --json nameWithOwner -q .nameWithOwner\" ]; then\n"
        "  printf '%s\\n' 'acme/widgets'\n"
        "  exit 0\n"
        "fi\n"
        "if [ \"$*\" = \"api /user\" ]; then\n"
        "  printf '%s\\n' '{\"login\":\"signal-fixture\"}'\n"
        "  exit 0\n"
        "fi\n"
        "printf '%s\\n' \"unexpected gh call: $*\" >&2\n"
        "exit 91\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return bin_dir, log_path


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_flushes_all_runner_recorders(
    tmp_path: Path,
    git_repo: Path,
    ext_dir: Any,
    signum: signal.Signals,
) -> None:
    """A real OS signal through CLI/runner flushes root and both live forks."""
    extension_dir = ext_dir.write_module(_SIGNAL_FIXTURE_EXTENSION)
    bin_dir, gh_log = _write_signal_fake_gh(tmp_path)
    ready = tmp_path / "signal-ready"
    child_env = dict(os.environ)
    for name in (
        "DAYDREAM_APP_ID",
        "DAYDREAM_APP_PRIVATE_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        child_env.pop(name, None)
    child_env.update(
        {
            "DAYDREAM_EXT_DIR": str(extension_dir),
            "DAYDREAM_SIGNAL_GH_LOG": str(gh_log),
            "DAYDREAM_SIGNAL_READY": str(ready),
            "PATH": os.pathsep.join((str(bin_dir), child_env.get("PATH", ""))),
        }
    )
    argv = [
        sys.executable,
        "-m",
        "daydream",
        "--non-interactive",
        "--no-archive",
        "--no-eval",
        "--flow",
        "signal-fixture",
        "--pr-number",
        "1",
        str(git_repo),
    ]
    proc = subprocess.Popen(  # noqa: S603 - controlled production entrypoint
        argv,
        cwd=Path(__file__).resolve().parents[1],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout = ""
    stderr = ""
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                pytest.fail(
                    f"signal fixture exited before READY: {proc.returncode}\n"
                    f"stdout={stdout}\nstderr={stderr}"
                )
            time.sleep(0.05)
        assert ready.exists(), "signal fixture did not expose two live sibling invocations"

        os.kill(proc.pid, signum)
        stdout, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 130, (
            f"signal={signum.name} exit={proc.returncode}\nstdout={stdout}\nstderr={stderr}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)

    run_root = git_repo / ".daydream" / "runs"
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    partials = sorted(run_dirs[0].rglob("*.partial"))
    assert len(partials) == 3
    by_name = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in partials}
    assert set(by_name) == {"trajectory.json.partial", "signal-a.json.partial", "signal-b.json.partial"}
    assert all(atif_validate(item, validate_images=False) for item in by_name.values())
    assert all(item.get("extra", {}).get("partial") is True for item in by_name.values())

    root_text = json.dumps(by_name["trajectory.json.partial"], sort_keys=True)
    a_text = json.dumps(by_name["signal-a.json.partial"], sort_keys=True)
    b_text = json.dumps(by_name["signal-b.json.partial"], sort_keys=True)
    assert "ROOT_SIGNAL_ONLY" in root_text
    assert "SIGNAL_A_ONLY" not in root_text and "SIGNAL_B_ONLY" not in root_text
    assert "SIGNAL_A_ONLY" in a_text
    assert "SIGNAL_B_ONLY" not in a_text and "ROOT_SIGNAL_ONLY" not in a_text
    assert "SIGNAL_B_ONLY" in b_text
    assert "SIGNAL_A_ONLY" not in b_text and "ROOT_SIGNAL_ONLY" not in b_text
    assert "Traceback" not in stdout + stderr
    assert "trajectory write failed" not in (stdout + stderr).lower()

    calls = gh_log.read_text(encoding="utf-8").splitlines()
    assert calls == [
        "repo view --json nameWithOwner -q .nameWithOwner",
        "api /user",
    ]


# corpus harvest / build subcommand wiring (Task 11 / corpus-pipeline-architecture)


def test_harvest_parser_accepts_repo_clone_root() -> None:
    """--repo-clone-root is parsed and forwarded to HarvestConfig."""
    from daydream.cli import _build_harvest_parser

    parser = _build_harvest_parser()
    args = parser.parse_args(["--repo-clone-root", "/tmp/clones"])
    assert args.repo_clone_root == Path("/tmp/clones")


def test_harvest_parser_repo_clone_root_defaults_to_none() -> None:
    """--repo-clone-root defaults to None (derived from cache_dir at runtime)."""
    from daydream.cli import _build_harvest_parser

    parser = _build_harvest_parser()
    args = parser.parse_args([])
    assert args.repo_clone_root is None


def test_pr_repo_falls_back_to_cwd_without_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With no target positional, slug detection falls back to the cwd (#128)."""
    invoking_repo = tmp_path / "invoking-repo"
    invoking_repo.mkdir()
    monkeypatch.chdir(invoking_repo)

    def fake_gh_repo_view(repo: Any) -> tuple[Any, ...]:
        assert Path(repo) == invoking_repo
        return ("existential-birds", "daydream")

    monkeypatch.setattr("daydream.git_ops.gh_repo_view", fake_gh_repo_view)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, _branch: None)
    monkeypatch.setattr(sys, "argv", ["daydream"])

    config = _parse_args()

    assert config.target is None
    assert config.pr_repo == "existential-birds/daydream"


def test_cli_stack_selector_and_skill_rejected() -> None:
    from daydream.cli import _build_main_parser

    p = _build_main_parser()
    args = p.parse_args(["--stack", "python", "/tmp"])
    assert args.stack == "python"
    # --skill is rejected as an unknown option (no alias).
    with pytest.raises(SystemExit) as e:
        p.parse_args(["--skill", "python", "/tmp"])
    assert e.value.code == 2   # argparse unknown-option exit code


def test_runconfig_uses_stack_terminology() -> None:
    from daydream.runner import RunConfig

    cfg = RunConfig(target="/tmp", stack="go")
    assert cfg.stack == "go"
    assert not hasattr(cfg, "skill")   # old name removed


def test_real_cli_stack_entry(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real entrypoint runs a selected stack and rejects the removed alias."""
    from daydream import cli
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    _silence(monkeypatch)
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda _repo: None)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda _repo, _branch: None)
    _install_stub_backend(monkeypatch, multi_stack_target)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "daydream",
            "--review",
            "--stack",
            "python",
            "--no-archive",
            "--no-eval",
            str(multi_stack_target),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0

    monkeypatch.setattr(
        sys,
        "argv",
        ["daydream", "--review", "--skill", "python", str(multi_stack_target)],
    )
    with pytest.raises(SystemExit) as skill_exc:
        cli.main()
    assert skill_exc.value.code == 2
