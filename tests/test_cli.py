"""Tests for CLI argument parsing."""

import json
import os
import signal
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, cast

import anyio
import pytest
from rich.console import Console

from daydream import cli, remote_ci
from daydream.atif import validate as atif_validate
from daydream.commands.improve import _parse_improve_args
from daydream.commands.review import _parse_args
from daydream.dataset import LocalRecordStore
from daydream.run_config import RunConfig
from daydream.ui import NEON_THEME, print_issues_table
from tests.harness.dataset import read_records
from tests.harness.git_helpers import bare_remote, commit, git, init_repo
from tests.harness.remote_ci import (
    _accelerate_empty_ci_discovery,
    finish_remote_ci_fake,
    seed_pr_identity,
    start_remote_ci_fake,
)
from tests.test_deep_orchestrator import _install_stub_backend, _silence
from tests.test_integration import (
    _FULL_FLOW_ISSUE,
    _remote_ci_push_project,
    _WorktreeMutatingBackend,
)


def test_signal_handler_flushes_before_backend_registry_snapshot(monkeypatch: pytest.MonkeyPatch,) -> None:

    events: list[str] = []
    class FakePanel:
        def __init__(self, _console: object) -> None:
            pass
        def start(self, _message: str) -> None:
            events.append("panel-start")
        def add_step(self, _message: str) -> None:
            events.append("backend-step")
    monkeypatch.setattr(cli, "flush_active_signal_recorders", lambda: events.append("flush"),)
    def snapshot() -> tuple[object, ...]:
        events.append("snapshot")
        return (object(),)
    monkeypatch.setattr(cli, "active_backends", snapshot)
    monkeypatch.setattr(cli, "ShutdownPanel", FakePanel)
    monkeypatch.setattr(cli, "set_shutdown_panel", lambda _panel: None)
    with pytest.raises(KeyboardInterrupt):
        cli._signal_handler(signal.SIGTERM, None)

    assert events == ["flush", "panel-start", "snapshot", "backend-step"]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            ["/tmp/project"],
            {
                "output_mode": "loop",
                "approved_head_sha": None,
                "findings_out": None,
                "ignore_paths": [],
                "dataset_capture": False,
                "dataset_store_path": None,
                "archive": True,
                "diagram": None,
                "extra_copy": [],
                "non_interactive": False,
            },
        ),
        (
            [
                "--review",
                "--approved-head-sha",
                "a" * 40,
                "--pr-number",
                "42",
                "--findings-out",
                "findings/out.json",
                "--capture-data",
                "--dataset-store",
                "/tmp/records",
                "--no-archive",
                "--worktree",
                "--non-interactive",
                "--branch",
                "feat/x",
                "--base",
                "develop",
                "-s",
                "python",
                "--ignore-path",
                ".planning",
                "--ignore-path",
                "vendor",
                "--copy",
                "a.env",
                "--copy",
                "b.env",
                "/tmp/project",
            ],
            {
                "output_mode": "review",
                "approved_head_sha": "a" * 40,
                "pr_number": 42,
                "findings_out": "findings/out.json",
                "dataset_capture": True,
                "dataset_store_path": Path("/tmp/records"),
                "archive": False,
                "force_worktree": True,
                "non_interactive": True,
                "branch": "feat/x",
                "base": "develop",
                "stack": "python",
                "ignore_paths": [".planning", "vendor"],
                "extra_copy": [Path("a.env"), Path("b.env")],
            },
        ),
        (
            [
                "--diagram-only",
                "flowchart",
                "--findings-out",
                "diagram.json",
                "--dataset-store",
                "/tmp/records",
                "/tmp/project",
            ],
            {
                "output_mode": "diagram",
                "diagram": "flowchart",
                "findings_out": "diagram.json",
                "dataset_capture": True,
                "dataset_store_path": Path("/tmp/records"),
            },
        ),
        (
            ["--shallow", "--diagram", "off", "--stack", "rust", "/tmp/project"],
            {"output_mode": "loop", "shallow": True, "diagram": "off", "stack": "rust"},
        ),
        (["--comment", "/tmp/project"], {"output_mode": "comment", "shallow": False}),
        (
            ["--findings-out", "findings/out.json", "/tmp/project"],
            {"output_mode": "loop", "findings_out": "findings/out.json", "shallow": False},
        ),
    ],
    ids=["defaults", "configured-review", "diagram-export", "shallow", "comment", "default-deep-export"],
)
def test_compatible_cli_vectors_resolve_coherent_configuration(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: dict[str, Any],
) -> None:
    """Compatible options compose in one real parse, including defaults and repeatable flags."""
    config = _cfg(monkeypatch, argv)
    for field, value in expected.items():
        actual = getattr(config, field)
        if value is None or isinstance(value, bool):
            assert actual is value, field
        else:
            assert actual == value, field


def test_invalid_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        _parse_args()


def _cfg(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> RunConfig:
    """Parse ``daydream <args>`` into a RunConfig via the real CLI parser."""
    monkeypatch.setattr(sys, "argv", ["daydream", *args])
    return _parse_args()


@pytest.mark.parametrize("conflict", [["--review"], ["--comment"], ["--shallow"]])
def test_flow_conflicts_rejected(monkeypatch: pytest.MonkeyPatch, conflict: Any) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--flow", "x", *conflict, "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_loop_flag_rejected(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--loop", "/tmp/project"])
    with pytest.raises(SystemExit):
        _parse_args()
    assert "unrecognized arguments" in capsys.readouterr().err


def test_log_flag_rejected_review(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--log", "/tmp/project"])
    with pytest.raises(SystemExit) as exc_info:
        _parse_args()
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_log_flag_rejected_improve(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _parse_improve_args(["improve", "/tmp/x", "--log"])
    assert exc_info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_verbose_joined_equals_rejected() -> None:
    with pytest.raises(SystemExit):
        _parse_improve_args(["improve", "/tmp/x", "--verbose=true"])
    with pytest.raises(SystemExit):
        _parse_improve_args(["improve", "/tmp/x", "--log=true"])


# Consolidated CLI surface (worktree-isolation refactor)


def test_parse_args_comment_mode_excludes_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "--comment", "--review", "/tmp/repo"])
    with pytest.raises(SystemExit):
        _parse_args()


@pytest.mark.parametrize("extra", [["--comment"], ["--shallow"]])
def test_findings_out_rejects_flows_without_pipeline_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], extra: Any,
) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", *extra, "--findings-out", "f.json", "/tmp/repo"])
    with pytest.raises(SystemExit) as exc_info:
        _parse_args()
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "--findings-out" in err


@pytest.mark.parametrize(("argv", "expected"),
    [(["--diagram", "both", "--diagram-only", "sequence"], "--diagram-only"),
        (["--diagram-only", "sequence", "--comment"], "not allowed with argument"),
        (["--diagram-only", "sequence", "--review"], "not allowed with argument"),
        (["--diagram-only", "sequence", "--flow", "improve"], "--flow"),
        (["--diagram-only", "sequence", "--start-at", "merge"], "--start-at merge"),
        (["--diagram-only", "sequence", "--yes"], "--yes"),
    ],
    ids=["diagram_with_diagram_only", "diagram_only_with_comment", "diagram_only_with_review", "diagram_only_with_flow",
        "diagram_only_with_start_at", "diagram_only_with_yes",
    ],
)
def test_diagram_only_conflicts_are_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str], expected: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", *argv, "/tmp/repo"])
    with pytest.raises(SystemExit) as exc_info:
        _parse_args()
    assert exc_info.value.code == 2
    assert expected in capsys.readouterr().err


def test_feedback_subcommand_is_unknown(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],) -> None:
    called = False
    async def fake_run(*_args: Any, **_kwargs: Any) -> int:
        nonlocal called
        called = True
        return 0
    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["daydream", "feedback", "7", "--bot", "x[bot]", "/tmp/repo"],)
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert not called
    assert "unrecognized arguments: 7 --bot x[bot] /tmp/repo" in capsys.readouterr().err


@pytest.mark.parametrize(("remote_outcome", "expected_code", "repo_override"), [
    ("no_ci", 0, True), ("failed", 1, True), ("no_ci", 0, False),
])
def test_explicit_review_argv_uses_target_remote_ci_verdict_drives_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, install_backend: Any, fake_gh: Any, remote_outcome: str,
    expected_code: int, repo_override: bool, capsys: pytest.CaptureFixture[str],
) -> None:
    """A fork origin must not replace gh's upstream when no GH_REPO is supplied.

    Exercise CLI configuration through a retained fix, real push, and bound CI
    verdict; explicit GH_REPO coverage previously hid origin preselection.
    """
    project, remote, hook_marker, raw_remote = _remote_ci_push_project(tmp_path)
    if repo_override:
        monkeypatch.setenv("GH_REPO", "base-user/project")
    else:
        monkeypatch.delenv("GH_REPO", raising=False)
    seed_pr_identity(fake_gh, head_sha=git(project, "rev-parse", "HEAD"))
    seeder = start_remote_ci_fake(project, fake_gh, hook_marker, outcome=remote_outcome)

    monkeypatch.setattr(remote_ci, "DEFAULT_LIMITS",
        remote_ci.RemoteCILimits(poll_seconds=0.01,
            # Allow ample polling time under xdist CPU load; polling discovers an observation and
            # does not define the expected result.
            discovery_seconds=30, completion_seconds=60, request_seconds=3,
        ),
    )
    if remote_outcome == "no_ci":
        _accelerate_empty_ci_discovery(monkeypatch)
    remote_rendered = StringIO()
    monkeypatch.setattr("daydream.deep.remote_ci_steps.console", Console(file=remote_rendered, width=160))
    install_backend(_WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]]))
    elsewhere = tmp_path / "different cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    trajectory_path = tmp_path / "trajectory.json"
    try:
        with pytest.raises(SystemExit) as exc_info:
            cli.main(["review", str(project), "--stack", "python", "--shallow", "--yes", "--test-command", "true",
                      "--trajectory", str(trajectory_path)])
    finally:
        finish_remote_ci_fake(seeder)

    assert exc_info.value.code == expected_code
    assert hook_marker.read_text() == "ran\n"
    pushed_sha = git(project, "rev-parse", "HEAD")
    assert git(remote, "rev-parse", "refs/heads/feature") == pushed_sha
    assert git(project, "config", "--get", "remote.origin.url") == raw_remote
    verdict = json.loads((project / ".daydream" / "deep" / "remote-ci-verdict.json").read_text())
    assert verdict["status"] == remote_outcome
    assert verdict["target"]["pushed_sha"] == pushed_sha
    assert verdict["target"]["base_repository"] == "base-user/project"
    assert verdict["target"]["head_repository"] == "fork-user/project"
    assert git(remote, "show", "refs/heads/feature:main.py") == "def hello() -> str:\n    return 'world'"
    trajectory = json.loads(trajectory_path.read_text())
    assert trajectory["extra"]["pr_repo"] == "base-user/project"
    assert trajectory["extra"]["pr_number"] == 7
    assert any(event["phase"] == "remote-ci" and event["event"] == "phase_end"
               for event in trajectory["extra"]["phase_events"])
    assert all(call.cwd == project.resolve() for call in fake_gh.process_calls())
    output = capsys.readouterr().out + remote_rendered.getvalue()
    if remote_outcome == "no_ci":
        assert verdict["evidence_sha"] == pushed_sha
        assert verdict["polling"]["stable_polls"] >= verdict["polling"]["required_stable_polls"]
        assert verdict["polling"]["elapsed_seconds"] >= verdict["polling"]["discovery_seconds"]
        assert "Remote CI was observably not configured" in output
        assert "Linux verified" not in output
        assert "coverage verified" not in output.lower()
        assert not (project / ".daydream" / "deep" / "remote-ci-handoff.json").exists()
    else:
        assert "Exact pushed-SHA remote CI passed" not in output
        assert (project / ".daydream" / "deep" / "remote-ci-handoff.json").is_file()


def test_print_issues_table_renders() -> None:



    test_console = Console(file=StringIO(), theme=NEON_THEME, force_terminal=True)
    issues = [{"id": 1, "title": "Bad pattern", "severity": "high", "description": "Uses antipattern",
         "recommendation": "Refactor", "files": ["src/main.py"]},
        {"id": 2, "title": "Missing test", "severity": "low", "description": "No test coverage",
         "recommendation": "Add tests", "files": ["src/utils.py"]},
    ]
    print_issues_table(test_console, issues)
    output = cast("StringIO", test_console.file).getvalue()
    assert "Bad pattern" in output
    assert "Missing test" in output


@pytest.mark.parametrize(
    ("flag", "phase", "sep"),
    [
        ("--review-backend", "review", " "),
        ("--fix-backend", "fix", "="),
        ("--exploration-model", "exploration", " "),
        ("--parse-model", "parse", "="),
    ],
)
def test_per_phase_flag_rejected_with_config_pointer(
    flag: Any,
    phase: Any,
    sep: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [flag, "claude-opus-5", str(tmp_path)] if sep == " " else [f"{flag}=claude-opus-5", str(tmp_path)]
    with pytest.raises(SystemExit):
        _parse_args(argv)
    err = capsys.readouterr().err
    assert flag in err
    assert f"[tool.daydream.phases.{phase}]" in err


def _write_fake_gh(bin_dir: Path, log_env: str, login: str) -> None:
    """Install a child-PATH ``gh`` shim answering exactly the two calls these tests allow.

    Both tests spawn a real ``python -m daydream`` subprocess, so the shim must be on the
    *child's* PATH rather than patched into this process. Every argv is appended to the
    env-named log the test asserts on; any other invocation exits 91, which fails the
    subprocess loudly instead of letting an unexpected API call pass unnoticed.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> \"${log_env}\"\n"
        "case \"$*\" in\n"
        "  'repo view --json nameWithOwner -q .nameWithOwner') printf '%s\\n' 'acme/widgets' ;;\n"
        f"  'api /user') printf '%s\\n' '{{\"login\":\"{login}\"}}' ;;\n"
        "  *) printf '%s\\n' \"unexpected gh call: $*\" >&2; exit 91 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)


@pytest.mark.parametrize("backend_name", ["codex", "pi", "osprey"])
def test_improve_audit_isolation_rejects_unsupported_cli_before_spawn(tmp_path: Path, backend_name: str,) -> None:
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
    _write_fake_gh(fake_bin, "GH_LOG", "fixture-user")
    process_tmp = tmp_path / "process-tmp"
    process_tmp.mkdir()

    before_head = git(repo, "rev-parse", "HEAD")
    before_refs = git(repo, "show-ref")
    before_status = git(repo, "status", "--porcelain=v1")
    before_origin_refs = git(origin, "show-ref")
    env = {key: value
        for key, value in os.environ.items()
        if key
        not in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GH_TOKEN", "GITHUB_TOKEN", "DAYDREAM_GITHUB_APP_ID",
            "DAYDREAM_GITHUB_APP_PRIVATE_KEY",
        }
    }
    env.update({"PATH": f"{fake_bin}{os.pathsep}{env.get('PATH', '')}", "BACKEND_MARKER": str(backend_marker),
            "GH_LOG": str(gh_log), "TMPDIR": str(process_tmp), "CI": "1",
        }
    )
    result = subprocess.run(  # noqa: S603 - fixed module and enum parameter
        [sys.executable, "-m", "daydream", "improve", "--backend", backend_name, "--no-archive", "--no-eval",
            str(repo),
        ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, env=env, timeout=30,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert f"backend '{backend_name}'" in combined
    assert "Use backend 'claude'" in combined
    assert not backend_marker.exists()
    assert gh_log.read_text(encoding="utf-8").splitlines() == [
        "api /user", "repo view --json nameWithOwner -q .nameWithOwner",
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
    log_path = tmp_path / "signal-gh.log"
    _write_fake_gh(bin_dir, "DAYDREAM_SIGNAL_GH_LOG", "signal-fixture")
    return bin_dir, log_path


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signal_flushes_all_runner_recorders(tmp_path: Path, git_repo: Path, ext_dir: Any, signum: signal.Signals,
) -> None:
    extension_dir = ext_dir.write_module(_SIGNAL_FIXTURE_EXTENSION)
    bin_dir, gh_log = _write_signal_fake_gh(tmp_path)
    ready = tmp_path / "signal-ready"
    child_env = dict(os.environ)
    for name in (
        "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY", "GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    ):
        child_env.pop(name, None)
    child_env.update({"DAYDREAM_EXT_DIR": str(extension_dir), "DAYDREAM_SIGNAL_GH_LOG": str(gh_log),
            "DAYDREAM_SIGNAL_READY": str(ready), "PATH": os.pathsep.join((str(bin_dir), child_env.get("PATH", ""))),
        }
    )
    argv = [
        sys.executable, "-m", "daydream", "--non-interactive", "--no-archive", "--no-eval", "--flow", "signal-fixture",
        "--pr-number", "1", str(git_repo),
    ]
    # Files, not pipes: nothing drains a pipe before READY, and a full one blocks the child's banner.
    stdout_path = tmp_path / "signal.stdout"
    stderr_path = tmp_path / "signal.stderr"
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        proc = subprocess.Popen(  # noqa: S603 - controlled production entrypoint
            argv, cwd=Path(__file__).resolve().parents[1], env=child_env, stdout=stdout_file, stderr=stderr_file,
        )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(
                    f"signal fixture exited before READY: {proc.returncode}\n"
                    f"stdout={stdout_path.read_text()}\nstderr={stderr_path.read_text()}"
                )
            time.sleep(0.05)
        assert ready.exists(), "signal fixture did not expose two live sibling invocations"

        os.kill(proc.pid, signum)
        proc.wait(timeout=10)
        assert proc.returncode == 130, (
            f"signal={signum.name} exit={proc.returncode}\n"
            f"stdout={stdout_path.read_text()}\nstderr={stderr_path.read_text()}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
    stdout = stdout_path.read_text()
    stderr = stderr_path.read_text()

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
    assert calls == ["api /user", "repo view --json nameWithOwner -q .nameWithOwner"]


def test_cli_maps_grouped_interrupt_to_shutdown_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise BaseExceptionGroup("task group", [BaseExceptionGroup("nested", [KeyboardInterrupt()])])
    monkeypatch.setattr(anyio, "run", interrupted)
    with pytest.raises(SystemExit) as caught:
        cli.main([str(tmp_path), "--non-interactive"])
    assert caught.value.code == 130


def test_cli_preserves_other_errors_beside_grouped_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:

    failure = BaseExceptionGroup("task group", [KeyboardInterrupt(), RuntimeError("sibling failed")])
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise failure
    monkeypatch.setattr(anyio, "run", interrupted)
    with pytest.raises(BaseExceptionGroup) as caught:
        cli.main([str(tmp_path), "--non-interactive"])
    assert caught.value is failure


def test_cli_defers_repository_discovery_until_session_resolution(
    monkeypatch: pytest.MonkeyPatch, git_repo: Path,
) -> None:
    git(git_repo, "remote", "add", "origin", "https://github.com/existential-birds/daydream.git")
    monkeypatch.chdir(git_repo)
    monkeypatch.setattr(sys, "argv", ["daydream"])
    config = _parse_args()

    assert config.target is None
    assert config.pr_repo is None
    assert config.discover_github_provenance is True


@pytest.mark.parametrize("capture", [False, True], ids=["capture-off", "store-only"])
def test_real_cli_stack_entry(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capture: bool,
) -> None:
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda _repo, **_kwargs: None)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda _repo, _branch, **_kwargs: None)
    _install_stub_backend(monkeypatch, multi_stack_target)
    store = tmp_path / "records"
    argv = ["daydream", "--review", "--stack", "python", "--no-archive", "--no-eval", str(multi_stack_target)]
    if capture:
        argv.extend(["--dataset-store", str(store)])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert store.exists() is capture
    if capture:
        records = read_records(LocalRecordStore(store))
        assert len(records.runs) == 1
        assert records.runs[0]["outcome"] == "success"

    monkeypatch.setattr(sys, "argv", ["daydream", "--review", "--skill", "python", str(multi_stack_target)],)
    with pytest.raises(SystemExit) as skill_exc:
        cli.main()
    assert skill_exc.value.code == 2
