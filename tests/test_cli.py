"""Tests for CLI argument parsing."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import cli
from daydream.atif import validate as atif_validate
from daydream.commands.improve import _parse_improve_args
from daydream.commands.review import _parse_args
from daydream.run_config import RunConfig


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
                "ignore_paths": [".planning", "vendor"],
                "extra_copy": [Path("a.env"), Path("b.env")],
            },
        ),
        (
            ["--non-interactive", "--pr-number", "42", "--approved-head-sha", "a" * 40,
             "--findings-out", "findings/findings.json", "--base", "origin/main", "/tmp/project"],
            {"output_mode": "review", "findings_out": "findings/findings.json", "assume": None},
        ),
        (["--fix", "--yes", "/tmp/project"], {"output_mode": "loop", "assume": "yes", "start_at": "review"}),
        (
            ["--flow", "x", "--yes", "--start-at", "fix", "/tmp/project"],
            {"output_mode": "review", "assume": "yes", "flow_name": "x", "start_at": "fix"},
        ),
    ],
    ids=["configured-review", "packaged-single-workflow-export", "selected-fixing", "custom-flow-gates"],
)
def test_compatible_cli_vectors_resolve_coherent_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
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
    improve = _parse_improve_args(["improve", "plan", "add rate limiting", "/tmp/x"])
    assert improve.improve_plan_description == "add rate limiting"
    # The two-tier help: --help-all shows --verbose but never the removed --log.
    with pytest.raises(SystemExit):
        _parse_args(["--help-all"])
    help_out = capsys.readouterr().out
    assert "--verbose" in help_out and "--log" not in help_out


def _cfg(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> RunConfig:
    """Parse ``daydream <args>`` into a RunConfig via the real CLI parser."""
    monkeypatch.setattr(sys, "argv", ["daydream", *args])
    return _parse_args()


def test_incompatible_cli_flags_are_rejected_with_the_offending_flag(
    capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    """Each conflicting or misspelled option exits 2 and names what to fix."""
    rejected = [
        (["--flow", "x", "--review", "/tmp/project"], ""),
        (["--flow", "x", "--shallow", "/tmp/project"], ""),
        (["--flow", "x", "--fix", "/tmp/project"], "--flow cannot be combined"),
        (["--yes", "/tmp/repo"], "--yes only answers the gates of a fixing run; add --fix"),
        (["--flow", "deep", "--yes", "/tmp/repo"], "--yes only answers the gates of a fixing run; add --fix"),
        # Options of the replaced staged reviewer are usage errors that name what replaces them.
        (["--start-at", "fix", "/tmp/repo"], "--start-at: every review starts fresh; rerun without it"),
        (["--fix", "--yes", "--start-at", "fix", "/tmp/repo"], "--start-at: every review starts fresh"),
        (["--fix", "--yes", "--shallow", "/tmp/repo"], "--shallow: the reviewer covers the whole change"),
        (["-s", "python", "/tmp/repo"], "--stack: the reviewer covers the whole change in one review; drop it"),
        (["--precision", "/tmp/repo"], "--precision: every finding is already challenged in a fresh context"),
        (["--latency-profile", "fast", "/tmp/repo"], "--latency-profile: review time and tool allowances come from"),
        (["--no-review-cache", "/tmp/repo"], "--no-review-cache: no review cache exists; drop it"),
        (["--flow", "shallow", "/tmp/repo"], "--shallow: the reviewer covers the whole change"),
        (["--fix", "--non-interactive", "/tmp/repo"], "cannot confirm the fix gate; add --yes"),
        (["--fix", "--yes", "--findings-out", "f.json", "/tmp/repo"], "--findings-out exports"),
        (["--diagram", "both", "--diagram-only", "sequence", "/tmp/repo"], "--diagram-only"),
        (["--diagram-only", "sequence", "--start-at", "merge", "/tmp/repo"], "--start-at: every review starts fresh"),
        (["--diagram-only", "sequence", "--yes", "/tmp/repo"], "--yes"),
        (["--fix-backend=claude-opus-5", str(tmp_path)], "[tool.daydream.phases.fix]"),
    ]
    for argv, expected in rejected:
        with pytest.raises(SystemExit) as exc_info:
            _parse_args(argv)
        assert exc_info.value.code == 2, argv
        assert expected in capsys.readouterr().err, argv
    for improve_flag in ("--verbose=true", "--log=true"):
        with pytest.raises(SystemExit):
            _parse_improve_args(["improve", "/tmp/x", improve_flag])


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


@pytest.mark.parametrize("signum", [signal.SIGTERM])
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


def test_cli_maps_only_a_purely_grouped_interrupt_to_shutdown_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Nested interrupts exit 130; a sibling failure beside an interrupt is preserved."""
    failure = BaseExceptionGroup("task group", [KeyboardInterrupt(), RuntimeError("sibling failed")])
    raised: list[BaseException] = [
        BaseExceptionGroup("task group", [BaseExceptionGroup("nested", [KeyboardInterrupt()])]), failure]

    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise raised.pop(0)
    monkeypatch.setattr(anyio, "run", interrupted)
    with pytest.raises(SystemExit) as caught:
        cli.main([str(tmp_path), "--non-interactive"])
    assert caught.value.code == 130
    with pytest.raises(BaseExceptionGroup) as grouped:
        cli.main([str(tmp_path), "--non-interactive"])
    assert grouped.value is failure
