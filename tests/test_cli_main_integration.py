"""Exercise the real CLI entrypoint, event-loop ownership and exit codes.

Tests are synchronous because cli.main owns anyio.run; an async test would
already own a loop. Pipeline cases stub network/backend/UI boundaries while
running real phases and Git. Visibility cases use an executable protocol fixture
and inspect what the child actually received.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from daydream import cli, git_ops
from daydream.artifact_visibility import (
    operational_worktree_root,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.backends.codex import CodexError
from daydream.phases import UnconfinedFindingError
from tests.harness.backend import ScriptedBackend
from tests.harness.blocked_mode import blocking_releaser
from tests.harness.git_helpers import bare_remote, git, tracked_source_state as _tracked_source_state
from tests.harness.protocol_cli import ProtocolCli, install_protocol_cli
from tests.test_artifact_visibility_integration import (
    _assert_frozen_outputs,
    _seed_visibility_canaries,
    _write_probe_extension,
)
from tests.test_deep_orchestrator import (
    _install_stub_backend,
    _silence,
)
from tests.test_improve_plans import _repo


def _silence_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence the shared agent harness plus the cli/runner gh and UI seams."""
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)


def _silence_cli_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub GitHub at the transport seam and silence the runner banner.

    Keep provenance detection real: acme/<inspected basename> identifies which
    checkout the detector inspected. Signal installation also remains real."""
    def repo_view(repo: Path, *, auth: git_ops.GitHubAuth,) -> tuple[str, str]:
        assert auth is git_ops.INHERIT_GITHUB_AUTH
        return "acme", Path(repo).name

    def pr_view(_repo: Path, _branch: int | None, *, auth: git_ops.GitHubAuth,) -> None:
        assert auth is git_ops.INHERIT_GITHUB_AUTH

    monkeypatch.setattr("daydream.git_ops.gh_repo_view", repo_view)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", pr_view)
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)


def _cli_main_exit(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int | str | None:
    monkeypatch.setattr(sys, "argv", ["daydream", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


def _denied_observability(*a: Any, **k: Any) -> Any:
    """Injected failing observability resolver: the pre-config failure case."""
    raise RuntimeError("observability boom")

def test_cli_main_clean_deep_run_exits_0(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _silence_all(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    assert _cli_main_exit(monkeypatch, str(multi_stack_target)) == 0

def test_cli_main_trajectory_pr_repo_is_target_not_cwd(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist the target checkout's slug in the trajectory.

    The GitHub seam encodes the inspected directory basename, so the artifact
    proves provenance follows the target even when the invoking cwd differs."""

    _silence_all(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    trajectory_path = tmp_path / "trajectory.json"
    assert (_cli_main_exit(monkeypatch, "--trajectory", str(trajectory_path), str(multi_stack_target)) == 0)

    assert trajectory_path.exists(), "deep run must write the trajectory to disk"
    data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert data["extra"]["pr_repo"] == f"acme/{multi_stack_target.name}"
    assert data["extra"]["pr_repo"] != f"acme/{Path.cwd().name}"

def test_cli_main_wrong_branch_exits_1(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _silence_all(monkeypatch)
    _install_stub_backend(monkeypatch, git_repo)
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **kw: None)

    assert _cli_main_exit(monkeypatch, str(git_repo)) == 1

def test_cli_main_confinement_valueerror_is_actionable_not_fatal(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Render escaped UnconfinedFindingError with actionable finding details.

    The normal fix path handles it earlier; this fallback must dispatch by type."""
    _silence_all(monkeypatch)

    async def _raising_run(*a: Any, **k: Any) -> int:
        raise UnconfinedFindingError("Finding file must be a confined repository-relative path")

    monkeypatch.setattr("daydream.cli.run", _raising_run)

    assert _cli_main_exit(monkeypatch, str(git_repo)) == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Finding file must be a confined repository-relative path" in out
    assert "Fatal Error" not in out  # actionable, not the bare generic string

def test_cli_main_rejects_workspace_copy_traversal(repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject traversal through --copy with exit 1 and no external-file/worktree residue.

    --worktree reaches copying before the ordinary WrongBranch guard."""
    _silence_all(monkeypatch)
    _install_stub_backend(monkeypatch, repo_with_origin)
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)

    code = _cli_main_exit(
        monkeypatch, str(repo_with_origin), "--worktree", "--copy", "safe.txt", "--copy", "../outside-source.txt",
    )
    assert code == 1
    wt_root = repo_with_origin / ".daydream" / "worktrees"
    assert not wt_root.exists() or not any(wt_root.iterdir())

def test_non_tty_auto_enables_non_interactive(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _silence_all(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)  # piped stdin
    monkeypatch.delenv("CI", raising=False)
    def forbidden_input(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("non-TTY run must not prompt")

    monkeypatch.setattr("daydream.run_context._prompt_user", forbidden_input)

    _cli_main_exit(monkeypatch, str(multi_stack_target))

    assert (multi_stack_target / ".review-output.md").exists()

def test_cli_main_prune_reanchor_removes_and_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    repo, _ = _repo(tmp_path)
    owner = resolve_private_workspace_owner(repo, locations=private_root_locations())
    target = operational_worktree_root(owner) / "run-abcd-reanchor"
    git(repo, "worktree", "add", "--detach", str(target), "HEAD")

    _silence_all(monkeypatch)
    monkeypatch.setattr("daydream.commands.improve.print_success", lambda *a, **k: None)
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **k: None)
    assert _cli_main_exit(monkeypatch, "improve", "prune-reanchor", "run-abcd-reanchor", str(repo)) == 0
    assert not target.exists()
    assert "run-abcd-reanchor" not in git(repo, "worktree", "list")

def test_cli_main_prune_reanchor_rejects_name_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    repo, _ = _repo(tmp_path)
    _silence_all(monkeypatch)
    monkeypatch.setattr("daydream.commands.improve.print_success", lambda *a, **k: None)
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **k: None)
    assert _cli_main_exit(monkeypatch, "improve", "prune-reanchor", "run-abc", str(repo)) == 1
    assert not any(p.name == "run-abc" for p in (repo / ".daydream" / "worktrees").glob("*"))
    assert not (repo / ".daydream" / "worktrees" / "run-abc").exists()

def test_cli_main_list_reanchor_lists_and_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    repo, _ = _repo(tmp_path)
    owner = resolve_private_workspace_owner(repo, locations=private_root_locations())
    target = operational_worktree_root(owner) / "run-abcd-reanchor"
    git(repo, "worktree", "add", "--detach", str(target), "HEAD")

    _silence_all(monkeypatch)
    listed: list[str] = []
    monkeypatch.setattr("daydream.commands.improve.print_info", lambda _console, name: listed.append(str(name)),)
    assert _cli_main_exit(monkeypatch, "improve", "list-reanchor", str(repo)) == 0
    assert listed == ["run-abcd-reanchor"]

def test_cli_main_list_reanchor_empty_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    repo, _ = _repo(tmp_path)
    _silence_all(monkeypatch)
    monkeypatch.setattr("daydream.commands.improve.print_info", lambda *a, **k: None)
    assert _cli_main_exit(monkeypatch, "improve", "list-reanchor", str(repo)) == 0


# Run the real CLI on the main thread for loop and signal behavior, with real Git, a Codex
# executable fixture, and a probe extension. Child observations establish what reached the agent;
# host callbacks cannot substitute.

SOURCE_CANARY = "SOURCE_CANARY"
PRIOR_REASONING_CANARY = "PRIOR_REASONING_CANARY"
CURRENT_REASONING_CANARY = "CURRENT_REASONING_CANARY"
SIBLING_REASONING_CANARY = "SIBLING_REASONING_CANARY"
RESUME_CACHE_CANARY = "RESUME_CACHE_CANARY"
SANCTIONED_INPUT_CANARY = "SANCTIONED_INPUT_CANARY"
PRIVATE_ROOT_CANARY = "PRIVATE_ROOT_CANARY"
RUNTIME_STATE_CANARY = "RUNTIME_STATE_CANARY"
# Everything that must stay dark in the model's cwd for the whole run.
_DARK_CANARIES = (PRIOR_REASONING_CANARY, CURRENT_REASONING_CANARY, SIBLING_REASONING_CANARY,
    RESUME_CACHE_CANARY, SANCTIONED_INPUT_CANARY, PRIVATE_ROOT_CANARY, RUNTIME_STATE_CANARY,
)
_PROBE_ARGV = ("--flow", "artifact-visibility-probe", "--backend", "codex", "--model", "fixture-model")
# One byte beyond the 12,288-byte inline sanctioned-input budget.
_OVERSIZE_PAYLOAD = ("OVERSIZE-1234" * 1000)[:12_289]


@dataclass(frozen=True)
class _VisibilityCase:
    """One seeded repo + probe extension + real Codex executable on ``PATH``."""

    repo: Path
    private_base: Path
    sanctioned_path: Path
    fixture: ProtocolCli


@pytest.fixture
def visibility_case(
    tiny_diff_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ext_dir: Any, artifact_runtime_root: Path,
) -> Callable[..., _VisibilityCase]:
    """Seed the canaries, register the probe flow, install the Codex fixture."""

    def _setup(*, oversize: bool = False,
        response_mode: Literal["success", "model_error", "process_error", "block"] = "success",
        payload: str = SANCTIONED_INPUT_CANARY,
    ) -> _VisibilityCase:
        repo = tiny_diff_target
        private_base = artifact_runtime_root.parent
        _seed_visibility_canaries(repo, private_base)
        sanctioned_dir = tmp_path / "external sanctioned artifacts"
        sanctioned_dir.mkdir()
        sanctioned_path = (sanctioned_dir / "evidence artifact.txt").resolve()
        sanctioned_path.write_text(payload, encoding="utf-8")
        _write_probe_extension(ext_dir, sanctioned_path, read_only=oversize, oversize=oversize)
        fixture = install_protocol_cli(
            tmp_path / "codex protocol fixture with spaces", "codex", response_mode=response_mode,
            sanctioned_files=(sanctioned_path,),
        )
        monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
        _silence_cli_and_runner(monkeypatch)
        return _VisibilityCase(repo, private_base, sanctioned_path, fixture)

    return _setup


def _assert_no_hidden_reasoning(observation: dict[str, Any]) -> None:
    assert observation["cwd_canaries"][SOURCE_CANARY] is True
    assert not any(observation["cwd_canaries"][canary] for canary in _DARK_CANARIES)
    assert observation["prompt_canaries"][CURRENT_REASONING_CANARY] is False
    assert observation["prompt_canaries"][SANCTIONED_INPUT_CANARY] is False
    entries = observation["cwd_entries"]
    assert "private sentinel.txt" not in entries
    assert "runtime state.txt" not in entries
    # Publication follows model execution; the child must not observe run state in its working
    # directory.
    assert not any(entry == ".daydream" or entry.startswith(".daydream/") for entry in entries)


def _assert_codex_observations(case: _VisibilityCase, *, expected_cwd: Path | None = None) -> set[Path]:
    """External observation is the authority for what the child really saw.

    Returns the distinct model cwds the two invocations actually ran in.
    """
    expected_digest = hashlib.sha256(case.sanctioned_path.read_bytes()).hexdigest()
    observations = case.fixture.read_observations()
    assert len(observations) == 2
    model_cwds: set[Path] = set()
    for observation in observations:
        assert observation["backend"] == "codex"
        assert observation["response_mode"] == "success"
        assert observation["process_outcome"] == "success"
        argv = observation["argv"]
        assert argv[argv.index("--sandbox") + 1] == "danger-full-access"
        cwd = Path(observation["effective_cwd"])
        model_cwds.add(cwd)
        assert argv[argv.index("--cd") + 1] == str(cwd)
        if expected_cwd is not None:
            assert cwd == expected_cwd.resolve()
        assert observation["sanctioned_reads"] == {str(case.sanctioned_path): expected_digest}
        _assert_no_hidden_reasoning(observation)
    return model_cwds


def test_artifact_visibility_cli_codex_in_place_publishes_after_model(
    visibility_case: Callable[..., _VisibilityCase], tmp_path: Path, archive_dir: Path
) -> None:
    """The real child sees source only; publication follows model execution.

    Explicit, public, archive, evaluation and dump outputs share one frozen identity."""
    case = visibility_case()
    explicit_trajectory = tmp_path / "explicit trajectory output.json"
    dump_dir = tmp_path / "explicit artifact dump"

    with pytest.raises(SystemExit) as exc:
        cli.main([
            str(case.repo), *_PROBE_ARGV, "--trajectory", str(explicit_trajectory), "--dump-artifacts", str(dump_dir),
        ])
    assert exc.value.code == 0

    _assert_codex_observations(case, expected_cwd=case.repo)
    _assert_frozen_outputs(
        case.repo, archive_dir, explicit_trajectory, dump_dir, backend="codex", model="fixture-model",
    )

def test_artifact_visibility_cli_codex_worktree_branch_and_paths_with_spaces(
    visibility_case: Callable[..., _VisibilityCase], tmp_path: Path, archive_dir: Path
) -> None:
    """``--worktree --branch <feature-ref>`` reviews a disposable worktree and
    leaves the source checkout byte-for-byte unchanged; destinations with
    spaces route through the real publication pipeline."""
    case = visibility_case()
    repo = case.repo
    # Real origin branches are required for branch resolution.
    origin = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-u", "origin", "main")
    git(repo, "push", "-u", "origin", "feature")
    git(repo, "fetch", "origin")
    source_before = _tracked_source_state(repo)
    explicit_trajectory = tmp_path / "worktree trajectory output with spaces.json"
    dump_dir = tmp_path / "worktree artifact dump with spaces"

    with pytest.raises(SystemExit) as exc:
        cli.main([str(repo), *_PROBE_ARGV, "--worktree", "--branch", "feature",
            "--trajectory", str(explicit_trajectory), "--dump-artifacts", str(dump_dir),
        ])
    assert exc.value.code == 0

    assert _tracked_source_state(repo) == source_before

    model_cwds = _assert_codex_observations(case)
    assert len(model_cwds) == 1
    model_cwd = model_cwds.pop()
    assert model_cwd != repo.resolve()
    assert model_cwd.is_relative_to(case.private_base / "workspaces")
    assert not model_cwd.exists(), "ephemeral worktree must be gone after the run"

    session_id = _assert_frozen_outputs(
        repo, archive_dir, explicit_trajectory, dump_dir, backend="codex", model="fixture-model",
    )
    assert (repo / ".daydream" / "runs" / session_id / "trajectory.json").exists()

def test_artifact_visibility_cli_codex_oversize_inline_input_fails_before_spawn(
    visibility_case: Callable[..., _VisibilityCase],
) -> None:
    assert len(_OVERSIZE_PAYLOAD.encode("utf-8")) == 12_289
    case = visibility_case(oversize=True, payload=_OVERSIZE_PAYLOAD)

    with pytest.raises(SystemExit) as exc:
        cli.main([str(case.repo), *_PROBE_ARGV])
    assert exc.value.code == 1

    assert list(case.fixture.observations.iterdir()) == []
    assert not case.fixture.entered.exists()

def test_artifact_visibility_cli_codex_publication_collision_restores_and_exits_one(
    visibility_case: Callable[..., _VisibilityCase], tmp_path: Path
) -> None:
    """Preserve a concurrent destination replacement and exit 1 on publication collision.

    The host replaces it while the child is blocked; cli.main stays on the main
    thread so signal handling and the FIFO handshake remain real."""
    case = visibility_case(response_mode="block")
    explicit_trajectory = tmp_path / "collision trajectory output.json"
    explicit_trajectory.write_bytes(b"pre-existing published trajectory bytes")
    replacement = b"concurrent replacement trajectory bytes"
    with blocking_releaser(case.fixture, 2, replacement=(explicit_trajectory, replacement), timeout_s=30,
            join_timeout_s=15, name="artifact-visibility-cli-collision",
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main([str(case.repo), *_PROBE_ARGV, "--trajectory", str(explicit_trajectory)])
    assert exc.value.code == 1

    observations = case.fixture.read_observations()
    assert len(observations) == 2
    for observation in observations:
        assert observation["response_mode"] == "block"
        assert observation["process_outcome"] == "block"

    assert explicit_trajectory.read_bytes() == replacement


def _install_chained_failure_backend(monkeypatch: pytest.MonkeyPatch, target: Path, outer: BaseException | None = None,
) -> None:

    if outer is None:
        outer = CodexError("failed to create disposable read-only checkout")
        outer.__cause__ = git_ops.GitError("isolation probe failure")
    monkeypatch.setattr("daydream.runner.create_backend",
        lambda name, model=None, **kwargs: ScriptedBackend(events=[outer], retryable=False),
    )

def test_verbose_token_scan_semantics() -> None:
    scan = cli._verbose_token_in_argv
    assert scan(["--verbose", "/t"]) is True
    assert scan(["review", "--verbose", "/t"]) is True
    assert scan(["improve", "/t", "--verbose"]) is True
    assert scan(["/t", "--", "--verbose"]) is False
    assert scan(["--verbose=true", "/t"]) is False
    assert scan(["--log", "/t"]) is False
    assert scan(["/t"]) is False
    assert scan([]) is False

def test_cli_main_fatal_default_concise_no_traceback(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _silence_all(monkeypatch)
    _install_chained_failure_backend(monkeypatch, multi_stack_target)

    code = _cli_main_exit(monkeypatch, "--review", str(multi_stack_target))
    out, err = capsys.readouterr()

    assert code == 1
    assert "failed to create disposable read-only checkout" in out + err
    assert "Traceback (most recent call last)" not in err
    assert "isolation probe failure" not in out + err
    assert "During handling" not in err
    assert "GitError" not in err

def test_cli_main_verbose_prints_redacted_chain_on_stderr(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _silence_all(monkeypatch)
    _install_chained_failure_backend(monkeypatch, multi_stack_target)

    code = _cli_main_exit(monkeypatch, "--verbose", "--review", str(multi_stack_target))
    out, err = capsys.readouterr()

    assert code == 1
    assert "failed to create disposable read-only checkout" in out + err
    assert "CodexError" in err and "failed to create disposable read-only checkout" in err
    assert "GitError" in err and "isolation probe failure" in err
    assert "directly caused by the following exception" in err
    assert err.index("CodexError") < err.index("GitError")
    assert "isolation probe failure" not in out

def test_cli_main_verbose_neutralizes_canaries_on_both_streams(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _silence_all(monkeypatch)
    sentinel = "ghp_" + "Q" * 12
    outer = CodexError(f"failed to create disposable read-only checkout token={sentinel}\x1b[31m")
    outer.__cause__ = git_ops.GitError("isolation probe failure\rBEEP\x07")
    _install_chained_failure_backend(monkeypatch, multi_stack_target, outer)

    code = _cli_main_exit(monkeypatch, "--verbose", "--review", str(multi_stack_target))
    out, err = capsys.readouterr()

    assert code == 1
    for stream in (out, err):
        assert sentinel not in stream
        assert "\x1b" not in stream and "\r" not in stream
    assert "[REDACTED" in err
    assert "CodexError" in err

def test_cli_main_formatter_failure_emits_fixed_marker_keeps_exit(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _silence_all(monkeypatch)
    _install_chained_failure_backend(monkeypatch, multi_stack_target)

    def _boom(*a: Any, **k: Any) -> str:
        raise RuntimeError("formatter exploded")

    monkeypatch.setattr("daydream.cli.format_verbose_exception", _boom)
    code = _cli_main_exit(monkeypatch, "--verbose", "--review", str(multi_stack_target))
    out, err = capsys.readouterr()

    assert code == 1
    assert "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]" in err
    assert "formatter exploded" not in err
    assert "failed to create disposable read-only checkout" in out + err

@pytest.mark.parametrize("verbose", [True, False], ids=["verbose-diagnoses", "default-hides"])
def test_cli_main_pre_config_failure(
    verbose: bool, git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.commands.common._resolve_cli_observability", _denied_observability)

    code = _cli_main_exit(monkeypatch, *(["--verbose"] if verbose else []), str(git_repo))
    _out, err = capsys.readouterr()

    assert code == 1
    assert ("RuntimeError" in err) is verbose
    if verbose:
        assert "observability boom" in err


class _ExplodingStrError(RuntimeError):

    def __str__(self) -> str:
        raise RuntimeError("cannot stringify hostile exception")


def _install_exploding_str_backend(monkeypatch: pytest.MonkeyPatch,) -> None:
    monkeypatch.setattr("daydream.runner.create_backend",
        lambda name, model=None, **kwargs: ScriptedBackend(events=[_ExplodingStrError("hostile fatal")], retryable=False
        ),
    )

@pytest.mark.parametrize("verbose", [False, True], ids=["default-fails-closed", "verbose-unavailable-marker"])
def test_cli_main_hostile_str_fatal(
    verbose: bool, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A failing exception __str__ must preserve exit 1 and a safe fatal panel.

    Verbose mode adds the fixed unavailable marker, never a secondary traceback.
    """
    _silence_all(monkeypatch)
    _install_exploding_str_backend(monkeypatch)

    code = _cli_main_exit(monkeypatch, *(["--verbose"] if verbose else []), "--review", str(multi_stack_target))
    out, err = capsys.readouterr()

    assert code == 1
    if verbose:
        assert "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]" in err
    assert "cannot stringify" not in out + err
    assert "Traceback (most recent call last)" not in err
    assert "Fatal Error" in out + err
