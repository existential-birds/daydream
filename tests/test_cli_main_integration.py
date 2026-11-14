"""Exercise the real CLI entrypoint, event-loop ownership and exit codes.

Tests are synchronous because cli.main owns anyio.run; an async test would
already own a loop. Pipeline cases stub network/backend/UI boundaries while
running real phases and Git. Visibility cases use an executable protocol fixture
and inspect what the child actually received.
"""

from __future__ import annotations

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
from daydream.phases import UnconfinedFindingError
from tests.harness.blocked_mode import blocking_releaser
from tests.harness.git_helpers import git
from tests.harness.protocol_cli import ProtocolCli, install_protocol_cli
from tests.test_artifact_visibility_integration import (
    _seed_visibility_canaries,
    _write_probe_extension,
)
from tests.test_deep_orchestrator import (
    _force_interactive,
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

def test_cli_main_only_fix_selects_fixing(
    git_repo: Path, tiny_diff_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only --fix selects fixing: a bare review ends with findings and --yes alone is refused.

    The operator answers yes to every question, so an offered fix gate would edit the tree."""
    _silence_all(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, tiny_diff_target)
    stub.fix_edit_line = "# fixed by the selected loop\n"
    stub.review_claims = [{"title": "Sample issue", "file": "api.py", "line": 1}]
    mute_side_effects()
    asked: list[str] = []

    def answer(_console: object, message: str, *_args: object, **_kwargs: object) -> str:
        asked.append(message)
        return "y"

    monkeypatch.setattr("daydream.run_context._prompt_user", answer)
    target = (str(tiny_diff_target), "--no-archive", "--no-cleanup", "--test-command", f"{sys.executable} -c pass")

    code = _cli_main_exit(monkeypatch, *target)
    assert (
        code, "### 1. Sample issue" in (tiny_diff_target / ".review-output.md").read_text(),
        [question for question in asked if "Apply fixes" in question],
        git(tiny_diff_target, "status", "--porcelain", "--untracked-files=no"),
    ) == (0, True, [], "")

    capsys.readouterr()
    assert _cli_main_exit(monkeypatch, *target, "--yes") == 2
    assert "add --fix" in capsys.readouterr().err

    assert _cli_main_exit(monkeypatch, *target, "--fix", "--yes") == 0
    assert "# fixed by the selected loop" in (tiny_diff_target / "api.py").read_text()

    # With nothing to review, a review and a selected fix loop both say so; the base-branch guard binds fixing only.
    empty = (str(git_repo), "--no-archive")
    capsys.readouterr()
    assert (_cli_main_exit(monkeypatch, *empty), "No diff found" in capsys.readouterr().out) == (0, True)
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **kw: None)
    assert _cli_main_exit(monkeypatch, *empty, "--fix", "--yes") == 1
    git(git_repo, "checkout", "-b", "unchanged")
    assert (
        _cli_main_exit(monkeypatch, *empty, "--fix", "--yes"), "No diff found" in capsys.readouterr().out,
    ) == (0, True)

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
        monkeypatch, str(repo_with_origin), "--fix", "--yes", "--worktree", "--copy", "safe.txt",
        "--copy", "../outside-source.txt",
    )
    assert code == 1
    wt_root = repo_with_origin / ".daydream" / "worktrees"
    assert not wt_root.exists() or not any(wt_root.iterdir())

def test_cli_main_prune_reanchor_removes_or_rejects_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    repo, _ = _repo(tmp_path)
    owner = resolve_private_workspace_owner(repo, locations=private_root_locations())
    target = operational_worktree_root(owner) / "run-abcd-reanchor"
    git(repo, "worktree", "add", "--detach", str(target), "HEAD")

    _silence_all(monkeypatch)
    monkeypatch.setattr("daydream.commands.improve.print_success", lambda *a, **k: None)
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **k: None)
    listed: list[str] = []
    monkeypatch.setattr("daydream.commands.improve.print_info", lambda _console, name: listed.append(str(name)))
    assert _cli_main_exit(monkeypatch, "improve", "list-reanchor", str(repo)) == 0
    assert listed == ["run-abcd-reanchor"]
    assert _cli_main_exit(monkeypatch, "improve", "prune-reanchor", "run-abcd-reanchor", str(repo)) == 0
    assert not target.exists()
    assert "run-abcd-reanchor" not in git(repo, "worktree", "list")
    listed.clear()
    assert _cli_main_exit(monkeypatch, "improve", "list-reanchor", str(repo)) == 0
    assert listed == []
    # A name outside the reanchor namespace is rejected with exit 1 and nothing is created.
    assert _cli_main_exit(monkeypatch, "improve", "prune-reanchor", "run-abc", str(repo)) == 1
    assert not any(p.name == "run-abc" for p in (repo / ".daydream" / "worktrees").glob("*"))
    assert not (repo / ".daydream" / "worktrees" / "run-abc").exists()

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
_PROBE_ARGV = ("--flow", "artifact-visibility-probe", "--backend", "codex", "--model", "fixture-model")
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


def test_cli_main_pre_config_failure_diagnoses_only_with_a_real_verbose_token(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Only an actual ``--verbose`` argv token (not ``--verbose=``, ``--log`` or after ``--``) diagnoses."""
    scan = cli._verbose_token_in_argv
    assert [scan(argv) for argv in (["--verbose", "/t"], ["review", "--verbose", "/t"], ["improve", "/t", "--verbose"])
            ] == [True, True, True]
    assert not any(scan(argv) for argv in (["/t", "--", "--verbose"], ["--verbose=true", "/t"], ["--log", "/t"],
                                           ["/t"], []))
    _silence(monkeypatch)
    monkeypatch.setattr("daydream.commands.common._resolve_cli_observability", _denied_observability)
    for verbose in (True, False):
        code = _cli_main_exit(monkeypatch, *(["--verbose"] if verbose else []), str(git_repo))
        _out, err = capsys.readouterr()
        assert code == 1
        assert ("RuntimeError" in err) is verbose
        assert ("observability boom" in err) is verbose
