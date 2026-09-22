"""Real-path integration tests for ``daydream.cli.main`` exit-code propagation.

``cli.main()`` is the true process entrypoint: it installs signal handlers,
routes subcommands, parses argv via ``_parse_args``, then drives
``anyio.run(run, config)`` and finally ``sys.exit(exit_code)``. Until now only
``_parse_args`` was unit-tested; ``main()`` itself — including its
``anyio.run`` ownership of the event loop and its dedicated except clauses —
had zero coverage.

These tests INVOKE ``cli.main()`` for real and assert the PROCESS EXIT CODE:

  * a clean default deep run -> ``0`` (the code returned by ``runner.run``
    must flow through ``anyio.run`` -> ``sys.exit``), and
  * the ``WrongBranchError`` guard -> ``1`` (exercising ``main()``'s dedicated
    ``except git_ops.WrongBranchError`` clause).

They are deliberately SYNC ``def test_...`` functions, NOT ``async def``.
``cli.main()`` calls ``anyio.run(...)``, which starts its own event loop. Under
``asyncio_mode = "auto"`` an ``async def`` test already runs inside a running
loop, so calling ``anyio.run`` from there raises "Already running asyncio in
this thread". A plain sync test lets ``anyio.run`` own the loop — which is the
exact production code path we need to cover.

Only the external seams are mocked: the network/SDK ``Backend`` (via
``create_backend``), the ``gh``-shelling detection helpers, and interactive
UI prompts/heroes. ``run``, ``_dispatch``, ``run_deep`` and every ``phase_*``
run for real against a real temp git worktree.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest

from daydream import cli, git_ops
from daydream.backends.codex import CodexError
from daydream.phases import UnconfinedFindingError
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import bare_remote, git
from tests.harness.git_helpers import tracked_source_state as _tracked_source_state
from tests.harness.protocol_cli import ProtocolCli, install_protocol_cli

# Reuse the artifact-visibility seeding/assertion helpers instead of duplicating them.
from tests.test_artifact_visibility_integration import (
    _assert_frozen_outputs,
    _seed_visibility_canaries,
)

# Reuse the deep-pipeline stub from the exemplar instead of duplicating it.
from tests.test_deep_orchestrator import (
    _install_stub_backend,
    _silence,
)


def _silence_cli_and_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock only the external seams cli.main touches before/around the loop.

    - The provenance detectors (`_auto_detect_pr_number`/`_detect_repo_slug`)
      shell out to ``gh``. We mock one layer deeper — the ``git_ops`` ``gh``
      seam — rather than stubbing the detectors themselves, so the REAL
      cwd-vs-target path-threading logic runs (the thing #128 fixed). The
      ``gh_repo_view`` stub returns a slug keyed on the inspected path's
      basename (``acme/<dir name>``), making provenance deterministic and
      letting a test prove which directory was used. ``gh_pr_view`` -> None.
    - ``runner.print_phase_hero`` renders the DAYDREAM banner on the real run
      path; silence it so the test output stays clean. (It does not block, but
      patching keeps the captured output focused.)
    - signal-handler install is a no-op concern here; leave it real — it is a
      cheap, side-effect-free part of the production entrypoint we want covered.
    """
    def repo_view(
        repo: Path,
        *,
        auth: git_ops.GitHubAuth,
    ) -> tuple[str, str]:
        assert auth is git_ops.INHERIT_GITHUB_AUTH
        return "acme", Path(repo).name

    def pr_view(
        _repo: Path,
        _branch: int | None,
        *,
        auth: git_ops.GitHubAuth,
    ) -> None:
        assert auth is git_ops.INHERIT_GITHUB_AUTH

    monkeypatch.setattr("daydream.git_ops.gh_repo_view", repo_view)
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", pr_view)
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)


def _cli_main_exit(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int | str | None:
    """Drive the real ``cli.main`` with *argv* and return its process exit code."""
    monkeypatch.setattr(sys, "argv", ["daydream", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return exc.value.code


def test_cli_main_clean_deep_run_exits_0(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean default deep run drives cli.main -> anyio.run -> sys.exit(0).

    ``multi_stack_target`` is a real git repo checked out on ``feature`` with a
    real cross-stack diff. Driving ``sys.argv`` with just the target positional
    exercises the production default (deep multi-stack) pipeline end to end. The
    only mocks are the Backend (stub) and the gh/UI seams — ``run``,
    ``_dispatch``, ``run_deep`` and the ``phase_*`` functions all run for real.
    """
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    # SystemExit.code is the int returned by runner.run, propagated through
    # anyio.run -> sys.exit. Not a hardcoded 0 (see TDD proof in the PR).
    assert _cli_main_exit(monkeypatch, str(multi_stack_target)) == 0


def test_cli_main_trajectory_pr_repo_is_target_not_cwd(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end provenance: the written trajectory's extra.pr_repo is the
    target checkout's slug, not the invoking cwd (#128).

    This drives the full production entrypoint — ``cli.main`` -> ``anyio.run``
    -> ``run_deep`` -> ``TrajectoryRecorder._write`` — and asserts the OBSERVABLE
    on-disk artifact: the ATIF trajectory JSON. The ``gh`` seam returns
    ``acme/<dir basename>``, so the target yields ``acme/multi_stack`` while the
    cwd (the daydream repo) would yield a different basename. Asserting
    ``acme/multi_stack`` proves provenance is attributed to the target — the
    benchmark-harness pattern that regressed before the fix.
    """
    import json

    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    trajectory_path = tmp_path / "trajectory.json"
    assert (
        _cli_main_exit(monkeypatch, "--trajectory", str(trajectory_path), str(multi_stack_target))
        == 0
    )

    assert trajectory_path.exists(), "deep run must write the trajectory to disk"
    data = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert data["extra"]["pr_repo"] == f"acme/{multi_stack_target.name}"
    assert data["extra"]["pr_repo"] != f"acme/{Path.cwd().name}"


def test_cli_main_wrong_branch_exits_1(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The WrongBranch guard drives cli.main's dedicated except clause -> exit 1.

    ``git_repo`` is a real repo checked out on ``main`` (the base branch) with a
    single commit and no feature branch. Running ``daydream <repo>`` with no
    ``--branch``/``--worktree`` hits the ``_dispatch`` guard, which raises
    ``git_ops.WrongBranchError``; ``runner.run`` re-raises it and
    ``cli.main``'s ``except git_ops.WrongBranchError`` clause calls
    ``sys.exit(1)``. The stub backend is installed but never reached.
    """
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_stub_backend(monkeypatch, git_repo)
    # The error path renders a panel via print_error; silence it.
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **kw: None)

    assert _cli_main_exit(monkeypatch, str(git_repo)) == 1


def test_cli_main_confinement_valueerror_is_actionable_not_fatal(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A confinement rejection that reaches cli.main renders actionably.

    Defense-in-depth fallback: the primary fix routes the rejection through
    ``_step_fix`` before it reaches this handler, but if it ever escapes, the
    operator must see which class of finding is at fault, not a bare
    "Fatal Error". The handler matches by type (``UnconfinedFindingError``),
    not by message string.
    """
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)

    async def _raising_run(*a: Any, **k: Any) -> int:
        raise UnconfinedFindingError("Finding file must be a confined repository-relative path")

    monkeypatch.setattr("daydream.cli.run", _raising_run)

    assert _cli_main_exit(monkeypatch, str(git_repo)) == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Finding file must be a confined repository-relative path" in out
    assert "Fatal Error" not in out  # actionable, not the bare generic string


def test_cli_main_rejects_workspace_copy_traversal(
    repo_with_origin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid --copy entry through the real entrypoint exits 1, touches no
    external file, and leaves no stray ephemeral worktree.

    ``--worktree`` forces an ephemeral workspace, so ``copy_files_into_ephemeral``
    runs (with ``--copy`` supplying the offending entry). The WorkspaceCopyPathError
    propagates out of ``open_workspace`` to runner.run's ``except git_ops.GitError``
    -> return 1. The WrongBranch guard is bypassed because force_worktree is set
    (runner.py:762), so the copy error fires first.
    """
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_stub_backend(monkeypatch, repo_with_origin)
    # The error path renders a "Workspace Error" panel via runner.print_error; silence it.
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)

    code = _cli_main_exit(
        monkeypatch,
        str(repo_with_origin),
        "--worktree",
        "--copy",
        "safe.txt",
        "--copy",
        "../outside-source.txt",
    )
    assert code == 1
    # No stray ephemeral worktree child remains after cleanup.
    wt_root = repo_with_origin / ".daydream" / "worktrees"
    assert not wt_root.exists() or not any(wt_root.iterdir())


def test_non_tty_auto_enables_non_interactive(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A piped (non-TTY) stdin auto-enables non-interactive with no flag.

    This drives the production entrypoint (``cli.main`` -> ``runner.run``) with a
    bare target and a non-TTY stdin. The interactivity axis must resolve from the
    environment (non-TTY) rather than requiring an explicit ``--non-interactive``
    flag. The raw input boundary must never be called, and the review completes.
    """
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)  # piped stdin
    monkeypatch.delenv("CI", raising=False)
    def forbidden_input(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("non-TTY run must not prompt")

    monkeypatch.setattr("daydream.run_context._prompt_user", forbidden_input)

    _cli_main_exit(monkeypatch, str(multi_stack_target))

    assert (multi_stack_target / ".review-output.md").exists()


def test_cli_main_prune_reanchor_removes_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.harness.git_helpers import git
    from tests.test_improve_plans import _repo

    repo, _ = _repo(tmp_path)
    target = repo / ".daydream" / "worktrees" / "run-abcd-reanchor"
    git(repo, "worktree", "add", "--detach", str(target), "HEAD")

    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    monkeypatch.setattr("daydream.cli.print_success", lambda *a, **k: None)
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **k: None)
    assert _cli_main_exit(monkeypatch, "improve", "prune-reanchor", "run-abcd-reanchor", str(repo)) == 0
    assert not target.exists()
    assert "run-abcd-reanchor" not in git(repo, "worktree", "list")


def test_cli_main_prune_reanchor_rejects_name_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_improve_plans import _repo

    repo, _ = _repo(tmp_path)
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    monkeypatch.setattr("daydream.cli.print_success", lambda *a, **k: None)
    monkeypatch.setattr("daydream.cli.print_error", lambda *a, **k: None)
    assert _cli_main_exit(monkeypatch, "improve", "prune-reanchor", "run-abc", str(repo)) == 1
    assert not any(
        p.name == "run-abc" for p in (repo / ".daydream" / "worktrees").glob("*")
    )
    assert not (repo / ".daydream" / "worktrees" / "run-abc").exists()


def test_cli_main_list_reanchor_lists_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``improve list-reanchor`` through cli.main exits 0 and names the
    re-anchor worktrees the automatic prune would remove.

    The sync sub-verb short-circuits in ``main()`` (no anyio/agent work), so no
    stub backend is needed; the observable outcome is the exit code and the
    printed worktree names.
    """
    from tests.harness.git_helpers import git
    from tests.test_improve_plans import _repo

    repo, _ = _repo(tmp_path)
    target = repo / ".daydream" / "worktrees" / "run-abcd-reanchor"
    git(repo, "worktree", "add", "--detach", str(target), "HEAD")

    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    listed: list[str] = []
    monkeypatch.setattr(
        "daydream.cli.print_info",
        lambda _console, name: listed.append(str(name)),
    )
    assert _cli_main_exit(monkeypatch, "improve", "list-reanchor", str(repo)) == 0
    assert listed == ["run-abcd-reanchor"]


def test_cli_main_list_reanchor_empty_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty re-anchor set still exits 0 — listing nothing is not an error."""
    from tests.test_improve_plans import _repo

    repo, _ = _repo(tmp_path)
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    monkeypatch.setattr("daydream.cli.print_info", lambda *a, **k: None)
    assert _cli_main_exit(monkeypatch, "improve", "list-reanchor", str(repo)) == 0


# ---------------------------------------------------------------------------
# Artifact-visibility CLI acceptance rows (P10 Task 6).
#
# These drive the REAL production entrypoint — synchronous ``cli.main([...])``
# on the main thread (real argv parsing, real signal installation, real
# ``anyio.run`` ownership, real ``sys.exit``) — with a real Git repo, a real
# Codex executable fixture on ``PATH`` (``tests/harness/protocol_cli.py``), and
# a real probe extension registered through the ``ext_dir`` seam. External
# observation is the authority: the fixture executable records what actually
# reached the child (argv, stdin, cwd contents as canary booleans), never a
# host-side callback. No backend stub, no spawn mock, no fake Git.
# ---------------------------------------------------------------------------

SOURCE_CANARY = "SOURCE_CANARY"
PRIOR_REASONING_CANARY = "PRIOR_REASONING_CANARY"
CURRENT_REASONING_CANARY = "CURRENT_REASONING_CANARY"
SIBLING_REASONING_CANARY = "SIBLING_REASONING_CANARY"
RESUME_CACHE_CANARY = "RESUME_CACHE_CANARY"
SANCTIONED_INPUT_CANARY = "SANCTIONED_INPUT_CANARY"
PRIVATE_ROOT_CANARY = "PRIVATE_ROOT_CANARY"
RUNTIME_STATE_CANARY = "RUNTIME_STATE_CANARY"
# Everything that must stay dark in the model's cwd for the whole run.
_DARK_CANARIES = (
    PRIOR_REASONING_CANARY, CURRENT_REASONING_CANARY, SIBLING_REASONING_CANARY,
    RESUME_CACHE_CANARY, SANCTIONED_INPUT_CANARY, PRIVATE_ROOT_CANARY, RUNTIME_STATE_CANARY,
)
# Shared argv tail: the probe flow on the real Codex fixture executable.
_PROBE_ARGV = ("--flow", "artifact-visibility-probe", "--backend", "codex", "--model", "fixture-model")
# One byte beyond the 12,288-byte inline sanctioned-input budget.
_OVERSIZE_PAYLOAD = ("OVERSIZE-1234" * 1000)[:12_289]


def _write_visibility_probe_extension(
    ext_dir: Any, sanctioned_path: Path, *, oversize: bool = False
) -> None:
    """Register the ``artifact-visibility-probe`` extension flow.

    Two real ``run_agent`` calls against the resolved backend: the first
    response's canary lands in the live trajectory, then the second external
    invocation searches its actual model cwd — proving an earlier response is
    not discoverable during the same run. With ``oversize`` the sanctioned
    input is one byte beyond the 12,288 inline budget, so
    ``prepare_sanctioned_inputs`` must fail closed before anything spawns.
    """
    ext_dir.write_module(
        "from pathlib import Path\n"
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.prompt_budget import prepare_sanctioned_inputs\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def _probe(ctx):\n"
        "    assert ctx.artifacts is not None\n"
        "    backend = ctx.backend_for('review')\n"
        f"    sanctioned_path = Path({str(sanctioned_path)!r})\n"
        "    prepared = prepare_sanctioned_inputs(\n"
        "        backend, ctx.work.repo, {'external evidence': sanctioned_path},\n"
        f"        read_only={oversize!r},\n"
        "    )\n"
        + (
            # Oversize row: preparation itself must raise; nothing below runs.
            "    raise AssertionError('oversize sanctioned input must not reach run_agent')\n"
            if oversize
            else
            "    first, _discarded_continuation, _reason = await run_agent(\n"
            "        backend, ctx.work.repo, 'FIRST VISIBILITY PROBE',\n"
            "        phase=DaydreamPhase.REVIEW, sanctioned_inputs=prepared,\n"
            "    )\n"
            "    assert 'CURRENT_REASONING_CANARY' in first\n"
            "    await run_agent(\n"
            "        backend, ctx.work.repo, 'SECOND VISIBILITY PROBE',\n"
            "        phase=DaydreamPhase.REVIEW, sanctioned_inputs=prepared,\n"
            "    )\n"
        )
        + "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='artifact-visibility-probe', run=_probe))\n"
        "    registry.set_flow('artifact-visibility-probe', ['artifact-visibility-probe'])\n"
    )


@dataclass(frozen=True)
class _VisibilityCase:
    """One seeded repo + probe extension + real Codex executable on ``PATH``."""

    repo: Path
    private_base: Path
    sanctioned_path: Path
    fixture: ProtocolCli


@pytest.fixture
def visibility_case(
    tiny_diff_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: Any,
    artifact_runtime_root: Path,
) -> Callable[..., _VisibilityCase]:
    """Seed the canaries, register the probe flow, install the Codex fixture."""

    def _setup(
        *,
        oversize: bool = False,
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
        _write_visibility_probe_extension(ext_dir, sanctioned_path, oversize=oversize)
        fixture = install_protocol_cli(
            tmp_path / "codex protocol fixture with spaces",
            "codex",
            response_mode=response_mode,
            sanctioned_files=(sanctioned_path,),
        )
        monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
        _silence_cli_and_runner(monkeypatch)
        return _VisibilityCase(repo, private_base, sanctioned_path, fixture)

    return _setup


def _assert_no_hidden_reasoning(observation: dict[str, Any]) -> None:
    """During the model calls, nothing but the committed source may be visible."""
    assert observation["cwd_canaries"][SOURCE_CANARY] is True
    assert not any(observation["cwd_canaries"][canary] for canary in _DARK_CANARIES)
    assert observation["prompt_canaries"][CURRENT_REASONING_CANARY] is False
    assert observation["prompt_canaries"][SANCTIONED_INPUT_CANARY] is False
    entries = observation["cwd_entries"]
    assert "private sentinel.txt" not in entries
    assert "runtime state.txt" not in entries
    # Publication happens only after the model: no run state existed in the
    # model cwd while the external executable was searching it.
    assert not any(entry == ".daydream" or entry.startswith(".daydream/") for entry in entries)


def _assert_codex_observations(
    case: _VisibilityCase, *, expected_cwd: Path | None = None
) -> set[Path]:
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


def _replace_destination_after_entered(
    fixture: ProtocolCli, destination: Path, replacement: bytes, *,
    expected_pids: int, stop: threading.Event, failures: list[BaseException],
) -> None:
    """Host helper thread: coordinate with the blocked executable over its FIFO.

    Waits for each blocked invocation's atomic ``entered`` marker, replaces the
    explicit trajectory destination exactly once (while the model is still
    running), and publishes the ``release`` byte so the executable can finish.
    """
    seen: set[int] = set()
    replaced = False
    deadline = time.monotonic() + 30
    try:
        while len(seen) < expected_pids:
            if stop.is_set():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("blocked protocol invocation did not enter")
            try:
                entered = json.loads(fixture.entered.read_text(encoding="utf-8"))
                pid = entered["pid"]
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                stop.wait(0.01)
                continue
            if not isinstance(pid, int) or pid in seen:
                stop.wait(0.01)
                continue
            seen.add(pid)
            if not replaced:
                destination.write_bytes(replacement)
                replaced = True
            with fixture.release.open("wb", buffering=0) as fifo:
                fifo.write(b"release")
    except BaseException as exc:  # surfaced by the test body
        failures.append(exc)


def test_artifact_visibility_cli_codex_in_place_publishes_after_model(
    visibility_case: Callable[..., _VisibilityCase], tmp_path: Path, archive_dir: Path
) -> None:
    """In-place CLI run: publication happens only after the model, and every
    final output shares one frozen identity.

    ``cli.main`` parses the real argv, installs real signal handlers, drives
    ``anyio.run`` -> ``runner.run`` -> the extension probe flow -> the real
    CodexBackend -> the real fixture executable, then freezes and publishes.
    The external observation proves the model saw the source and nothing else;
    the post-run assertions prove explicit/public/archive/eval/dump are
    byte-bound to one session.
    """
    case = visibility_case()
    explicit_trajectory = tmp_path / "explicit trajectory output.json"
    dump_dir = tmp_path / "explicit artifact dump"

    with pytest.raises(SystemExit) as exc:
        cli.main([
            str(case.repo), *_PROBE_ARGV,
            "--trajectory", str(explicit_trajectory), "--dump-artifacts", str(dump_dir),
        ])
    assert exc.value.code == 0

    _assert_codex_observations(case, expected_cwd=case.repo)
    _assert_frozen_outputs(
        case.repo, archive_dir, explicit_trajectory, dump_dir,
        backend="codex", model="fixture-model",
    )


def test_artifact_visibility_cli_codex_worktree_branch_and_paths_with_spaces(
    visibility_case: Callable[..., _VisibilityCase], tmp_path: Path, archive_dir: Path
) -> None:
    """``--worktree --branch <feature-ref>`` reviews a disposable worktree and
    leaves the source checkout byte-for-byte unchanged; destinations with
    spaces route through the real publication pipeline."""
    case = visibility_case()
    repo = case.repo
    # ``--branch`` resolves the feature ref against a real origin: publish a
    # bare remote and push both branches (real Git, no fakes).
    origin = bare_remote(tmp_path / "origin.git")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "-u", "origin", "main")
    git(repo, "push", "-u", "origin", "feature")
    git(repo, "fetch", "origin")
    source_before = _tracked_source_state(repo)
    explicit_trajectory = tmp_path / "worktree trajectory output with spaces.json"
    dump_dir = tmp_path / "worktree artifact dump with spaces"

    with pytest.raises(SystemExit) as exc:
        cli.main([
            str(repo), *_PROBE_ARGV, "--worktree", "--branch", "feature",
            "--trajectory", str(explicit_trajectory), "--dump-artifacts", str(dump_dir),
        ])
    assert exc.value.code == 0

    # Source Git refs, index, and tracked bytes are exactly as before.
    assert _tracked_source_state(repo) == source_before

    model_cwds = _assert_codex_observations(case)
    assert len(model_cwds) == 1
    model_cwd = model_cwds.pop()
    assert model_cwd != repo.resolve()
    assert model_cwd.is_relative_to(case.private_base / "workspaces")
    assert not model_cwd.exists(), "ephemeral worktree must be gone after the run"

    session_id = _assert_frozen_outputs(
        repo, archive_dir, explicit_trajectory, dump_dir,
        backend="codex", model="fixture-model",
    )
    # The public run is published to the SOURCE checkout, not the worktree.
    assert (repo / ".daydream" / "runs" / session_id / "trajectory.json").exists()


def test_artifact_visibility_cli_codex_oversize_inline_input_fails_before_spawn(
    visibility_case: Callable[..., _VisibilityCase],
) -> None:
    """A 12,289-byte sanctioned input exceeds the 12,288-byte inline budget.

    The extension selects read-only Codex (inline transport), so
    ``prepare_sanctioned_inputs`` must fail closed before any model process is
    spawned: ``cli.main`` exits 1 and the observation directory stays empty.
    """
    assert len(_OVERSIZE_PAYLOAD.encode("utf-8")) == 12_289
    case = visibility_case(oversize=True, payload=_OVERSIZE_PAYLOAD)

    with pytest.raises(SystemExit) as exc:
        cli.main([str(case.repo), *_PROBE_ARGV])
    assert exc.value.code == 1

    # Nothing was spawned: no observation, no entered marker.
    assert list(case.fixture.observations.iterdir()) == []
    assert not case.fixture.entered.exists()


def test_artifact_visibility_cli_codex_publication_collision_restores_and_exits_one(
    visibility_case: Callable[..., _VisibilityCase], tmp_path: Path
) -> None:
    """A destination replaced while the model runs must never be clobbered.

    The explicit trajectory destination pre-exists. A host helper thread waits
    for the blocked executable's atomic ``entered`` marker, replaces the
    destination, then publishes the FIFO ``release`` byte — all while
    ``cli.main`` stays blocked on the main thread (real signal installation
    exercised). Publication must detect the collision, exit 1, and preserve
    the concurrent replacement exactly.
    """
    case = visibility_case(response_mode="block")
    explicit_trajectory = tmp_path / "collision trajectory output.json"
    explicit_trajectory.write_bytes(b"pre-existing published trajectory bytes")
    replacement = b"concurrent replacement trajectory bytes"
    stop = threading.Event()
    failures: list[BaseException] = []
    releaser = threading.Thread(
        target=_replace_destination_after_entered,
        args=(case.fixture, explicit_trajectory, replacement),
        kwargs={"expected_pids": 2, "stop": stop, "failures": failures},
        name="artifact-visibility-cli-collision",
        daemon=True,
    )
    releaser.start()
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main([str(case.repo), *_PROBE_ARGV, "--trajectory", str(explicit_trajectory)])
    finally:
        stop.set()
        releaser.join(timeout=15)
    assert exc.value.code == 1
    assert not releaser.is_alive()
    assert failures == []

    # Both blocked invocations were observed and released.
    observations = case.fixture.read_observations()
    assert len(observations) == 2
    for observation in observations:
        assert observation["response_mode"] == "block"
        assert observation["process_outcome"] == "block"

    # Exact preservation: the concurrent replacement was not clobbered.
    assert explicit_trajectory.read_bytes() == replacement


def _install_chained_failure_backend(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    outer: BaseException | None = None,
) -> None:
    """Install a ScriptedBackend that raises a CodexError chained to a GitError."""
    from daydream.backends.codex import CodexError

    if outer is None:
        outer = CodexError("failed to create disposable read-only checkout")
        outer.__cause__ = git_ops.GitError("isolation probe failure")
    monkeypatch.setattr(
        "daydream.runner.create_backend",
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
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


def test_cli_main_verbose_diagnoses_pre_config_failure(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _silence(monkeypatch)

    def _denied(*a: Any, **k: Any) -> Any:
        raise RuntimeError("observability boom")

    monkeypatch.setattr("daydream.cli._resolve_cli_observability", _denied)

    code = _cli_main_exit(monkeypatch, "--verbose", str(git_repo))
    _out, err = capsys.readouterr()
    assert code == 1
    assert "RuntimeError" in err
    assert "observability boom" in err


def test_cli_main_default_hides_pre_config_failure_details(
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _silence(monkeypatch)

    def _denied(*a: Any, **k: Any) -> Any:
        raise RuntimeError("observability boom")

    monkeypatch.setattr("daydream.cli._resolve_cli_observability", _denied)

    code = _cli_main_exit(monkeypatch, str(git_repo))
    _out, err = capsys.readouterr()
    assert code == 1
    assert "RuntimeError" not in err


class _ExplodingStrError(RuntimeError):
    """An exception whose ``__str__`` raises: the hostile-message case."""

    def __str__(self) -> str:
        raise RuntimeError("cannot stringify hostile exception")


def _install_exploding_str_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install a ScriptedBackend whose failure exception cannot be str()ed.

    The exception escapes ``runner.run`` into ``cli.main``'s generic fatal
    handler — the exact shape issue #1236's fail-closed contract covers.
    """
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: ScriptedBackend(
            events=[_ExplodingStrError("hostile fatal")], retryable=False
        ),
    )


def test_cli_main_hostile_str_fatal_default_fails_closed(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A fatal exception whose ``__str__`` raises must not escape the generic
    handler: the panel is empty-safe, never a raw interpreter traceback, never
    the hostile message's own text (issue #1236 fail-closed contract)."""
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_exploding_str_backend(monkeypatch)

    code = _cli_main_exit(monkeypatch, "--review", str(multi_stack_target))
    out, err = capsys.readouterr()

    assert code == 1
    assert "cannot stringify" not in out + err
    assert "Traceback (most recent call last)" not in err
    assert "Fatal Error" in out + err


def test_cli_main_hostile_str_fatal_verbose_emits_unavailable_marker(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The verbose variant still fails closed: the verbose diagnostic is the
    fixed unavailable marker (never a secondary traceback), the panel stays
    safe, and exit code 1 is preserved."""
    _silence(monkeypatch)
    _silence_cli_and_runner(monkeypatch)
    _install_exploding_str_backend(monkeypatch)

    code = _cli_main_exit(monkeypatch, "--verbose", "--review", str(multi_stack_target))
    out, err = capsys.readouterr()

    assert code == 1
    assert "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]" in err
    assert "cannot stringify" not in out + err
    assert "Traceback (most recent call last)" not in err
    assert "Fatal Error" in out + err
