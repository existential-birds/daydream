"""Shared pytest fixtures for the daydream test suite."""

import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from daydream.exploration import (
    Convention,
    Dependency,
    ExplorationContext,
    FileInfo,
)
from daydream.workspace import WorkContext

# Isolate the test process from any inherited git environment. When the suite
# runs under a git command that exports them — most importantly a pre-push hook
# — variables like GIT_DIR/GIT_INDEX_FILE/GIT_WORK_TREE point every `git`
# subprocess at the real repository. Tests that create temp git repos would
# then commit, branch, and checkout against the live worktree instead of their
# fixtures, corrupting it. Strip these at collection time so every git
# invocation (test helpers and production git_ops alike) resolves its repo from
# the working directory, exactly as it does when run outside a hook.
for _git_env_var in (
    "GIT_DIR",
    "GIT_INDEX_FILE",
    "GIT_WORK_TREE",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
):
    os.environ.pop(_git_env_var, None)

# --- Real-git fixtures ------------------------------------------------------
#
# Mirrors the helpers that previously lived only in tests/test_git_ops.py.
# Lifted here so other test modules (notably tests/test_pr_review.py) can
# build real repos instead of mocking subprocess. tests/test_workspace.py
# still has its own helpers (it needs additional plumbing for bare-origin
# push semantics) — left untouched on purpose.


def _git(repo: Path, *args: str, check: bool = True) -> str:
    """Run a git command in *repo* and return stripped stdout (test helper)."""
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )
    return proc.stdout.strip()


def _configure_identity(repo: Path) -> None:
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _configure_identity(repo)


def _bare_remote(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "--bare", "-b", "main")
    return path


def _make_repo_with_main(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    _init_repo(repo)
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _commit(repo, "initial")
    return repo


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Initialize a fresh git repo at tmp_path with one initial commit on `main`."""
    return _make_repo_with_main(tmp_path)


@pytest.fixture
def feature_branch_repo(tmp_path: Path) -> Path:
    """Git repo with a committed Python diff on a non-``main`` feature branch.

    Built on ``_make_repo_with_main`` + the shared ``_git``/``_commit`` helpers
    (not inline ``subprocess.run``). Seeds ``main.py`` on ``main`` and commits,
    then checks out a ``feature`` branch and commits a modification — so the
    branch carries a real ``main...HEAD`` diff. Leaves a clean working tree on a
    non-``main`` branch, satisfying loop mode's dirty-tree preflight check.

    Intentionally does NOT pre-write ``.review-output.md``: under replay the
    parse phase consumes structured output, so the file's absence is the correct
    shape (and a Spike-2 regression guard).
    """
    repo = _make_repo_with_main(tmp_path, name="loop_project")
    main_py = repo / "main.py"
    main_py.write_text("def hello():\n    return 'world'\n")
    _git(repo, "add", "main.py")
    _commit(repo, "add main.py")
    _git(repo, "checkout", "-b", "feature")
    main_py.write_text("def hello():\n    return 'universe'\n")
    _git(repo, "add", "main.py")
    _commit(repo, "modify main.py")
    return repo


@pytest.fixture
def bare_origin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A bare repo suitable for use as `origin`."""
    path = tmp_path_factory.mktemp("origin") / "remote.git"
    return _bare_remote(path)


@pytest.fixture
def repo_with_origin(tmp_path: Path, bare_origin: Path) -> Path:
    """A working repo cloned from bare_origin, ready for push/fetch."""
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "remote", "add", "origin", str(bare_origin))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "remote", "set-head", "origin", "main")
    return repo


@pytest.fixture
def exploration_context_fixture() -> ExplorationContext:
    """Populated ExplorationContext used by Phase 03 review-integration tests.

    Provides a minimal but realistic context: one modified file, one
    convention, and one dependency edge. Wave 1 implementation work
    will consume this fixture to verify prompt-injection behavior.
    """
    return ExplorationContext(
        affected_files=[
            FileInfo(
                path="daydream/runner.py",
                role="modified",
                summary="Run orchestrator",
            ),
        ],
        conventions=[
            Convention(
                name="snake_case_modules",
                description="All modules use snake_case filenames",
                source="inferred from code",
            ),
        ],
        dependencies=[
            Dependency(
                source="daydream/runner.py",
                target="daydream/phases.py",
                relationship="imports",
            ),
        ],
    )


@pytest.fixture
def exploration_dir_fixture(tmp_path: Path, exploration_context_fixture: ExplorationContext) -> Path:
    """Write exploration context to a temp dir for phase prompt tests."""
    exploration_dir = tmp_path / "exploration"
    exploration_context_fixture.write_to_dir(exploration_dir)
    return exploration_dir


@pytest.fixture
def multi_stack_target(tmp_path: Path) -> Path:
    """Git repo with a Python + React + Markdown diff on a feature branch.

    Used by deep-mode orchestrator and integration tests (plan 05-09 / 05-10)
    to exercise the multi-stack routing path. The repo has an ``init`` commit
    on ``main`` and a ``change`` commit on a ``feature`` branch that modifies
    one file per stack (``api.py``, ``App.tsx``, ``README.md``).
    """
    project = tmp_path / "multi_stack"
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'world'\n")
    (project / "App.tsx").write_text("export const App = () => <div>hello</div>;\n")
    (project / "README.md").write_text("# Project\n")
    subprocess.run(  # noqa: S603
        ["git", "init", "-b", "main"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "config", "user.email", "test@test.com"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "config", "user.name", "Test"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "add", "."],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "commit", "-m", "init"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "checkout", "-b", "feature"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    (project / "api.py").write_text("def hello():\n    return 'universe'\n")
    (project / "App.tsx").write_text("export const App = () => <div>universe</div>;\n")
    (project / "README.md").write_text("# Project\n\nUpdated.\n")
    subprocess.run(  # noqa: S603
        ["git", "add", "."],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    subprocess.run(  # noqa: S603
        ["git", "commit", "-m", "change"],  # noqa: S607
        cwd=project,
        capture_output=True,
        check=True,
    )
    return project


@pytest.fixture
def make_work() -> Callable[..., WorkContext]:
    """Builder for synthetic ``WorkContext`` instances.

    Stage 3 threads ``WorkContext`` through every ``phase_*`` function. Tests
    that previously passed a raw ``Path`` as the second positional argument
    use this fixture to construct a context anchored on *repo* with stable
    fake SHAs. The builder mirrors the production fields so tests don't need
    to spin up a real git repo just to call a phase.
    """

    def _make(
        repo: Path,
        *,
        base_branch: str = "main",
        base_sha: str = "DEADBEEF",
        head_sha: str = "CAFEBABE",
        head_branch: str | None = "feat/x",
        is_ephemeral: bool = False,
    ) -> WorkContext:
        return WorkContext(
            repo=repo,
            source=repo,
            base_branch=base_branch,
            base_sha=base_sha,
            head_branch=head_branch,
            head_sha=head_sha,
            is_ephemeral=is_ephemeral,
            run_id="20260101000000-deadbeef",
        )

    return _make


@pytest.fixture(autouse=True)
def _reset_agent_state():
    """Reset the ``AgentState`` singleton before AND after every test.

    The interaction axes (``non_interactive``, ``assume``) and ``quiet_mode``
    live on a module-level ``AgentState`` singleton in ``daydream.agent``. A test
    that drives ``run()`` calls ``set_non_interactive``/``set_assume``, leaving
    the singleton dirty for the next test — which silently changes the resolved
    answer at every ``resolve_gate`` call site. Reset on both edges so each test
    starts from defaults regardless of order.
    """
    from daydream.agent import reset_state

    reset_state()
    yield
    reset_state()


@pytest.fixture(autouse=True)
def _reset_gh_token_env():
    """Reset the ``gh`` token-env singleton before AND after every test.

    The ``gh`` subprocess environment lives on a module-level singleton in
    ``daydream.git_ops`` (``set_gh_token_env``/``reset_gh_token_env``). A test
    that drives ``run()`` with GitHub App credentials sets it, leaving the
    singleton dirty for the next test — which would silently inject a stale
    ``GH_TOKEN`` into every subsequent ``gh`` call. Reset on both edges so each
    test starts from parent-env inheritance regardless of order.
    """
    from daydream import git_ops

    git_ops.reset_gh_token_env()
    yield
    git_ops.reset_gh_token_env()


@pytest.fixture(autouse=True)
def _reset_trajectory_recorder():
    """Clear the trajectory ContextVar before AND after every test.

    Mirrors ``daydream.agent.reset_state()`` for ``AgentState`` (CORE-10 / D-17).
    Prevents cross-test bleed when a test forgets to wrap recorder usage in
    ``async with TrajectoryRecorder(...)``.

    Lazy-imports ``_reset_recorder_for_tests`` to avoid eagerly loading
    ``daydream.trajectory`` (and its Pydantic-heavy ATIF imports) at
    pytest-collect time.
    """
    from daydream.trajectory import _reset_recorder_for_tests

    _reset_recorder_for_tests()
    yield
    _reset_recorder_for_tests()


@pytest.fixture(autouse=True)
def archive_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate DAYDREAM_ARCHIVE_DIR to a per-test tmpdir so tests never touch ~/.daydream/archive/."""
    path = tmp_path / "archive"
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(path))
    yield path
