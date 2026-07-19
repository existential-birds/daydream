"""Shared pytest fixtures for the daydream test suite."""

import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from daydream.workspace import WorkContext
from tests.harness.fake_gh import FakeGh, install_fake_gh
from tests.harness.git_helpers import bare_remote as _bare_remote
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import configure_identity as _configure_identity  # noqa: F401 - test_git_ops re-import
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo

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

# Test-created repositories commit non-interactively and must not inherit a
# developer's global signing requirement. Append an environment-scoped Git
# override so every descendant git process is deterministic, including helpers
# defined outside this file.
_git_config_count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
os.environ[f"GIT_CONFIG_KEY_{_git_config_count}"] = "commit.gpgsign"
os.environ[f"GIT_CONFIG_VALUE_{_git_config_count}"] = "false"
os.environ["GIT_CONFIG_COUNT"] = str(_git_config_count + 1)

# --- Real-git fixtures ------------------------------------------------------
#
# Mirrors the helpers that previously lived only in tests/test_git_ops.py.
# Lifted here so other test modules (notably tests/test_pr_review.py) can
# build real repos instead of mocking subprocess. tests/test_workspace.py
# still has its own helpers (it needs additional plumbing for bare-origin
# push semantics) — left untouched on purpose. The helper bodies now live in
# tests/harness/git_helpers.py; the aliased imports above keep the local
# `_git`/`_commit`/... names (and external `from tests.conftest import _git`
# call sites) unchanged.


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
def linked_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A main worktree + a linked worktree on a feature branch (issue #221).

    The main repo lives at ``main_repo`` on ``main`` and contains only
    ``base.txt``. A ``feature`` branch adds files under ``services/taste/`` that
    do NOT exist on ``main``. A linked worktree is checked out on ``feature`` at
    ``linked_worktree``, sharing the main repo's git dir.

    This reproduces the bug's trap: from the linked worktree, git topology
    (``git rev-parse --git-common-dir`` / ``git worktree list``) points at the
    MAIN worktree, where ``services/taste/`` does not exist. An agent that
    re-roots a relative path via git topology reads the wrong worktree.

    Returns:
        ``(main_repo_path, linked_worktree_path)``.
    """
    main_repo = _make_repo_with_main(tmp_path, name="main_repo")
    _git(main_repo, "checkout", "-b", "feature")
    taste = main_repo / "services" / "taste"
    taste.mkdir(parents=True, exist_ok=True)
    for name in ("parser.go", "lexer.go", "token.go", "ast.go"):
        (taste / name).write_text(f"package taste\n\n// {name}\nfunc {name[:-3].title()}() {{}}\n")
    _git(main_repo, "add", "services/taste")
    _commit(main_repo, "add taste service")
    # Return the main worktree to `main` so it does NOT contain services/taste/.
    _git(main_repo, "checkout", "main")
    linked = tmp_path / "linked_worktree"
    _git(main_repo, "worktree", "add", str(linked), "feature")
    return main_repo, linked


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
def improve_monorepo_target(tmp_path: Path) -> Path:
    """Committed multi-service repository for improve-flow real-path tests."""
    project = tmp_path / "improve_monorepo"
    for service in ("billing", "catalog"):
        root = project / "apps" / service
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(f"[project]\nname = \"{service}\"\n")
        (root / "api.py").write_text(f'def service_name():\n    return "{service}"\n')
    web = project / "web"
    web.mkdir()
    (web / "App.tsx").write_text("export const App = () => <div>daydream</div>;\n")
    (project / "README.md").write_text("# Improve monorepo\n")
    (project / "pyproject.toml").write_text("[project]\nname = \"improve-monorepo\"\n")
    _init_repo(project)
    _git(project, "add", ".")
    _commit(project, "initial")
    return project


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
def tiny_diff_target(tmp_path: Path) -> Path:
    """Git repo with a 2-file two-language diff on a feature branch (issue #172).

    Mirrors ``multi_stack_target`` but with only two changed files (``api.py`` +
    ``App.tsx``) so the deep pipeline's tiny-diff short-circuit fires: the two
    language stacks collapse into one combined generic-fallback assignment and
    the merge agent + arbiter are skipped. Used by AC2 / AC5 real-path tests.
    """
    project = tmp_path / "tiny_diff"
    project.mkdir()
    (project / "api.py").write_text("def hello():\n    return 'world'\n")
    (project / "App.tsx").write_text("export const App = () => <div>hello</div>;\n")
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
def _isolate_github_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip GitHub App credentials from the environment for every test.

    The development machine exports ``DAYDREAM_APP_ID`` and
    ``DAYDREAM_APP_PRIVATE_KEY`` via direnv for local bot runs. Tests must
    never see real credentials: the credentials-absent path is the default
    under test, and a real key reaching ``mint_installation_token`` would
    send live JWTs at the GitHub API mid-suite. Tests that need credentials
    set fakes explicitly via ``monkeypatch.setenv``.
    """
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)


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


@pytest.fixture(autouse=True)
def _no_harvest_row_spacing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the harvest inter-row rate-limit delay so tests don't sleep for real.

    ``run_harvest`` awaits ``_row_spacing_sleep(gh_request_spacing_sec)`` (default
    0.8s) between every row to spread ``gh`` calls under GitHub's secondary rate
    limits — real-time politeness with no bearing on test logic. Left unstubbed,
    every ``run_harvest`` test paid 0.8s/row (the 10-row degrade test alone burned
    8.17s of wall clock). Stub the module-level seam (mirroring how
    ``_rate_limit_sleep`` is stubbed) so the delay is a no-op; no test asserts it
    is honored. Lazy-imported so unrelated tests don't eagerly load the harvest
    stack.
    """
    from daydream.training import harvest

    async def _noop(_seconds: float) -> None:
        return None

    monkeypatch.setattr(harvest, "_row_spacing_sleep", _noop)


class ExtDir:
    """Helper for the ``ext_dir`` fixture: writes a ``daydream_ext`` package to tmp."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._root = root
        self._monkeypatch = monkeypatch

    def write_module(self, source: str) -> Path:
        """Write ``<tmp>/daydream_ext/__init__.py`` and point ``$DAYDREAM_EXT_DIR`` at it."""
        package = self._root / "daydream_ext"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text(source)
        self._monkeypatch.setenv("DAYDREAM_EXT_DIR", str(package))
        return package


@pytest.fixture
def ext_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExtDir:
    """A ``daydream_ext`` package writer wired to ``$DAYDREAM_EXT_DIR``.

    The extension loader's explicit-path override (mirroring the
    ``$DAYDREAM_SKILLS_DIR`` convention) is the test seam: tests write an
    extension module to tmp and the loader picks it up without touching
    ``sys.modules``.
    """
    return ExtDir(tmp_path, monkeypatch)


@pytest.fixture
def fake_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    """Install the subprocess ``gh`` shim and shrink the ``gh`` timeout budget.

    Every test using this fixture spawns a real ``gh`` subprocess (the fake
    shim), so it is auto-marked ``integration`` by
    :func:`pytest_collection_modifyitems` and excluded from the pre-push gate.

    The shim replies in milliseconds, so the production 60s / 2-retry budget
    only serves to turn a CPU-starved interpreter cold start into a 3 x 60s =
    180s stall (the original fake-gh flake). Cap each attempt at 15s with a
    single retry: the typical run stays sub-second, the worst case is 30s.
    """
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("DAYDREAM_GH_TIMEOUT_RETRIES", "1")
    return install_fake_gh(tmp_path / "fake-gh-bin", monkeypatch)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-mark every test that drives the subprocess ``gh`` shim ``integration``.

    These tests spawn a real ``gh`` subprocess and so carry process cold-start
    cost that flakes under host CPU saturation. Marking them keeps the pre-push
    gate (``-m "not integration"``) fast and deterministic; CI runs the full
    suite, preserving their coverage.
    """
    for item in items:
        if "fake_gh" in getattr(item, "fixturenames", ()):
            item.add_marker("integration")
