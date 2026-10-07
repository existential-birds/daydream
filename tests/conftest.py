"""Shared pytest fixtures for the daydream test suite."""

import importlib.util
import os
import sys
import tempfile
import tomllib
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from daydream.artifacts import ownership as artifact_ownership
from daydream.extensions import EXTENSION_API_VERSION
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.run_config import RunConfig
from tests.harness.fake_gh import FakeGh, install_fake_gh
from tests.harness.git_helpers import (
    bare_remote as _bare_remote,
    commit as _commit,
    git as _git,
    init_repo as _init_repo,
)
from tests.harness.remote_ci import NoCIRemote

# Git hooks can export paths to the live repository. Clear them at collection
# so temporary-repo tests cannot mutate the user's worktree instead.
for _git_env_var in (
    "GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_PREFIX", "GIT_OBJECT_DIRECTORY", "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
):
    os.environ.pop(_git_env_var, None)

# Temporary test repositories must not inherit signing or global ignores.
# Append scoped overrides for all descendant Git processes, including helpers.
_git_config_count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
os.environ[f"GIT_CONFIG_KEY_{_git_config_count}"] = "commit.gpgsign"
os.environ[f"GIT_CONFIG_VALUE_{_git_config_count}"] = "false"
_git_config_count += 1
os.environ[f"GIT_CONFIG_KEY_{_git_config_count}"] = "core.excludesFile"
os.environ[f"GIT_CONFIG_VALUE_{_git_config_count}"] = "/dev/null"
os.environ["GIT_CONFIG_COUNT"] = str(_git_config_count + 1)

# Real-Git fixtures share tests.harness.git_helpers.


def _make_repo_with_main(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    _init_repo(repo)
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _commit(repo, "initial")
    return repo


def _feature_repo(tmp_path: Path, name: str, initial: dict[str, str], changed: dict[str, str]) -> Path:
    """Commit initial files on main, then changed files on feature."""
    repo = tmp_path / name
    for path, content in initial.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    _init_repo(repo)
    _git(repo, "add", ".")
    _commit(repo, "init")
    _git(repo, "checkout", "-b", "feature")
    for path, content in changed.items():
        (repo / path).write_text(content)
    _git(repo, "add", ".")
    _commit(repo, "change")
    return repo


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Initialize a fresh git repo at tmp_path with one initial commit on `main`."""
    return _make_repo_with_main(tmp_path)


@pytest.fixture
def feature_branch_repo(tmp_path: Path) -> Path:
    """Clean feature branch with a committed Python diff and no review output file."""
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
def deep_target(tmp_path: Path) -> Path:
    """Single-file Python diff that exercises the deep-review skip tier."""
    return _feature_repo(tmp_path, "deep_repo",
        {"foo.py": "def foo():\n    return 1\n"},
        {"foo.py": "def foo():\n    return 2\n"},
    )


@pytest.fixture
def linked_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Return main and linked feature worktrees sharing one Git directory.

    Only the feature worktree contains services/taste. Re-rooting through Git
    topology would therefore resolve its relative paths against the wrong tree."""
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


def _improve_monorepo(tmp_path: Path, name: str, *, with_web: bool = True, branch_changes: tuple[str, ...] = (),
) -> Path:
    """Build the shared committed apps/{billing,catalog} monorepo scaffold."""
    project = tmp_path / name
    for service in ("billing", "catalog"):
        root = project / "apps" / service
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(f"[project]\nname = \"{service}\"\n")
        (root / "api.py").write_text(f'def service_name():\n    return "{service}"\n')
    if with_web:
        web = project / "web"
        web.mkdir()
        (web / "App.tsx").write_text("export const App = () => <div>daydream</div>;\n")
    (project / "README.md").write_text("# Improve monorepo\n")
    (project / "pyproject.toml").write_text(
        "[project]\n"
        'name = "improve-monorepo"\n'
        "\n"
        "[tool.daydream]\n"
        'test-command = "uv run pytest"\n'
        'scope-command = "git diff --exit-code"\n'
    )
    _init_repo(project)
    _git(project, "add", ".")
    _commit(project, "initial")
    if branch_changes:
        _git(project, "checkout", "-b", "feature")
        for service in branch_changes:
            (project / "apps" / service / "api.py").write_text(f'def service_name():\n    return "{service}-v2"\n')
        _git(project, "add", *(f"apps/{service}/api.py" for service in branch_changes))
        _commit(project, "change " + " and ".join(branch_changes) + " api")
    return project


@pytest.fixture
def improve_monorepo_target(tmp_path: Path) -> Path:
    """Committed multi-service repository for improve-flow real-path tests."""
    return _improve_monorepo(tmp_path, "improve_monorepo")


@pytest.fixture
def improve_scaled_monorepo_target(tmp_path: Path) -> Path:
    """Committed monorepo large enough that partition fan-out must split."""
    project = tmp_path / "improve_scaled"
    for index in range(12):  # 12 conventional-root services
        root = project / "apps" / f"svc{index:02d}"
        root.mkdir(parents=True)
        (root / "pyproject.toml").write_text(f'[project]\nname = "svc{index:02d}"\n')
        (root / "api.py").write_text(f'def service_name():\n    return "svc{index:02d}"\n')
    for sub in ("alpha", "beta", "gamma"):  # uncovered react tree, no service signal
        pkg = project / "frontend" / "src" / sub
        pkg.mkdir(parents=True)
        for index in range(4):
            (pkg / f"view{index}.tsx").write_text("export const V = () => <div/>;\n")
    (project / "README.md").write_text("# scaled\n")
    (project / "pyproject.toml").write_text(
        '[project]\nname = "improve-scaled"\n\n[tool.daydream]\ntest-command = "uv run pytest"\n'
    )
    _init_repo(project)
    _git(project, "add", ".")
    _commit(project, "initial")
    return project


@pytest.fixture
def improve_branch_target(tmp_path: Path) -> Path:
    """Improve monorepo with one billing change committed on a feature branch."""
    return _improve_monorepo(tmp_path, "improve_branch", branch_changes=("billing",))


@pytest.fixture
def improve_branch_two_services_target(tmp_path: Path) -> Path:
    """Improve monorepo whose feature branch changes billing AND catalog."""
    return _improve_monorepo(tmp_path, "improve_branch_two", with_web=False, branch_changes=("billing", "catalog"))


@pytest.fixture
def multi_stack_target(tmp_path: Path) -> Path:
    """Feature-branch diff with one Python, React, and Markdown file."""
    return _feature_repo(tmp_path, "multi_stack", {
        "api.py": "def hello():\n    return 'world'\n",
        "App.tsx": "export const App = () => <div>hello</div>;\n",
        "README.md": "# Project\n",
    }, {
        "api.py": "def hello():\n    return 'universe'\n",
        "App.tsx": "export const App = () => <div>universe</div>;\n",
        "README.md": "# Project\n\nUpdated.\n",
    })


@pytest.fixture
def shard_many_python_target(tmp_path: Path) -> Path:
    """Three changed Python files plus docs, allowing max_files to force sharding."""
    before = {f"mod{i}.py": f"def f{i}():\n    return {i}\n" for i in range(3)}
    before["README.md"] = "# Project\n"
    after = {f"mod{i}.py": f"def f{i}():\n    return 'x{i}'\n" for i in range(3)}
    after["README.md"] = "# Project\n\nUpdated.\n"
    return _feature_repo(tmp_path, "shard_many", before, after)


@pytest.fixture
def sibling_frontier_target(tmp_path: Path) -> Path:
    """Thirteen changed Python files with parseable cross-shard import edges.

    Twelve spokes import core_helper, forcing the sibling frontier to supply
    context when the Python stack is split across shards."""
    before = {
        f"mod{i}.py": f"from core import core_helper\ndef mod{i}_fn(): return core_helper() + {i}\n"
        for i in range(12)
    }
    before["core.py"] = "def core_helper():\n    return 1\n"
    after = {f"mod{i}.py": (
            f"from core import core_helper\ndef mod{i}_fn() -> int: return core_helper() + {i}\n# v2\n"
            + ("# extra0\n# extra1\n# extra2\n# extra3\n# extra4\n# extra5\n" if i == 5 else "")
        )
        for i in range(12)
    }
    after["core.py"] = "def core_helper():\n    return 1\n# v2\n"
    return _feature_repo(tmp_path, "canary", before, after)


@pytest.fixture
def rust_wire_target(tmp_path: Path) -> Path:
    """Rust and Markdown diff exercising Rust and generic wire-contract prompts."""
    return _feature_repo(tmp_path, "rust_wire", {
        "src/main.rs": "fn main() {\n    println!(\"hi\");\n}\n",
        "README.md": "# Project\n",
    }, {
        "src/main.rs": "fn main() {\n    println!(\"hello from wire\");\n}\n",
        "README.md": "# Project\n\nUpdated.\n",
    })


@pytest.fixture
def tiny_diff_target(tmp_path: Path) -> Path:
    """Two-language diff small enough to combine stacks and skip merge/arbiter."""
    return _feature_repo(tmp_path, "tiny_diff", {
        "api.py": "def hello():\n    return 'world'\n",
        "App.tsx": "export const App = () => <div>hello</div>;\n",
    }, {
        "api.py": "def hello():\n    return 'universe'\n",
        "App.tsx": "export const App = () => <div>universe</div>;\n",
    })


@pytest.fixture
def make_work() -> Callable[..., WorkContext]:
    """Build a synthetic WorkContext anchored at repo, with stable fake SHAs."""

    def _make(repo: Path, *, base_branch: str = "main", base_sha: str = "DEADBEEF", head_sha: str = "CAFEBABE",
        head_branch: str | None = "feat/x", is_ephemeral: bool = False,
    ) -> WorkContext:
        return WorkContext(repo=repo, source=repo, base_branch=base_branch, base_sha=base_sha, head_branch=head_branch,
            head_sha=head_sha, is_ephemeral=is_ephemeral, run_id="20260101000000-deadbeef",
        )

    return _make


@pytest.fixture
def make_config() -> Callable[..., "RunConfig"]:
    """Build RunConfig with non_interactive=True, cleanup=False, archive=False.

    Other fields retain production defaults; explicit overrides always win.
    Paths are stringified to match RunConfig.target."""
    from daydream.run_config import RunConfig

    def _make(target: Path | str, **overrides: object) -> RunConfig:
        fields: dict[str, object] = {"target": str(target), "non_interactive": True, "cleanup": False, "archive": False,
        }
        fields.update(overrides)
        return RunConfig(**fields)  # type: ignore[arg-type]

    return _make


@pytest.fixture
def install_backend(monkeypatch: pytest.MonkeyPatch) -> Callable[[object], object]:
    """Install and return a fake at the runner.create_backend seam."""

    def _install(backend: object) -> object:
        monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
        return backend

    return _install


def silence_module_console(monkeypatch: pytest.MonkeyPatch, module: str, *, keep: tuple[str, ...] = ()) -> None:
    """Silence a module's callable print_* helpers and console.

    Pass names in keep to preserve output under observation, or install spies
    after silencing. Discovering bound helpers avoids a separate name registry."""
    mod = importlib.import_module(module)
    for name in dir(mod):
        if not name.startswith("print_") or name in keep:
            continue
        if callable(getattr(mod, name, None)):
            monkeypatch.setattr(f"{module}.{name}", lambda *a, **kw: None)
    if "console" not in keep and hasattr(getattr(mod, "console", None), "print"):
        monkeypatch.setattr(f"{module}.console", type("C", (), {"print": lambda *a, **kw: None})())


@pytest.fixture
def silence_console(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Return a callable that silences a module, preserving names passed in keep."""

    def _silence(module: str, *, keep: tuple[str, ...] = ()) -> None:
        silence_module_console(monkeypatch, module, keep=keep)

    return _silence


@pytest.fixture
def mute_side_effects(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Stub publication, healing, and commit phases for flow tests.

    All stubs default on. Set post=False, heal=False, or commit=False when that
    action is under test and install its recording fake explicitly. module owns
    the heal/commit bindings; publication is patched on daydream.pr_review."""

    def _mute(module: str = "daydream.deep.fix_steps", *, post: bool = True, heal: bool = True, commit: bool = True,
    ) -> None:
        async def _no_post(*_args: object, **_kwargs: object) -> None:
            return None

        async def _ok(*_args: object, **kwargs: object) -> object:
            from daydream.phases import TestAndHealResult, TestAttemptEvidence

            capture = kwargs["capture_tree_key"]
            session_id = str(kwargs["session_id"])
            key = capture()  # type: ignore[operator]
            return TestAndHealResult(passed=True, retries=0, proceed=True, ignored=False,
                attempts=(TestAttemptEvidence(
                        session_id=session_id, kind="agent", command=None, passed=True, input_tree_key=key,
                        output_tree_key=key,
                    ),
                ),
            )

        if post:
            monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)
        if heal:
            monkeypatch.setattr(f"{module}.phase_test_and_heal", _ok)
        if commit:
            monkeypatch.setattr(f"{module}.phase_commit_push", _no_post)

    return _mute


@pytest.fixture(autouse=True)
def _isolate_github_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove real GitHub App credentials so tests cannot mint live tokens.

    Credential-present scenarios set their own fakes through monkeypatch."""
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)


@pytest.fixture(autouse=True)
def _isolate_github_cli_env(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hide host gh credentials; explicit test auth and live opt-ins remain available."""
    if request.node.get_closest_marker("live_gh") is not None:
        return
    for name in (
        "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN", "GH_HOST", "GH_REPO",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh-config"))


@pytest.fixture(autouse=True)
def _isolate_trace_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip tracing endpoints and credentials for every test."""
    for key in os.environ:
        if key.startswith(("OTEL_", "LANGSMITH_", "HH_", "DAYDREAM_TRACE_", "_OTEL_")):
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def _hermetic_skill_availability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give native Claude clients a fresh private configuration with an empty registry.

    Tests may override CLAUDE_CONFIG_DIR after this autouse fixture."""
    cfg = Path(tempfile.mkdtemp(prefix=f"{tmp_path.name}-claude-config-", dir=tmp_path.parent))
    (cfg / "plugins").mkdir()
    (cfg / "plugins" / "installed_plugins.json").write_text('{"plugins": {}}')
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg))


@pytest.fixture(autouse=True)
def _reset_trajectory_recorder() -> Iterator[Any]:
    """Clear recorder and signal-run state before and after each test.

    Lazy imports keep Pydantic-heavy trajectory modules out of collection."""
    from daydream.trajectory.context import _RECORDER_VAR
    from daydream.trajectory.recorder import _ACTIVE_SIGNAL_RUNS

    def _reset() -> None:
        _RECORDER_VAR.set(None)
        _ACTIVE_SIGNAL_RUNS.clear()

    _reset()
    yield
    _reset()


@pytest.fixture
def recorder(tmp_path: Path) -> Any:
    """Build a recorder at tmp_path, importing its Pydantic-heavy harness lazily."""
    from tests.harness.trajectory import make_recorder

    return make_recorder(tmp_path)


@pytest.fixture(autouse=True)
def artifact_runtime_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use per-test private storage while retaining real ownership checks and IO."""

    # Repositories may occupy tmp_path itself; private storage must be a sibling
    # to remain outside source ownership for both fixture layouts.
    base = tmp_path.parent / f"{tmp_path.name}-artifact-private"
    monkeypatch.setattr(artifact_ownership, "_default_private_base", lambda: base)
    return base / "runtime"


@pytest.fixture(autouse=True)
def archive_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Isolate DAYDREAM_ARCHIVE_DIR to a per-test tmpdir so tests never touch ~/.daydream/archive/."""
    path = tmp_path / "archive"
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(path))
    yield path


_CURRENT_API = object()  # sentinel: "use the tool's current version"


class ExtDir:
    """Helper for the ``ext_dir`` fixture: writes a ``daydream_ext`` package to tmp."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._root = root
        self._monkeypatch = monkeypatch

    def write_module(self, source: str, *, api_version: Any = _CURRENT_API) -> Path:
        """Write a package and point DAYDREAM_EXT_DIR at it.

        api_version defaults to the current API. Explicit values are interpolated
        as raw Python (including malformed test declarations); None omits it."""
        if api_version is _CURRENT_API:
            api_version = EXTENSION_API_VERSION
        if api_version is not None:
            source = f"DAYDREAM_EXT_API = {api_version}\n{source}"
        package = self._root / "daydream_ext"
        package.mkdir(exist_ok=True)
        (package / "__init__.py").write_text(source)
        self._monkeypatch.setenv("DAYDREAM_EXT_DIR", str(package))
        return package


@pytest.fixture
def ext_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ExtDir:
    """Write extension packages through DAYDREAM_EXT_DIR without altering sys.modules."""
    return ExtDir(tmp_path, monkeypatch)


@pytest.fixture
def cli_runner() -> Any:
    """Invoke ``daydream`` in-process; the SystemExit code becomes ``exit_code``."""

    class _Runner:
        def invoke(self, argv: list[str]) -> SimpleNamespace:
            from daydream import cli

            saved = sys.argv
            sys.argv = ["daydream", *argv]
            code = 0
            try:
                cli.main()
            except SystemExit as exc:
                code = int(exc.code or 0)
            finally:
                sys.argv = saved
            return SimpleNamespace(exit_code=code)

    return _Runner()


@pytest.fixture
def fake_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeGh:
    """Answer gh synchronously at the subprocess boundary; real Git still runs."""
    return install_fake_gh(tmp_path / "fake-gh-state", monkeypatch)


@pytest.fixture
def no_ci_remote(fake_gh: FakeGh, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,) -> Iterator[NoCIRemote]:
    """A real bare transport behind a GitHub URL with external no-CI evidence."""
    harness = NoCIRemote(fake_gh, monkeypatch, tmp_path)
    yield harness
    harness.finish()


def improve_fixture_service(apps_dir: Path) -> str:
    """Choose the first service whose pyproject starts with [project].

    Host evidence quotes line 1 verbatim, so reject a scaffold with no service
    or an incompatible first line instead of silently changing attribution."""
    service_entries: list[Path] = []
    if apps_dir.is_dir():
        service_entries = sorted(p for p in apps_dir.iterdir() if p.is_dir() and (p / "pyproject.toml").is_file())
    if not service_entries:
        raise AssertionError("improve_monorepo_target fixture must contain at least one "
            "service directory under apps/"
        )
    chosen = service_entries[0]
    if chosen.joinpath("pyproject.toml").read_text(encoding="utf-8").splitlines()[:1] != ["[project]"]:
        raise AssertionError(f"improve_monorepo_target service {chosen.name!r} must declare "
            "'[project]' on pyproject.toml line 1 to anchor host evidence"
        )
    return chosen.name


# Harbor assets use bare verifier_core imports without a daydream dependency.
# Load the canonical module under that name, matching compiled-task resolution.
_TEMPLATES = Path(__file__).resolve().parents[1] / "daydream" / "benchmark" / "harbor" / "templates"


def _load_template_asset(path: Path, name: str) -> Any:
    """Load a non-package template asset; bare `import verifier_core` hits the canonical module."""
    from daydream.benchmark.harbor import verifier_core as _canonical_vc

    registered = "verifier_core" not in sys.modules
    if registered:
        sys.modules["verifier_core"] = _canonical_vc
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if registered:
            sys.modules.pop("verifier_core", None)


@pytest.fixture(scope="session")
def sr_module() -> Any:
    """The loaded ``templates/tests/score_review.py`` verifier entry module."""
    return _load_template_asset(_TEMPLATES / "tests" / "score_review.py", "score_review")



def improve_fixture_test_command_anchor(pyproject: Path) -> int:
    """Find the exact test-command key after validating its TOML value.

    The value must be uv run pytest; prefix matches such as test-command-timeout
    must not provide the evidence anchor."""
    cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    if cfg.get("tool", {}).get("daydream", {}).get("test-command") != "uv run pytest":
        raise AssertionError("improve_monorepo_target fixture must declare test command "
            "'uv run pytest' in its root pyproject.toml"
        )
    for line_number, line in enumerate(pyproject.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip().split("=", 1)[0].strip() == "test-command":
            return line_number
    raise AssertionError("improve_monorepo_target fixture must declare test command "
        "'uv run pytest' in its root pyproject.toml"
    )
