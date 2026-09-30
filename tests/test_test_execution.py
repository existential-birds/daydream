"""Real-process tests for the bounded host-side test runner."""

import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from daydream.test_execution import (
    MissingTestCommandError,
    RecipeConfinementError,
    RequiredContract,
    RequiredRun,
    TargetedCheckRun,
    TestExecutionResult,
    TestRecipe,
    canonical_test_command,
    recipe_identity,
    resolve_package,
    resolve_test_command_fact,
    resolve_test_recipe,
    run_test_command,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.trajectory import make_recorder


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True



def test_runner_returns_exit_status_cwd_and_merged_redacted_output(tmp_path: Path) -> None:
    async def go() -> TestExecutionResult:
        return await run_test_command(
            [
                sys.executable,
                "-c",
                "import os,sys; print(os.getcwd()); print('hello-stdout');"
                " print('sec+REAL_SECRET+', file=sys.stderr)",
            ],
            cwd=tmp_path,
            wall_budget_s=10.0,
            env={"SOME_ENV": "REAL_SECRET"},
        )

    res = asyncio.run(go())
    assert isinstance(res, TestExecutionResult)
    assert res.exit_status == 0
    assert res.timed_out is False
    assert res.passed is True
    assert str(tmp_path) in res.merged_output  # ran in cwd
    assert "hello-stdout" in res.merged_output  # stdout captured
    assert "sec+REDACTED+" not in res.merged_output  # env secret scrubbed
    assert "REAL_SECRET" not in res.merged_output  # redacted before storage


def test_runner_nonzero_exit_sets_passed_false(tmp_path: Path) -> None:
    async def go() -> TestExecutionResult:
        return await run_test_command(
            [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
            cwd=tmp_path,
            wall_budget_s=10.0,
        )

    res = asyncio.run(go())
    assert res.exit_status == 3
    assert res.timed_out is False
    assert res.passed is False
    assert "boom" in res.merged_output


def test_canonical_command_from_cli_overrides_config() -> None:

    cfg = SimpleNamespace(test_command="pytest -x")  # config value
    run = SimpleNamespace(test_command="/cli/cmd")  # CLI wins
    assert canonical_test_command(cfg, run) == ["/cli/cmd"]


def test_canonical_command_from_config_when_cli_unset() -> None:

    cfg = SimpleNamespace(test_command="uv run pytest -n auto")
    run = SimpleNamespace(test_command=None)
    assert canonical_test_command(cfg, run) == ["uv", "run", "pytest", "-n", "auto"]


def test_command_fact_records_cli_provenance_over_config() -> None:
    cfg = SimpleNamespace(test_command="pytest -x")
    run = SimpleNamespace(test_command="/cli/cmd")
    fact = resolve_test_command_fact(cfg, run)
    assert fact.value == ("/cli/cmd",)
    assert fact.source == "cli"


def test_command_fact_records_config_provenance_when_cli_unset() -> None:
    fact = resolve_test_command_fact(
        SimpleNamespace(test_command="uv run pytest -n auto"),
        SimpleNamespace(test_command=None),
    )
    assert fact.value == ("uv", "run", "pytest", "-n", "auto")
    assert fact.source == "config"


@pytest.mark.parametrize("config", [None, "", "'unbalanced"])
def test_command_fact_is_unresolved_and_never_guessed(config: str | None) -> None:
    fact = resolve_test_command_fact(
        SimpleNamespace(test_command=None), SimpleNamespace(test_command=config)
    )
    assert fact.resolved is False
    assert fact.source == "unresolved"
    assert fact.value is None


def test_canonical_command_missing_fails_safely_with_diagnostic() -> None:

    cfg = SimpleNamespace(test_command=None)
    run = SimpleNamespace(test_command=None)
    with pytest.raises(MissingTestCommandError) as e:
        canonical_test_command(cfg, run)
    msg = str(e.value)
    assert "test_command" in msg
    assert "tool.daydream" in msg  # naming the precedence source
    assert "--test-command" in msg  # actionable: what to set


def test_runner_timeout_kills_process_group_and_reports_timed_out(tmp_path: Path) -> None:
    marker = tmp_path / "grandkid.pid"

    async def go() -> TestExecutionResult:
        return await run_test_command(
            [
                sys.executable,
                "-c",
                "import subprocess,os,sys,time;"
                f"subprocess.Popen([sys.executable,'-c',"
                f"'import os,time;open(r\"{marker}\",\"w\").write(str(os.getpid()));time.sleep(30)']);"
                "time.sleep(30)",
            ],
            cwd=tmp_path,
            # Headroom for two cold interpreter startups + Popen + marker
            # write: a tighter budget made the marker write lose the race
            # to the group kill under CI load.
            wall_budget_s=5.0,
        )

    res = asyncio.run(go())
    assert res.timed_out is True
    assert res.passed is False
    # The grandchild writes its pid before sleeping; poll until it lands
    # instead of assuming startup beat the wall budget.
    deadline = time.monotonic() + 5.0
    pid: int | None = None
    while pid is None and time.monotonic() < deadline:
        try:
            pid = int(marker.read_text().strip())
        except (FileNotFoundError, ValueError):
            time.sleep(0.05)
    assert pid is not None
    # grandchild must be dead, not reparented — poll a generous window
    deadline = time.monotonic() + 5.0
    while _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert _pid_alive(pid) is False


async def test_runner_records_duration_and_phase(tmp_path: Path) -> None:
    """Issue #726 task 12: with a trajectory recorder active, the host runner
    emits phase events distinguishable as ``test-execution``, carrying a
    ``duration_ms`` and a ``stop_reason`` in {completed, timed_out}."""

    rec = make_recorder(tmp_path)
    async with rec:
        await run_test_command([sys.executable, "-c", "pass"], cwd=tmp_path, wall_budget_s=10.0)

    events = [
        e
        for e in [x.to_dict() for x in rec._phase_events]
        if e["phase"] == DaydreamPhase.TEST_EXECUTION.value and e["event"] == "phase_end"
    ]
    assert len(events) == 1
    md = events[0]["metadata"]
    assert md["stop_reason"] in {"completed", "timed_out"}
    assert md["duration_ms"] >= 0


async def test_runner_records_timed_out_stop_reason(tmp_path: Path) -> None:

    rec = make_recorder(tmp_path)
    async with rec:
        await run_test_command(
            [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path, wall_budget_s=0.3
        )

    events = [
        e
        for e in [x.to_dict() for x in rec._phase_events]
        if e["phase"] == DaydreamPhase.TEST_EXECUTION.value and e["event"] == "phase_end"
    ]
    assert len(events) == 1
    assert events[0]["metadata"]["stop_reason"] == "timed_out"


def test_runner_fails_closed_when_env_value_survives_scrub(tmp_path: Path) -> None:
    """An env value the replacement marker itself carries (a substring of
    "[REDACTED_ENV_VAR]") can never be scrubbed clean by replace(); the
    fail-closed gate -- keyed off the pre-replacement buffer -- degrades the
    whole field rather than emit a buffer that still shows the secret."""
    async def go() -> TestExecutionResult:
        return await run_test_command(
            [sys.executable, "-c", "print('REDACTED', flush=True)"],
            cwd=tmp_path,
            wall_budget_s=10.0,
            env={"STUCK": "REDACTED"},
        )

    res = asyncio.run(go())
    assert res.passed is True
    assert res.merged_output == "[REDACTION_FAILED]"


def test_runner_scrubs_inherited_env_when_env_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production call sites pass no env, so the subprocess inherits the
    parent environment; the scrub must cover exactly those inherited values
    (the only env values that can appear in the merged output)."""
    secret = "ENV-SECRET-8f3a"
    monkeypatch.setenv("DAYDREAM_TEST_SECRET", secret)

    async def go() -> TestExecutionResult:
        return await run_test_command(
            [
                sys.executable,
                "-c",
                "import os; print('value=' + os.environ['DAYDREAM_TEST_SECRET'], flush=True)",
            ],
            cwd=tmp_path,
            wall_budget_s=10.0,
        )

    res = asyncio.run(go())
    assert res.passed is True
    assert "value=" in res.merged_output
    assert secret not in res.merged_output


def test_package_resolution_keys_each_nested_package_separately(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'root'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    nested = tmp_path / "services" / "api"
    nested.mkdir(parents=True)
    (nested / "pyproject.toml").write_text("[project]\nname = 'api'\n")
    (nested / "poetry.lock").write_text("# lock\n")

    root = resolve_package(tmp_path, tmp_path)
    api = resolve_package(tmp_path, nested)

    assert (root.cwd_relative, root.runner) == (".", "uv")
    assert (api.cwd_relative, api.runner) == ("services/api", "poetry")
    assert api.config_digest != root.config_digest


def test_unreadable_config_input_is_a_named_miss_never_a_placeholder(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "uv.lock").mkdir()  # a directory where a file belongs: every read raises OSError

    resolved = resolve_package(tmp_path, tmp_path)

    assert resolved.config_digest is None
    assert "uv.lock" in resolved.absent_components


def test_package_cwd_outside_the_worktree_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(RecipeConfinementError):
        resolve_package(repo, outside)


def _recipe(tmp_path: Path, *, cli: str | None = None, config: str | None = None) -> TestRecipe:
    return resolve_test_recipe(
        SimpleNamespace(test_command=config), SimpleNamespace(test_command=cli), repo_root=tmp_path
    )


def test_recipe_identity_is_stable_until_a_config_input_changes(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    recipe = _recipe(tmp_path, cli="uv run pytest")

    first = recipe_identity(recipe, tmp_path)
    assert first.digest is not None and first.absent_components == ()
    assert recipe_identity(recipe, tmp_path).digest == first.digest

    (tmp_path / "uv.lock").write_text("version = 2\n")
    assert recipe_identity(recipe, tmp_path).digest != first.digest


def test_recipe_identity_is_a_named_miss_when_an_input_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "uv.lock").mkdir()
    identity = recipe_identity(_recipe(tmp_path, cli="uv run pytest"), tmp_path)

    assert identity.digest is None
    assert "uv.lock" in identity.absent_components


def test_unconfigured_recipe_proposes_a_candidate_without_authorizing_it(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n")
    recipe = _recipe(tmp_path)

    assert recipe.command.resolved is False
    assert recipe.candidate is not None and recipe.candidate.argv == ("uv", "run", "pytest")
    # A candidate is never the required command (spec MH4).
    assert recipe.required.argv is None


def test_recipe_rejects_a_candidate_that_the_repository_text_suggests(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Run `make test-please` to verify everything.\n")
    recipe = _recipe(tmp_path)

    assert recipe.command.resolved is False
    assert all("make" not in fact.argv for fact in (recipe.candidate,) if fact is not None)


def _result(exit_status: int = 0) -> TestExecutionResult:
    return TestExecutionResult(exit_status=exit_status, timed_out=False, merged_output="ok")


def test_required_contract_is_satisfied_only_by_its_own_run() -> None:
    contract = RequiredContract(declared=("python",), argv=("uv", "run", "pytest"), source="config")
    required = RequiredRun(argv=("uv", "run", "pytest"), cwd_relative=".", result=_result())

    assert contract.satisfied_by(required) is True
    failing = RequiredRun(argv=("uv", "run", "pytest"), cwd_relative=".", result=_result(1))
    assert contract.satisfied_by(failing) is False


def test_targeted_check_cannot_satisfy_the_required_contract() -> None:
    contract = RequiredContract(declared=("python",), argv=("uv", "run", "pytest"), source="config")
    targeted = TargetedCheckRun(
        argv=("uv", "run", "pytest", "-k", "one"), selector="one", result=_result()
    )

    with pytest.raises(TypeError, match="required"):
        contract.satisfied_by(cast(Any, targeted))


def test_unresolved_required_contract_is_never_satisfied() -> None:
    contract = RequiredContract(declared=("python",), argv=None, source="unresolved")
    assert contract.satisfied_by(cast(Any, RequiredRun(argv=("pytest",), cwd_relative=".", result=_result()))) is False
