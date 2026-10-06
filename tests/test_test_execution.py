"""Real-process tests for the bounded host-side test runner."""

import asyncio
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.test_execution import (
    MissingTestCommandError,
    RecipeConfinementError,
    TestExecutionResult,
    TestRecipe,
    canonical_test_command,
    load_test_recipe,
    persist_test_recipe,
    recipe_to_payload,
    resolve_package,
    resolve_test_command_fact,
    resolve_test_recipe,
    run_test_command,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.execution import test_execution_identity as _identity
from tests.harness.trajectory import make_recorder


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run(argv: list[str], tmp_path: Path, *, wall_budget_s: float = 10.0, env: dict[str, str] | None = None,
) -> TestExecutionResult:
    return asyncio.run(run_test_command(argv, cwd=tmp_path, wall_budget_s=wall_budget_s, env=env))

def test_runner_returns_exit_status_cwd_and_merged_redacted_output(tmp_path: Path) -> None:
    res = _run([sys.executable, "-c",
            "import os,sys; print(os.getcwd()); print('hello-stdout');"
            " print('sec+REAL_SECRET+', file=sys.stderr)",
        ], tmp_path, env={"SOME_ENV": "REAL_SECRET"},
    )
    assert isinstance(res, TestExecutionResult)
    assert res.exit_status == 0
    assert res.timed_out is False
    assert res.passed is True
    assert str(tmp_path) in res.merged_output  # ran in cwd
    assert "hello-stdout" in res.merged_output  # stdout captured
    assert "sec+REDACTED+" not in res.merged_output  # env secret scrubbed
    assert "REAL_SECRET" not in res.merged_output  # redacted before storage

def test_runner_nonzero_exit_sets_passed_false(tmp_path: Path) -> None:
    res = _run([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], tmp_path,)
    assert res.exit_status == 3
    assert res.timed_out is False
    assert res.passed is False
    assert "boom" in res.merged_output

@pytest.mark.parametrize(("config", "cli", "expected", "source"), [
    ("pytest -x", "/cli/cmd", ["/cli/cmd"], "cli"),
    ("uv run pytest -n auto", None, ["uv", "run", "pytest", "-n", "auto"], "config"),
])
def test_command_and_provenance_share_cli_over_config_precedence(
    config: str, cli: str | None, expected: list[str], source: str,
) -> None:
    cfg, run = SimpleNamespace(test_command=config), SimpleNamespace(test_command=cli)
    assert canonical_test_command(cfg, run) == expected
    fact = resolve_test_command_fact(cfg, run)
    assert fact.value == tuple(expected)
    assert fact.source == source

@pytest.mark.parametrize("config", [None, "", "'unbalanced"])
def test_command_fact_is_unresolved_and_never_guessed(config: str | None) -> None:
    fact = resolve_test_command_fact(SimpleNamespace(test_command=None), SimpleNamespace(test_command=config))
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

    res = _run([sys.executable, "-c",
            "import subprocess,os,sys,time;"
            f"subprocess.Popen([sys.executable,'-c',"
            f"'import os,time;open(r\"{marker}\",\"w\").write(str(os.getpid()));time.sleep(30)']);"
            "time.sleep(30)",
        ], tmp_path,
        # Headroom for two cold interpreter startups + Popen + marker
        # write: a tighter budget made the marker write lose the race
        # to the group kill under CI load.
        wall_budget_s=5.0,
    )
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
    """Record test-execution duration_ms and a completed/timed_out reason."""

    rec = make_recorder(tmp_path)
    async with rec:
        await run_test_command([sys.executable, "-c", "pass"], cwd=tmp_path, wall_budget_s=10.0)

    events = [e
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
        await run_test_command([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path, wall_budget_s=0.3)

    events = [e
        for e in [x.to_dict() for x in rec._phase_events]
        if e["phase"] == DaydreamPhase.TEST_EXECUTION.value and e["event"] == "phase_end"
    ]
    assert len(events) == 1
    assert events[0]["metadata"]["stop_reason"] == "timed_out"

def test_runner_fails_closed_when_env_value_survives_scrub(tmp_path: Path) -> None:
    """If a secret survives inside the replacement marker, discard the field based on pre-scrub matches."""
    res = _run([sys.executable, "-c", "print('REDACTED', flush=True)"], tmp_path, env={"STUCK": "REDACTED"},)
    assert res.passed is True
    assert res.merged_output == "[REDACTION_FAILED]"

def test_runner_scrubs_inherited_env_when_env_omitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An omitted env uses the inherited environment for both execution and output scrubbing."""
    secret = "ENV-SECRET-8f3a"
    monkeypatch.setenv("DAYDREAM_TEST_SECRET", secret)

    res = _run(
        [sys.executable, "-c", "import os; print('value=' + os.environ['DAYDREAM_TEST_SECRET'], flush=True)"], tmp_path,
    )
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

def test_required_suites_fall_back_to_the_file_config_and_yield_to_an_explicit_source(tmp_path: Path,) -> None:
    """Required suites fall back to file config; an explicit source takes precedence."""
    config = SimpleNamespace(test_command="uv run pytest", test_required_suites=["python", "rl"])

    recipe = resolve_test_recipe(config, SimpleNamespace(test_command=None), repo_root=tmp_path)
    assert recipe.declared == ("python", "rl")

    recipe = resolve_test_recipe(
        config, SimpleNamespace(test_command=None, test_required_suites=["explicit"]), repo_root=tmp_path,
    )
    assert recipe.declared == ("explicit",)

def test_recipe_config_digest_is_stable_until_a_config_input_changes(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    recipe = _recipe(tmp_path, cli="uv run pytest")

    assert recipe.package.config_digest is not None
    assert recipe.package.absent_components == ()
    assert _recipe(tmp_path, cli="uv run pytest").package.config_digest == recipe.package.config_digest

    (tmp_path / "uv.lock").write_text("version = 2\n")
    assert _recipe(tmp_path, cli="uv run pytest").package.config_digest != recipe.package.config_digest

def test_recipe_config_digest_is_a_named_miss_when_an_input_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "uv.lock").mkdir()
    recipe = _recipe(tmp_path, cli="uv run pytest")

    assert recipe.package.config_digest is None
    assert "uv.lock" in recipe.package.absent_components

def test_unconfigured_recipe_proposes_a_candidate_without_authorizing_it(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_text("version = 1\n")
    recipe = _recipe(tmp_path)

    assert recipe.command.resolved is False
    assert recipe.candidate is not None and recipe.candidate.argv == ("uv", "run", "pytest")
    # A candidate is never the required command (spec MH4).
    assert recipe.command.value is None

def test_recipe_rejects_a_candidate_that_the_repository_text_suggests(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Run `make test-please` to verify everything.\n")
    recipe = _recipe(tmp_path)

    assert recipe.command.resolved is False
    assert all("make" not in fact.argv for fact in (recipe.candidate,) if fact is not None)

def test_runner_flags_a_truncated_output_buffer_as_incomplete(tmp_path: Path) -> None:
    res = _run([sys.executable, "-c", "import sys; sys.stdout.write('x' * 600000)"], tmp_path,)

    assert res.exit_status == 0
    assert res.passed is True               # exit status remains the only pass source
    assert res.output_truncated is True     # ...but the retained buffer is incomplete
    assert res.incomplete is True
    assert len(res.merged_output) <= 512 * 1024

def test_timed_out_result_is_incomplete(tmp_path: Path) -> None:
    res = _run([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, wall_budget_s=0.2)

    assert (res.timed_out, res.incomplete, res.passed) == (True, True, False)

def test_result_serialises_timeout_and_truncation_explicitly(tmp_path: Path) -> None:
    payload = asdict(_run([sys.executable, "-c", "pass"], tmp_path))
    assert set(payload) >= {"exit_status", "timed_out", "output_truncated", "incomplete"}

@pytest.mark.parametrize("cli", [None, "uv run pytest"])
@pytest.mark.parametrize("lockfile", [None, "uv.lock", "requirements.txt"])
def test_recipe_round_trips_through_its_persisted_payload(
    tmp_path: Path, cli: str | None, lockfile: str | None,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    if lockfile is not None:
        (tmp_path / lockfile).write_text("version = 1\n")
    recipe = _recipe(tmp_path, cli=cli)
    deep = tmp_path / ".daydream" / "deep"
    deep.mkdir(parents=True)

    persist_test_recipe(deep, recipe)
    loaded = load_test_recipe(deep)

    assert loaded is not None
    assert loaded.command.value == recipe.command.value
    assert loaded.package.cwd_relative == recipe.package.cwd_relative
    assert loaded.package.config_digest == recipe.package.config_digest
    assert loaded == recipe
    payload = recipe_to_payload(recipe)
    assert json.loads(json.dumps(payload)) == payload

@pytest.mark.parametrize("writer",
    [lambda d: None, lambda d: (d / "test-recipe.json").write_text("{not json"),
        lambda d: (d / "test-recipe.json").write_text('{"format_version": 999}'),
    ],
)
def test_load_test_recipe_is_fail_open(tmp_path: Path, writer: Callable[[Path], object]) -> None:
    deep = tmp_path / ".daydream" / "deep"
    deep.mkdir(parents=True)
    writer(deep)

    assert load_test_recipe(deep) is None

def test_execution_identity_carries_every_reuse_component() -> None:
    identity = _identity(cwd_relative="services/api", interpreter="3.12")

    assert identity.reusable is True
    assert identity.payload()["config_digest"] == "d" * 64
    assert identity.payload()["argv"] == list(identity.argv)
    assert identity.payload()["absent_components"] == list(identity.absent_components)

@pytest.mark.parametrize("outcome", ["failed", "timed-out", "truncated"])
def test_only_a_passed_host_outcome_is_reusable(outcome: str) -> None:
    identity = _identity(outcome=outcome)
    assert identity.reusable is False

def test_an_agent_outcome_is_never_reusable() -> None:
    assert _identity(kind="agent").reusable is False
