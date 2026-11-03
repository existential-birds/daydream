"""Real-process tests for the bounded host-side test runner."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.test_execution import (
    RecipeConfinementError,
    TestExecutionResult,
    TestRecipe,
    canonical_test_command,
    resolve_package,
    resolve_test_command_fact,
    resolve_test_recipe,
    run_test_command,
)


def _run(argv: list[str], tmp_path: Path, *, wall_budget_s: float = 10.0, env: dict[str, str] | None = None,
) -> TestExecutionResult:
    return asyncio.run(run_test_command(argv, cwd=tmp_path, wall_budget_s=wall_budget_s, env=env))

def test_command_and_provenance_share_cli_over_config_precedence() -> None:
    cfg, run = SimpleNamespace(test_command="pytest -x"), SimpleNamespace(test_command="/cli/cmd")
    assert canonical_test_command(cfg, run) == ["/cli/cmd"]
    fact = resolve_test_command_fact(cfg, run)
    assert fact.value == ("/cli/cmd",)
    assert fact.source == "cli"
    none = SimpleNamespace(test_command=None)
    for config, run_config, reason in (
        (SimpleNamespace(test_command="pytest", test_command_invalid=True), none, "invalid-type"),
        (SimpleNamespace(test_command=["pytest"]), none, "invalid-type"),
        (none, SimpleNamespace(test_command="   "), "blank"),
    ):
        miss = resolve_test_command_fact(config, run_config)
        assert miss.value is None and miss.diagnostic is not None and miss.diagnostic.reason_code == reason
    # A bare executable is looked up on PATH; a missing one is a named "unavailable" miss.
    from daydream.test_execution import ResolvedFact, check_test_command_executable
    missing = check_test_command_executable(ResolvedFact(("daydream-no-such-runner",), "cli"), cwd=Path.cwd())
    assert missing.diagnostic is not None and missing.diagnostic.reason_code == "unavailable"

def test_runner_fails_closed_when_env_value_survives_scrub(tmp_path: Path) -> None:
    """If a secret survives inside the replacement marker, discard the field based on pre-scrub matches."""
    res = _run([sys.executable, "-c", "print('REDACTED', flush=True)"], tmp_path, env={"STUCK": "REDACTED"},)
    assert res.passed is True
    assert res.merged_output == "[REDACTION_FAILED]"
    res = _run([sys.executable, "-c", "import os,sys; print(os.getcwd()); print('sec+REAL_SECRET+', file=sys.stderr)"],
               tmp_path, env={"SOME_ENV": "REAL_SECRET"})
    assert (res.exit_status, res.timed_out, res.passed) == (0, False, True)
    assert str(tmp_path) in res.merged_output and "sec+[REDACTED_ENV_VAR]+" in res.merged_output
    assert "REAL_SECRET" not in res.merged_output

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

def test_recipe_config_digest_is_a_named_miss_when_an_input_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (tmp_path / "uv.lock").mkdir()
    recipe = _recipe(tmp_path, cli="uv run pytest")

    assert recipe.package.config_digest is None
    assert "uv.lock" in recipe.package.absent_components

def test_runner_flags_a_truncated_output_buffer_as_incomplete(tmp_path: Path) -> None:
    res = _run([sys.executable, "-c", "import sys; sys.stdout.write('x' * 600000)"], tmp_path,)

    assert res.exit_status == 0
    assert res.passed is True               # exit status remains the only pass source
    assert res.output_truncated is True     # ...but the retained buffer is incomplete
    assert res.incomplete is True
    assert len(res.merged_output) <= 512 * 1024
