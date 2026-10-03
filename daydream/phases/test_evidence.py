"""Test evidence for review and fix phases."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from daydream import agent, config as phase_config, git_ops, ui
from daydream.agent import (
    detect_test_success,
)
from daydream.backends import (
    Backend,
    ContinuationToken,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.evidence_reuse import (
    ReuseTarget,
)
from daydream.git_ops import GitError
from daydream.phases.inputs import append_extended_facts
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    MissingTestCommandError,
    TestExecutionIdentity,
    TestExecutionResult,
    TestRecipe,
    canonical_test_command,
    run_test_command,
)
from daydream.trajectory import (
    DaydreamPhase,
)
from daydream.workspace import WorkContext

# Deprecated unconfigured fallback: the host parses the final test summary.
# The suite must finish in the turn; "still running" prose is a failure.
# Claude also enforces foreground execution mechanically.
_TEST_RUN_INSTRUCTIONS = (
    "Run it in the foreground and wait for it to finish; never run it in the background. "
    "Report if tests pass or fail, quoting the test runner's final summary line verbatim."
)


def _canonical_test_cmd(config: Any) -> list[str] | None:
    """Resolve CLI > file-config test argv; warn and return None on absent/malformed commands.

    None selects the deprecated agent-reported fallback, which cannot prove a real
    subprocess exit status.
    """
    file_config = getattr(config, "file_config", None)
    if not isinstance(file_config, DaydreamFileConfig):
        file_config = DaydreamFileConfig()
    try:
        return canonical_test_command(file_config, config)
    except MissingTestCommandError as exc:
        ui.print_warning(
            agent.console,
            f"{exc} Falling back to agent-run tests (deprecated by issue #726).",
        )
        return None


def _test_command_wall_budget(config: Any) -> float:
    """Resolve file-config test_command_wall_s over the default TEST_WALL_BUDGET_S."""
    file_config = getattr(config, "file_config", None)
    if isinstance(file_config, DaydreamFileConfig):
        configured = file_config.test_command_wall_s
        if configured is not None:
            return configured
    return phase_config.TEST_WALL_BUDGET_S


def _recipe_command(recipe: TestRecipe) -> list[str] | None:
    """Return resolved recipe argv, or None for its named miss; never guess a command."""
    if not recipe.command.resolved:
        return None
    value = recipe.command.value
    if isinstance(value, tuple):
        return list(value)
    return [value] if value else None


async def _run_host_test_command(
    cmd: list[str],
    work: WorkContext,
    config: Any,
    *,
    recipe: TestRecipe | None = None,
) -> TestExecutionResult:
    """Run with the resolved wall budget in the recipe package cwd, else the repository.

    Spawn errors propagate; callers choose their own gate failure policy.
    """
    cwd = work.repo if recipe is None else Path(work.repo, recipe.package.cwd_relative)
    return await run_test_command(
        cmd=cmd,
        cwd=cwd,
        wall_budget_s=_test_command_wall_budget(config),
    )


def _git_revision_facts(repo: Path) -> tuple[str, str]:
    """Read HEAD/branch tolerantly: unborn checkouts compare empty revision facts like any other identity."""
    try:
        return git_ops.head_sha(repo), git_ops.current_branch(repo) or ""
    except GitError:
        return "", ""


def reuse_target(
    work: WorkContext,
    recipe: TestRecipe | None,
    *,
    session_id: str,
    retained_tree_key: str,
    post_commit_verified: bool = False,
) -> ReuseTarget | None:
    """Build from resolved recipe facts and the caller's already-proven retained tree key.

    Missing recipes return None and require real validation.
    """
    if recipe is None:
        return None
    command = recipe.command.value if recipe.command.resolved else None
    head_sha, branch = _git_revision_facts(work.repo)
    return ReuseTarget(
        session_id=session_id,
        tree_key=retained_tree_key,
        argv=tuple(command) if isinstance(command, tuple) else (),
        cwd_relative=recipe.package.cwd_relative,
        runner=recipe.package.runner,
        interpreter=recipe.package.interpreter,
        config_digest=recipe.package.config_digest,
        absent_components=recipe.package.absent_components,
        head_sha=head_sha,
        branch=branch,
        post_commit_verified=post_commit_verified,
    )


def _host_test_identity(
    *,
    session_id: str,
    argv: tuple[str, ...],
    work: WorkContext,
    recipe: TestRecipe | None,
    input_tree_key: str,
    output_tree_key: str,
    result: TestExecutionResult | None,
    passed: bool,
) -> TestExecutionIdentity:
    """Bind resolved recipe facts and revision to the host outcome.

    Without a recipe, use worktree-root facts. Timeout takes precedence over clipped
    output, then exit-status pass/fail.
    """
    package = recipe.package if recipe is not None else None
    if result is not None and result.timed_out:
        outcome: Literal["passed", "failed", "timed-out", "truncated"] = "timed-out"
    elif result is not None and result.output_truncated:
        outcome = "truncated"
    elif passed:
        outcome = "passed"
    else:
        outcome = "failed"
    head_sha, branch = _git_revision_facts(work.repo)
    return TestExecutionIdentity(
        session_id=session_id,
        argv=argv,
        cwd_relative=package.cwd_relative if package is not None else ".",
        runner=package.runner if package is not None else None,
        interpreter=package.interpreter if package is not None else None,
        config_digest=package.config_digest if package is not None else None,
        absent_components=package.absent_components if package is not None else (),
        input_tree_key=input_tree_key,
        output_tree_key=output_tree_key,
        head_sha=head_sha,
        branch=branch,
        kind="host",
        outcome=outcome,
    )


@dataclass(frozen=True)
class TestAttemptEvidence:
    """Tree-bound evidence from a host execution or TEST-agent report.

    Only host runs carry a reusable execution identity; prose verdicts never do.
    """

    session_id: str
    kind: Literal["host", "agent"]
    command: tuple[str, ...] | None
    passed: bool
    input_tree_key: str
    output_tree_key: str
    identity: TestExecutionIdentity | None = None


@dataclass(frozen=True)
class TestAndHealResult:
    """Typed outcome of the bounded test-and-heal interaction."""

    passed: bool
    retries: int
    proceed: bool
    ignored: bool
    attempts: tuple[TestAttemptEvidence, ...]

@bind_resolved_run_context
async def phase_test_once(
    backend: Backend,
    work: WorkContext,
    *,
    config: Any,
    session_id: str,
    capture_tree_key: Callable[[], str],
    continuation: ContinuationToken | None = None,
    command_override: list[str] | None = None,
    recipe: TestRecipe | None = None,
    run_context: RunContext | None = None,
) -> tuple[TestAttemptEvidence, ContinuationToken | None, str]:
    """Execute one test attempt and capture the supplied before/after tree identity.

    Configured or approved commands run on the host; absent commands use the agent
    fallback. A recipe supplies command and cwd without re-resolution.
    """
    run_context = resolve_run_context(run_context)
    cmd: list[str] | None
    if command_override is not None:
        cmd = command_override
    elif recipe is not None:
        cmd = _recipe_command(recipe)
    else:
        cmd = _canonical_test_cmd(config)
    input_tree_key = capture_tree_key()
    next_continuation: ContinuationToken | None = None
    identity: TestExecutionIdentity | None = None
    host_result: TestExecutionResult | None = None
    if cmd is not None:
        try:
            host_result = await _run_host_test_command(cmd, work, config, recipe=recipe)
        except (OSError, ValueError) as exc:
            output = f"The configured test command failed to run (spawn): {exc}"
            passed = False
        else:
            if host_result.timed_out:
                ui.print_warning(
                    agent.console,
                    f"Test command hit the {_test_command_wall_budget(config):g}s "
                    "wall budget and was killed.",
                )
            output = host_result.merged_output
            passed = host_result.passed
        kind: Literal["host", "agent"] = "host"
        command: tuple[str, ...] | None = tuple(cmd)
    else:
        prompt = f"Run the project's test suite. {_TEST_RUN_INSTRUCTIONS}"
        prompt = append_extended_facts(prompt, recipe)
        output, next_continuation, _ = await agent.run_agent(
            backend,
            work.repo,
            prompt,
            continuation=continuation,
            phase=DaydreamPhase.TEST,
            tool_call_budget=phase_config.DEFAULT_TOOL_CALL_BUDGET,
            wall_budget_s=phase_config.TEST_WALL_BUDGET_S,
            run_context=run_context,
        )
        passed = detect_test_success(output)
        kind = "agent"
        command = None
    output_tree_key = capture_tree_key()
    if kind == "host" and command is not None:
        identity = _host_test_identity(
            session_id=session_id,
            argv=command,
            work=work,
            recipe=recipe,
            input_tree_key=input_tree_key,
            output_tree_key=output_tree_key,
            result=host_result,
            passed=passed,
        )
    return (
        TestAttemptEvidence(
            session_id=session_id,
            kind=kind,
            command=command,
            passed=passed,
            input_tree_key=input_tree_key,
            output_tree_key=output_tree_key,
            identity=identity,
        ),
        next_continuation,
        output,
    )
