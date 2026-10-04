"""Test evidence for review and fix phases."""

from collections.abc import Callable, Mapping
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
from daydream.phases.repair_outcome import RepairOutcome, repair_reason_code
from daydream.review_result import ReasonCode
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
    ``abort_reason`` names why an agent turn's result is not a result: a turn the
    host cut short is never evidence of a passing suite, however much its partial
    output looks like one.
    """

    session_id: str
    kind: Literal["host", "agent"]
    command: tuple[str, ...] | None
    passed: bool
    input_tree_key: str
    output_tree_key: str
    identity: TestExecutionIdentity | None = None
    abort_reason: str | None = None


@dataclass(frozen=True)
class RepairAttemptEvidence:
    """Host-recorded outcome of one bounded test-repair turn inside a repair job.

    This record is the only thing that distinguishes an interrupted repair from a
    completed one, so it is built from Git-observed state and host signals alone:
    ``changed_paths`` is what the tree actually holds, never what the turn claimed.
    ``focused_evidence`` holds bounded redacted excerpts of the failure output the
    turn worked from, and ``scope_request`` is the structured request the turn
    returned in its final message (parsed by ``phases.fix.parse_fix_scope_request``),
    if any.
    """

    job_id: str
    execution_id: str
    run_id: str
    outcome: RepairOutcome
    abort_reason: str | None
    backend_name: str
    model: str
    execution_elapsed_s: float
    job_elapsed_s: float
    input_tree_key: str
    output_tree_key: str
    changed_paths: tuple[str, ...]
    checkpoint_ref: str | None = None
    focused_evidence: tuple[str, ...] = ()
    scope_request: Mapping[str, Any] | None = None
    #: Opaque digest of the turn's continuation token, never the token itself:
    #: a resumed context may carry opaque provider data.
    continuation_ref: str | None = None
    #: Named degradation of the host's own accounting (e.g. a Git read that
    #: failed), never a swallowed failure.
    diagnostics: tuple[str, ...] = ()

    @property
    def reason_code(self) -> ReasonCode | None:
        """The converged public stop reason for this turn, or ``None`` if it ended on its own."""
        return repair_reason_code(self.abort_reason)

    def payload(self) -> dict[str, Any]:
        """JSON-serializable form, naming the converged public reason for the stop.

        Nothing is defaulted away: a field the host could not produce must fail
        record construction rather than be invented here.
        """
        return {
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "outcome": self.outcome.value,
            "reason_code": repair_reason_code(self.abort_reason),
            "abort_reason": self.abort_reason,
            "backend_name": self.backend_name,
            "model": self.model,
            "execution_elapsed_s": self.execution_elapsed_s,
            "job_elapsed_s": self.job_elapsed_s,
            "input_tree_key": self.input_tree_key,
            "output_tree_key": self.output_tree_key,
            "changed_paths": list(self.changed_paths),
            "checkpoint_ref": self.checkpoint_ref,
            "focused_evidence": list(self.focused_evidence),
            "scope_request": dict(self.scope_request) if self.scope_request is not None else None,
            "continuation_ref": self.continuation_ref,
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class TestAndHealResult:
    """Typed outcome of the bounded test-and-heal interaction.

    ``attempts`` holds one entry per test execution and ``repairs`` one entry per
    repair turn, so a caller can tell an interrupted repair from a completed one
    without re-reading the transcript. ``repairs`` defaults empty for callers that
    never run a repair turn.
    """

    passed: bool
    retries: int
    proceed: bool
    ignored: bool
    attempts: tuple[TestAttemptEvidence, ...]
    repairs: tuple[RepairAttemptEvidence, ...] = ()

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
    abort_reason: str | None = None
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
        output, next_continuation, abort_reason = await agent.run_agent(
            backend,
            work.repo,
            prompt,
            continuation=continuation,
            phase=DaydreamPhase.TEST,
            tool_call_budget=phase_config.DEFAULT_TOOL_CALL_BUDGET,
            wall_budget_s=phase_config.TEST_WALL_BUDGET_S,
            run_context=run_context,
        )
        # An incomplete result is not a result. A turn the host cut off carries
        # whatever text it had streamed, and a partial summary of a suite that
        # never finished running is exactly the shape a false green takes, so
        # the attempt is red and says why.
        passed = abort_reason is None and detect_test_success(output)
        if abort_reason is not None:
            ui.print_warning(
                agent.console,
                f"Test-suite turn did not complete ({abort_reason}); it is not a result.",
            )
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
            abort_reason=abort_reason,
        ),
        next_continuation,
        output,
    )
