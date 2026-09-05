"""Agent interaction and backend management."""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import random
import re
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import anyio
from jsonschema import Draft202012Validator
from rich.console import Console

if TYPE_CHECKING:
    from claude_agent_sdk.types import AgentDefinition
    from rich.text import Text

from daydream.backends import (
    AgentEventStream,
    Backend,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)
from daydream.extensions import get_registry
from daydream.json_utils import extract_json
from daydream.trajectory import DaydreamPhase, get_current_recorder, redact_structured_text, redact_text, redact_value
from daydream.ui import (
    NEON_THEME,
    AgentTextRenderer,
    LiveToolPanelRegistry,
    format_callback_progress,
    format_callback_text,
    print_cost,
    print_error,
    print_thinking,
    print_warning,
)
from daydream.ui import (
    prompt_user as prompt_user,
)
from daydream.ui.tools import _BASH_COMMAND_MAX_CHARS, _PRIMARY_TOOL_ARG

_logger = logging.getLogger(__name__)


class _ToolSupervisorFailure(Exception):
    """Internal marker that keeps supervisor failures out of backend retries."""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))

    @property
    def subtype(self) -> str:
        """Expose the original error's type name for trajectory recording."""
        return type(self.original).__name__


class _RedactedSupervisorError(RuntimeError):
    """Scrubbed stand-in for a supervisor exception that cannot be rebuilt clean.

    A tool supervisor is arbitrary extension code and may raise an exception
    whose ``str()`` is not derived from ``args`` (e.g. ``OSError`` built from
    errno/strerror, or a type overriding ``__str__``/``__repr__``); such a value
    cannot be scrubbed in place. This stand-in carries the original type name
    (for recognizable diagnostics) and a message already run through
    ``redact_text``, so ``str(exc)`` re-printed by outer handlers never re-
    surfaces the raw credential.
    """

    def __init__(self, original_type_name: str, message: str) -> None:
        self.original_type_name = original_type_name
        self.retryable: bool = False
        super().__init__(message)


def _scrubbed_supervisor_error(original: BaseException) -> BaseException:
    """Return a re-propagatable copy of ``original`` whose ``str()`` is scrubbed.

    Reconstruct the same exception type from its args (each string run through
    ``redact_text``) where possible -- this handles RuntimeError-derived types
    and OSError alike, since a fresh instance rebuilds errno/strerror from the
    scrubbed args. Where reconstruction is impossible or the resulting str()
    still carries a redactable value (a type overriding ``__str__``/``__repr__``),
    fall back to ``_RedactedSupervisorError``. Either way the re-raised
    exception's ``str()`` is clean and ``retryable`` (a discriminator consumers
    like improve-run retry checks read via ``getattr``) is preserved.
    """
    scrubbed_args = tuple(
        redact_text(a) if isinstance(a, str) else a for a in original.args
    )
    try:
        clone = type(original)(*scrubbed_args)
    except (AttributeError, TypeError):
        clone = None
    if clone is not None and redact_text(str(clone)) == str(clone):
        setattr(clone, "retryable", getattr(original, "retryable", False))
        return clone
    stand_in = _RedactedSupervisorError(
        type(original).__name__, redact_text(str(original))
    )
    stand_in.retryable = getattr(original, "retryable", False)
    return stand_in


class _EventStreamScope:
    """Idempotent owner for one backend invocation's event stream."""

    def __init__(self, event_iter: AgentEventStream) -> None:
        self.event_iter = event_iter
        self._closed = False

    async def __aenter__(self) -> AgentEventStream:
        return self.event_iter

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the invocation once without masking its outcome."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.event_iter.aclose()
        except Exception:  # noqa: BLE001 - cleanup must not mask the invocation outcome
            pass


@dataclass
class AgentState:
    """Consolidated state for agent module.

    Attributes:
        assume: A forced yes/no answer for interactive gates — ``"yes"`` (``--yes``),
            ``"no"`` (a future ``--no``), or ``None`` (no assumption). Orthogonal to
            ``non_interactive``: ``non_interactive`` controls *whether* we may block on
            stdin; ``assume`` supplies a *pre-decided answer* regardless of TTY.
        log_mode: When True, bypass Rich UI and emit redacted agent events as plain text
            to stdout (for CI log capture). Default False.
    """

    quiet_mode: bool = False
    non_interactive: bool = False
    assume: str | None = None
    log_mode: bool = False
    current_backends: list[Backend] = field(default_factory=list)


class _LogRedactingConsole(Console):
    """Console that redacts string payloads while ``--log`` mode is active.

    phases.py, runner.py, and the other importers bind to this module-level
    console, so their Rich output would otherwise bypass the run_agent-event
    emitter and leak raw secrets via the UI path in ``--log`` mode.
    """

    def print(self, *objects: Any, **kwargs: Any) -> None:
        if _state.log_mode:
            objects = tuple(
                redact_text(obj) if isinstance(obj, str) else obj
                for obj in objects
            )
        super().print(*objects, **kwargs)


# Module-level singletons: access/mutate via the getter/setter functions below,
# never _state directly. reset_state() restores defaults between test runs.

_state = AgentState()
console = _LogRedactingConsole(theme=NEON_THEME)


def reset_state() -> None:
    """Reset the global agent state to defaults (restores between test runs)."""
    global _state
    _state = AgentState()


def set_quiet_mode(quiet: bool) -> None:
    """Set quiet mode for agent output."""
    _state.quiet_mode = quiet


def get_quiet_mode() -> bool:
    """Get current quiet mode setting."""
    return _state.quiet_mode


def set_non_interactive(value: bool) -> None:
    """Set non-interactive mode for prompts."""
    _state.non_interactive = value


def get_non_interactive() -> bool:
    """Get current non-interactive mode setting."""
    return _state.non_interactive


def set_assume(value: str | None) -> None:
    """Set the forced yes/no answer for interactive gates.

    Args:
        value: ``"yes"`` to auto-approve gates (``--yes``), ``"no"`` to auto-decline,
            or ``None`` for no assumption (gates fall back to prompting or their
            unattended safe default).
    """
    _state.assume = value


def get_assume() -> str | None:
    """Get the forced yes/no answer for interactive gates.

    Returns:
        ``"yes"``, ``"no"``, or ``None`` when no assumption is set.
    """
    return _state.assume


def set_log_mode(log_mode: bool) -> None:
    """Set log mode for agent output."""
    _state.log_mode = log_mode


def get_log_mode() -> bool:
    """Get current log mode setting."""
    return _state.log_mode


def resolve_gate(*, assume: str | None, interactive: bool, safe_default: bool) -> bool | None:
    """Resolve a yes/no interaction gate across the two orthogonal axes.

    Collapses *assume* (a forced answer) and *interactivity* (may we block on
    stdin?) into a single decision. Pure — performs no I/O.

    Args:
        assume: A forced answer: ``"yes"`` → True, ``"no"`` → False, ``None`` →
            no assumption (defer to interactivity).
        interactive: True when prompts may read stdin.
        safe_default: The answer to use when unattended and no assumption is set
            (e.g. ``False`` to decline a fix-apply, ``True`` to auto-commit).

    Returns:
        ``True``/``False`` to use the resolved answer directly, or ``None`` when
        the caller should fall back to an interactive prompt.
    """
    if assume is not None:
        return assume == "yes"
    if not interactive:
        return safe_default
    return None


def resolve_or_prompt(
    *,
    assume: str | None,
    interactive: bool,
    safe_default: bool,
    question: str,
    default: str,
) -> bool:
    """Resolve a yes/no gate, falling back to an interactive prompt when needed.

    Wraps :func:`resolve_gate` with the canonical prompt-and-coerce step so
    callers don't each re-implement the ``decision is None → prompt_user →
    lower() in ("y", "yes")`` idiom.

    Args:
        assume: Forwarded to :func:`resolve_gate` — ``"yes"`` → ``True``,
            ``"no"`` → ``False``, ``None`` → defer to interactivity.
        interactive: Forwarded to :func:`resolve_gate` — True when stdin may
            be read.
        safe_default: Forwarded to :func:`resolve_gate` — the answer used when
            unattended and no assumption is set.
        question: The prompt string shown to the user when interactive (e.g.
            ``"Apply fixes now? [y/N]"``).
        default: The default hint shown alongside the question (e.g. ``"n"``).

    Returns:
        ``True`` if the gate is approved, ``False`` if declined.
    """
    decision = resolve_gate(assume=assume, interactive=interactive, safe_default=safe_default)
    if decision is None:
        response = prompt_user(console, question, default)
        decision = response.strip().lower() in ("y", "yes")
    return decision


def get_current_backends() -> list[Backend]:
    """Get all currently running backends."""
    return list(_state.current_backends)


def detect_test_success(output: str) -> bool:
    """Detect if tests passed using pattern matching.

    Extracts structured pass/fail counts first (tolerating "N tests failed"
    wording and any separator between counts), then falls through to sentinel
    pass-phrases emitted by tooling or agents.
    """
    if not output:
        return False

    output_lower = output.lower()

    # finditer so a later non-zero count isn't hidden by an earlier "0 failed".
    # pytest "errors" (collection errors) are genuine non-passes — counted
    # alongside failures here.
    failed_counts = [
        int(match.group(1).replace(",", ""))
        for match in re.finditer(r"(\d[\d,]*)\s+(?:tests?\s+)?(?:fail(?:ed|ures?)|errors?)\b", output_lower)
    ]
    passed_counts = [
        int(match.group(1).replace(",", ""))
        for match in re.finditer(r"(\d[\d,]*)\s+(?:tests?\s+)?passed\b", output_lower)
    ]

    if any(count > 0 for count in failed_counts):
        return False

    # Hard negative signals win over success sentinels — a late traceback must not be
    # masked by an earlier "all tests pass" phrase.
    error_patterns = [
        r"tests? failing",
        r"test failure",
        r"assertion error",
        r"traceback",
    ]
    for pattern in error_patterns:
        if re.search(pattern, output_lower):
            return False

    # Explicit sentinels emitted by tooling / the test agent.
    success_sentinels = [
        r"test result:\s*ok",           # cargo / rust native
        r"tests?\s+pass(?:ed)?\s*[✅✓]", # agent emoji summary ("Tests PASS ✅")
        r"all \d+ tests? passed",
        r"tests? passed successfully",
        r"test suite passed",
        r"all tests pass",
        r"no (?:test )?failures?",
        r"\b0\s+failures?\b",
        r"\d+\s+passed(?:,\s*\d+\s+(?:deselected|skipped|xfailed))*(?:,\s*\d+\s+warnings?)?",
    ]
    for pattern in success_sentinels:
        if re.search(pattern, output_lower):
            return True

    # Structured: positive passed count and no failures at all.
    # pytest omits "0 failed" entirely when there are zero failures — an empty
    # failed_counts means "no failures mentioned". When "0 failed" IS present,
    # failed_counts is [0] (non-empty). Both cases are passes.
    max_passed = max(passed_counts) if passed_counts else None
    no_failures = not failed_counts or all(c == 0 for c in failed_counts)
    if max_passed is not None and max_passed > 0 and no_failures:
        return True

    # Conservative fallback: bare "passed" with no count is not enough.
    return False


def is_environmental_failure(test_output: str) -> bool:
    """Detect whether a test failure stems from missing infrastructure, not the code.

    Conservative, case-insensitive match on infra signatures (database/cache not
    reachable). Used to short-circuit the heal loop: re-running an agent fix turn
    cannot bring up a Postgres/Redis container, so an environmental failure must
    abort rather than burn turns on a non-code problem.
    """
    if not test_output:
        return False

    output_lower = test_output.lower()

    infra_signatures = [
        "connection refused",
        "localhost:5432",
        ":6379",
        "container is not running",
        "make db-up",
        "econnrefused",
    ]
    return any(signature in output_lower for signature in infra_signatures)


def _summarize_input(input_data: dict[str, Any], name: str) -> str:
    """One-line summary of tool input for log output."""
    if not input_data:
        return ""
    # The COMPLETE selected string is redacted before any [:_BASH_COMMAND_MAX_CHARS]
    # slice — redact-after-slice would truncate a credential into an unmatchable fragment.
    # Key the shared primary table by tool name the way ui.tools._primary_tool_value
    # does instead of hard-applying the Bash-only (command, description) preference:
    # TaskCreate/Agent inputs also carry "description", and letting the Bash pair
    # shadow it would replace their short subject with the long field.
    for key in _PRIMARY_TOOL_ARG.get(name, ()):
        value = input_data.get(key)
        if isinstance(value, str) and value:
            return redact_structured_text(value)[:_BASH_COMMAND_MAX_CHARS]
    if "path" in input_data:
        complete = f"{input_data['path']}" + (
            f" -> {input_data.get('new_path', '')}" if "new_path" in input_data else ""
        )
        return redact_structured_text(complete)
    # Generic: first value that's a string
    for v in input_data.values():
        if isinstance(v, str):
            return redact_structured_text(v)[:_BASH_COMMAND_MAX_CHARS]
    return redact_structured_text(str(input_data))[:_BASH_COMMAND_MAX_CHARS]


def _summarize_output(output: str) -> str:
    """One-line summary of tool output for log output."""
    if not output:
        return "(empty)"
    # Redact the COMPLETE output before strip/first-line/[:200] — a credential
    # straddling the summary boundary must be caught before the slice.
    redacted = redact_structured_text(output)
    # Take first non-empty line or first 200 chars
    first_line = redacted.strip().split("\n")[0]
    return first_line[:200]


def _validates_schema(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether ``value`` validates against ``schema`` (shape + required)."""
    return not any(Draft202012Validator(schema).iter_errors(value))


def _salvageable(value: Any, schema: dict[str, Any]) -> bool:
    """Return whether ``value`` is usable by a salvage-tolerant consumer.

    Full validation is the baseline, but the structured-output gate must not
    be all-or-nothing: it guards the backend-supplied primary result and the
    extraction fallback alike, and the per-stack parse, the recommendation
    verifier, and the cross-stack merge all normalize partial agent output
    (dropping invalid records rather than losing the whole payload, or
    accepting a bare item array), so a dict whose required top-level fields
    are present — with array-typed fields holding actual lists — is still
    returned for them to salvage, and so is a bare JSON array, which
    ``phase_cross_stack_merge`` normalizes to its item list (a bare array can
    never validate against the object-typed ``MERGED_ITEMS_SCHEMA``). Nested
    item validity is deliberately not checked here: that is the consumers'
    salvage domain.
    """
    if _validates_schema(value, schema):
        return True
    if isinstance(value, list):
        return True
    if not isinstance(value, dict):
        return False
    required = schema.get("required")
    if not isinstance(required, list):
        return False
    properties = schema.get("properties", {})
    for key in required:
        if key not in value:
            return False
        prop = properties.get(key)
        if isinstance(prop, dict) and prop.get("type") == "array":
            if not isinstance(value[key], list):
                return False
    return True


def _redact_log_value(value: Any) -> Any:
    """Recursively redact a log-mode value without mutating its argument.

    Delegates to the canonical :func:`daydream.trajectory.redact_value`
    redactor so the security-relevant recursion lives in exactly one place.
    """
    return redact_value(value)


def _print_log(value: str) -> None:
    """The safe ``--log`` emitter for run_agent events: redact, then print.

    Phase/UI output flows through the module-level ``console``, which redacts
    string payloads in log mode via the same fail-closed boundary.
    """
    print(redact_structured_text(value), flush=True)


async def run_agent(
    backend: Backend,
    cwd: Path,
    prompt: str,
    *,
    phase: DaydreamPhase,
    output_schema: dict[str, Any] | None = None,
    progress_callback: Callable[[Text], Any] | None = None,
    continuation: ContinuationToken | None = None,
    agents: dict[str, AgentDefinition] | None = None,
    max_turns: int | None = None,
    read_only: bool = False,
    persist_session: bool = True,
    wall_budget_s: float | None = None,
    tool_call_budget: int | None = None,
    validate_structured_output: bool = True,
) -> tuple[str | Any, ContinuationToken | None, str | None]:
    """Run agent with the given prompt and return output plus continuation token.

    Streams verbose output to stdout as it's received. When progress_callback
    is provided, runs in quiet mode and routes status updates through the
    callback instead of printing to the console.

    All keyword arguments after ``prompt`` are keyword-only (the ``*``
    separator was added in Phase 2). Existing call sites pass them by name,
    so this is non-breaking — but the new ``phase`` argument is REQUIRED
    with no default (D-05). Calls that omit it raise ``TypeError`` from the
    Python interpreter at call time.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the agent.
        prompt: The prompt to send to the agent.
        phase: Required DaydreamPhase label for ATIF Step.extra (MAP-08, D-05).
            Must be a literal DaydreamPhase enum member. Required keyword-only
            with no default — Python raises TypeError if omitted.
        output_schema: Optional JSON schema for structured output.
        progress_callback: Optional callback for status updates (quiet mode).
        continuation: Optional continuation token for multi-turn.
        agents: Optional mapping of specialist name -> AgentDefinition.
        max_turns: Optional cap on the number of model turns.
        read_only: When True, enforcement delegates to the backend: Claude
            rejects mutating tools via its PreToolUse guard, and Codex
            combines its read-only sandbox with a disposable standalone Git
            checkout whenever *cwd* is a worktree root, so a read-only commit
            can only update the disposable clone's refs and index. Callers
            select this flag explicitly per call site; the diagnostic
            subagents (setup-investigator, recommendation-verifier), the
            failure summarizer, and the exploration and repository
            reconnaissance specialists (pre_scan, repo_scan, improve recon)
            pass True, while mutating phases keep the False default.
        persist_session: When False, request an ephemeral backend invocation.
            The default preserves existing continuation behavior.
        wall_budget_s: Opt-in per-invocation wall-clock budget. When exceeded
            the loop and this invocation's event iterator are closed, the ATIF
            turn is marked aborted, and the partial output is returned — no
            exception reaches the caller. ``None`` (the default) disables the
            wall budget.
        tool_call_budget: Opt-in ceiling on ToolStartEvents in this turn. When
            exceeded the loop breaks with the same abort/partial-return path.
            ``None`` (the default) means no tool-call ceiling.
        validate_structured_output: When True (default), structured output —
            the backend-supplied primary result and the extraction fallback alike
            — is returned only when its shape is usable by downstream consumers
            (see ``_salvageable``). Set False for call sites that re-validate
            or salvage wholesale downstream — the improve recon and plan
            author, whose downstream validators (``validate_recon_commands`` /
            ``assemble_plan``) are the fail-closed enforcement point.

    Returns:
        Tuple of (output, continuation_token, budget_reason). Output is text
        or structured data. ``budget_reason`` is ``None`` on a normal
        completion, or a string such as ``"wall_budget_exceeded"`` /
        ``"tool_call_budget_exceeded"`` / ``"tool_vetoed:Write"`` when the
        turn was cut short.

    Raises:
        TypeError: If the keyword-only ``phase`` argument is not provided
            (raised by the Python interpreter at call time).
    """
    output_parts: list[str] = []
    structured_result: Any = None
    result_continuation: ContinuationToken | None = None
    aborted_reason: str | None = None
    use_callback = progress_callback is not None
    tool_supervisor = get_registry().tool_supervisor_if_registered()

    _state.current_backends.append(backend)
    try:
        # Open Invocation scope when a recorder is active; nullcontext keeps the
        # with-shape uniform otherwise (CORE-09 no-op). D-19: no ATIF construction
        # here — only inv.observe()/inv.observe_user_step() against the recorder.
        recorder = get_current_recorder()
        try:
            _default_attempts = int(os.environ.get("DAYDREAM_PI_RETRY_ATTEMPTS", "20"))
        except ValueError:
            _default_attempts = 20
        if _default_attempts < 0:
            _default_attempts = 20

        def _retry_delay_from_env(name: str, default: float) -> float:
            try:
                value = float(os.environ.get(name, str(default)))
            except ValueError:
                return default
            return value if math.isfinite(value) and value >= 0 else default

        _default_delay = _retry_delay_from_env("DAYDREAM_PI_RETRY_BASE_DELAY_S", 2.0)
        _default_max_delay = _retry_delay_from_env("DAYDREAM_PI_RETRY_MAX_DELAY_S", 120.0)
        max_attempts = getattr(backend, "retry_attempts", _default_attempts)
        base_delay = getattr(backend, "retry_base_delay_s", _default_delay)
        max_delay = getattr(backend, "retry_max_delay_s", _default_max_delay)
        if max_attempts < 0:
            raise ValueError("retry attempts must be >= 0")
        if not math.isfinite(base_delay):
            raise ValueError("retry base delay must be finite")
        if base_delay < 0:
            raise ValueError("retry base delay must be >= 0")
        if not math.isfinite(max_delay):
            raise ValueError("retry max delay must be finite")
        if max_delay < 0:
            raise ValueError("retry max delay must be >= 0")

        for attempt in range(max_attempts + 1):
            # Reset accumulated state so a failed attempt's partial output
            # does not leak into the next attempt's return value.
            output_parts = []
            structured_result = None
            result_continuation = None
            tool_calls = 0
            budget_reason: str | None = None
            # Track tool names by id for log mode output
            tool_names: dict[str, str] = {}
            callback_text_parts: list[str] = []

            async def _flush_callback_text() -> None:
                """Render one line for a consecutive run of streamed text deltas."""
                if progress_callback is None or not callback_text_parts:
                    return
                text = "".join(callback_text_parts)
                callback_text_parts.clear()
                last_line = text.strip().split("\n")[-1]
                if last_line:
                    result = progress_callback(format_callback_text(last_line))
                    if inspect.isawaitable(result):
                        await result

            # Created per attempt so a failed retry's UI panels and task-label
            # mappings cannot be flushed or reused by a later successful attempt.
            tool_registry = LiveToolPanelRegistry(console, _state.quiet_mode)
            agent_renderer = AgentTextRenderer(console)

            try:
                execute_kwargs: dict[str, Any] = {
                    "agents": agents,
                    "max_turns": max_turns,
                    "read_only": read_only,
                }
                if not persist_session:
                    execute_kwargs["persist_session"] = False
                event_iter = backend.execute(
                    cwd, prompt, output_schema, continuation,
                    **execute_kwargs,
                )
                invocation_cm: Any = (
                    recorder.invocation(phase=phase) if recorder is not None else nullcontext(None)
                )
                event_stream_scope = _EventStreamScope(event_iter)

                async with invocation_cm as inv, event_stream_scope:
                    if inv is not None:
                        inv.observe_user_step(prompt=prompt)

                    # Per-invocation abort controls live here so both backends are
                    # covered without a backend-signature change. The wall budget
                    # cancels the async-for via move_on_after; the tool-call ceiling
                    # and supervisor veto break in-loop.
                    wall_scope: Any = (
                        anyio.move_on_after(wall_budget_s) if wall_budget_s is not None else nullcontext()
                    )

                    with wall_scope:
                        async for event in event_iter:
                            if use_callback and not isinstance(event, TextEvent):
                                await _flush_callback_text()

                            if isinstance(event, TextEvent):
                                output_parts.append(event.text)

                                if _state.log_mode:
                                    _print_log(event.text)
                                elif use_callback and progress_callback is not None:
                                    callback_text_parts.append(event.text)
                                elif output_schema is None:
                                    # Structured-output text is the JSON payload, redundant with
                                    # the returned structured result — don't echo it to the terminal.
                                    agent_renderer.append(event.text)

                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, ThinkingEvent):
                                if _state.log_mode:
                                    _print_log(f"[thinking] {event.text}")
                                elif not use_callback:
                                    if agent_renderer.has_content:
                                        agent_renderer.finish()
                                    print_thinking(console, event.text)

                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, ToolStartEvent):
                                if _state.log_mode:
                                    tool_names[event.id] = event.name
                                    _print_log(f"[tool:{event.name}] {_summarize_input(event.input, event.name)}")
                                elif progress_callback is not None:
                                    # Record the originating call so a backgrounded launch's result
                                    # can later resolve a Task-family label for the progress line.
                                    tool_registry.note_call(event.id, event.name, event.input)
                                    label = tool_registry.resolve_call_label(event.name, event.input)
                                    result = progress_callback(format_callback_progress(event.name, event.input, label))
                                    if inspect.isawaitable(result):
                                        await result
                                else:
                                    if agent_renderer.has_content:
                                        agent_renderer.finish()
                                    tool_registry.create(event.id, event.name, event.input)

                                if inv is not None:
                                    inv.observe(event)

                                if tool_supervisor is not None:
                                    try:
                                        decision = tool_supervisor(event.name, event.input, phase=phase)
                                    except Exception as exc:  # noqa: BLE001 - policy failures must propagate
                                        raise _ToolSupervisorFailure(exc) from exc
                                    if decision.veto:
                                        if recorder is not None:
                                            recorder.emit_tool_veto(
                                                event.name, decision.reason, phase=phase
                                            )
                                        budget_reason = f"tool_vetoed:{event.name}"
                                        break

                                tool_calls += 1
                                if tool_call_budget is not None and tool_calls > tool_call_budget:
                                    budget_reason = "tool_call_budget_exceeded"
                                    break

                            elif isinstance(event, ToolResultEvent):
                                if _state.log_mode:
                                    tool_name = tool_names.get(event.id, "unknown")
                                    prefix = (
                                        f"[tool:{tool_name} ERROR]" if event.is_error
                                        else f"[tool:{tool_name} result]"
                                    )
                                    _print_log(f"{prefix} {_summarize_output(event.output)}")
                                else:
                                    # Populate the task_id→label map in both modes, so a later
                                    # TaskOutput/TaskStop resolves its originating label.
                                    tool_registry.observe_result(event.id, event.output)
                                    if not use_callback:
                                        panel = tool_registry.get(event.id)
                                        if panel:
                                            panel.set_result(event.output, event.is_error)
                                            panel.finish()
                                            tool_registry.remove(event.id)

                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, MetricsEvent):
                                if _state.log_mode:
                                    _print_log(
                                        f"[metrics] prompt={event.prompt_tokens} completion={event.completion_tokens}",
                                    )
                                # EVNT-02 / MAP-06: recorder-only, no UI in normal mode. Must precede the
                                # CostEvent branch so isinstance order is correct.
                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, CostEvent):
                                if _state.log_mode:
                                    cost_str = f"${event.cost_usd:.4f}" if event.cost_usd is not None else "unknown"
                                    _print_log(f"[cost] {cost_str}")
                                elif event.cost_usd and not use_callback:
                                    if agent_renderer.has_content:
                                        agent_renderer.finish()
                                    console.print()
                                    print_cost(console, event.cost_usd)

                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, TurnEndEvent):
                                # Per-turn close (issue #747): forward the
                                # turn boundary so each turn's already-emitted
                                # MetricsEvent lands on its own Step instead of
                                # collapsing into one. Pure recorder
                                # forwarding — no UI, no logging. The recorder's
                                # no-open-step no-op guard prevents empty-step
                                # invention.
                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, ResultEvent):
                                # Capture the structured result unconditionally: the log-mode
                                # print is an additive side effect, never a substitute for
                                # capture (otherwise --log silently drops every structured
                                # result — exploration conventions, review findings, etc.).
                                structured_result = event.structured_output
                                if event.structured_output is not None:
                                    if _state.log_mode:
                                        redacted = _redact_log_value(event.structured_output)
                                        _print_log(
                                            f"[result] {json.dumps(redacted)[:500]}",
                                        )
                                    elif not use_callback:
                                        issues = (
                                            structured_result.get("issues", [])
                                            if isinstance(structured_result, dict)
                                            else []
                                        )
                                        if issues:
                                            formatted = []
                                            for i in issues:
                                                if "file" in i and "line" in i:
                                                    desc = i.get("description", "")
                                                    issue_id = i.get("id", "?")
                                                    formatted.append(
                                                        f"[{issue_id}] {i['file']}:{i['line']} - {desc}"
                                                    )
                                                else:
                                                    label = i.get("title", i.get("description", ""))
                                                    formatted.append(f"[{i.get('id', '?')}] {label}")
                                            agent_renderer.append("\n".join(formatted))
                                result_continuation = event.continuation

                                if inv is not None:
                                    inv.observe(event)

                        if use_callback:
                            await _flush_callback_text()

                    # Abort handling: the wall scope cancelled the loop, a quantitative
                    # tool ceiling fired, or a supervisor veto broke out. Mark the
                    # ATIF turn aborted and let the invocation's event-stream scope
                    # close its resources before returning partial output.
                    wall_cancelled = bool(getattr(wall_scope, "cancelled_caught", False))
                    if budget_reason is None and wall_cancelled:
                        budget_reason = "wall_budget_exceeded"
                    aborted_reason = budget_reason
                    if budget_reason is not None:
                        await event_stream_scope.aclose()
                        if inv is not None:
                            inv.mark_aborted(budget_reason)
                            inv.observe(TurnEndEvent())
                        if _state.log_mode:
                            _print_log(f"[aborted] {budget_reason}")
                        elif use_callback and progress_callback is not None:
                            result = progress_callback(format_callback_text(f"[budget] aborted: {budget_reason}"))
                            if inspect.isawaitable(result):
                                await result
                        elif not use_callback:
                            print_warning(console, f"Turn aborted: {budget_reason}")

                    if not use_callback and not _state.log_mode:
                        if agent_renderer.has_content:
                            agent_renderer.finish()
                        tool_registry.finish_all()
                        console.print()

                break  # success — exit the retry loop

            except _ToolSupervisorFailure:
                raise
            except Exception as exc:
                if use_callback:
                    await _flush_callback_text()
                exception_max_retries = min(
                    max_attempts, getattr(exc, "max_retries", max_attempts)
                )
                if attempt < exception_max_retries and getattr(exc, "retryable", False):
                    delay = min(
                        base_delay * (2 ** attempt) + random.uniform(0, 1),
                        max_delay,
                    )
                    retry_msg = (
                        f"Backend error ({type(exc).__name__}), retrying "
                        f"attempt {attempt + 2}/{exception_max_retries + 1} after {delay:.1f}s..."
                    )
                    if _state.log_mode:
                        _print_log(f"[retry] {retry_msg}")
                    elif use_callback and progress_callback is not None:
                        result = progress_callback(format_callback_text(f"[retry] {retry_msg}"))
                        if inspect.isawaitable(result):
                            await result
                    elif not use_callback:
                        print_warning(console, retry_msg)
                    # The event-stream scope has already closed only this failed
                    # invocation. Backend-wide cancel() is reserved for shutdown.
                    tool_registry.discard_all()
                    await anyio.sleep(delay)
                    continue
                raise

    except _ToolSupervisorFailure as exc:
        original = exc.original
        print_error(
            console, "Extension Failure", redact_text(f"{type(original).__name__}: {original}")
        )
        # The exception itself still propagates to outer handlers that re-print
        # str(exc) without redaction (e.g. the CLI's "Fatal Error" panel on
        # `daydream <target>`, improve-run retry checks). Rewriting .args in
        # place is not enough: a supervisor can raise OSError (whose str() is
        # built from errno/strerror) or a type overriding __str__/__repr__, for
        # which the raw credential would survive. Rebuild a scrubbed exception
        # here, failing closed instead of silently passing the raw value onward.
        raise _scrubbed_supervisor_error(original) from original
    except Exception as exc:
        category = getattr(exc, "category", None)
        msg = str(exc).strip()
        diagnostic = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
        if isinstance(category, str):
            diagnostic += f" [{category}]"
        # Error messages can embed secrets (a leaked env var, an API key in a
        # provider error); redact at this host boundary like every other surfaced text.
        print_error(console, "Backend Execution Error", redact_text(diagnostic))
        raise
    except BaseException:
        # Shutdown path: SIGINT (KeyboardInterrupt) / task cancellation
        # (CancelledError) are BaseException, so the generic `except Exception`
        # above never sees them. Deterministically reap the tracked subprocesses
        # via backend.cancel() before unwinding.
        try:
            await backend.cancel()
        except Exception:  # cancel() must not mask the original signal
            _logger.exception("backend.cancel() failed during shutdown")
        raise
    finally:
        _state.current_backends.remove(backend)

    def _usable(value: Any) -> bool:
        """Whether ``value`` passes the structured-output gate.

        One predicate shared by the primary-result and extraction-fallback
        return paths: either this call site opted out of validation, or the
        value is salvageable (see ``_salvageable``).
        """
        return not validate_structured_output or (
            output_schema is not None and _salvageable(value, output_schema)
        )

    if output_schema is not None and structured_result is not None and _usable(structured_result):
        return structured_result, result_continuation, aborted_reason
    if output_schema is not None:
        raw = "".join(output_parts)
        # Fallback: extract JSON from the raw text when structured output
        # failed. Uses robust extraction (handles prose-wrapped JSON and
        # markdown code fences — common with GLM and other OpenAI-compat models).
        # The parsed result must also pass the fallback gate — an unvalidated
        # fallback would be asymmetric with the success path. The gate is
        # salvage-tolerant, not all-or-nothing (see _salvageable): consumers
        # like the per-stack parse, the recommendation verifier, and the
        # cross-stack merge normalize partial output (dropping invalid records
        # rather than losing the whole payload, or accepting a bare item
        # array), so salvageable structures still reach them. Only output that
        # is unusable in any shape falls through to the plain-text return.
        # Callers that salvage wholesale downstream — the improve recon and
        # the plan author, whose downstream validators (validate_recon_commands
        # / assemble_plan) are the fail-closed enforcement point — opt out
        # with validate_structured_output=False; the downstream validator
        # remains the fail-closed enforcement point for those call sites.
        if raw.strip():
            parsed = extract_json(raw)
            if parsed is not None and _usable(parsed):
                return parsed, result_continuation, aborted_reason
    return "".join(output_parts), result_continuation, aborted_reason
