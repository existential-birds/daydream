"""Agent interaction and backend management."""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import re
from collections.abc import AsyncGenerator, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anyio

if TYPE_CHECKING:
    from claude_agent_sdk.types import AgentDefinition
    from rich.text import Text

from daydream.backends import (
    Backend,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.config import UNKNOWN_SKILL_PATTERN
from daydream.extensions import get_registry
from daydream.json_utils import extract_json
from daydream.trajectory import DaydreamPhase, get_current_recorder
from daydream.ui import (
    AgentTextRenderer,
    LiveToolPanelRegistry,
    create_console,
    format_callback_progress,
    format_callback_text,
    print_cost,
    print_error,
    print_thinking,
    print_warning,
    prompt_user,
)


class MissingSkillError(Exception):
    """Raised when a required skill is not available."""

    def __init__(self, skill_name: str):
        self.skill_name = skill_name
        super().__init__(f"Skill '{skill_name}' is not available")


class _ToolSupervisorFailure(Exception):
    """Internal marker that keeps supervisor failures out of backend retries."""

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(str(original))

    @property
    def subtype(self) -> str:
        """Expose the original error's type name for trajectory recording."""
        return type(self.original).__name__


async def _emit_progress(callback: Callable[[Text], Any], text: Text) -> None:
    """Invoke a progress callback, awaiting its result when it is a coroutine."""
    result = callback(text)
    if inspect.isawaitable(result):
        await result


@dataclass
class AgentState:
    """Consolidated state for agent module.

    Attributes:
        assume: A forced yes/no answer for interactive gates — ``"yes"`` (``--yes``),
            ``"no"`` (a future ``--no``), or ``None`` (no assumption). Orthogonal to
            ``non_interactive``: ``non_interactive`` controls *whether* we may block on
            stdin; ``assume`` supplies a *pre-decided answer* regardless of TTY.
        log_mode: When True, bypass Rich UI and emit raw agent events as plain text
            to stdout (for CI log capture). Default False.
    """

    quiet_mode: bool = False
    non_interactive: bool = False
    assume: str | None = None
    log_mode: bool = False
    current_backends: list[Backend] = field(default_factory=list)


# Module-level singletons: access/mutate via the getter/setter functions below,
# never _state directly. reset_state() restores defaults between test runs.

_state = AgentState()
console = create_console()


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


def _summarize_input(input_data: dict[str, Any]) -> str:
    """One-line summary of tool input for log output."""
    if not input_data:
        return ""
    # For known tools, pick the most informative key
    if "command" in input_data:
        return input_data["command"][:200]
    if "path" in input_data:
        return f"{input_data['path']}" + (f" -> {input_data.get('new_path', '')}" if "new_path" in input_data else "")
    # Generic: first value that's a string
    for v in input_data.values():
        if isinstance(v, str):
            return v[:200]
    return str(input_data)[:200]


def _summarize_output(output: str) -> str:
    """One-line summary of tool output for log output."""
    if not output:
        return "(empty)"
    # Take first non-empty line or first 200 chars
    first_line = output.strip().split("\n")[0]
    return first_line[:200]


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
    wall_budget_s: float | None = None,
    tool_call_budget: int | None = None,
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
        read_only: When True, the backend enforces a non-mutating tool profile
            at the tool layer (Claude PreToolUse guard hook; Codex
            ``--sandbox read-only``). Wired True only for the read-only
            failure-summarizer call; all other call sites keep the default.
        wall_budget_s: Opt-in per-invocation wall-clock budget. When exceeded
            the loop is cancelled, ``backend.cancel()`` is awaited, the ATIF
            turn is marked aborted, and the partial output is returned — no
            exception reaches the caller. ``None`` (the default) disables the
            wall budget.
        tool_call_budget: Opt-in ceiling on ToolStartEvents in this turn. When
            exceeded the loop breaks with the same abort/partial-return path.
            ``None`` (the default) means no tool-call ceiling.

    Returns:
        Tuple of (output, continuation_token, budget_reason). Output is text
        or structured data. ``budget_reason`` is ``None`` on a normal
        completion, or a string such as ``"wall_budget_exceeded"`` /
        ``"tool_call_budget_exceeded"`` / ``"tool_vetoed:Write"`` when the
        turn was cut short.

    Raises:
        MissingSkillError: If a required skill is not available.
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
    event_iter: AsyncGenerator[Any, None] | None = None
    try:
        # Open Invocation scope when a recorder is active; nullcontext keeps the
        # with-shape uniform otherwise (CORE-09 no-op). D-19: no ATIF construction
        # here — only inv.observe()/inv.observe_user_step() against the recorder.
        recorder = get_current_recorder()
        _default_attempts = int(os.environ.get("DAYDREAM_PI_RETRY_ATTEMPTS", "3"))
        _default_delay = float(os.environ.get("DAYDREAM_PI_RETRY_BASE_DELAY_S", "2.0"))
        max_attempts = getattr(backend, "retry_attempts", _default_attempts)
        base_delay = getattr(backend, "retry_base_delay_s", _default_delay)
        if max_attempts < 0:
            raise ValueError("retry attempts must be >= 0")
        if not math.isfinite(base_delay):
            raise ValueError("retry base delay must be finite")
        if base_delay < 0:
            raise ValueError("retry base delay must be >= 0")

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
            # Created per attempt so a failed retry's UI panels and task-label
            # mappings cannot be flushed or reused by a later successful attempt.
            tool_registry = LiveToolPanelRegistry(console, _state.quiet_mode)
            agent_renderer = AgentTextRenderer(console)

            try:
                event_iter = cast(
                    AsyncGenerator[Any, None],
                    backend.execute(
                        cwd, prompt, output_schema, continuation,
                        agents=agents, max_turns=max_turns, read_only=read_only,
                    ),
                )
                invocation_cm: Any = (
                    recorder.invocation(phase=phase) if recorder is not None else nullcontext(None)
                )

                async with invocation_cm as inv:
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
                            if isinstance(event, TextEvent):
                                output_parts.append(event.text)

                                skill_match = re.search(UNKNOWN_SKILL_PATTERN, event.text)
                                if skill_match:
                                    if not use_callback and not _state.log_mode:
                                        agent_renderer.finish()
                                        tool_registry.finish_all()
                                    raise MissingSkillError(skill_match.group(1))

                                if _state.log_mode:
                                    print(event.text, flush=True)
                                elif use_callback and progress_callback is not None:
                                    last_line = event.text.strip().split("\n")[-1]
                                    if last_line:
                                        await _emit_progress(progress_callback, format_callback_text(last_line))
                                elif output_schema is None:
                                    # Structured-output text is the JSON payload, redundant with
                                    # the returned structured result — don't echo it to the terminal.
                                    agent_renderer.append(event.text)

                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, ThinkingEvent):
                                if _state.log_mode:
                                    print(f"[thinking] {event.text}", flush=True)
                                elif not use_callback:
                                    if agent_renderer.has_content:
                                        agent_renderer.finish()
                                    print_thinking(console, event.text)

                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, ToolStartEvent):
                                if _state.log_mode:
                                    tool_names[event.id] = event.name
                                    print(f"[tool:{event.name}] {_summarize_input(event.input)}", flush=True)
                                elif progress_callback is not None:
                                    # Record the originating call so a backgrounded launch's result
                                    # can later resolve a Task-family label for the progress line.
                                    tool_registry.note_call(event.id, event.name, event.input)
                                    label = tool_registry.resolve_call_label(event.name, event.input)
                                    await _emit_progress(
                                        progress_callback, format_callback_progress(event.name, event.input, label)
                                    )
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
                                    print(f"{prefix} {_summarize_output(event.output)}", flush=True)
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
                                    print(
                                        f"[metrics] prompt={event.prompt_tokens} completion={event.completion_tokens}",
                                        flush=True
                                    )
                                # EVNT-02 / MAP-06: recorder-only, no UI in normal mode. Must precede the
                                # CostEvent branch so isinstance order is correct.
                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, CostEvent):
                                if _state.log_mode:
                                    cost_str = f"${event.cost_usd:.4f}" if event.cost_usd is not None else "unknown"
                                    print(f"[cost] {cost_str}", flush=True)
                                elif event.cost_usd and not use_callback:
                                    if agent_renderer.has_content:
                                        agent_renderer.finish()
                                    console.print()
                                    print_cost(console, event.cost_usd)

                                if inv is not None:
                                    inv.observe(event)

                            elif isinstance(event, ResultEvent):
                                if _state.log_mode and event.structured_output is not None:
                                    print(f"[result] {json.dumps(event.structured_output)[:500]}", flush=True)
                                elif event.structured_output is not None:
                                    structured_result = event.structured_output
                                    if not use_callback:
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
                                else:
                                    structured_result = event.structured_output
                                result_continuation = event.continuation

                                if inv is not None:
                                    inv.observe(event)

                    # Abort handling: the wall scope cancelled the loop, a quantitative
                    # tool ceiling fired, or a supervisor veto broke out. Cancel the
                    # backend (swallow any error — an abort must NEVER raise into the
                    # CLI), mark the ATIF turn aborted, surface a marker, then fall
                    # through to the partial-output return.
                    wall_cancelled = bool(getattr(wall_scope, "cancelled_caught", False))
                    if budget_reason is None and wall_cancelled:
                        budget_reason = "wall_budget_exceeded"
                    aborted_reason = budget_reason
                    if budget_reason is not None:
                        try:
                            await event_iter.aclose()
                        except Exception:  # noqa: BLE001 - abort must not raise into the CLI
                            pass
                        try:
                            await backend.cancel()
                        except Exception:  # noqa: BLE001 - abort must not raise into the CLI
                            pass
                        if inv is not None:
                            inv.mark_aborted(budget_reason)
                        if _state.log_mode:
                            print(f"[aborted] {budget_reason}", flush=True)
                        elif use_callback and progress_callback is not None:
                            await _emit_progress(
                                progress_callback, format_callback_text(f"[budget] aborted: {budget_reason}")
                            )
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
                if attempt < max_attempts and getattr(exc, "retryable", False):
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    retry_msg = (
                        f"Backend error ({type(exc).__name__}), retrying "
                        f"attempt {attempt + 2}/{max_attempts + 1} after {delay:.1f}s..."
                    )
                    if _state.log_mode:
                        print(f"[retry] {retry_msg}", flush=True)
                    elif use_callback and progress_callback is not None:
                        await _emit_progress(progress_callback, format_callback_text(f"[retry] {retry_msg}"))
                    elif not use_callback:
                        print_warning(console, retry_msg)
                    if event_iter is not None:
                        try:
                            await event_iter.aclose()
                        except Exception:  # noqa: BLE001
                            pass
                        event_iter = None
                    # Do NOT call backend.cancel() here: the execute() generator's
                    # finally block already terminates and removes the failed
                    # invocation's process when event_iter.aclose() is called above.
                    # Calling cancel() would terminate all processes on the shared
                    # backend instance, killing sibling concurrent tasks under
                    # fan-out (phases.py phase_per_stack_reviews).
                    tool_registry.discard_all()
                    await anyio.sleep(delay)
                    continue
                raise

    except _ToolSupervisorFailure as exc:
        print_error(console, "Extension Failure", f"{type(exc.original).__name__}: {exc.original}")
        raise exc.original from None
    except Exception as exc:
        print_error(console, "Backend Execution Error", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        _state.current_backends.remove(backend)
        # Always close the async generator explicitly so the SDK's internal
        # TaskGroup / cancel scope exits in the same task that entered it.
        # Without this, the async-gen finalizer can fire during GC in a
        # different task, causing "Attempted to exit a cancel scope in a
        # different task" RuntimeError from anyio (D-20).
        if event_iter is not None:
            try:
                await event_iter.aclose()
            except Exception:  # noqa: BLE001 — cleanup must not raise
                pass

    if output_schema is not None and structured_result is not None:
        return structured_result, result_continuation, aborted_reason
    if output_schema is not None:
        raw = "".join(output_parts)
        # Fallback: extract JSON from the raw text when structured output
        # failed. Uses robust extraction (handles prose-wrapped JSON and
        # markdown code fences — common with GLM and other OpenAI-compat models).
        if raw.strip():
            parsed = extract_json(raw)
            if parsed is not None:
                return parsed, result_continuation, aborted_reason
    return "".join(output_parts), result_continuation, aborted_reason
