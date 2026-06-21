"""ATIF v1.6 trajectory recorder for daydream runs.

This module is the SOLE home for ATIF Pydantic model construction (D-19
module-bloat ban). Other modules (agent.py, phases.py, ui.py, runner.py,
backends/*) import only the public surface — never `daydream.atif.*`.

Lifecycle: ``runner.py`` opens ``async with TrajectoryRecorder(...) as
recorder`` once per run. ``agent.run_agent()`` opens an ``Invocation`` per
call against the recorder via ``get_current_recorder()``. Backends emit
``AgentEvent`` instances; the Invocation buffers them into ATIF Steps and
flushes to the parent Trajectory at scope exit. The Recorder writes the
Trajectory JSON on clean ``__aexit__``.

Phase 2 shipped the minimum surface: ONE ``ContextVar`` (``_RECORDER_VAR``),
and a no-op ``Redactor``. Phase 3 adds ``fork()`` for parallel task groups
(sibling trajectory files) reusing the single ``_RECORDER_VAR`` (D-04). Phase 4 fills
in the Redactor rule list.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import nullcontext, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import daydream
from daydream.atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from daydream.timeutil import parse_iso_timestamp
from daydream.ui import create_console, print_error, print_warning

# Run-directory layout. Live + archive trajectories share an identical on-disk
# shape (<root>/runs/<session_id>/trajectory.json and .../trajectories/<descriptor>.json);
# live root is <target>/.daydream, archive root is per daydream.archive.
_RUNS_SUBDIR = "runs"
_TRAJECTORIES_SUBDIR = "trajectories"
_DAYDREAM_DIRNAME = ".daydream"

if TYPE_CHECKING:
    from daydream.backends import AgentEvent

_console = create_console()
_INITIAL_TOTALS: dict[str, Any] = {"prompt": 0, "completion": 0, "cached": 0, "cost": 0.0, "any_cost_seen": False}  # noqa: E501 - module-level constant cloned via dict.copy() at recorder init

# Generic backend labels that should be replaced as soon as a real SDK
# model id arrives via MetricsEvent / CostEvent. Runner stamps the recorder
# with one of these (or empty) at init since the real model id isn't known
# until the first agent turn streams back.
_GENERIC_MODEL_LABELS: frozenset[str] = frozenset({"claude", "codex", ""})

# Redaction patterns (REDA-01..04). Order in _REDACTION_RULES matters: URL-credential
# before bare API-key (so the captured credential isn't re-matched), PEM before env-var
# (so `VAR=<PEM>` collapses whole instead of leaking the key body), env-var before bare
# API-key (so `OPENAI_API_KEY=sk-1234` keeps its name per D-03).
_URL_CREDENTIAL_PATTERN = re.compile(r"(https?://)([^:@/\s]+):([^@/\s]+)@")
_API_KEY_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_\-]{6,}|ghp_[A-Za-z0-9]{6,}|ghs_[A-Za-z0-9]{6,}|xoxb-[A-Za-z0-9\-]{6,}|AKIA[A-Z0-9]{16})\b"
)
_JWT_PATTERN = re.compile(
    r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b"
)
_USERNAME_PATH_PATTERN = re.compile(r"(/Users/|/home/|[A-Z]:\\Users\\)([^/\\\s]+)")
# PEM private-key blocks (PKCS1 and PKCS8). Multi-line body collapsed before the
# bare API-key rule scans it. CERTIFICATE blocks are public material — not matched.
_PEM_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA )?PRIVATE KEY-----.*?-----END (?:RSA )?PRIVATE KEY-----",
    re.DOTALL,
)
# Match env-var assignment where one of the underscore-separated SEGMENTS of
# the var name is a secret keyword. Substring matching (the original) over-
# redacted MONKEY_PATCH/KEYBOARD_LAYOUT/AUTHOR/TOKENIZED — segment-aware
# matching keeps the secret list precise without false positives.
_ENV_VAR_PATTERN = re.compile(
    r"\b((?:[A-Z][A-Z0-9]*_)*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|CREDENTIALS|API_?KEY|APIKEY|AUTH)(?:_[A-Z0-9]+)*)\s*=\s*([^\s\n\r;]+)"  # noqa: E501 - secret-segment alternation
)
_REDACTION_RULES: tuple[tuple[Any, str], ...] = (
    (_URL_CREDENTIAL_PATTERN, r"\1[REDACTED_USER]:[REDACTED_API_KEY]@"),
    (_PEM_KEY_PATTERN, "[REDACTED_PEM_KEY]"),
    (_ENV_VAR_PATTERN, r"\1=[REDACTED_ENV_VAR]"),
    (_API_KEY_PATTERN, "[REDACTED_API_KEY]"),
    (_JWT_PATTERN, "[REDACTED_JWT]"),
    (_USERNAME_PATH_PATTERN, r"\1[REDACTED_USER]"),
)


def _safe_descriptor(raw: str) -> str:
    """Slugify a descriptor to filesystem-safe characters (D-06).

    Raises:
        ValueError: If *raw* produces an empty slug after sanitization.
    """
    slug = re.sub(r"[^a-z0-9-]", "-", raw.lower())
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError(f"Descriptor {raw!r} produces empty slug after sanitization")
    return slug


def default_trajectory_path(target_dir: Path, session_id: str) -> Path:
    """Return the default trajectory path under ``<target>/.daydream/runs/<session_id>/``.

    The session_id segment guarantees uniqueness per run; the recorder
    creates the directory before its first write.
    """
    return (
        target_dir
        / _DAYDREAM_DIRNAME
        / _RUNS_SUBDIR
        / session_id
        / "trajectory.json"
    )


def maybe_fork(recorder: "TrajectoryRecorder | None", descriptor: str) -> Any:
    """Return a fork CM if *recorder* is set, otherwise a no-op context manager."""
    if recorder is not None:
        return recorder.fork(descriptor)
    return nullcontext()


def now_iso() -> str:
    """Return current UTC time as ISO 8601 with trailing 'Z'.

    The single source of truth for timestamps in daydream's trajectory
    recording. Used by ``AgentEvent`` dataclass ``field(default_factory=...)``
    in ``daydream/backends/__init__.py`` (Plan 02), by recorder Step
    construction here, and by Phase 4 partial-write paths.

    Banned alternatives: the deprecated naive-utc helper from ``datetime``
    (Pitfall 2: lacks tzinfo, deprecated in 3.12+); ad-hoc
    ``datetime.now().isoformat()`` (no ``Z`` suffix — Pydantic timestamp
    validator requires ``Z`` or ``+00:00``).

    Returns:
        Timestamp string parseable by ``Step.validate_timestamp``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").removesuffix("+00:00") + "Z"


class DaydreamPhase(str, Enum):
    """Phase label for ``Step.extra['daydream_phase']`` (MAP-08).

    Values match ATIF ``extra`` field literals exactly. Required keyword-only
    arg on ``run_agent()`` (D-05); every call site in ``phases.py`` passes a
    literal member.
    """

    REVIEW = "review"
    PARSE = "parse"
    FIX = "fix"
    TEST = "test"
    INTENT = "intent"
    ALTERNATIVES = "alternatives"
    PLAN = "plan"
    PR_FEEDBACK = "pr_feedback"
    DEEP = "deep"
    EXPLORATION = "exploration"
    VERIFY = "verify"


class DaydreamRunFlow(str, Enum):
    """Run-flow label for ``Step.extra['daydream_run_flow']`` (MAP-09).

    Set once at recorder construction (D-07); recorder stamps every Step.
    """

    NORMAL = "normal"
    TTT = "ttt"
    PR = "pr"
    DEEP = "deep"


class Redactor:
    """Regex-driven redactor (REDA-01..06).

    Applies ``_REDACTION_RULES`` uniformly to all four ATIF text surfaces:
    ``Step.message``, ``Step.reasoning_content``, every
    ``ToolCall.arguments`` value, and every ``ObservationResult.content``
    string. Per D-04 the dispatch is flat regex on serialized text — no
    JSON-aware deep walk. Per REDA-05 the failure mode is "redact-or-omit":
    any internal exception replaces the offending value with
    ``"[REDACTION_FAILED]"`` rather than letting the raw value through.
    """

    def _redact_text(self, s: str) -> str:
        """Apply every redaction rule to *s* and return the result."""
        for pattern, replacement in _REDACTION_RULES:
            s = pattern.sub(replacement, s)
        return s

    def _redact_optional_text(self, value: str | None) -> str | None:
        """Redact a possibly-None text field; degrade to [REDACTION_FAILED] on error."""
        if value is None:
            return None
        try:
            return self._redact_text(value)
        except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
            return "[REDACTION_FAILED]"

    def _redact_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Redact every value inside a ToolCall.arguments dict.

        Per D-04 the dispatch is flat regex on the serialized form of each
        value. String values are redacted directly. Non-string values are
        ``json.dumps``'d, redacted, then re-parsed back to their Python
        structure so ``ToolCall.arguments`` keeps its declared shape. If
        re-parse fails (regex broke JSON syntax) the value is replaced with
        ``"[REDACTION_FAILED]"`` per REDA-05.
        """
        out: dict[str, Any] = {}
        for key, val in arguments.items():
            try:
                if isinstance(val, str):
                    out[key] = self._redact_text(val)
                else:
                    serialized = json.dumps(val)
                    redacted = self._redact_text(serialized)
                    out[key] = json.loads(redacted)
            except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
                out[key] = "[REDACTION_FAILED]"
        return out

    def _redact_observation(self, observation: Observation | None) -> Observation | None:
        """Redact every string-valued ObservationResult.content in *observation*."""
        if observation is None:
            return None
        new_results: list[ObservationResult] = []
        for r in observation.results:
            new_content: Any = r.content
            if isinstance(r.content, str):
                try:
                    new_content = self._redact_text(r.content)
                except Exception:  # noqa: BLE001 - REDA-05 redact-or-omit
                    new_content = "[REDACTION_FAILED]"
            elif isinstance(r.content, list):
                new_content = [
                    part.model_copy(update={"text": self._redact_optional_text(part.text)})
                    if part.type == "text"
                    else part
                    for part in r.content
                ]
            new_results.append(r.model_copy(update={"content": new_content}))
        return observation.model_copy(update={"results": new_results})

    def redact_step(self, step: Step) -> Step:
        """Return a redacted copy of *step* (REDA-04, REDA-05).

        Applies the redaction rules uniformly to ``message``,
        ``reasoning_content``, every ``ToolCall.arguments`` value, and
        every ``ObservationResult.content`` string. Internal exceptions
        degrade to ``"[REDACTION_FAILED]"`` for the offending field — never
        raw pass-through.

        Args:
            step: ATIF Step about to be appended to the Trajectory.

        Returns:
            A new Step instance whose text-bearing fields have been run
            through the redaction rules.
        """
        try:
            updates: dict[str, Any] = {}
            if isinstance(step.message, str):
                updates["message"] = self._redact_optional_text(step.message)
            elif isinstance(step.message, list):
                updates["message"] = [
                    part.model_copy(update={"text": self._redact_optional_text(part.text)})
                    if part.type == "text"
                    else part
                    for part in step.message
                ]
            if step.reasoning_content is not None:
                updates["reasoning_content"] = self._redact_optional_text(step.reasoning_content)
            if step.tool_calls is not None:
                redacted_calls = [
                    tc.model_copy(update={"arguments": self._redact_arguments(tc.arguments)})
                    for tc in step.tool_calls
                ]
                updates["tool_calls"] = redacted_calls
            if step.observation is not None:
                updates["observation"] = self._redact_observation(step.observation)
            if not updates:
                return step
            return step.model_copy(update=updates)
        except Exception as exc:  # noqa: BLE001 - REDA-05 redact-or-omit (top-level fallback)
            print_warning(_console, f"Redactor failure: {type(exc).__name__}: {exc}")
            # Wipe every text-bearing surface — partial wipes leak secrets
            # if redaction failed mid-arguments / mid-observation.
            safe_updates: dict[str, Any] = {"message": "[REDACTION_FAILED]"}
            if step.reasoning_content is not None:
                safe_updates["reasoning_content"] = "[REDACTION_FAILED]"
            if step.tool_calls is not None:
                safe_updates["tool_calls"] = [
                    tc.model_copy(update={"arguments": {"_redaction": "[REDACTION_FAILED]"}})
                    for tc in step.tool_calls
                ]
            if step.observation is not None:
                safe_updates["observation"] = step.observation.model_copy(
                    update={
                        "results": [
                            r.model_copy(update={"content": "[REDACTION_FAILED]"})
                            for r in step.observation.results
                        ]
                    }
                )
            return step.model_copy(update=safe_updates)


# Recorder propagation uses a ContextVar (not a module-level dataclass, per
# PROJECT.md "propagated via ContextVar (not AgentState)"). Access via
# get_current_recorder() ONLY; never import _RECORDER_VAR directly. Test isolation
# goes through _reset_recorder_for_tests() (CORE-10 / D-17).
_RECORDER_VAR: ContextVar["TrajectoryRecorder | None"] = ContextVar(
    "_RECORDER_VAR", default=None,
)

# Signal-handler-safe stack of active recorders (root + forks). Python signal
# handlers fire in the main thread at bytecode boundaries — ContextVar.get()
# from that handler returns whatever context the interpreter happened to be
# in, which is non-deterministic relative to async tasks. The signal-handler
# path reads the top of this stack instead so SIGINT-flush is reliable.
_ACTIVE_RECORDERS: list["TrajectoryRecorder"] = []


def get_current_recorder() -> "TrajectoryRecorder | None":
    """Return the recorder for the current async context, or None if none active.

    The single public accessor for ``_RECORDER_VAR`` (D-10). ``agent.py`` reads
    this at the top of ``run_agent()`` and skips the entire Invocation lifecycle
    when None — direct test invocation of ``run_agent()`` without an active
    recorder is therefore a clean no-op (CORE-09).

    Signal handlers MUST use :func:`get_signal_recorder` instead — ContextVar
    reads inside a signal handler are not deterministic with respect to the
    async context where the recorder was set.

    Returns:
        The active ``TrajectoryRecorder`` instance, or ``None`` if no
        ``async with TrajectoryRecorder(...)`` block is on the stack.
    """
    return _RECORDER_VAR.get()


def get_signal_recorder() -> "TrajectoryRecorder | None":
    """Return the most recently entered recorder for signal-handler use.

    Signal handlers run in the main thread outside the asyncio task context,
    so ``ContextVar.get()`` returns non-deterministic values depending on
    where the interpreter was when the signal fired. This accessor reads
    from a module-level stack populated by ``TrajectoryRecorder.__aenter__``,
    which is set synchronously and remains valid across the entire run.

    Returns:
        The most recently entered (top-of-stack) ``TrajectoryRecorder``, or
        ``None`` if no recorder is active. For nested forks, the innermost
        recorder is returned — partial flushes cascade to ancestors via
        each recorder's own ``write_partial``.
    """
    return _ACTIVE_RECORDERS[-1] if _ACTIVE_RECORDERS else None


def _reset_recorder_for_tests() -> None:
    """Test-only: clear the recorder ContextVar and signal-handler stack.

    Use exclusively from the autouse ``_reset_trajectory_recorder`` fixture
    in ``tests/conftest.py`` (CORE-10, D-17). Production code MUST go through
    ``TrajectoryRecorder.__aenter__`` / ``__aexit__``.
    """
    _RECORDER_VAR.set(None)
    _ACTIVE_RECORDERS.clear()


@dataclass
class Invocation:
    """Per-``run_agent()`` recording scope; closes one Step per assistant turn.

    Owns the Step buffer for one model conversation and the in-flight
    ``tool_call_id -> host-step`` map (CORE-06). ``parent`` linkage lives on
    ``TrajectoryRecorder`` (Phase 3, D-02), not on ``Invocation``.

    Each ``TurnEndEvent`` from the backend closes the open Step so multi-turn
    invocations produce N Steps (not a single collapsed Step). ``ResultEvent``
    also closes the open Step at the end of the invocation; ``finish()``
    performs a final idempotent close so partial turns are not dropped.

    Tool-result-after-close: a ``ToolStartEvent`` followed by a
    ``TurnEndEvent`` and then a ``ToolResultEvent`` is legal — the result
    lands on the closed Step (the one that hosts the ``ToolStartEvent``) via
    ``model_copy`` so the observation stays attached to its originating turn.

    Attributes:
        recorder: Owning TrajectoryRecorder (shares step_id counter, Redactor).
        phase: DaydreamPhase label stamped on every Step (MAP-08, D-05).
        steps: Steps accumulated; flushed to ``recorder.steps`` at scope exit.
        _open_step_dict: In-progress agent-step state before flush.
        _in_flight_tools: tool_call_id -> ``{open_dict, closed_index}`` entry.
            While the host Step is open, ``open_dict`` is the live in-progress
            dict and ``closed_index`` is None; once closed, ``open_dict`` is
            None and ``closed_index`` is the index in ``self.steps`` of the
            closed Step. ToolResultEvents route to whichever is set.
    """

    recorder: "TrajectoryRecorder"
    phase: DaydreamPhase
    steps: list[Step] = field(default_factory=list)
    _open_step_dict: dict[str, Any] | None = None
    _in_flight_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    _stop_reason: str | None = None

    def observe_user_step(self, prompt: str) -> None:
        """Append a user Step at invocation start (MAP-01, Pitfall 4).

        Constructs a minimal user Step — only step_id / timestamp / source /
        message / extra. NO agent-only fields (model_name, tool_calls,
        metrics, reasoning_content) so Step.validate_agent_only_fields
        passes.
        """
        try:
            self._close_open_step()
            user_step = Step(
                step_id=self.recorder._next_step_id(),
                timestamp=now_iso(),
                source="user",
                message=prompt,
                extra={
                    "daydream_phase": self.phase.value,
                    "daydream_run_flow": self.recorder.run_flow.value,
                },
            )
            self.steps.append(self.recorder.redactor.redact_step(user_step))
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def mark_aborted(self, reason: str) -> None:
        """Record that this invocation was aborted (e.g. budget exceeded).

        The reason is stamped onto the closing Step's ``extra["stop_reason"]``
        when the open step is finalized (mirrors the ``extra["partial_step"]``
        mechanism). ATIF's Step model has no dedicated status field, so the
        ``extra`` dict is the established extension point.

        If the budget fires before any event is received, no step is open yet,
        so we open one here to ensure ``_close_open_step`` (called from
        ``finish()``) has a Step to stamp the reason onto.
        """
        self._stop_reason = reason
        self._ensure_open_step()

    def observe(self, event: "AgentEvent") -> None:
        """Dispatch an AgentEvent into the active Step buffer.

        Catch-and-degrade boundary (Architecture Q7): exceptions are caught
        here so trajectory recording NEVER crashes the user's review/fix
        run. The catch is local to this method — agent.py's event loop
        continues to surface its own errors.
        """
        try:
            self._dispatch(event)
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def _dispatch(self, event: Any) -> None:
        # Function-local imports avoid load-order cycles with daydream.backends.
        from daydream.backends import (
            CostEvent,
            MetricsEvent,
            ResultEvent,
            TextEvent,
            ThinkingEvent,
            ToolResultEvent,
            ToolStartEvent,
            TurnEndEvent,
        )

        if isinstance(event, TextEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_text_chunks"].append(event.text)
        elif isinstance(event, ThinkingEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_thinking_chunks"].append(event.text)
        elif isinstance(event, ToolStartEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_tool_calls"].append(
                ToolCall(tool_call_id=event.id, function_name=event.name, arguments=event.input or {})
            )
            # Map tool_call_id -> THIS open step so paired ToolResultEvent lands
            # on the SAME step (CORE-06, Pitfall 3). The closed_index slot is
            # filled in by _close_open_step() if the host Step closes before
            # the matching ToolResultEvent arrives.
            self._in_flight_tools[event.id] = {
                "open_dict": self._open_step_dict,
                "closed_index": None,
            }
        elif isinstance(event, ToolResultEvent):
            host = self._in_flight_tools.pop(event.id, None)
            if host is None:
                # Dangling ToolResultEvent (Codex pending-id miss, Pitfall 3).
                # Mark via extra.unmatched_tool_results; do NOT emit a dangling
                # source_call_id reference (Trajectory validator hard-fail).
                self._ensure_open_step()
                assert self._open_step_dict is not None
                self._open_step_dict["_unmatched_tool_results"].append(event.id)
                return
            open_dict = host["open_dict"]
            if open_dict is not None:
                # Host Step is still open — append to its observation buffer.
                open_dict["_observation_results"].append(
                    ObservationResult(source_call_id=event.id, content=event.output)
                )
            else:
                # Host Step was closed by an intervening TurnEndEvent. Patch the
                # closed Step in-place via model_copy so the observation stays
                # bound to its originating turn.
                self._amend_closed_step_observation(
                    closed_index=host["closed_index"],
                    result=ObservationResult(source_call_id=event.id, content=event.output),
                )
        elif isinstance(event, MetricsEvent):
            # EVNT-02 attribute names verbatim (D-15: cached_tokens is a
            # SUBSET of prompt_tokens, not added).
            #
            # D-04 correlation fallback (Codex): Codex emits no per-message
            # id, so MetricsEvent.message_id is always '' on the Codex path.
            # In the common Codex case a TurnEndEvent closes the content Step
            # before turn.completed fires, so this MetricsEvent arrives with
            # no open Step and the ``target is None`` branch below opens a
            # fresh Step to hold the metrics. Correlation is therefore
            # TURN-granular for Codex — one MetricsEvent per turn.completed →
            # one metrics-bearing Step per turn — which is coarser than
            # Claude's per-message correlation via message_id. This is the
            # documented, tested fallback for the missing id surface (see
            # tests/contract/test_backend_codex_trajectory.py); it is not a
            # silent coarsening.
            target = self._open_step_dict
            if target is None:
                self._ensure_open_step()
                target = self._open_step_dict
            assert target is not None
            target["_metrics"] = Metrics(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
            )
            if event.model_name:
                target["_model_name"] = event.model_name
                self.recorder._upgrade_model_name(event.model_name)
            # Aggregate into recorder-level totals for FinalMetrics (MAP-07).
            self.recorder._accumulate_metrics(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
            )
        elif isinstance(event, CostEvent):
            # End-of-call signal — also fold per-step metrics onto the open
            # Step so the renderer's per-step rollup sees real cost / tokens
            # (Bug C: previously CostEvent only updated _final_totals).
            self._ensure_open_step()
            assert self._open_step_dict is not None
            target = self._open_step_dict
            existing = target["_metrics"]
            if existing is None:
                target["_metrics"] = Metrics(
                    prompt_tokens=event.input_tokens,
                    completion_tokens=event.output_tokens,
                    cached_tokens=event.cached_tokens,
                    cost_usd=event.cost_usd,
                )
            else:
                # MetricsEvent already populated this step. Prefer the
                # existing token counts but backfill cost_usd if it wasn't
                # surfaced per-message.
                if existing.cost_usd is None and event.cost_usd is not None:
                    target["_metrics"] = existing.model_copy(
                        update={"cost_usd": event.cost_usd}
                    )
            if event.model_name:
                target["_model_name"] = event.model_name
                self.recorder._upgrade_model_name(event.model_name)
            # Aggregate into recorder-level totals so FinalMetrics reflects
            # per-step cost_usd from the backend.
            self.recorder._accumulate_metrics(
                prompt_tokens=event.input_tokens,
                completion_tokens=event.output_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
            )
        elif isinstance(event, ResultEvent):
            self._close_open_step()
        elif isinstance(event, TurnEndEvent):
            # Per-turn close: a TurnEndEvent arriving while no Step is open is
            # a no-op (never invent an empty Step just to close it).
            if self._open_step_dict is not None:
                self._close_open_step()

    def _ensure_open_step(self) -> None:
        """Open a new agent step if none currently in flight."""
        if self._open_step_dict is not None:
            return
        self._open_step_dict = {
            "_text_chunks": [],
            "_thinking_chunks": [],
            "_tool_calls": [],
            "_observation_results": [],
            "_metrics": None,
            "_model_name": self.recorder.agent_model_name,
            "_unmatched_tool_results": [],
        }

    def _close_open_step(self) -> None:
        """Finalize the current open step into a Pydantic Step + redact + append.

        Called per ``TurnEndEvent`` (assistant-turn boundary), per
        ``ResultEvent`` (end-of-call), and once more from ``finish()`` for an
        idempotent final flush. After appending, any ``_in_flight_tools``
        entry whose host was the just-closed dict is amended to reference the
        closed Step by its index in ``self.steps`` so a ToolResultEvent
        arriving after the close still lands on the right turn.
        """
        if self._open_step_dict is None:
            return
        d = self._open_step_dict
        self._open_step_dict = None

        message_text = "".join(d["_text_chunks"])
        reasoning = "\n".join(d["_thinking_chunks"]) if d["_thinking_chunks"] else None
        tool_calls = list(d["_tool_calls"]) or None
        observation = (
            Observation(results=list(d["_observation_results"]))
            if d["_observation_results"]
            else None
        )
        extra: dict[str, Any] = {
            "daydream_phase": self.phase.value,
            "daydream_run_flow": self.recorder.run_flow.value,
        }
        if d["_unmatched_tool_results"]:
            extra["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
        if self._stop_reason is not None:
            extra["stop_reason"] = self._stop_reason

        agent_step = Step(
            step_id=self.recorder._next_step_id(),
            timestamp=now_iso(),
            source="agent",
            message=message_text,
            model_name=d["_model_name"],
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            observation=observation,
            metrics=d["_metrics"],
            extra=extra,
        )
        self.steps.append(self.recorder.redactor.redact_step(agent_step))
        closed_index = len(self.steps) - 1
        # Amend in-flight entries whose host Step just closed so a delayed
        # ToolResultEvent can still find its host via closed_index.
        for entry in self._in_flight_tools.values():
            if entry["open_dict"] is d:
                entry["open_dict"] = None
                entry["closed_index"] = closed_index

    def _amend_closed_step_observation(
        self, *, closed_index: int, result: ObservationResult
    ) -> None:
        """Attach *result* to a closed Step via ``model_copy``.

        Used when a ToolResultEvent arrives after a ``TurnEndEvent`` closed
        the host Step. The replacement is redacted again because the new
        ObservationResult content has not yet been run through the redactor.
        """
        existing = self.steps[closed_index]
        if existing.observation is None:
            new_observation = Observation(results=[result])
        else:
            new_observation = existing.observation.model_copy(
                update={"results": [*existing.observation.results, result]}
            )
        updated = existing.model_copy(update={"observation": new_observation})
        self.steps[closed_index] = self.recorder.redactor.redact_step(updated)

    def snapshot_steps(self, *, snapshot_step_id: int | None = None) -> list[Step]:
        """Return steps including a materialized copy of any open step (signal-safe, non-mutating).

        Args:
            snapshot_step_id: Pre-allocated step ID for the partial step. When multiple
                invocations are active, the caller allocates unique IDs to avoid duplicates.
        """
        if self._open_step_dict is None:
            return list(self.steps)
        d = self._open_step_dict
        message_text = "".join(d["_text_chunks"])
        reasoning = "\n".join(d["_thinking_chunks"]) if d["_thinking_chunks"] else None
        tool_calls = list(d["_tool_calls"]) or None
        observation = (
            Observation(results=list(d["_observation_results"]))
            if d["_observation_results"]
            else None
        )
        extra: dict[str, Any] = {
            "daydream_phase": self.phase.value,
            "daydream_run_flow": self.recorder.run_flow.value,
            "partial_step": True,
        }
        if d["_unmatched_tool_results"]:
            extra["unmatched_tool_results"] = list(d["_unmatched_tool_results"])
        step_id = snapshot_step_id if snapshot_step_id is not None else self.recorder._step_id_counter + 1
        partial_step = Step(
            step_id=step_id,
            timestamp=now_iso(),
            source="agent",
            message=message_text,
            model_name=d["_model_name"],
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            observation=observation,
            metrics=d["_metrics"],
            extra=extra,
        )
        return [*self.steps, self.recorder.redactor.redact_step(partial_step)]

    def finish(self) -> None:
        """Close any open step and flush all steps to the parent recorder."""
        self._close_open_step()
        self.recorder._extend_steps(self.steps)


@dataclass
class TrajectoryRecorder:
    """Owns the per-run ATIF Trajectory and writes it to disk on clean exit.

    Phase 2 surface: ONE recorder per run, opened via ``async with`` from
    ``runner.py``. ``__aenter__`` sets ``_RECORDER_VAR``; ``__aexit__`` writes
    the trajectory and clears the ContextVar. Disk-write failure degrades to
    ``print_warning`` per D-11 (Phase 4 adds the explicit fail-loud branch).

    Attributes:
        path: Output JSON path; default ``<target>/.daydream/runs/<session_id>/trajectory.json``.
        run_flow: Per-trajectory invariant (D-07) stamped on every Step.
        target_dir: Repo/target directory; recorded into Trajectory.extra.
        agent_model_name: Active model name; stamped into Agent and every
            agent Step's model_name.
        redactor: No-op in Phase 2 (D-12); Phase 4 fills in rule list.
        session_id: UUID4 for this run, supplied by the caller (CORE-07).
        steps: Sequential Steps from every Invocation, step_id 1..N.
        pr_number: GitHub PR number if reviewing a PR. Stored in trajectory extra.
        pr_repo: GitHub repo (``owner/repo``) if reviewing a PR. Stored in trajectory extra.
        _step_id_counter: Monotonic; never decreases (Pitfall 1).
        _final_totals: Running tally for FinalMetrics aggregation (MAP-07).
        _previous_token: ContextVar reset token; used by __aexit__ to restore.
    """

    path: Path
    run_flow: DaydreamRunFlow
    target_dir: Path
    agent_model_name: str
    session_id: str
    redactor: Redactor = field(default_factory=Redactor)
    steps: list[Step] = field(default_factory=list)
    parent: TrajectoryRecorder | None = None
    descriptor: str = ""
    explicit_path: bool = False
    pr_number: int | None = None
    pr_repo: str | None = None
    _step_id_counter: int = 0
    _final_totals: dict[str, Any] = field(default_factory=lambda: _INITIAL_TOTALS.copy())
    _previous_token: Any = None
    _registered_siblings: list[tuple[Path, str]] = field(default_factory=list)
    # Active invocations whose in-flight steps haven't been flushed yet.
    # write_partial reads this so SIGINT mid-run_agent() captures partial
    # work rather than dropping it.
    _active_invocations: list[Invocation] = field(default_factory=list)
    on_write: Callable[[TrajectoryRecorder, str], None] | None = None

    async def __aenter__(self) -> "TrajectoryRecorder":
        self._previous_token = _RECORDER_VAR.set(self)
        _ACTIVE_RECORDERS.append(self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self._write()
        except Exception as exc:  # noqa: BLE001 - branch on explicit_path per D-06
            if self.explicit_path:
                # D-06: user asked for it, deliver or fail loud
                print_error(
                    _console,
                    "Trajectory write failed",
                    f"{type(exc).__name__}: {exc}",
                )
                raise SystemExit(2) from exc
            # Implicit/default path — degrade with warning per CORE-09 / D-11
            print_warning(
                _console,
                f"Trajectory write failed: {type(exc).__name__}: {exc}",
            )
        finally:
            if self._previous_token is not None:
                _RECORDER_VAR.reset(self._previous_token)
                self._previous_token = None
            try:
                _ACTIVE_RECORDERS.remove(self)
            except ValueError:
                pass  # already removed by reset_recorder_for_tests or never registered

    def invocation(self, *, phase: DaydreamPhase) -> "_InvocationCM":
        """Open an Invocation scope for one ``run_agent()`` call.

        Returns an async-context-manager that flushes its accumulated Steps
        to ``self.steps`` on exit. Phase 2 has no parent linkage — flat
        sequential append per D-08.
        """
        return _InvocationCM(self, phase)

    def current_phase(self) -> DaydreamPhase | None:
        """Return the firing :class:`DaydreamPhase`, or None if no invocation is active.

        The public read-seam for the phase of the innermost open Invocation,
        complementing :func:`get_current_recorder`. The replay harness reads this
        during ``execute()`` iteration to serve the right per-phase fixture: by
        the time a backend's first event is pulled, ``agent.py`` has already
        opened ``recorder.invocation(phase=...)`` around the stream, so the
        active phase is observable here.

        Returns:
            The ``.phase`` of the last (innermost) active Invocation, or
            ``None`` when ``self._active_invocations`` is empty — the documented,
            correct default for the direct-call no-op path (no active invocation),
            mirroring :func:`get_current_recorder`.
        """
        return self._active_invocations[-1].phase if self._active_invocations else None

    def _next_step_id(self) -> int:
        self._step_id_counter += 1
        return self._step_id_counter

    def _extend_steps(self, steps: list[Step]) -> None:
        self.steps.extend(steps)

    def _upgrade_model_name(self, candidate: str) -> None:
        """Promote *candidate* over a generic backend label.

        Runner stamps the recorder with a generic alias (``"claude"`` /
        ``"codex"``) or empty string at init since the real SDK model id is
        only known after the first agent turn streams back. The first real
        model id observed from MetricsEvent / CostEvent upgrades the
        recorder's ``agent_model_name`` so the rendered Trajectory.agent
        carries the real id rather than the alias.
        """
        if not candidate:
            return
        current = self.agent_model_name or ""
        if current and current not in _GENERIC_MODEL_LABELS and current == candidate:
            return
        if current in _GENERIC_MODEL_LABELS or not current:
            self.agent_model_name = candidate

    def _accumulate_metrics(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        if prompt_tokens is not None:
            self._final_totals["prompt"] += prompt_tokens
        if completion_tokens is not None:
            self._final_totals["completion"] += completion_tokens
        if cached_tokens is not None:
            self._final_totals["cached"] += cached_tokens
        if cost_usd is not None:
            self._final_totals["cost"] += cost_usd
            self._final_totals["any_cost_seen"] = True

    def compute_wall_clock_seconds(self) -> float | None:
        """Total wall-clock seconds spanned by recorded step timestamps.

        Derived from the earliest and latest ``Step.timestamp`` across the
        recorder's steps. Returns ``None`` when fewer than two timestamped
        steps exist (no measurable span).

        Independent of ``--eval``: this mirrors the timestamp-span derivation
        in :func:`daydream.eval.analyzer.analyze_timing`, but reads in-memory
        steps so every archived run captures duration without the deterministic
        evaluation pass. Fork-only steps live in sibling recorders and are not
        included here; the main flow's span bounds them because forks are
        dispatched and merged within it.

        Returns:
            Rounded duration in seconds, or ``None`` when unmeasurable —
            fewer than two timestamped steps, or an unparseable timestamp.
        """
        try:
            timestamps = [parse_iso_timestamp(s.timestamp) for s in self.steps if s.timestamp]
        except ValueError:
            return None
        if len(timestamps) < 2:
            return None
        return round((max(timestamps) - min(timestamps)).total_seconds(), 1)

    def _sibling_path_for(self, descriptor: str) -> Path:
        """Return the sibling trajectory file path for *descriptor*.

        Layout: ``<target>/.daydream/runs/<session_id>/trajectories/<slug>.json``.
        Sibling files live under the same per-run directory as the parent
        trajectory, so every fork in the run dir belongs to this run by
        construction (no prefix filtering required).
        """
        slug = _safe_descriptor(descriptor)
        return (
            self.target_dir
            / _DAYDREAM_DIRNAME
            / _RUNS_SUBDIR
            / self.session_id
            / _TRAJECTORIES_SUBDIR
            / f"{slug}.json"
        )

    def fork(self, descriptor: str) -> "_ForkCM":
        """Create a child recorder for a parallel task group.

        Args:
            descriptor: Semantic label for the sibling (e.g. ``"fix-0"``).

        Returns:
            An async context manager yielding a child ``TrajectoryRecorder``.
        """
        return _ForkCM(parent=self, descriptor=descriptor)

    def _register_sibling(self, path: Path, descriptor: str) -> None:
        """Register a completed sibling trajectory (synchronous, no await)."""
        self._registered_siblings.append((path, descriptor))

    def create_dispatch_step(self, *, phase: DaydreamPhase) -> None:
        """Create an agent Step referencing all registered sibling trajectories.

        No-op when ``_registered_siblings`` is empty.
        """
        if not self._registered_siblings:
            return
        results: list[ObservationResult] = []
        for sibling_path, desc in self._registered_siblings:
            try:
                rel = str(sibling_path.relative_to(self.target_dir / ".daydream"))
            except ValueError:
                rel = sibling_path.name
            results.append(
                ObservationResult(
                    content=f"Dispatched to {desc}",
                    subagent_trajectory_ref=[
                        SubagentTrajectoryRef(session_id=self.session_id, trajectory_path=rel),
                    ],
                )
            )
        count = len(self._registered_siblings)
        step = Step(
            step_id=self._next_step_id(),
            timestamp=now_iso(),
            source="agent",
            model_name=self.agent_model_name,
            message=f"Dispatching {count} parallel {phase.value} tasks",
            observation=Observation(results=results),
            extra={
                "daydream_phase": phase.value,
                "daydream_run_flow": self.run_flow.value,
            },
        )
        self.steps.append(self.redactor.redact_step(step))
        self._registered_siblings.clear()

    def build_trajectory(self, steps: list[Step] | None = None) -> Trajectory:
        if steps is None:
            steps = self.steps
        version = daydream.__version__

        final_metrics = FinalMetrics(
            total_prompt_tokens=self._final_totals["prompt"] or None,
            total_completion_tokens=self._final_totals["completion"] or None,
            total_cached_tokens=self._final_totals["cached"] or None,
            total_cost_usd=(
                self._final_totals["cost"] if self._final_totals["any_cost_seen"] else None
            ),
            total_steps=len(steps),
        )
        extra: dict[str, Any] = {"target_dir": str(self.target_dir)}
        if self.pr_number is not None:
            extra["pr_number"] = self.pr_number
        if self.pr_repo is not None:
            extra["pr_repo"] = self.pr_repo
        return Trajectory(
            schema_version="ATIF-v1.6",
            session_id=self.session_id,
            agent=Agent(name="daydream", version=version, model_name=self.agent_model_name),
            steps=list(steps),
            final_metrics=final_metrics,
            extra=extra,
        )

    def _write(self) -> None:
        # Empty trajectory: skip — Pydantic Trajectory.steps has min_length=1.
        # Phase 4 may revisit if empty runs need a stub file on disk.
        if not self.steps:
            return
        trajectory = self.build_trajectory()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(trajectory.to_json_dict(), indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            if self.on_write is not None:
                try:
                    self.on_write(self, "complete")
                except Exception:  # noqa: BLE001 - archive failure must never affect the run
                    pass
        except BaseException:
            with suppress(OSError):
                os.unlink(tmp)
            raise

    def _snapshot_in_flight_steps(self) -> list[Step]:
        """Concatenate flushed steps with steps from any active invocations.

        Mid-``run_agent()``, an Invocation has accumulated steps that haven't
        been flushed back to ``self.steps`` (the flush happens in
        ``_InvocationCM.__aexit__`` via ``Invocation.finish``). For a partial
        flush we want those in-flight steps too; this helper concatenates
        them in registration order without mutating either buffer.
        """
        if not self._active_invocations:
            return list(self.steps)
        snapshot = list(self.steps)
        next_id = self._step_id_counter + 1
        for inv in self._active_invocations:
            snapshot.extend(inv.snapshot_steps(snapshot_step_id=next_id))
            if inv._open_step_dict is not None:
                next_id += 1
        return snapshot

    def write_partial(self) -> None:
        """SIGINT/SIGTERM flush path — write in-flight steps to ``<path>.partial``.

        Per D-07 the partial trajectory lives at a sibling path with the
        ``.partial`` suffix appended to the full filename (e.g.
        ``trajectory.json.partial``). The Trajectory's ``extra`` dict carries
        ``partial=true`` so consumers can detect incomplete runs without
        path-string parsing. Steps from any in-flight Invocation are
        included so SIGINT mid-``run_agent()`` does not lose work; empty
        trajectories are skipped (matches ``_write``).

        Idempotent: callable from a signal handler synchronously without
        awaiting ``__aexit__``; safe to invoke from outside the async context.
        Disk-write failures degrade with a warning per D-11 — partial flush
        must never crash shutdown.
        """
        snapshot_steps = self._snapshot_in_flight_steps()
        if not snapshot_steps:
            return
        try:
            trajectory = self.build_trajectory(steps=snapshot_steps)
            partial_path = self.path.with_suffix(self.path.suffix + ".partial")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            json_dict = trajectory.to_json_dict()
            extra = json_dict.setdefault("extra", {})
            extra["partial"] = True
            partial_path.write_text(json.dumps(json_dict, indent=2), encoding="utf-8")
            if self.on_write is not None:
                try:
                    self.on_write(self, "partial")
                except Exception:  # noqa: BLE001 - archive failure must never crash shutdown
                    pass
            if self.parent is not None:
                self.parent.write_partial()
        except Exception as exc:  # noqa: BLE001 - partial flush must never crash shutdown
            print_warning(
                _console, f"Partial trajectory write failed: {type(exc).__name__}: {exc}"
            )


class _ForkCM:
    """Async context manager for forking a child recorder (D-01, D-02, D-03)."""

    def __init__(self, parent: TrajectoryRecorder, descriptor: str) -> None:
        self._parent = parent
        self._descriptor = descriptor
        self._child: TrajectoryRecorder | None = None

    async def __aenter__(self) -> TrajectoryRecorder:
        child = TrajectoryRecorder(
            path=self._parent._sibling_path_for(self._descriptor),
            run_flow=self._parent.run_flow,
            target_dir=self._parent.target_dir,
            agent_model_name=self._parent.agent_model_name,
            redactor=self._parent.redactor,
            session_id=self._parent.session_id,
            pr_number=self._parent.pr_number,
            pr_repo=self._parent.pr_repo,
        )
        child.parent = self._parent
        child.descriptor = self._descriptor
        child._previous_token = _RECORDER_VAR.set(child)
        _ACTIVE_RECORDERS.append(child)
        self._child = child
        return child

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        child = self._child
        if child is None:
            return
        write_ok = False
        try:
            child._write()
            write_ok = bool(child.steps)
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run
            print_warning(_console, f"Sibling trajectory write failed: {type(exc).__name__}: {exc}")
        finally:
            if child._previous_token is not None:
                _RECORDER_VAR.reset(child._previous_token)
                child._previous_token = None
        try:
            _ACTIVE_RECORDERS.remove(child)
        except ValueError:
            pass
        if write_ok and child.parent is not None:
            child.parent._register_sibling(child.path, self._descriptor)


class _InvocationCM:
    """Async context manager wrapping an Invocation (internal helper)."""

    def __init__(self, recorder: TrajectoryRecorder, phase: DaydreamPhase) -> None:
        self._recorder = recorder
        self._phase = phase
        self._invocation: Invocation | None = None

    async def __aenter__(self) -> Invocation:
        self._invocation = Invocation(recorder=self._recorder, phase=self._phase)
        # Register with recorder so write_partial can capture in-flight steps
        # if SIGINT fires mid-invocation.
        self._recorder._active_invocations.append(self._invocation)
        return self._invocation

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._invocation is not None:
            try:
                self._invocation.finish()
            finally:
                with suppress(ValueError):
                    self._recorder._active_invocations.remove(self._invocation)
                self._invocation = None
