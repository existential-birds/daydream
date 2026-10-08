"""Codex CLI subprocess backend for daydream.

Spawns `codex exec --experimental-json` as an async subprocess,
writes the prompt to stdin, and reads JSONL events from stdout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import shutil
import subprocess as subprocess
import sys as sys
import tempfile
import threading
import uuid
from collections import Counter
from collections.abc import AsyncGenerator, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from daydream import git_ops
from daydream.backends import (
    AgentEvent,
    BackendExecutionInput,
    CodexRequestConfig,
    ContinuationToken,
    CostEvent,
    DiagnosticEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    resolve_fanout_concurrency,
)
from daydream.backends._subprocess import stream_idle_timeout_s
from daydream.backends._transport import (
    CliTransport,
    StderrPolicy,
    StdinMode,
    process_exit_message,
    raise_for_exit,
    reap,
    teardown,
    write_temp_json_schema,
)
from daydream.pricing import ModelPrice, compute_cost_from_totals, load_user_prices, resolve_prices
from daydream.trajectory import redact_structured_text

_CODEX_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024
_DIAGNOSTIC_LABEL_MAX_CHARS = 64
_DIAGNOSTIC_LABEL_MAX_DISTINCT = 32
_NON_JSON_EXCERPT_MAX_LINES = 20
_NON_JSON_EXCERPT_MAX_CHARS_PER_LINE = 256
# Only public error items activate this contract; absent markers cannot
# justify invented tool events or invocation-wide coverage claims.
# External contract: this literal identifies Codex CLI release/wire mode.
_TRANSPORT_COVERAGE_CONTRACT = "codex-cli-0.153.4-json-code-mode"

_logger = logging.getLogger(__name__)


def _prepare_read_only_checkout(source: Path, destination: Path) -> Path:
    """Build the shared standalone snapshot, including nonignored untracked files.

    Generic Codex read-only behavior remains path-hiding, not an audit sandbox.
    """
    return git_ops.prepare_independent_snapshot(
        source, destination, include_untracked=True,
    ).repo


# Child-environment variables whose value would give an isolated codex
# subprocess a handle on the caller's repo: the inherited ``$PWD``/``$OLDPWD``
# and the ``GIT_*`` redirection vars that could point the clone's git ops back
# at the source's refs/index/worktree.
_GIT_REDIRECT_STRIP_VARS = (
    "PWD",
    "OLDPWD",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_COMMON_DIR",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_PREFIX",
)

# Cached at-most-once per process, including negative resolution results.
_REAL_GIT_DIR: str | None = None
_REAL_GIT_RESOLVED = False
# to_thread fan-out can race first resolution. Hold the lock through the
# probe: the resolved flag is set before the cached directory is assigned.
_REAL_GIT_RESOLUTION_LOCK = threading.Lock()

# Bound for the parent-side xcrun resolution, mirroring ``git_process._run_git``'s
# default timeout (5s): a hung ``xcrun`` must fail open, never wedge the process.
_XCRUN_TIMEOUT_S = 5


def _probe_real_git_dir(environment: Mapping[str, str] | None) -> str | None:
    """Run and validate one bounded ``xcrun --find git`` probe.

    ``None`` preserves ordinary subprocess inheritance. An explicit mapping is
    complete and is passed through without merging process globals.
    """
    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": _XCRUN_TIMEOUT_S,
    }
    if environment is not None:
        kwargs["env"] = dict(environment)
    try:
        proc = subprocess.run(["xcrun", "--find", "git"], **kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        _logger.warning(
            "codex: could not resolve real git via 'xcrun --find git' (%s); leaving child PATH unchanged",
            exc,
        )
        return None
    if proc.returncode != 0:
        _logger.warning(
            "codex: could not resolve real git via 'xcrun --find git' (exit %s: %s); "
            "leaving child PATH unchanged",
            proc.returncode,
            proc.stderr.strip(),
        )
        return None
    git_path = proc.stdout.strip()
    if not git_path or not Path(git_path).is_file() or not os.access(git_path, os.X_OK):
        _logger.warning(
            "codex: could not resolve real git via 'xcrun --find git' (invalid target %r); "
            "leaving child PATH unchanged",
            git_path,
        )
        return None
    return str(Path(git_path).parent)


def _resolve_real_git_dir() -> str | None:
    """Cache ambient macOS real-git resolution, including misses, once per process.

    Explicit execution environments probe directly without consulting this cache.
    """
    global _REAL_GIT_DIR, _REAL_GIT_RESOLVED
    with _REAL_GIT_RESOLUTION_LOCK:
        if _REAL_GIT_RESOLVED:
            return _REAL_GIT_DIR
        _REAL_GIT_RESOLVED = True
        if sys.platform != "darwin":
            return None
        _REAL_GIT_DIR = _probe_real_git_dir(None)
        return _REAL_GIT_DIR


class _SharedCheckout:
    """Reference-count one disposable checkout per concurrent (backend, cwd) group.

    Delete it when the last generator exits. Sequential calls rebuild for freshness.
    """

    __slots__ = ("cwd", "path", "temp_dir", "refs")

    def __init__(self, cwd: Path, path: Path, temp_dir: tempfile.TemporaryDirectory[str]) -> None:
        self.cwd = cwd
        self.path = path
        self.temp_dir = temp_dir
        self.refs = 0


def _isolated_child_env(
    cwd: Path,
    execution_cwd: Path,
    *,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """Build the child environment without paths back to an isolated source repo.

    For clones, remove PWD/OLDPWD and Git redirects. On Darwin prepend real git
    to PATH to bypass the sandbox-incompatible xcrun shim; probe failure leaves
    PATH unchanged. Without isolation, return an explicit environment intact or
    None for ambient inheritance. Independently discovered source paths remain
    accessible; this is path hiding, not filesystem confinement.
    """
    if execution_cwd == cwd:
        return dict(base_environment) if base_environment is not None else None
    child_env = (
        dict(base_environment) if base_environment is not None else os.environ.copy()
    )
    for var in _GIT_REDIRECT_STRIP_VARS:
        child_env.pop(var, None)
    # Explicit environments probe uncached; ordinary callers use the cache.
    if sys.platform == "darwin":
        real_git_dir = (
            _probe_real_git_dir(child_env)
            if base_environment is not None
            else _resolve_real_git_dir()
        )
        if real_git_dir and "PATH" in child_env:
            child_env["PATH"] = real_git_dir + os.pathsep + child_env["PATH"]
    return child_env


def _resolved_prices_for_execution(
    execution_input: BackendExecutionInput | None,
) -> dict[str, ModelPrice]:
    """Load prices from the run environment, or ambient overrides for direct callers.

    An injected environment lacking both a prices path and HOME uses built-ins.
    """
    if execution_input is None:
        return resolve_prices(load_user_prices())

    environment = execution_input.child_environment()
    prices_path = environment.get("DAYDREAM_PRICES_FILE")
    if prices_path:
        overrides = load_user_prices(Path(prices_path))
    elif home := environment.get("HOME"):
        overrides = load_user_prices(Path(home) / ".daydream" / "prices.toml")
    else:
        overrides = {}
    return resolve_prices(overrides)


def _rebind_source_paths(prompt: str, source: Path, execution: Path) -> str:
    """Replace source paths in prompts, including resolved and doubled-slash forms.

    Match path boundaries so sibling prefixes such as work-2/workspace/work.py
    remain unchanged; no recognized source spelling may reach isolated stdin.
    """
    for candidate in {str(source), str(source.resolve())}:
        pattern = re.escape(candidate).replace("/", "/+") + r"(?![\w.-])"
        prompt = re.sub(pattern, str(execution), prompt)
    return prompt


_SHELL_LC_PREFIX_RE = re.compile(r"^/bin/(?:zsh|bash|sh)\s+-lc\s+")


def _unwrap_shell_command(command: str) -> str:
    """Decode /bin/{zsh,bash,sh} -lc payloads without changing replayable command bytes.

    A single shlex payload is returned verbatim, including leading cd. For bare
    multi-word payloads, preserve the raw remainder after -lc so embedded quotes
    survive. Unknown wrappers, missing payloads, malformed quoting, and quoted
    payloads followed by extra arguments return the original input unchanged.
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return command
    if len(argv) >= 2 and argv[0] in ("/bin/zsh", "/bin/bash", "/bin/sh") and argv[1] == "-lc":
        if len(argv) == 3:
            return argv[2]
        # More than one word after '-lc': a shell-quoted payload with trailing
        # argv is not a valid wrapper (fail open), but a bare, unquoted payload
        # is the real-Codex shape for simple commands ('/bin/zsh -lc ls -la').
        # Recover it from the raw command so embedded quoting is preserved.
        wrapper = _SHELL_LC_PREFIX_RE.match(command)
        if wrapper is not None:
            payload = command[wrapper.end() :]
            if payload and payload[0] not in ("'", '"'):
                return payload
    return command


_CD_PREFIX_RE = re.compile(r"^cd\s+\S+\s*&&\s*")


def display_shell_command(command: str) -> str:
    """Decode a shell wrapper and strip leading cd <dir> && for display.

    Stored tool inputs use _unwrap_shell_command instead to preserve replay.
    Malformed wrappers pass through; an unmatched cd prefix stays unchanged.
    """
    decoded = _unwrap_shell_command(command)
    return _CD_PREFIX_RE.sub("", decoded, count=1)


def supervisor_shell_command(command: str) -> str:
    """Decode and strip leading cd for start-anchored supervisor rules such as ^make.

    Never redact or cap: either operation could change what a supervisor matches.
    """
    return display_shell_command(command)


def _bounded_diagnostic_label(value: Any) -> str:
    """Return one redacted, bounded scalar label for diagnostic aggregation."""
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        return "<non-scalar>"
    return redact_structured_text(str(value))[:_DIAGNOSTIC_LABEL_MAX_CHARS]


def _bounded_process_excerpt(value: str) -> str:
    """Redact a complete non-JSON line before applying the exception cap."""
    return redact_structured_text(value)[:_NON_JSON_EXCERPT_MAX_CHARS_PER_LINE]


def _record_unknown(
    counter: Counter[str],
    value: Any,
    *,
    overflow: list[int],
) -> None:
    """Count an unknown label with bounded retained cardinality."""
    label = _bounded_diagnostic_label(value)
    if label in counter:
        counter[label] += 1
    elif len(counter) < _DIAGNOSTIC_LABEL_MAX_DISTINCT:
        counter[label] = 1
    else:
        overflow[0] += 1


def _counter_summary(counter: Counter[str], overflow: list[int]) -> dict[str, Any]:
    return {
        "total": sum(counter.values()) + overflow[0],
        "labels": dict(counter),
        "overflow": overflow[0],
    }


def _parser_diagnostics(
    *,
    error_sentinel_count: int,
    unknown_event_types: Counter[str],
    unknown_event_overflow: list[int],
    unknown_item_types: Counter[str],
    unknown_item_overflow: list[int],
    malformed_shapes: Counter[str],
    non_json_count: int,
    parse_warnings: Counter[str],
) -> list[DiagnosticEvent]:
    """Build deterministic conditional diagnostics from bounded parser state."""
    diagnostics: list[DiagnosticEvent] = []
    if error_sentinel_count:
        diagnostics.append(
            DiagnosticEvent(
                code="codex_transport_coverage",
                message=(
                    "The current Codex public stream contains uncorrelated error items; "
                    "tool coverage is incomplete."
                ),
                metadata={
                    "coverage": "incomplete",
                    "reason": "uncorrelated_public_error_item",
                    "occurrences": error_sentinel_count,
                    "contract": _TRANSPORT_COVERAGE_CONTRACT,
                },
            )
        )
    if (
        unknown_event_types
        or unknown_event_overflow[0]
        or unknown_item_types
        or unknown_item_overflow[0]
        or malformed_shapes
        or non_json_count
        or parse_warnings
    ):
        diagnostics.append(
            DiagnosticEvent(
                code="codex_parser_coverage",
                message="The Codex public stream contained parser coverage gaps.",
                metadata={
                    "unknown_event_types": _counter_summary(
                        unknown_event_types, unknown_event_overflow
                    ),
                    "unknown_item_types": _counter_summary(
                        unknown_item_types, unknown_item_overflow
                    ),
                    "malformed_shapes": dict(malformed_shapes),
                    "non_json_lines": non_json_count,
                    "warnings": {
                        "total": sum(parse_warnings.values()),
                        "reasons": dict(parse_warnings),
                    },
                },
            )
        )
    return diagnostics


class CodexError(Exception):
    """Codex failure: category=None for turn.failed, PROCESS_EXIT for a nonzero CLI exit."""

    def __init__(self, message: str, *, category: str | None = None):
        super().__init__(message)
        self.category = category


def _file_change_events(
    item: dict[str, Any],
    execution_cwd: Path,
) -> Iterator[ToolStartEvent | ToolResultEvent]:
    """Emit a synthetic patch start/result pair; file changes have no native start.

    Preserve legacy scalar inputs for extension tool supervisors. Modern changes
    additionally carry the full path/kind list, with in-checkout absolute paths
    made relative. Missing status means success; pathless input is an observable
    error. Result excerpts are capped at 500 characters per section.
    """
    item_id = item.get("id", str(uuid.uuid4()))
    changes = item.get("changes")
    if isinstance(changes, (dict, list)):
        if isinstance(changes, list):
            changes = {
                str(change.get("path") or change.get("file_path")): change
                for change in changes
                if isinstance(change, dict) and (change.get("path") or change.get("file_path"))
            }
        parsed = []
        for raw_path, entry in changes.items():
            kind = entry.get("type", "unknown") if isinstance(entry, dict) else "unknown"
            path = str(raw_path)
            try:
                if os.path.isabs(path) and os.path.commonpath([path, str(execution_cwd)]) == str(execution_cwd):
                    path = os.path.relpath(path, execution_cwd)
            except ValueError:
                pass  # Keep absolute paths on disjoint drives.
            parsed.append({"path": path, "kind": kind})
        status = item.get("status") or "completed"
        output = ", ".join(f"{change['kind']}: {change['path']}" for change in parsed)[:500]
        if status == "declined":
            output = f"File change declined by sandbox: {output}"
        if status in ("failed", "declined"):
            for stream in ("stdout", "stderr"):
                excerpt = item.get(stream, "")
                if excerpt:
                    output += f"\n{stream}: {excerpt[:500]}"
        start_input: dict[str, Any] = {"changes": parsed, "file": "unknown", "action": "modified"}
        if len(parsed) == 1:
            start_input.update(file=parsed[0]["path"], action=parsed[0]["kind"])
        is_error = status != "completed"
    elif "file_path" in item:
        file_path = item.get("file_path", "unknown")
        action = item.get("action", "modified")
        start_input = {"file": file_path, "action": action}
        output = f"{action}: {file_path}"
        is_error = False
        status = None
    else:
        fields = {key: value for key, value in item.items() if key != "type" and value != "unknown"}
        start_input = {"file_change": fields}
        output = f"unparseable file_change item: {json.dumps(fields)[:500]}"
        is_error = True
        status = item.get("status") or None
    yield ToolStartEvent(
        id=item_id,
        name="patch",
        input=start_input,
    )
    yield ToolResultEvent(
        id=item_id,
        output=output,
        is_error=is_error,
        status=status,
    )


class CodexBackend:
    """Backend that wraps the Codex CLI subprocess.

    Translates Codex JSONL events into the unified AgentEvent stream.
    """

    supports_finalization = True

    # Codex operates in a disposable read-only clone of the workspace, so it
    # can safely have over-budget diffs inlined (truncated) and exploration
    # summaries inlined rather than pointed at on-disk artifact files.
    read_only_disposable_clone = True

    def __init__(
        self,
        model: str,
        reasoning_effort: str | None = None,
        *,
        execution_input: BackendExecutionInput | None = None,
    ):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._execution_input = execution_input
        if execution_input is not None:
            self.fanout_concurrency = execution_input.fanout_concurrency
            self.retry_policy = execution_input.retry_policy
        else:
            self.fanout_concurrency = resolve_fanout_concurrency(
                "DAYDREAM_FANOUT_CONCURRENCY", 8
            )
        self._transports: list[CliTransport] = []
        # Disposable read-only checkouts shared across concurrent execute() calls
        # (built once per cwd, refcounted; cleaned up when the last holder exits).
        self._checkout_lock = asyncio.Lock()
        self._checkouts: dict[Path, _SharedCheckout] = {}

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
        finalization: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield Codex events, rejecting unsupported agents with NotImplementedError.

        read_only adds a sandbox and, at Git worktree roots, a disposable standalone
        clone of HEAD, index, tracked/nonignored files, and refs without a source
        remote. Git metadata mutations therefore affect the clone. Exclude source
        paths from argv/stdin/env/cwd; independently discovered paths remain a
        limitation. Non-root cwd uses the sandbox in place. Clone preparation and
        read-only resumption failures raise CodexError; never fall back to source.
        Keep native session identity but omit continuation for deleted checkouts.
        The default uses danger-full-access in the caller's cwd.

        finalization caps effort at low, preserving lower levels. Codex has no tool
        disable control, so callers retain the host zero-tool guard. persist_session
        is accepted but ignored. Turn failures raise CodexError; stdout silence raises
        retryable StreamStalledError and run_agent retries with a fresh subprocess.
        """
        if agents:
            raise NotImplementedError(
                "Codex backend does not support exploration subagents; use --backend claude for exploration."
            )

        sandbox_mode = "read-only" if read_only else "danger-full-access"

        schema_path: str | None = None
        if output_schema:
            schema_path = write_temp_json_schema(output_schema, prefix="daydream-schema-")

        thread_id: str | None = None
        provider_name: str | None = None
        model_name = self.model
        last_agent_text: str | None = None
        structured_result: Any = None
        structured_output_origin: Literal["native", "text"] = "native"

        # Pair idless starts/completions by FIFO; misses get
        # observable orphan ids. Accumulate item.updated text because completion
        # may contain no text. See _claim_tool_id for the correlation contract.
        pending_fifo: dict[str, list[str]] = {}  # item_type → [ids] in start order
        updated_text: dict[str, list[str]] = {}  # item_id → [text deltas]
        parse_warnings: Counter[str] = Counter()  # persisted bounded reasons
        unknown_event_types: Counter[str] = Counter()
        unknown_event_overflow = [0]
        unknown_item_types: Counter[str] = Counter()
        unknown_item_overflow = [0]
        malformed_shapes: Counter[str] = Counter()
        non_json_count = 0
        error_sentinel_count = 0
        _pending_result: ResultEvent | None = None
        non_json_lines: list[str] = []  # redacted/capped process-exit excerpts
        unmatched_seq = 0  # monotonic source for orphaned tool-result ids

        def _warn(reason: str) -> None:
            """Log and retain only a fixed parser-warning reason."""
            parse_warnings[reason] += 1
            _logger.warning("codex parser warning: %s", reason.replace("_", " "))

        emitted_diagnostic_signatures: dict[str, str] = {}
        early_diagnostic_codes: set[str] = set()

        def _current_diagnostics() -> list[DiagnosticEvent]:
            return _parser_diagnostics(
                error_sentinel_count=error_sentinel_count,
                unknown_event_types=unknown_event_types,
                unknown_event_overflow=unknown_event_overflow,
                unknown_item_types=unknown_item_types,
                unknown_item_overflow=unknown_item_overflow,
                malformed_shapes=malformed_shapes,
                non_json_count=non_json_count,
                parse_warnings=parse_warnings,
            )

        def _diagnostic_signature(event: DiagnosticEvent) -> str:
            return json.dumps(
                {
                    "code": event.code,
                    "message": event.message,
                    "metadata": event.metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
            )

        def _take_early_diagnostics() -> list[DiagnosticEvent]:
            """Emit the first observable marker for each diagnostic code."""
            fresh = [
                event
                for event in _current_diagnostics()
                if event.code not in early_diagnostic_codes
            ]
            for event in fresh:
                early_diagnostic_codes.add(event.code)
                emitted_diagnostic_signatures[event.code] = _diagnostic_signature(event)
            return fresh

        def _take_final_diagnostics() -> list[DiagnosticEvent]:
            """Emit only aggregates that changed since their first marker."""
            changed = []
            for event in _current_diagnostics():
                signature = _diagnostic_signature(event)
                if emitted_diagnostic_signatures.get(event.code) != signature:
                    emitted_diagnostic_signatures[event.code] = signature
                    changed.append(event)
            return changed

        def _claim_tool_id(item_type: str) -> str:
            """Correlate no-id completions with unconsumed starts in FIFO order.

            On a miss, warn and assign a deterministic orphan id for unmatched_tool_results;
            a dangling source_call_id would fail trajectory validation.
            """
            nonlocal unmatched_seq
            fifo = pending_fifo.get(item_type, [])
            item_id = fifo.pop(0) if fifo else None
            if item_id is None:
                _warn("unmatched_tool_result")
                item_id = f"codex-unmatched-{unmatched_seq}"
                unmatched_seq += 1
            return item_id

        transport: CliTransport | None = None
        execution_cwd = cwd
        shared_checkout: _SharedCheckout | None = None

        try:
            if read_only:
                try:
                    is_worktree_root = await asyncio.to_thread(git_ops.is_inside_worktree, cwd)
                except git_ops.GitError as exc:
                    # Wrap the pre-check's git failures too — the documented
                    # CodexError on read-only prep failure covers the full prep.
                    raise CodexError("failed to create disposable read-only checkout") from exc
                if is_worktree_root:
                    if continuation is not None and continuation.backend == "codex":
                        # The thread retains its deleted clone cwd, not the newly rebound
                        # prompt path; refuse resumption rather than silently changing cwd.
                        raise CodexError(
                            "read-only Codex sessions cannot be resumed: the thread's stored "
                            "cwd is a per-call disposable clone deleted when the turn ends"
                        )
                    async with self._checkout_lock:
                        shared_checkout = self._checkouts.get(cwd)
                        if shared_checkout is None:
                            temp_dir = tempfile.TemporaryDirectory(prefix="daydream-codex-read-only-")
                            destination = Path(temp_dir.name) / "repo"
                            try:
                                prepared = await asyncio.to_thread(
                                    _prepare_read_only_checkout, cwd, destination,
                                )
                            except (git_ops.GitError, OSError, shutil.Error) as exc:
                                temp_dir.cleanup()
                                raise CodexError(
                                    "failed to create disposable read-only checkout"
                                ) from exc
                            shared_checkout = _SharedCheckout(cwd, prepared, temp_dir)
                            self._checkouts[cwd] = shared_checkout
                        shared_checkout.refs += 1
                    execution_cwd = shared_checkout.path

            args = [
                "codex",
                "exec",
                "--experimental-json",
                "--model",
                self.model,
                "--sandbox",
                sandbox_mode,
                "--cd",
                str(execution_cwd),
            ]
            effort = self.reasoning_effort
            if finalization:
                effort = effort if effort in {"none", "minimal", "low"} else "low"
            if effort:
                args.extend(["-c", f'model_reasoning_effort="{effort}"'])
            if schema_path:
                args.extend(["--output-schema", schema_path])
            if continuation is not None and continuation.backend == "codex":
                args.extend(["resume", continuation.data["thread_id"]])

            # Built off the event loop (asyncio.to_thread, like the sibling git
            # calls above), so the env copy and bounded xcrun probe never stall
            # concurrent fan-out execute() calls.
            base_environment = (
                self._execution_input.child_environment()
                if self._execution_input is not None
                else None
            )
            child_env = await asyncio.to_thread(
                _isolated_child_env,
                cwd,
                execution_cwd,
                base_environment=base_environment,
            )

            if execution_cwd != cwd:
                # Rebind the prompt so no rendering of the caller's source
                # path appears in the bytes written to the isolated subprocess.
                prompt = _rebind_source_paths(prompt, cwd, execution_cwd)

            # P18 Task 1: closed typed effective-config admission from the
            # exact argv built above. max_turns and persist_session are
            # accepted-but-not-passed on this CLI surface, so they stay
            # None (documented unavailability, never an effective claim).
            codex_resume_applied = (
                continuation is not None and continuation.backend == "codex"
            )
            codex_resume_thread: str | None = None
            if continuation is not None and codex_resume_applied:
                thread_value = continuation.data.get("thread_id")
                codex_resume_thread = thread_value if isinstance(thread_value, str) else None
            yield RequestEvent(
                prompt=prompt, model_name=model_name, output_schema=output_schema,
                reasoning_effort=effort,
                session_id=codex_resume_thread,
                config=CodexRequestConfig(
                    finalization=finalization,
                    sandbox_mode=(
                        "read-only" if read_only else "danger-full-access"
                    ),
                    experimental_json=True,
                    native_output_schema=schema_path is not None,
                    read_only_isolation=read_only and execution_cwd != cwd,
                    continuation_mode="resume" if codex_resume_applied else "fresh",
                    model_mode="single",
                ),
                model_source="configured",
                session_source="configured" if codex_resume_applied else None,
            )

            transport = CliTransport(
                "codex",
                args,
                stdin_mode=StdinMode.PIPE,
                stdin_data=prompt.encode(),
                stderr_policy=StderrPolicy.MERGE_INTO_STDOUT,
                limit=_CODEX_STDOUT_LIMIT_BYTES,
                env=child_env,
                cwd=str(execution_cwd) if execution_cwd != cwd else None,
            )
            self._transports.append(transport)
            await transport.start()

            idle_timeout_s = (
                self._execution_input.stream_idle_timeout_s
                if self._execution_input is not None
                else stream_idle_timeout_s()
            )
            async for raw_line in transport.lines(lambda: idle_timeout_s):
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    # Keep bounded raw lines only for the existing process-exit
                    # exception excerpt. Persistent diagnostics and logs retain
                    # only the count, never the line payload.
                    non_json_count += 1
                    if len(non_json_lines) >= _NON_JSON_EXCERPT_MAX_LINES:
                        non_json_lines.pop(0)
                    non_json_lines.append(_bounded_process_excerpt(raw_line))
                    _logger.debug("codex: non-JSON line skipped")
                    for diagnostic in _take_early_diagnostics():
                        yield diagnostic
                    continue

                if not isinstance(event, dict):
                    malformed_shapes["event_not_object"] += 1
                    for diagnostic in _take_early_diagnostics():
                        yield diagnostic
                    continue
                event_type = event.get("type", "")
                if isinstance(event_type, (dict, list)):
                    malformed_shapes["event_type_not_scalar"] += 1
                    for diagnostic in _take_early_diagnostics():
                        yield diagnostic
                    continue

                if event_type in ("item.started", "item.updated", "item.completed"):
                    item = event.get("item", {})
                    if not isinstance(item, dict):
                        malformed_shapes["item_not_object"] += 1
                        for diagnostic in _take_early_diagnostics():
                            yield diagnostic
                        continue
                    item_type = item.get("type", "")
                    if isinstance(item_type, (dict, list)):
                        malformed_shapes["item_type_not_scalar"] += 1
                        for diagnostic in _take_early_diagnostics():
                            yield diagnostic
                        continue

                if event_type == "thread.started":
                    thread_id = event.get("thread_id")

                elif event_type == "item.started":
                    if item_type in ("command_execution", "mcp_tool_call"):
                        item_id = item.get("id")
                        if not item_id:
                            item_id = str(uuid.uuid4())
                            pending_fifo.setdefault(item_type, []).append(item_id)
                    if item_type == "command_execution":
                        raw_cmd = item.get("command", "")
                        if not isinstance(raw_cmd, str):
                            _warn("command_not_string")
                            raw_cmd = ""
                            for diagnostic in _take_early_diagnostics():
                                yield diagnostic
                        yield ToolStartEvent(
                            id=item_id,
                            name="shell",
                            input={"command": _unwrap_shell_command(raw_cmd)},
                        )
                    elif item_type == "mcp_tool_call":
                        tool_name = item.get("tool", "unknown")
                        if not isinstance(tool_name, str):
                            _warn("tool_not_string")
                            tool_name = "unknown"
                            for diagnostic in _take_early_diagnostics():
                                yield diagnostic
                        arguments = item.get("arguments", {})
                        if not isinstance(arguments, dict):
                            _warn("tool_arguments_not_object")
                            arguments = {}
                            for diagnostic in _take_early_diagnostics():
                                yield diagnostic
                        yield ToolStartEvent(
                            id=item_id,
                            name=tool_name,
                            input=arguments,
                        )
                    elif item_type not in ("agent_message", "reasoning", "file_change", "error"):
                        _record_unknown(
                            unknown_item_types,
                            item_type,
                            overflow=unknown_item_overflow,
                        )
                    # agent_message and reasoning item.started are no-ops
                    # (text is empty, we wait for item.completed)

                elif event_type == "item.updated":
                    item_id = item.get("id", "")

                    if item_type in ("agent_message", "reasoning"):
                        text = self._extract_text(item)
                        if text and item_id:
                            updated_text.setdefault(item_id, []).append(text)
                    elif item_type not in ("command_execution", "mcp_tool_call", "file_change", "error"):
                        _record_unknown(
                            unknown_item_types,
                            item_type,
                            overflow=unknown_item_overflow,
                        )

                elif event_type == "item.completed":
                    if item_type in ("command_execution", "mcp_tool_call"):
                        item_id = item.get("id")
                        if not item_id:
                            item_id = _claim_tool_id(item_type)
                            for diagnostic in _take_early_diagnostics():
                                yield diagnostic


                    if item_type in ("agent_message", "reasoning"):
                        text = self._extract_text(item)
                        # Fall back to text accumulated from item.updated deltas.
                        if not text:
                            item_id = item.get("id", "")
                            parts = updated_text.pop(item_id, [])
                            text = "".join(parts)
                        if text:
                            if item_type == "agent_message":
                                last_agent_text = text
                                yield TextEvent(text=text)
                                # Codex has no per-message id; message_id stays empty (D-04).
                                yield TurnEndEvent(message_id="")
                            else:
                                yield ThinkingEvent(text=text)

                    elif item_type == "command_execution":
                        exit_code = item.get("exit_code", -1)
                        output = item.get("aggregated_output", "")
                        status = item.get("status", "")

                        if status == "declined":
                            yield ToolResultEvent(
                                id=item_id,
                                output="Command declined by sandbox",
                                is_error=True,
                                status="declined",
                            )
                        else:
                            # Forward exit status metadata alongside is_error so
                            # trajectories preserve the structured signal.
                            yield ToolResultEvent(
                                id=item_id,
                                output=output,
                                is_error=exit_code != 0,
                                exit_code=exit_code,
                                status=status or None,
                            )

                    elif item_type == "file_change":
                        for patch_event in _file_change_events(item, execution_cwd):
                            yield patch_event

                    elif item_type == "mcp_tool_call":
                        result_content = ""
                        if "result" in item:
                            result = item["result"]
                            if isinstance(result, dict):
                                result_content = str(result.get("content", ""))
                            else:
                                _warn("tool_result_not_object")
                                for diagnostic in _take_early_diagnostics():
                                    yield diagnostic
                        error = item.get("error")
                        yield ToolResultEvent(
                            id=item_id,
                            output=result_content,
                            is_error=bool(error),
                        )

                    elif item_type == "error":
                        error_sentinel_count += 1

                    else:
                        _record_unknown(
                            unknown_item_types,
                            item_type,
                            overflow=unknown_item_overflow,
                        )

                elif event_type == "turn.completed":
                    usage = event.get("usage", {})
                    if not isinstance(usage, dict):
                        malformed_shapes["usage_not_object"] += 1
                        usage = {}
                    native_model = event.get("model")
                    native_provider = event.get("provider")
                    if isinstance(native_model, str) and native_model:
                        model_name = native_model
                    if isinstance(native_provider, str) and native_provider:
                        provider_name = native_provider
                    # Codex has no message id or cost: use an empty id and estimate cost
                    # from user-overridable prices (None for unknown models). Require both
                    # token counts; reasoning tokens are an output subset, never additive.
                    cached_tokens = usage.get("cached_input_tokens")
                    reasoning_tokens = usage.get("reasoning_output_tokens")
                    in_tok = usage.get("input_tokens")
                    out_tok = usage.get("output_tokens")
                    synth_cost = compute_cost_from_totals(
                        model_name,
                        total_input_tokens=in_tok or 0,
                        cached_input_tokens=cached_tokens or 0,
                        output_tokens=out_tok or 0,
                        prices=_resolved_prices_for_execution(self._execution_input),
                    )
                    if in_tok is not None and out_tok is not None:
                        yield MetricsEvent(
                            message_id="",
                            prompt_tokens=in_tok,
                            completion_tokens=out_tok,
                            cached_tokens=cached_tokens,
                            cost_usd=synth_cost,
                            reasoning_tokens=reasoning_tokens,
                            model_name=model_name,
                            provider_name=provider_name,
                            usage_scope="invocation",
                            measurement_source="turn_end",
                        )
                    yield CostEvent(
                        cost_usd=synth_cost,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cached_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                        model_name=model_name,
                        provider_name=provider_name,
                        measurement_source="turn_end",
                        cost_source="estimated",
                    )

                    if output_schema and last_agent_text:
                        try:
                            structured_result = json.loads(last_agent_text)
                            structured_output_origin = "text"
                        except json.JSONDecodeError:
                            # Observable failure path — surface the bad payload
                            # instead of silently degrading to None.
                            _warn("structured_output_parse_failed")
                            for diagnostic in _take_early_diagnostics():
                                yield diagnostic

                    # Fallback: result/output field directly on turn.completed.
                    if output_schema and structured_result is None:
                        structured_output_origin = "native"
                        for key in ("result", "output"):
                            raw = event.get(key)
                            if raw is not None:
                                if isinstance(raw, dict):
                                    structured_result = raw
                                elif isinstance(raw, str) and raw.strip():
                                    try:
                                        structured_result = json.loads(raw)
                                    except json.JSONDecodeError:
                                        _warn("structured_output_parse_failed")
                                        for diagnostic in _take_early_diagnostics():
                                            yield diagnostic
                                if structured_result is not None:
                                    break

                    continuation_token = None
                    # A disposable checkout disappears after this invocation;
                    # only sessions with a stable cwd can offer native resume.
                    if thread_id and shared_checkout is None:
                        continuation_token = ContinuationToken(
                            backend="codex",
                            data={"thread_id": thread_id},
                        )

                    _pending_result = ResultEvent(
                        structured_output=structured_result,
                        continuation=continuation_token,
                        model_name=model_name,
                        provider_name=provider_name,
                        session_id=thread_id,
                        finish_reason=event.get("finish_reason"),
                        structured_output_origin=structured_output_origin,
                    )

                elif event_type == "turn.failed":
                    error = event.get("error", {})
                    message = (
                        error.get("message", "Unknown Codex error")
                        if isinstance(error, dict)
                        else "Unknown Codex error"
                    )
                    for diagnostic in _take_final_diagnostics():
                        yield diagnostic
                    raise CodexError(str(message))

                elif event_type not in ("turn.started",):
                    _record_unknown(
                        unknown_event_types,
                        event_type,
                        overflow=unknown_event_overflow,
                    )

                for diagnostic in _take_early_diagnostics():
                    yield diagnostic

            # Reap the child, then surface its final diagnostics before the
            # shared exit check formats the backend-specific message from the
            # code and captured diagnostics.
            returncode = await reap(transport)

            for diagnostic in _take_final_diagnostics():
                yield diagnostic

            # Fail fast on non-zero exit: if codex crashed without emitting a
            # turn.failed event, surface the failure with diagnostic output
            # instead of reporting a successful completion with empty/partial
            # output.
            raise_for_exit(
                returncode,
                error_type=CodexError,
                category="PROCESS_EXIT",
                build_message=lambda rc: process_exit_message(display="Codex", returncode=rc, lines=non_json_lines),
            )
            if _pending_result is not None:
                yield _pending_result

        finally:
            if transport is not None:
                await teardown(transport, self._transports)
            if schema_path:
                Path(schema_path).unlink(missing_ok=True)
            if shared_checkout is not None:
                async with self._checkout_lock:
                    shared_checkout.refs -= 1
                    if shared_checkout.refs <= 0:
                        self._checkouts.pop(shared_checkout.cwd, None)
                        shared_checkout.temp_dir.cleanup()

    async def cancel(self) -> None:
        """Terminate and reap every active Codex transport."""
        await CliTransport.cancel_all(self._transports)

    @staticmethod
    def _extract_text(item: dict[str, Any]) -> str:
        """Prefer nonempty top-level text, otherwise join text/output_text content blocks."""
        # Top-level text field (real Codex CLI format).
        top = item.get("text")
        if isinstance(top, str) and top:
            return top
        # Content-block format (legacy / alternative).
        content = item.get("content", [])
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "output_text")
        )
