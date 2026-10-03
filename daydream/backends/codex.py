"""Translate codex exec --experimental-json: stdin prompts, JSONL stdout events."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess as subprocess
import sys as sys
import tempfile
import threading
import uuid
from collections import Counter
from collections.abc import AsyncGenerator, Mapping
from functools import partial
from pathlib import Path
from typing import Any

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
from daydream.backends._codex_events import (
    _bounded_diagnostic_label as _bounded_diagnostic_label,
    _bounded_process_excerpt,
    _file_change_events,
    _parser_diagnostics,
    _record_unknown,
    _unwrap_shell_command as _unwrap_shell_command,
    display_shell_command as display_shell_command,
    supervisor_shell_command as supervisor_shell_command,
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

_CODEX_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024
_NON_JSON_EXCERPT_MAX_LINES = 20

_logger = logging.getLogger(__name__)


def _prepare_read_only_checkout(source: Path, destination: Path) -> Path:
    """Snapshot tracked/nonignored files and Git refs; Codex read-only remains path hiding, not an audit sandbox."""
    return git_ops.prepare_independent_snapshot(
        source, destination, include_untracked=True,
    ).repo


# Remove inherited cwd/Git redirects that could point isolated child operations at source refs/index/worktree.
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
    """Validate a bounded xcrun --find git probe. None preserves ambient inheritance;
    explicit environment mappings are complete and never merge host globals.
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
            "codex: could not resolve real git via 'xcrun --find git' (exit %s: %s); leaving child PATH unchanged",
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
    """Cache ambient macOS real-git hits/misses; explicit execution environments probe independently."""
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
    """Reference-count a disposable checkout for concurrent backend/cwd calls; rebuild after the last exit."""

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
    """Remove source-repo path redirects from clone environments. On Darwin prepend real git to
    bypass xcrun; probe failure leaves PATH intact. Non-isolated calls retain explicit environments
    or ambient inheritance. Independently discovered source paths remain accessible.
    """
    if execution_cwd == cwd:
        return dict(base_environment) if base_environment is not None else None
    child_env = dict(base_environment) if base_environment is not None else os.environ.copy()
    for var in _GIT_REDIRECT_STRIP_VARS:
        child_env.pop(var, None)
    # Explicit environments probe uncached; ordinary callers use the cache.
    if sys.platform == "darwin":
        real_git_dir = _probe_real_git_dir(child_env) if base_environment is not None else _resolve_real_git_dir()
        if real_git_dir and "PATH" in child_env:
            child_env["PATH"] = real_git_dir + os.pathsep + child_env["PATH"]
    return child_env


def _resolved_prices_for_execution(
    execution_input: BackendExecutionInput | None,
) -> dict[str, ModelPrice]:
    """Load run-owned prices or ambient direct-call overrides; missing prices path/HOME uses built-ins."""
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
    """Rebind recognized source paths, including resolved/double-slash forms, at path boundaries
    so sibling prefixes remain intact and source spellings cannot reach isolated stdin.
    """
    for candidate in {str(source), str(source.resolve())}:
        pattern = re.escape(candidate).replace("/", "/+") + r"(?![\w.-])"
        prompt = re.sub(pattern, str(execution), prompt)
    return prompt


class CodexError(Exception):
    """Codex failure: category=None for turn.failed, PROCESS_EXIT for a nonzero CLI exit."""

    def __init__(self, message: str, *, category: str | None = None):
        super().__init__(message)
        self.category = category


class CodexBackend:
    """Translate the Codex CLI JSONL stream into normalized events."""

    supports_finalization = True

    # Disposable read-only clones permit inline truncated diffs/context instead of source artifact paths.
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
            self.fanout_concurrency = resolve_fanout_concurrency("DAYDREAM_FANOUT_CONCURRENCY", 8)
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
        """Yield Codex events; nonempty agents raise NotImplementedError. read_only uses a sandbox and,
        at Git roots, a disposable snapshot of HEAD/index/tracked/nonignored files/refs without source
        remotes. Git mutations affect the clone; source paths are excluded from argv/stdin/env/cwd,
        but independently discovered source paths remain accessible. Non-root cwd uses the sandbox
        in place. Preparation/resume failures raise CodexError without falling back to source.
        Deleted clones expose native session identity without continuation; default runs use caller cwd
        with danger-full-access. Finalization caps effort at low and still needs a host zero-tool guard.
        persist_session is ignored. Turn failures raise CodexError; stdout stalls raise retryable
        StreamStalledError and retries start fresh subprocesses.
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

        # Pair idless tools FIFO with observable orphan ids; retain updated text when completion omits it.
        pending_fifo: dict[str, list[str]] = {}  # item_type → [ids] in start order
        updated_text: dict[str, list[str]] = {}  # item_id → [text deltas]
        parse_warnings: Counter[str] = Counter()  # persisted bounded reasons
        unknown_event_types: Counter[str] = Counter()
        unknown_event_overflow = [0]
        unknown_item_types: Counter[str] = Counter()
        unknown_item_overflow = [0]
        malformed_shapes: Counter[str] = Counter()
        record_unknown_item = partial(_record_unknown, unknown_item_types, overflow=unknown_item_overflow)
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
            fresh = [event for event in _current_diagnostics() if event.code not in early_diagnostic_codes]
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
            """Pair idless completions with starts in FIFO order; misses warn and get deterministic orphan ids."""
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
                                    _prepare_read_only_checkout,
                                    cwd,
                                    destination,
                                )
                            except (git_ops.GitError, OSError, shutil.Error) as exc:
                                temp_dir.cleanup()
                                raise CodexError("failed to create disposable read-only checkout") from exc
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

            # Copy/probe the environment off-loop so bounded xcrun lookup cannot stall sibling executions.
            base_environment = self._execution_input.child_environment() if self._execution_input is not None else None
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

            # Record exact applied argv controls; unsupported max_turns/persist_session remain None.
            codex_resume_applied = continuation is not None and continuation.backend == "codex"
            codex_resume_thread: str | None = None
            if continuation is not None and codex_resume_applied:
                thread_value = continuation.data.get("thread_id")
                codex_resume_thread = thread_value if isinstance(thread_value, str) else None
            yield RequestEvent(
                prompt=prompt,
                model_name=model_name,
                output_schema=output_schema,
                reasoning_effort=effort,
                session_id=codex_resume_thread,
                config=CodexRequestConfig(
                    finalization=finalization,
                    sandbox_mode=("read-only" if read_only else "danger-full-access"),
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
                    # Retain bounded lines only for exit exceptions; persistent diagnostics/logs keep counts without
                    # payloads.
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
                        record_unknown_item(item_type)
                    # agent_message and reasoning item.started are no-ops
                    # (text is empty, we wait for item.completed)

                elif event_type == "item.updated":
                    item_id = item.get("id", "")

                    if item_type in ("agent_message", "reasoning"):
                        text = self._extract_text(item)
                        if text and item_id:
                            updated_text.setdefault(item_id, []).append(text)
                    elif item_type not in ("command_execution", "mcp_tool_call", "file_change", "error"):
                        record_unknown_item(item_type)

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

                        declined = status == "declined"
                        yield ToolResultEvent(
                            id=item_id,
                            output="Command declined by sandbox" if declined else output,
                            is_error=declined or exit_code != 0,
                            exit_code=None if declined else exit_code,
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
                        record_unknown_item(item_type)

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
                    # Use an empty message id and price-table cost estimates; unknown models yield no cost.
                    # Metrics require both token counts; reasoning remains an output subset.
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
                        except json.JSONDecodeError:
                            # Observable failure path — surface the bad payload
                            # instead of silently degrading to None.
                            _warn("structured_output_parse_failed")
                            for diagnostic in _take_early_diagnostics():
                                yield diagnostic

                    # Fallback: result/output field directly on turn.completed.
                    if output_schema and structured_result is None:
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

            # Reap, emit final diagnostics, then raise a backend-specific exit error.
            returncode = await reap(transport)

            for diagnostic in _take_final_diagnostics():
                yield diagnostic

            # Report nonzero process exits even without turn.failed; empty/partial output is not success.
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
