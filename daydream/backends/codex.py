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
import shutil
import tempfile
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
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
    TransportExitError,
)
from daydream.pricing import compute_cost_from_totals, load_user_prices, resolve_prices

_SHELL_WRAPPER_RE = re.compile(r"/bin/(?:zsh|bash|sh)\s+-lc\s+(.+)$", re.DOTALL)
_CD_PREFIX_RE = re.compile(r"^cd\s+\S+\s*&&\s*")
_CODEX_STDOUT_LIMIT_BYTES = 10 * 1024 * 1024

_logger = logging.getLogger(__name__)


def _prepare_read_only_checkout(source: Path, destination: Path) -> Path:
    """Build a disposable standalone clone of *source* at *destination*.

    Mirrors source's HEAD, staged index, tracked working files, and nonignored
    untracked files into a fresh clone with no source remote. Uses only
    :mod:`daydream.git_ops` primitives plus shutil/pathlib — never
    ``git rev-parse --git-common-dir``, ``git worktree list``, or the source
    ``.git`` file (the linked-worktree #221 trap). Unstaged deletions are
    mirrored (the worktree file is removed from the clone) and symlinks are
    recreated as links — never materialised as their targets. Submodule gitlink
    entries are skipped: they have no copyable worktree file.

    Returns:
        The clone path.

    Raises:
        GitError / OSError / shutil.Error: Underlying git or filesystem
            failure; the caller wraps these in ``CodexError``.

    Local branch refs are snapshotted by OID from the source so base-branch
    names resolve and ``git diff <base>...HEAD`` works inside the clone,
    without keeping any remote.
    """
    git_ops.clone(str(source), destination)
    git_ops.checkout_detach(destination, git_ops.head_sha(source))
    # Snapshot every source local branch (name -> OID) into the clone before
    # the remote is removed: after a plain clone the clone only exposes the
    # source's checked-out branch under refs/heads/* (the rest exist only as
    # refs/remotes/origin/*), so re-create each same-named ref by explicit OID
    # — never a symbolic ref — with update_refs' validation as the gate. All
    # branches go through one `git update-ref --stdin` transaction, so prep
    # costs a single git call regardless of branch count, and any GitError
    # (invalid name, failed transaction) propagates fail-closed: no
    # half-snapshotted clone is used.
    branches = git_ops.list_local_branches(source)
    if branches:
        git_ops.update_refs(
            destination,
            {f"refs/heads/{name}": oid for name, oid in branches.items()},
        )
    git_ops.remove_remote(destination)
    # ls-files and ls-files --others --exclude-standard are disjoint by
    # construction, so one loop covers both. Enumerate strictly so a mid-prep
    # git failure raises (per this function's error-propagation contract) instead
    # of silently producing a clone missing tracked/untracked files.
    for rel in [*git_ops.ls_files(source, strict=True), *git_ops.list_untracked(source, strict=True)]:
        src = source / rel
        dst = destination / rel
        if src.is_symlink():
            # Mirror the LINK itself. copy2 follows the link (materialising a
            # regular file, or copying a directory target) and src.exists()/
            # src.is_dir() would drop directory-pointing and dangling links,
            # leaving a phantom 120000-mode typechange against the clone's index.
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.unlink(missing_ok=True)
            os.symlink(os.readlink(src), dst)
        elif src.is_dir():
            continue  # submodule gitlink — no copyable worktree file
        elif src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                # Committed symlink now a regular file in the source.
                dst.unlink()
            shutil.copy2(src, dst)
        else:
            # Unstaged deletion: the path is still tracked but the source's
            # worktree file is gone — mirror the missing file so the audit model
            # does not see a phantom file in every git status/ls it runs.
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink(missing_ok=True)
    patch = git_ops.staged_patch(source)
    if patch:
        git_ops.apply_staged_patch(destination, patch)
    return destination


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


class _SharedCheckout:
    """A disposable read-only checkout shared by concurrent ``execute()`` calls.

    The clone is built once per ``(backend, cwd)`` and reference-counted, so
    parallel read-only calls in a fan-out reuse the same checkout instead of each
    cloning the monorepo; the temp dir is removed when the last holder's
    generator exits. Sequential calls each rebuild (snapshot freshness is never
    traded for cache hits), so per-call cleanup semantics are preserved.
    """

    __slots__ = ("cwd", "path", "temp_dir", "refs")

    def __init__(self, cwd: Path, path: Path, temp_dir: tempfile.TemporaryDirectory[str]) -> None:
        self.cwd = cwd
        self.path = path
        self.temp_dir = temp_dir
        self.refs = 0


def _isolated_child_env(cwd: Path, execution_cwd: Path) -> dict[str, str] | None:
    """Return the isolated subprocess environment, or ``None`` when no isolation.

    When *execution_cwd* differs from *cwd* (the disposable-clone case), the
    child must not inherit a path to the caller's repo: the returned copy of the
    parent environment strips ``$PWD``/``$OLDPWD`` and the ``GIT_*`` redirect
    vars that could point the clone's git ops at the source. Returns ``None``
    when the process runs in *cwd* unchanged (nothing to strip). The isolation is
    path-hiding, not physical — a model that independently discovers the source
    path could still write to its refs.
    """
    if execution_cwd == cwd:
        return None
    child_env = os.environ.copy()
    for var in _GIT_REDIRECT_STRIP_VARS:
        child_env.pop(var, None)
    return child_env


def _rebind_source_paths(prompt: str, source: Path, execution: Path) -> str:
    """Rebind every rendering of *source* to *execution* in *prompt*.

    The match is anchored at path boundaries — a sibling path merely sharing
    *source*'s prefix (``/home/user/work-2``, ``/home/user/workspace``,
    ``/home/user/work.py``) is never rewritten — while ``//``-doubled renderings
    (a single run of ``+`` over each slash) and the symlink-resolved form of
    *source* are also rebound, so no textual variant of the source repo path can
    survive into the bytes written to the isolated subprocess's stdin.
    """
    for candidate in {str(source), str(source.resolve())}:
        pattern = re.escape(candidate).replace("/", "/+") + r"(?![\w.-])"
        prompt = re.sub(pattern, str(execution), prompt)
    return prompt


def _unwrap_shell_command(command: str) -> str:
    """Strip shell wrapper from Codex command_execution commands.

    Codex wraps commands in three forms::

        /bin/zsh -lc 'actual command'      (single-quoted)
        /bin/zsh -lc "actual command"      (double-quoted)
        /bin/zsh -lc actual command         (unquoted)

    This extracts just the inner command for display purposes.
    """
    m = _SHELL_WRAPPER_RE.match(command)
    if not m:
        return command
    inner = m.group(1)
    if (inner.startswith('"') and inner.endswith('"')) or (inner.startswith("'") and inner.endswith("'")):
        inner = inner[1:-1]
    inner = _CD_PREFIX_RE.sub("", inner)  # Strip leading "cd /some/path &&".
    return inner.strip()


class CodexError(Exception):
    """Raised when a Codex turn fails or the Codex CLI subprocess exits non-zero.

    Two failure shapes, discriminated by ``category``:

    * ``category is None`` — a structured ``turn.failed`` event (the default,
      used for model-level errors surfaced in the JSONL stream).
    * ``category == "PROCESS_EXIT"`` — the ``codex`` subprocess exited with a
      non-zero return code, carrying captured diagnostic output.
    """

    def __init__(self, message: str, *, category: str | None = None):
        super().__init__(message)
        self.category = category


class CodexBackend:
    """Backend that wraps the Codex CLI subprocess.

    Translates Codex JSONL events into the unified AgentEvent stream.
    """

    concise_fix_prompts = False
    # Codex operates in a disposable read-only clone of the workspace, so it
    # can safely have over-budget diffs inlined (truncated) and exploration
    # summaries inlined rather than pointed at on-disk artifact files.
    read_only_disposable_clone = True

    def __init__(self, model: str, reasoning_effort: str | None = None):
        self.model = model
        self.reasoning_effort = reasoning_effort
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
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute a prompt via Codex CLI and yield unified events.

        Args:
            agents: Optional subagent mapping. Codex does not support non-empty
                subagent maps and will raise if provided.
            read_only: When True, the agent runs under ``--sandbox read-only``
                and — whenever *cwd* is a Git worktree root — inside a
                disposable standalone clone (mirrored HEAD + staged index +
                tracked/nonignored untracked files, no source remote) that is
                deleted after the subprocess exits. The sandbox blocks
                filesystem writes but not Git index/object-store operations;
                the clone is the hard boundary that makes any commit, ref, or
                index change land only in the disposable clone's metadata and
                die with it — the caller's HEAD, staged index, refs, and
                remotes are unreachable via any path the subprocess is given
                (its argv, stdin, env, and cwd); a model that independently
                discovers the source path could still write to its refs. A
                non-Git *cwd* keeps the read-only sandbox in place (no clone).
                If the disposable checkout cannot be created or prepared, the
                call raises ``CodexError`` — never a fallback to the caller's
                cwd. Default False keeps ``danger-full-access`` in the
                caller's cwd.
            persist_session: Accepted for backend protocol parity. Codex does
                not expose persisted CLI sessions here, so this is ignored.

        Raises:
            CodexError: If the Codex turn fails, if the disposable read-only
                checkout cannot be created, or if a ``read_only`` session is
                resumed (a resumed thread's stored cwd is the per-call clone,
                deleted when the creating turn ends — fail closed rather than
                silently continuing in a nonexistent cwd).
            StreamStalledError: If the ``codex`` subprocess emits nothing on
                stdout for the idle window (see
                :func:`daydream.backends._subprocess.stream_idle_timeout_s`).
                Retryable — ``run_agent`` re-arms a fresh subprocess and retries.
            NotImplementedError: If ``agents`` is non-empty (Codex backend
                does not support exploration subagents).
        """
        if agents:
            raise NotImplementedError(
                "Codex backend does not support exploration subagents; use --backend claude for exploration."
            )

        sandbox_mode = "read-only" if read_only else "danger-full-access"

        schema_path: str | None = None
        if output_schema:
            schema_path = self._write_temp_schema(output_schema)

        thread_id: str | None = None
        last_agent_text: str | None = None
        structured_result: Any = None

        # Event correlation state. Codex events arrive item.started → updated* →
        # completed. (1) Some item.started lack an `id`; we pair start/result via
        # a deterministic FIFO per item_type (order-preserving, content-
        # independent) with the legacy content-key as a secondary fallback. When
        # both miss, the completion is an orphan — we emit an OBSERVABLE warning
        # and assign a deterministic sequence id so the trajectory recorder can
        # still bucket it via unmatched_tool_results without a silent drop.
        # (2) agent_message/reasoning text may stream as item.updated deltas with
        # an empty item.completed, so we accumulate deltas by id and join them.
        pending_fifo: dict[str, list[str]] = {}  # item_type → [ids] in start order
        pending_item_ids: dict[str, str] = {}  # "type:content" → generated id (legacy)
        updated_text: dict[str, list[str]] = {}  # item_id → [text deltas]
        parse_warnings: list[str] = []  # observable parse-failure surface
        _pending_result: ResultEvent | None = None
        non_json_lines: list[str] = []  # non-JSON lines (stdout merged with stderr) for error diagnostics
        unmatched_seq = 0  # monotonic source for orphaned tool-result ids

        def _warn(msg: str, **detail: Any) -> None:
            """Log a parser warning and record it for later trajectory surfacing."""
            parse_warnings.append(msg)
            _logger.warning("codex: %s %s", msg, detail)

        def _claim_tool_id(item_type: str, content_key: str, content_value: Any) -> str:
            """Pop the next correlated id for a no-id item.completed.

            Prefers the FIFO (content-independent order correlation), falls back
            to the legacy content-key, and finally — if both miss — emits an
            OBSERVABLE warning and returns a deterministic orphan id. The orphan
            still lands in ``trajectory.extra.unmatched_tool_results`` because
            the validator hard-fails on a dangling ``source_call_id``; the
            warning ensures the miss is no longer silent.
            """
            nonlocal unmatched_seq
            fifo = pending_fifo.get(item_type, [])
            item_id = fifo.pop(0) if fifo else None
            if item_id is None:
                item_id = pending_item_ids.pop(content_key, None)
            if item_id is None:
                _warn("unmatched tool result", item_type=item_type, key=content_key, value=content_value)
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
                        # A resumed thread's stored cwd is the per-call disposable
                        # clone, deleted when the creating turn exits — honoring the
                        # resume here would run the thread in a nonexistent cwd
                        # (the prompt rebind targets a fresh clone path, not the
                        # thread's stored one). Fail closed.
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
            if self.reasoning_effort:
                args.extend(["-c", f'model_reasoning_effort="{self.reasoning_effort}"'])
            if schema_path:
                args.extend(["--output-schema", schema_path])
            if continuation is not None and continuation.backend == "codex":
                args.extend(["resume", continuation.data["thread_id"]])

            # When running in the disposable clone, the child must not inherit a
            # path to the caller's repo: _isolated_child_env strips $PWD/$OLDPWD
            # and the GIT_* overrides that could redirect the clone's git ops back
            # at the source, and the child is started in the clone (below) so its
            # own cwd is never the source path. Path-hiding, not physical.
            child_env = _isolated_child_env(cwd, execution_cwd)

            if execution_cwd != cwd:
                # Rebind the prompt so no rendering of the caller's source
                # path appears in the bytes written to the isolated subprocess.
                prompt = _rebind_source_paths(prompt, cwd, execution_cwd)

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

            idle_timeout_s = stream_idle_timeout_s()
            async for raw_line in transport.lines(lambda: idle_timeout_s):
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    # Codex occasionally emits non-JSON status lines on stdout;
                    # skip them but leave a debug breadcrumb for triage.
                    # Capture non-JSON lines for diagnostic surfacing on non-zero exit.
                    if len(non_json_lines) >= 20:
                        non_json_lines.pop(0)
                    non_json_lines.append(raw_line)
                    _logger.debug("codex: non-JSON line skipped: %r", raw_line[:80])
                    continue

                event_type = event.get("type", "")

                if event_type == "thread.started":
                    thread_id = event.get("thread_id")

                elif event_type == "item.started":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "command_execution":
                        item_id = item.get("id")
                        if not item_id:
                            item_id = str(uuid.uuid4())
                            # FIFO is the primary correlation path (order-
                            # preserving); content-key stays as legacy fallback.
                            pending_fifo.setdefault("command_execution", []).append(item_id)
                            pending_item_ids[f"command_execution:{item.get('command', '')}"] = item_id
                        raw_cmd = item.get("command", "")
                        yield ToolStartEvent(
                            id=item_id,
                            name="shell",
                            input={"command": _unwrap_shell_command(raw_cmd)},
                        )
                    elif item_type == "mcp_tool_call":
                        item_id = item.get("id")
                        if not item_id:
                            item_id = str(uuid.uuid4())
                            # FIFO is the primary correlation path (order-
                            # preserving); content-key stays as legacy fallback.
                            pending_fifo.setdefault("mcp_tool_call", []).append(item_id)
                            pending_item_ids[f"mcp_tool_call:{item.get('tool', '')}"] = item_id
                        yield ToolStartEvent(
                            id=item_id,
                            name=item.get("tool", "unknown"),
                            input=item.get("arguments", {}),
                        )
                    # agent_message and reasoning item.started are no-ops
                    # (text is empty, we wait for item.completed)

                elif event_type == "item.updated":
                    item = event.get("item", {})
                    item_type = item.get("type", "")
                    item_id = item.get("id", "")

                    if item_type in ("agent_message", "reasoning"):
                        text = self._extract_text(item)
                        if text and item_id:
                            updated_text.setdefault(item_id, []).append(text)

                elif event_type == "item.completed":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "agent_message":
                        text = self._extract_text(item)
                        # Fall back to text accumulated from item.updated deltas.
                        if not text:
                            item_id = item.get("id", "")
                            parts = updated_text.pop(item_id, [])
                            text = "".join(parts)
                        if text:
                            last_agent_text = text
                            yield TextEvent(text=text)
                            # Codex has no per-message id; message_id stays empty (D-04).
                            yield TurnEndEvent(message_id="")

                    elif item_type == "reasoning":
                        text = self._extract_text(item)
                        if not text:
                            item_id = item.get("id", "")
                            parts = updated_text.pop(item_id, [])
                            text = "".join(parts)
                        if text:
                            yield ThinkingEvent(text=text)

                    elif item_type == "command_execution":
                        item_id = item.get("id")
                        if not item_id:
                            item_id = _claim_tool_id(
                                "command_execution",
                                f"command_execution:{item.get('command', '')}",
                                item.get("command", ""),
                            )
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
                        # file_change has no item.started — emit a synthetic pair.
                        changes = item.get("changes")
                        if isinstance(changes, dict) or isinstance(changes, list):
                            # Current CLI shape: a map of path -> {type, ...}.
                            # Normalize each path against execution_cwd; keep
                            # absolute paths that fall outside it. A
                            # list-shaped `changes` (plausible alternate CLI
                            # layout) is folded to the same path-keyed map;
                            # entries without a path-ish key carry no usable
                            # path and are dropped. An empty map is a no-op,
                            # never an error — the old code always recorded
                            # success for it.
                            if isinstance(changes, list):
                                changes = {
                                    str(c.get("path") or c.get("file_path")): c
                                    for c in changes
                                    if isinstance(c, dict) and (c.get("path") or c.get("file_path"))
                                }
                            item_id = item.get("id", str(uuid.uuid4()))
                            parsed = []
                            for raw_path, entry in changes.items():
                                kind = entry.get("type", "unknown") if isinstance(entry, dict) else "unknown"
                                path = str(raw_path)
                                try:
                                    if os.path.isabs(path) and os.path.commonpath([
                                        path,
                                        str(execution_cwd),
                                    ]) == str(execution_cwd):
                                        path = os.path.relpath(path, execution_cwd)
                                except ValueError:
                                    pass  # disjoint drives etc. — keep absolute
                                parsed.append({"path": path, "kind": kind})
                            # A missing `status` means "completed": the old
                            # code always recorded success, so an absent status
                            # (or an explicit `null`) must never be archived as
                            # an error.
                            status = item.get("status") or "completed"
                            # Cap the joined path listing with the same limit
                            # as the stdout/stderr excerpts so a large
                            # multi-file apply or a giant path cannot produce
                            # an unbounded ToolResult output.
                            joined = ", ".join(
                                f"{c['kind']}: {c['path']}" for c in parsed
                            )[:500]
                            if status == "declined":
                                # Name the affected paths and keep any
                                # stdout/stderr so a declined change is not
                                # silently path-free.
                                output = f"File change declined by sandbox: {joined}"
                            else:
                                output = joined
                            if status in ("failed", "declined"):
                                for stream in ("stdout", "stderr"):
                                    excerpt = item.get(stream, "")
                                    if excerpt:
                                        output += f"\n{stream}: {excerpt[:500]}"
                            # Preserve the legacy patch-input keys for
                            # extension tool supervisors keyed on
                            # {"file", "action"}: exact path/kind for
                            # single-file changes; multi-file (or empty)
                            # keeps the pre-diff scalar fallback
                            # ("unknown"/"modified") so the documented keys
                            # never go silent, with the full detail
                            # namespaced under `changes`.
                            start_input: dict[str, Any] = {"changes": parsed}
                            if len(parsed) == 1:
                                start_input["file"] = parsed[0]["path"]
                                start_input["action"] = parsed[0]["kind"]
                            else:
                                start_input["file"] = "unknown"
                                start_input["action"] = "modified"
                            yield ToolStartEvent(
                                id=item_id,
                                name="patch",
                                input=start_input,
                            )
                            yield ToolResultEvent(
                                id=item_id,
                                output=output,
                                is_error=status != "completed",
                                status=status or None,
                            )
                        elif "file_path" in item:
                            # Legacy scalar shape (older CLI): keep as-is.
                            item_id = item.get("id", str(uuid.uuid4()))
                            file_path = item.get("file_path", "unknown")
                            action = item.get("action", "modified")
                            yield ToolStartEvent(
                                id=item_id,
                                name="patch",
                                input={"file": file_path, "action": action},
                            )
                            yield ToolResultEvent(
                                id=item_id,
                                output=f"{action}: {file_path}",
                                is_error=False,
                            )
                        else:
                            # Pathless payload — neither `changes` nor `file_path`.
                            # Never silently archive (the old behavior recorded
                            # "modified: unknown"): emit a synthetic pair whose
                            # ToolResult is an observable error echoing the
                            # item's available fields. Fall back to a plain
                            # uuid, not `_claim_tool_id` — file_change never
                            # populates the FIFO/content-key maps, so that call
                            # could only fire a spurious "unmatched tool result"
                            # warning for an item that is not unmatched.
                            item_id = item.get("id", str(uuid.uuid4()))
                            fields = {
                                k: v for k, v in item.items()
                                if k != "type" and v != "unknown"
                            }
                            # Cap the diagnostic echo with the same limit as the
                            # stdout/stderr excerpts so a degenerate pathless item
                            # carrying a large field cannot produce an unbounded
                            # ToolResult output.
                            echo = json.dumps(fields)[:500]
                            yield ToolStartEvent(
                                id=item_id,
                                name="patch",
                                input={"file_change": fields},
                            )
                            yield ToolResultEvent(
                                id=item_id,
                                output=f"unparseable file_change item: {echo}",
                                is_error=True,
                                status=item.get("status") or None,
                            )

                    elif item_type == "mcp_tool_call":
                        item_id = item.get("id")
                        if not item_id:
                            item_id = _claim_tool_id(
                                "mcp_tool_call",
                                f"mcp_tool_call:{item.get('tool', '')}",
                                item.get("tool", ""),
                            )
                        result_content = ""
                        if "result" in item:
                            result_content = str(item["result"].get("content", ""))
                        error = item.get("error")
                        yield ToolResultEvent(
                            id=item_id,
                            output=result_content,
                            is_error=bool(error),
                        )

                elif event_type == "turn.completed":
                    usage = event.get("usage", {})
                    # EVNT-07: MetricsEvent per turn (empty message_id — Codex has no
                    # per-message id). #194 reverses D-16: the CLI emits no cost field,
                    # so we now synthesize from tokens via the #61 user-overridable price
                    # table (Claude/Pi parity). None when the model is unknown to the
                    # table (preserves the #156 observable-marker). cached_input_tokens
                    # IS surfaced (#65, K4). Rename input/output_tokens → prompt/completion;
                    # skip if either missing (EVNT-02 requires both). CostEvent below
                    # carries partials.
                    #
                    # #192: reasoning_output_tokens is the reasoning portion of
                    # output_tokens — a SUBSET, NOT additive (codex's own
                    # accounting.rs already counts these inside output_tokens, so
                    # cost synthesis is unchanged). Surfaced for cost attribution
                    # and perf observability (#171/#172/#186). OpenAI emits the
                    # COUNT only — no reasoning content (openai/codex#26428).
                    cached_tokens = usage.get("cached_input_tokens")
                    reasoning_tokens = usage.get("reasoning_output_tokens")
                    in_tok = usage.get("input_tokens")
                    out_tok = usage.get("output_tokens")
                    synth_cost = compute_cost_from_totals(
                        self.model,
                        total_input_tokens=in_tok or 0,
                        cached_input_tokens=cached_tokens or 0,
                        output_tokens=out_tok or 0,
                        prices=resolve_prices(load_user_prices()),
                    )
                    if in_tok is not None and out_tok is not None:
                        yield MetricsEvent(
                            message_id="",
                            prompt_tokens=in_tok,
                            completion_tokens=out_tok,
                            cached_tokens=cached_tokens,
                            cost_usd=synth_cost,
                            reasoning_tokens=reasoning_tokens,
                            model_name=self.model,
                        )
                    yield CostEvent(
                        cost_usd=synth_cost,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        cached_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                        model_name=self.model,
                    )

                    if output_schema and last_agent_text:
                        try:
                            structured_result = json.loads(last_agent_text)
                        except json.JSONDecodeError:
                            # Observable failure path — surface the bad payload
                            # instead of silently degrading to None.
                            _warn(
                                "structured output parse failed",
                                source="agent_text",
                                raw=last_agent_text[:200],
                            )

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
                                        _warn(
                                            "structured output parse failed",
                                            source=f"turn.completed.{key}",
                                            raw=raw[:200],
                                        )
                                if structured_result is not None:
                                    break

                    continuation_token = None
                    if thread_id:
                        continuation_token = ContinuationToken(
                            backend="codex",
                            data={"thread_id": thread_id},
                        )

                    _pending_result = ResultEvent(
                        structured_output=structured_result,
                        continuation=continuation_token,
                    )

                elif event_type == "turn.failed":
                    error = event.get("error", {})
                    raise CodexError(error.get("message", "Unknown Codex error"))

                elif event_type not in ("turn.started",):
                    pass

            # Reap the child (the transport raises TransportExitError on a
            # non-zero exit; _check_return_code below formats the backend-
            # specific message from the code and captured diagnostics).
            try:
                await transport.wait()
            except TransportExitError:
                pass

            # Fail fast on non-zero exit: if codex crashed without emitting a
            # turn.failed event, surface the failure with diagnostic output
            # instead of reporting a successful completion with empty/partial
            # output.
            self._check_return_code(transport.returncode, non_json_lines)
            if _pending_result is not None:
                yield _pending_result

        finally:
            if transport is not None:
                await transport.terminate()
                if transport in self._transports:
                    self._transports.remove(transport)
            if schema_path:
                Path(schema_path).unlink(missing_ok=True)
            if shared_checkout is not None:
                async with self._checkout_lock:
                    shared_checkout.refs -= 1
                    if shared_checkout.refs <= 0:
                        self._checkouts.pop(shared_checkout.cwd, None)
                        shared_checkout.temp_dir.cleanup()

    async def cancel(self) -> None:
        """Cancel all running Codex processes.

        Sends SIGTERM, waits briefly, then SIGKILL if still running.
        """
        await CliTransport.cancel_all(self._transports)

    @staticmethod
    def _extract_text(item: dict[str, Any]) -> str:
        """Extract text from a Codex item.

        Codex items may carry text either as a top-level ``text`` field
        or inside ``content`` blocks (with type ``text`` or ``output_text``).
        """
        # Top-level text field (real Codex CLI format).
        top = item.get("text")
        if isinstance(top, str) and top:
            return top
        # Content-block format (legacy / alternative).
        content = item.get("content", [])
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _check_return_code(
        returncode: int | None,
        non_json_lines: list[str],
    ) -> None:
        """Raise CodexError(PROCESS_EXIT) if the subprocess exited non-zero."""
        if returncode is not None and returncode != 0:
            tail = "\n".join(non_json_lines[-10:])
            if non_json_lines:
                detail = (
                    f"\nCodex CLI output (last {min(len(non_json_lines), 10)} "
                    f"non-JSON lines):\n{tail}"
                )
            else:
                detail = (
                    "\n(no non-JSON output captured — codex may have "
                    "crashed before writing to stdout)"
                )
            raise CodexError(
                f"Codex CLI exited with return code {returncode}.{detail}",
                category="PROCESS_EXIT",
            )

    @staticmethod
    def _write_temp_schema(schema: dict[str, Any]) -> str:
        """Write JSON schema to a temp file and return the path."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="daydream-schema-") as f:
            json.dump(schema, f)
            return f.name
