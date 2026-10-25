"""Tests for CodexBackend with canned JSONL fixtures."""

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream import git_ops
from daydream.agent import StructuredOutputFailure, run_agent
from daydream.atif import validate
from daydream.backends import (
    BackendExecutionInput,
    CodexRequestConfig,
    ContinuationToken,
    CostEvent,
    DiagnosticEvent,
    GenerationEndEvent,
    GenerationStartEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    codex,
)
from daydream.backends.codex import (
    _CODEX_STDOUT_LIMIT_BYTES,
    CodexBackend,
    CodexError,
    _isolated_child_env,
    _prepare_read_only_checkout,
    _rebind_source_paths,
)
from daydream.extensions import Registry, ToolDecision, set_registry
from daydream.retry_policy import FailureClass, classify_failure
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.codex_replay import (
    make_mock_process,
    make_mock_process_from_fixture,
)
from tests.harness.fake_cli_process import (
    LimitAwareStdout,
    assert_concurrent_streams_isolated,
)
from tests.harness.git_helpers import git as _git
from tests.harness.process_replay import replay_process
from tests.harness.protocol_cli import install_protocol_cli
from tests.harness.trajectory import make_recorder

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "codex_jsonl"


def _completed_command(command: str, output: str) -> dict[str, Any]:
    return {
        "type": "command_execution", "command": command, "status": "completed", "exit_code": 0,
        "aggregated_output": output,
    }


def _stage_executable(path: Path) -> Path:
    """Create a real 0o755 shell stub at *path*, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


async def test_artifact_visibility_protocol_cli_consumes_stdin_and_honors_cd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "model cwd with spaces").resolve()
    target.mkdir()
    (target / "source.py").write_text("SOURCE_CANARY\n", encoding="utf-8")
    fixture = install_protocol_cli(tmp_path / "external fixture", "codex")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    prompt = "Inspect the committed source only."
    backend = CodexBackend(model="fixture-model")
    events = [event async for event in backend.execute(target, prompt)]
    observation = fixture.read_observations()[0]
    assert observation["effective_cwd"] == str(target)
    assert observation["stdin_bytes"] == len(prompt.encode())
    assert observation["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert observation["cwd_canaries"]["SOURCE_CANARY"] is True
    assert observation["walk_truncated"] is False
    argv = observation["argv"]
    assert argv[:2] == ["exec", "--experimental-json"]
    assert argv[argv.index("--sandbox") + 1] == "danger-full-access"
    assert argv[argv.index("--cd") + 1] == str(target)
    assert any(isinstance(event, TextEvent) and event.text == "CURRENT_REASONING_CANARY" for event in events)
    assert len([event for event in events if isinstance(event, ResultEvent)]) == 1
    assert backend._transports == []


async def _run_fixture(backend: Any, prompt: Any, fixture: Any, **kwargs: Any) -> Any:
    mock_proc = make_mock_process_from_fixture(fixture)
    events, _ = await replay_process(backend, mock_proc, Path("/tmp"), prompt, **kwargs)
    return events


async def test_malformed_mcp_tool_name_is_string_safe_and_diagnosed(tmp_path: Path) -> None:
    null_tool = {"id": "null-tool", "type": "mcp_tool_call", "tool": None, "arguments": {"path": "api.py"}}
    mapping_tool = {
        "id": "mapping-tool", "type": "mcp_tool_call", "tool": {"token": "private-value"},
        "arguments": {"path": "cli.py"},
    }
    process = make_mock_process([json.dumps(row) for row in [
        {"type": "item.started", "item": null_tool},
        {"type": "item.completed", "item": {**null_tool, "result": {"content": []}}},
        {"type": "item.started", "item": mapping_tool},
        {"type": "item.completed", "item": {**mapping_tool, "result": {"content": []}}},
        {"type": "turn.completed", "usage": {}},
    ]])
    events, _ = await replay_process(CodexBackend(model="fixture-model"), process, tmp_path, "review")
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    diagnostics = [event for event in events if isinstance(event, DiagnosticEvent)]
    assert [(event.id, event.name) for event in starts] == [("null-tool", "unknown"), ("mapping-tool", "unknown")]
    assert [event.id for event in results] == ["null-tool", "mapping-tool"]
    assert [event.code for event in diagnostics] == ["codex_parser_coverage", "codex_parser_coverage"]
    assert [event.metadata["warnings"]["reasons"] for event in diagnostics] == [
        {"tool_not_string": 1}, {"tool_not_string": 2},
    ]
    assert events.index(diagnostics[0]) < events.index(starts[0])
    assert "private-value" not in repr(diagnostics)


async def test_file_changes_preserve_native_shapes_and_correlate_patch_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One edit session retains each native shape, failure diagnostic and synthetic tool correlation."""
    rows = [
        {"type": "thread.started", "thread_id": "patch-session"},
        {
            "type": "item.completed",
            "item": {
                "type": "file_change",
                "id": "map",
                "changes": {
                    "/tmp/spike-repo/a.py": {"type": "add"},
                    "/tmp/spike-repo/b.py": {"type": "update"},
                    "/tmp/spike-repo/c.py": {"type": "delete"},
                    "/tmp/spike-repo/d.py": {"type": "move"},
                },
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "file_change",
                "id": "list",
                "changes": [
                    {"path": "/tmp/spike-repo/e.py", "type": "add"},
                ],
            },
        },
        {"type": "item.completed", "item": {"type": "file_change", "id": "empty", "changes": []}},
        {"type": "item.completed", "item": {
            "type": "file_change", "id": "null-status", "status": None,
            "changes": {"/tmp/spike-repo/f.py": {"type": "update"}},
        }},
        {"type": "item.completed", "item": {
            "type": "file_change", "id": "declined", "status": "declined", "stderr": "sandbox denied write",
            "changes": {"/tmp/spike-repo/b.py": {"type": "update"}},
        }},
        {"type": "item.completed", "item": {
            "type": "file_change", "id": "failed", "status": "failed",
            "stderr": "patch: failed to apply hunk at b.py:12", "changes": {"/tmp/spike-repo/b.py": {"type": "update"}},
        }},
        {"type": "item.completed", "item": {"type": "file_change", "stderr": "x" * 100_000}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 50}},
    ]
    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_inline_lines(CodexBackend(model="fixture-model"), "Edit", [json.dumps(row) for row in rows])
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert [(event.id, event.name) for event in starts[:6]] == [
        ("map", "patch"), ("list", "patch"), ("empty", "patch"), ("null-status", "patch"),
        ("declined", "patch"), ("failed", "patch"),
    ]
    assert len(starts) == len(results) == 7 and starts[-1].name == "patch"
    assert [event.input for event in starts] == [
        {
            "changes": [
                {"path": "spike-repo/a.py", "kind": "add"},
                {"path": "spike-repo/b.py", "kind": "update"},
                {"path": "spike-repo/c.py", "kind": "delete"},
                {"path": "spike-repo/d.py", "kind": "move"},
            ],
            "file": "unknown",
            "action": "modified",
        },
        {"changes": [{"path": "spike-repo/e.py", "kind": "add"}], "file": "spike-repo/e.py", "action": "add"},
        {"changes": [], "file": "unknown", "action": "modified"},
        {"changes": [{"path": "spike-repo/f.py", "kind": "update"}], "file": "spike-repo/f.py", "action": "update"},
        {"changes": [{"path": "spike-repo/b.py", "kind": "update"}], "file": "spike-repo/b.py", "action": "update"},
        {"changes": [{"path": "spike-repo/b.py", "kind": "update"}], "file": "spike-repo/b.py", "action": "update"},
        {"file_change": {"stderr": "x" * 100_000}},
    ]
    assert [event.id for event in results] == [
        "map", "list", "empty", "null-status", "declined", "failed", starts[-1].id,
    ]
    assert [event.status for event in results] == [
        "completed", "completed", "completed", "completed", "declined", "failed", None,
    ]
    assert [event.is_error for event in results] == [False, False, False, False, True, True, True]
    assert [event.output for event in results[:6]] == [
        "add: spike-repo/a.py, update: spike-repo/b.py, delete: spike-repo/c.py, move: spike-repo/d.py",
        "add: spike-repo/e.py", "", "update: spike-repo/f.py",
        "File change declined by sandbox: update: spike-repo/b.py\nstderr: sandbox denied write",
        "update: spike-repo/b.py\nstderr: patch: failed to apply hunk at b.py:12",
    ]
    assert "unparseable file_change" in results[-1].output and "unknown" not in results[-1].output
    assert len(results[-1].output) < 600
    assert "unmatched tool result" not in caplog.text


async def _run_inline_lines(backend: Any, prompt: Any, lines: list[str]) -> list[Any]:
    mock_proc = make_mock_process(lines)
    events, _ = await replay_process(backend, mock_proc, Path("/tmp"), prompt)
    return events


@pytest.mark.parametrize('staged', [False, True])
async def test_empty_final_codex_message_cannot_reuse_planning_json(tmp_path: Path, staged: bool) -> None:
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}, "required": ["issues"]}
    lines = [json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "empty-final"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": '{"issues": []}'}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": ""}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
    ]]
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec",
               return_value=make_mock_process(lines)):
        result, token, reason = await run_agent(
            CodexBackend(model="fixture-model"), tmp_path, "review", phase=DaydreamPhase.REVIEW,
            output_schema=schema, require_full_schema=True,
            investigation_budget=ReviewInvestigationBudget.from_limits(ReviewLimits()) if staged else None,
        )
    assert isinstance(result, StructuredOutputFailure) and result.reason == "missing_output"
    assert token is not None and token.data == {"thread_id": "empty-final"}
    assert reason is None


async def test_nonzero_exit_raises_with_captured_output() -> None:
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process(["Error: authentication required. Run `codex login` to authenticate."])
    mock_proc.returncode = 1
    events = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="return code 1") as exc_info:
            async for event in backend.execute(Path("/tmp"), "Fail"):
                events.append(event)
    # The attempted request and bounded parse gap are observable before the
    # original process-exit failure is preserved.
    assert len(events) == 2
    assert isinstance(events[0], RequestEvent)
    assert isinstance(events[1], DiagnosticEvent)
    assert events[1].code == "codex_parser_coverage"
    msg = str(exc_info.value)
    assert "authentication required" in msg
    assert exc_info.value.category == "AUTH_CONFIG"
    assert classify_failure(exc_info.value).failure_class is FailureClass.AUTH_CONFIG


async def test_nonzero_exit_with_no_output_still_informative() -> None:
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process([])
    mock_proc.returncode = 1
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="return code 1") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass
    assert "no non-JSON output captured" in str(exc_info.value)
    assert exc_info.value.category == "PROCESS_EXIT"


@pytest.mark.parametrize("read_only", [False, True])
async def test_continuation_token_resumes(tmp_path: Path, read_only: bool) -> None:
    """Stable directories retain native resume, including non-Git read-only runs."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    token = ContinuationToken(backend="codex", data={"thread_id": "th_prev"})
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        events = [
            event
            async for event in backend.execute(
                tmp_path,
                "Continue",
                continuation=token,
                read_only=read_only,
                output_schema=schema,
            )
        ]
        call_args = mock_exec.call_args
        flat_args = list(call_args.args) if call_args.args else []
        assert "resume" in flat_args
        assert "th_prev" in flat_args and "--output-schema" in flat_args
        request = next(event for event in events if isinstance(event, RequestEvent))
        assert isinstance(request.config, CodexRequestConfig)
        assert request.config.continuation_mode == "resume" and request.config.native_output_schema is True
        assert request.output_schema == schema
        assert (request.session_id, request.session_source) == ("th_prev", "configured")
        result = next(event for event in events if isinstance(event, ResultEvent))
        assert result.continuation is not None
        assert result.continuation.data["thread_id"] == "th_abc123"
        assert result.session_id == "th_abc123"


async def test_codex_read_only_uses_read_only_sandbox(linked_worktree: tuple[Path, Path]) -> None:
    """A disposable clone mirrors HEAD, index, tracked worktree edits, and untracked files; rebind prompts,
    omit origin, and clean up without changing the source.
    """
    _main, source = linked_worktree
    parser = source / "services" / "taste" / "parser.go"
    parser.write_text("package taste\n\n// caller staged\nfunc CallerStaged() {}\n")
    _git(source, "add", "services/taste/parser.go")
    # The disposable clone includes tracked unstaged edits and untracked files.
    lexer = source / "services" / "taste" / "lexer.go"
    lexer.write_text("package taste\n\n// caller unstaged\nfunc CallerUnstaged() {}\n")
    notes = source / "notes.md"
    notes.write_text("untracked caller note\n")
    source_head = git_ops.head_sha(source)
    source_patch = git_ops.staged_patch(source)
    captured: dict[str, Any] = {}
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")

    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        flat = list(args)
        cd = flat[flat.index("--cd") + 1]
        isolated = Path(cd)
        captured["isolated"] = isolated
        captured["head"] = git_ops.head_sha(isolated)
        captured["patch"] = git_ops.staged_patch(isolated)
        captured["has_parser"] = (isolated / "services" / "taste" / "parser.go").exists()
        captured["lexer"] = (isolated / "services" / "taste" / "lexer.go").read_text()
        captured["has_notes"] = (isolated / "notes.md").exists()
        captured["notes"] = (isolated / "notes.md").read_text() if captured["has_notes"] else None
        captured["remote"] = git_ops.remote_url(isolated)
        captured["branches"] = git_ops.list_local_branches(isolated)
        captured["source_branches"] = git_ops.list_local_branches(source)
        captured["args"] = flat
        return mock_proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        events = [event async for event in CodexBackend(model="fixture-model").execute(
            source, f"Audit repository at {source}", read_only=True,
        )]
    flat = captured["args"]
    assert flat[flat.index("--sandbox") + 1] == "read-only"
    isolated = captured["isolated"]
    assert isolated != source
    assert captured["head"] == source_head
    assert captured["patch"] == source_patch
    assert captured["has_parser"] is True
    assert captured["lexer"] == lexer.read_text()
    assert captured["has_notes"] is True
    assert captured["notes"] == notes.read_text()
    assert captured["remote"] is None
    assert captured["branches"] == captured["source_branches"]
    assert "main" in captured["branches"]
    assert "feature" in captured["branches"]
    written = mock_proc.stdin.write.call_args.args[0]
    assert isinstance(written, bytes)
    assert str(isolated).encode() in written
    assert str(source).encode() not in written
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert request.prompt.encode() == written
    assert isinstance(request.config, CodexRequestConfig)
    assert request.config.sandbox_mode == "read-only" and request.config.read_only_isolation is True
    assert request.config.native_output_schema is False
    assert not isolated.exists()
    # The native session remains observable, but its deleted cwd cannot resume.
    result = next(event for event in events if isinstance(event, ResultEvent))
    assert result.session_id == "th_abc123"
    assert result.continuation is None


async def test_codex_read_only_snapshot_all_branches_diff_and_source_immutable(
    linked_worktree: tuple[Path, Path],
) -> None:
    """Snapshot every branch OID, including slash names; preserve source HEAD/refs/index.

    The clone supports base...HEAD diffs and has no remote or source config path.
    """
    _main, source = linked_worktree
    _git(source, "branch", "release/9.9", "main")
    source_branches = git_ops.list_local_branches(source)
    assert set(source_branches) == {"main", "feature", "release/9.9"}
    head_before = git_ops.head_sha(source)
    captured: dict[str, Any] = {}
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        isolated = Path(list(args)[list(args).index("--cd") + 1])
        captured["branches"] = git_ops.list_local_branches(isolated)
        # Resolve base...HEAD inside the clone.
        captured["diff_rc"] = subprocess.run(
            ["git", "diff", "main...HEAD", "--stat"], cwd=isolated, capture_output=True, text=True,
        ).returncode
        captured["diff_feature_rc"] = subprocess.run(
            ["git", "diff", "release/9.9...HEAD", "--stat"], cwd=isolated, capture_output=True, text=True,
        ).returncode
        captured["head"] = git_ops.head_sha(isolated)
        captured["config"] = subprocess.run(
            ["git", "config", "--local", "--list"], cwd=isolated, capture_output=True, text=True,
        ).stdout
        # Mutating clone refs must leave the source repository independent.
        git_ops.update_refs(isolated, {"refs/heads/main": git_ops.head_sha(isolated)})
        return mock_proc
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        async for _ in CodexBackend(model="fixture-model").execute(source, "Audit repository", read_only=True):
            pass
    assert captured["branches"] == source_branches  # all names, exact OIDs
    assert captured["head"] == head_before  # still detached at source HEAD
    assert captured["diff_rc"] == 0
    assert captured["diff_feature_rc"] == 0
    assert "remote" not in captured["config"]
    assert str(source) not in captured["config"]
    assert git_ops.head_sha(source) == head_before
    assert git_ops.list_local_branches(source) == source_branches


@pytest.mark.parametrize("operation", ["clone", "update_refs"])
async def test_codex_read_only_preparation_failure_is_fail_closed(
    linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    _main, source = linked_worktree
    (source / "services" / "taste" / "parser.go").write_text("package taste\n\n// staged\nfunc S() {}\n")
    _git(source, "add", "services/taste/parser.go")
    before_head = git_ops.head_sha(source)
    before_patch = git_ops.staged_patch(source)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise git_ops.GitError(f"{operation} failure")

    monkeypatch.setattr(f"daydream.git_ops.mutations.{operation}", fail)
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
        with pytest.raises(CodexError, match="failed to create disposable read-only checkout") as excinfo:
            async for _ in CodexBackend(model="fixture-model").execute(source, "Audit", read_only=True):
                pass
    assert isinstance(excinfo.value.__cause__, git_ops.GitError)
    assert git_ops.head_sha(source) == before_head
    assert git_ops.staged_patch(source) == before_patch
    spawn.assert_not_called()


def test_rebind_source_paths_preserves_sibling_paths() -> None:
    """Rebind exact source paths, descendants, and doubled separators; preserve siblings that merely share
    the prefix.
    """
    source = Path("/home/exedev/work")
    execution = Path("/tmp/daydream-codex-read-only-abc/repo")
    siblings = f"never {source}-2 or {source}2 or {source}space or {source}.py"
    out = _rebind_source_paths(siblings, source, execution)
    assert f"{source}-2" in out
    assert f"{source}2" in out
    assert f"{source}space" in out
    assert f"{source}.py" in out
    assert str(execution) not in out
    exact = f"Audit {source} and now {source}/sub/a then {source}/sub/b finally {source}"
    out2 = _rebind_source_paths(exact, source, execution)
    assert str(execution / "sub" / "a") in out2
    assert str(execution / "sub" / "b") in out2
    assert out2.startswith(f"Audit {execution}")
    assert out2.endswith(f"finally {execution}")
    assert str(source) not in out2
    doubled = f"Audit {source}//inner//file and {source}//two"
    out3 = _rebind_source_paths(doubled, source, execution)
    assert str(source) not in out3
    assert str(execution) in out3
    assert "/work//inner" not in out3


async def test_codex_execution_input_supplies_complete_native_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution = BackendExecutionInput.from_environment(
        {
            "HOME": str(tmp_path / "run-home"), "PATH": "/run/bin", "OPENAI_API_KEY": "run-key",
            "DAYDREAM_FANOUT_CONCURRENCY": "5", "DAYDREAM_PI_RETRY_ATTEMPTS": "2",
        }, backend="codex",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key")
    process = make_mock_process_from_fixture("simple_text.jsonl")
    backend = CodexBackend(model="fixture-model", execution_input=execution)
    _, mock_exec = await replay_process(backend, process, tmp_path, "review")
    assert mock_exec.call_args.kwargs["env"] == {
        "HOME": str(tmp_path / "run-home"), "PATH": "/run/bin", "OPENAI_API_KEY": "run-key",
        "DAYDREAM_FANOUT_CONCURRENCY": "5", "DAYDREAM_PI_RETRY_ATTEMPTS": "2",
    }
    assert backend.fanout_concurrency == 5
    assert backend.retry_policy == RetryPolicy(2, 2.0, 120.0)


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="darwin behavior is covered by TestIsolatedChildEnvDarwinPath (issue #1122 M3)",
)
def test_isolated_child_env_untouched_on_non_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/opt/bin")
    monkeypatch.setenv("GIT_DIR", "/leak")
    monkeypatch.setenv("PWD", "/work")
    monkeypatch.setenv("OLDPWD", "/previous")
    monkeypatch.setenv("HOME", "/home/exedev")
    assert _isolated_child_env(Path("/work"), Path("/work")) is None
    monkeypatch.setattr(codex.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("xcrun on Linux")))  # noqa: E501
    env = codex._isolated_child_env(Path("/work"), Path("/tmp/clone/repo"))
    assert env is not None
    assert env["PATH"] == "/usr/bin:/opt/bin"  # byte-for-byte, no prepend (M3)
    assert not {"PWD", "OLDPWD", "GIT_DIR"} & env.keys()
    assert env["HOME"] == "/home/exedev"


async def test_codex_read_only_resume_is_refused(linked_worktree: tuple[Path, Path]) -> None:
    """A read-only session passed a codex resume token fails closed: the resumed
    thread's stored cwd is the per-call clone, deleted when the turn ends."""
    _main, source = linked_worktree
    backend = CodexBackend(model="fixture-model")
    token = ContinuationToken(backend="codex", data={"thread_id": "th_prev"})
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="cannot be resumed"):
            async for _ in backend.execute(source, "Continue", continuation=token, read_only=True):
                pass


async def test_codex_read_only_parallel_calls_share_one_clone(
    linked_worktree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent read-only calls on one backend build a single disposable clone
    (parallel fan-out must not clone the monorepo once per call) and remove it
    only after the last holder's generator exits."""
    _main, source = linked_worktree
    build_calls: list[Path] = []
    real_prep = _prepare_read_only_checkout
    def counting_prep(src: Path, destination: Path) -> Path:
        build_calls.append(destination)
        return real_prep(src, destination)
    monkeypatch.setattr("daydream.backends.codex._prepare_read_only_checkout", counting_prep)
    seen: list[str] = []
    entered = asyncio.Event()
    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        flat = list(args)
        seen.append(flat[flat.index("--cd") + 1])
        if not entered.is_set():
            entered.set()
            # Keep both subprocesses alive so the second reaches the checkout lock while the first
            # still holds its reference.
            await asyncio.sleep(0.2)
        return make_mock_process_from_fixture("simple_text.jsonl")
    backend = CodexBackend(model="fixture-model")
    async def drive() -> None:
        async for _ in backend.execute(source, f"Audit {source}", read_only=True):
            pass
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        await asyncio.gather(drive(), drive())
    assert len(build_calls) == 1, "the two concurrent calls must share one clone build"
    assert len(seen) == 2
    assert seen[0] == seen[1], "both calls ran inside the same shared clone"
    assert not Path(seen[0]).exists()


async def test_codex_read_only_mirrors_symlinks_and_unstaged_deletions(linked_worktree: tuple[Path, Path]) -> None:
    _main, source = linked_worktree
    taste = source / "services" / "taste"
    target = taste / "real.go"
    target.write_text("package taste\n")
    (taste / "link.go").symlink_to("real.go")
    _git(source, "add", "services/taste/real.go", "services/taste/link.go")
    _git(source, "commit", "-m", "add symlink")
    (taste / "lexer.go").unlink()
    (taste / "deadlink").symlink_to("no-such-target")
    subdir = taste / "subdir"
    subdir.mkdir()
    (subdir / "inner.txt").write_text("i")
    (taste / "dir-link").symlink_to(subdir)
    captured: dict[str, Any] = {}
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    async def fake_exec(*args: Any, **kwargs: Any) -> Any:
        flat = list(args)
        isolated = Path(flat[flat.index("--cd") + 1])
        captured["isolated"] = isolated
        captured["link"] = (isolated / "services/taste/link.go").is_symlink()
        captured["link_target"] = str((isolated / "services/taste/link.go").readlink())
        captured["deadlink"] = (isolated / "services/taste/deadlink").is_symlink()
        captured["dir_link"] = (isolated / "services/taste/dir-link").is_symlink()
        captured["doomed_present"] = (isolated / "services/taste/lexer.go").exists()
        return mock_proc
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        async for _ in CodexBackend(model="fixture-model").execute(source, "Audit", read_only=True):
            pass
    assert captured["link"] is True
    assert captured["link_target"] == "real.go"
    assert captured["deadlink"] is True
    assert captured["dir_link"] is True
    assert captured["doomed_present"] is False
    assert not captured["isolated"].exists()


async def test_codex_default_full_access_at_worktree_skips_isolation(linked_worktree: tuple[Path, Path]) -> None:
    _main, source = linked_worktree
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(source, "p"):
            pass
        flat_args = list(mock_exec.call_args.args)
        assert flat_args[flat_args.index("--sandbox") + 1] == "danger-full-access"
        assert "read-only" not in flat_args
        assert flat_args[flat_args.index("--cd") + 1] == str(source)


@pytest.mark.parametrize("effort", ["high", None])
async def test_codex_reasoning_effort_config_override(effort: str | None) -> None:
    backend = CodexBackend(model="fixture-model", reasoning_effort=effort)
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        async for _ in backend.execute(Path("/tmp"), "p"):
            pass
        flat_args = list(mock_exec.call_args.args)
        if effort is None:
            assert "-c" not in flat_args
        else:
            assert flat_args[flat_args.index("-c") + 1] == 'model_reasoning_effort="high"'


async def test_codex_stdout_limit_allows_large_jsonl_events() -> None:
    backend = CodexBackend(model="fixture-model")
    large_text = "x" * (70 * 1024)
    large_line = (
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": large_text}}) + "\n"
    ).encode()
    lines = [large_line, b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n']
    captured_kwargs: dict[str, object] = {}
    async def fake_exec(*args: object, **kwargs: object) -> MagicMock:
        captured_kwargs.update(kwargs)
        raw_limit = kwargs.get("limit", 64 * 1024)
        limit = raw_limit if isinstance(raw_limit, int) else 64 * 1024
        process = MagicMock()
        process.stdout = LimitAwareStdout(lines, limit)
        process.stdin = MagicMock()
        process.stdin.write = MagicMock()
        process.stdin.close = MagicMock()
        process.wait = AsyncMock(return_value=0)
        process.returncode = 0
        process.terminate = MagicMock()
        process.kill = MagicMock()
        return process
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", fake_exec):
        events = [event async for event in backend.execute(Path("/tmp"), "large event")]
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert text_events[0].text == large_text
    assert captured_kwargs["limit"] == _CODEX_STDOUT_LIMIT_BYTES
    assert _CODEX_STDOUT_LIMIT_BYTES > len(large_line)


def _write_model_prices(path: Path, *, model: str, input_price: float, output_price: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'[prices."{model}"]\ninput = {input_price}\noutput = {output_price}\n', encoding="utf-8")
    return path


async def _codex_cost_for_execution_input(model: str, environment: dict[str, str]) -> float | None:
    backend = CodexBackend(
        model=model, execution_input=BackendExecutionInput.from_environment(environment, backend="codex"),
    )
    events = [event async for event in backend.execute(Path("/tmp"), "price this turn")]
    costs = [event for event in events if isinstance(event, CostEvent)]
    metrics = [event for event in events if isinstance(event, MetricsEvent)]
    assert len(costs) == 1
    assert len(metrics) == 1
    assert metrics[0].cost_usd == costs[0].cost_usd
    return costs[0].cost_usd


def _codex_price_process(*, cached_input_tokens: int = 0) -> Any:
    return make_mock_process([
        '{"type":"thread.started","thread_id":"th_price"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 1000000, "cached_input_tokens": cached_input_tokens, "output_tokens": 1000000,
        }}),
    ])


async def test_codex_injected_pricing_uses_each_captured_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent run inputs select their own price file, never the parent override."""
    model = "run-local-price-model"
    first = _write_model_prices(tmp_path / "first.toml", model=model, input_price=1.0, output_price=2.0)
    second = _write_model_prices(tmp_path / "second.toml", model=model, input_price=10.0, output_price=20.0)
    hostile = _write_model_prices(tmp_path / "ambient.toml", model=model, input_price=100.0, output_price=200.0)
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(hostile))
    monkeypatch.setenv("HOME", str(tmp_path / "ambient-home"))
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=[_codex_price_process(cached_input_tokens=500_000), _codex_price_process()],
    ):
        first_cost, second_cost = await asyncio.gather(
            _codex_cost_for_execution_input(model, {"DAYDREAM_PRICES_FILE": str(first)}),
            _codex_cost_for_execution_input(model, {"DAYDREAM_PRICES_FILE": str(second)}),
        )
    assert first_cost == pytest.approx(3.0)
    assert second_cost == pytest.approx(30.0)


async def test_codex_injected_pricing_falls_back_to_captured_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "captured-home-price-model"
    run_home = tmp_path / "run-home"
    _write_model_prices(run_home / ".daydream" / "prices.toml", model=model, input_price=4.0, output_price=5.0)
    hostile = _write_model_prices(tmp_path / "ambient.toml", model=model, input_price=40.0, output_price=50.0)
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(hostile))
    monkeypatch.setenv("HOME", str(tmp_path / "ambient-home"))
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=_codex_price_process()):
        cost = await _codex_cost_for_execution_input(model, {"DAYDREAM_PRICES_FILE": "", "HOME": str(run_home)})
    assert cost == pytest.approx(9.0)


async def test_codex_injected_pricing_without_home_uses_builtins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_model = "ambient-only-price-model"
    hostile = _write_model_prices(tmp_path / "ambient.toml", model=custom_model, input_price=100.0, output_price=200.0)
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(hostile))
    monkeypatch.setenv("HOME", str(tmp_path / "ambient-home"))
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=[_codex_price_process(), _codex_price_process()],
    ):
        custom_cost = await _codex_cost_for_execution_input(custom_model, {"PATH": "/run/bin"})
        builtin_cost = await _codex_cost_for_execution_input("gpt-5.6-sol", {"PATH": "/run/bin"})
    assert custom_cost is None
    # One million input tokens crosses the built-in long-context threshold;
    # this also verifies resolve_prices retains its pricing-policy metadata.
    assert builtin_cost == pytest.approx(55.0)


async def test_codex_injected_missing_or_malformed_price_file_falls_back_to_builtins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile = _write_model_prices(tmp_path / "ambient.toml", model="gpt-5.5", input_price=100.0, output_price=200.0)
    malformed = tmp_path / "malformed.toml"
    malformed.write_text("this is = = not toml", encoding="utf-8")
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(hostile))
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec",
        side_effect=[_codex_price_process(), _codex_price_process()],
    ):
        missing_cost = await _codex_cost_for_execution_input(
            "gpt-5.5", {"DAYDREAM_PRICES_FILE": str(tmp_path / "missing.toml")}
        )
        malformed_cost = await _codex_cost_for_execution_input("gpt-5.5", {"DAYDREAM_PRICES_FILE": str(malformed)})
    assert missing_cost == pytest.approx(35.0)
    assert malformed_cost == pytest.approx(35.0)


async def test_codex_without_execution_input_preserves_ambient_price_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = "ambient-price-model"
    ambient = _write_model_prices(tmp_path / "ambient.toml", model=model, input_price=6.0, output_price=7.0)
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(ambient))
    backend = CodexBackend(model=model)
    events = await _run_inline_lines(
        backend, "price this turn",
        [
            '{"type":"thread.started","thread_id":"th_ambient_price"}',
            '{"type":"turn.completed","usage":{"input_tokens":1000000,"output_tokens":1000000}}',
        ],
    )
    costs = [event for event in events if isinstance(event, CostEvent)]
    assert len(costs) == 1
    assert costs[0].cost_usd == pytest.approx(13.0)


async def test_concurrent_execute_calls_do_not_share_stdout_reader() -> None:
    await assert_concurrent_streams_isolated(
        CodexBackend(model="fixture-model"),
        [
            '{"type":"item.completed","item":{"type":"agent_message","text":"first"}}',
            '{"type":"turn.completed","usage":{}}',
        ],
        second_lines=['{"type":"turn.completed","usage":{}}'],
    )


class TestUnwrapShellCommand:
    async def test_duplicate_completion_does_not_reuse_consumed_start(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Each idless start is consumed once, including wrapped shell commands."""
        raw = '/bin/zsh -lc "ls -la"'
        def line(event_type: str, item: dict[str, Any]) -> str:
            return json.dumps({"type": event_type, "item": item}, separators=(",", ":"))
        lines = [
            json.dumps({"type": "thread.started", "thread_id": "th_m7"}),
            line("item.started", {"type": "command_execution", "command": "echo one"}),
            line("item.started", {"type": "command_execution", "command": raw}),
            line("item.completed", _completed_command('echo one', "one")),
            line("item.completed", _completed_command(raw, "ls")),
            line("item.completed", _completed_command(raw, "ls (dup)")),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
        ]
        backend = CodexBackend(model="fixture-model")
        mock_proc = make_mock_process(lines)
        with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
            events, _ = await replay_process(backend, mock_proc, Path("/tmp"), "M7 content key")
        starts = [e for e in events if isinstance(e, ToolStartEvent)]
        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert [s.input["command"] for s in starts] == ["echo one", "ls -la"]
        start_ids = [s.id for s in starts]
        assert [r.id for r in results] == start_ids + ["codex-unmatched-0"]
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == ["codex parser warning: unmatched tool result"]

    async def test_tool_supervisor_sees_cd_stripped_display_variant(self) -> None:
        """Supervisor ^make rules inspect decoded, cd-stripped commands.

        Storage retains the cd prefix for replay; forwarding it would evade the rule.
        """
        raw = "/bin/zsh -lc cd /home/user/project && make test"
        lines = [
            json.dumps({"type": "thread.started", "thread_id": "th_sup"}),
            json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": raw}}),
            json.dumps({"type": "item.completed", "item": _completed_command(raw, "ok")}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
        ]
        seen: dict[str, str] = {}
        def supervisor(tool_name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase) -> ToolDecision:
            del phase
            assert tool_name == "shell"
            seen["command"] = str(tool_input.get("command", ""))
            if seen["command"].startswith("make"):
                return ToolDecision(veto=True, reason="^make deny")
            return ToolDecision(veto=False)
        registry = Registry()
        registry.register_tool_supervisor(supervisor)
        set_registry(registry)
        backend = CodexBackend(model="fixture-model")
        mock_proc = make_mock_process(lines)
        with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
            _, _, budget_reason = await run_agent(backend, Path("/tmp"), "run", phase=DaydreamPhase.REVIEW)
        assert seen["command"] == "make test"
        assert budget_reason == "tool_vetoed:shell"

    async def test_tool_supervisor_sees_stripped_and_unredacted_command(self) -> None:
        """The supervisor value is strip-only: no redaction, no cap (issue #1227)."""
        token = "ghp_" + "K" * 30
        command = "deploy " + "a" * 180 + " --token " + token + " tail"
        raw = '/bin/zsh -lc "cd /srv/app && ' + command + '"'
        lines = [
            json.dumps({"type": "thread.started", "thread_id": "th_sup1227"}),
            json.dumps({"type": "item.started", "item": {"type": "command_execution", "command": raw}}),
            json.dumps({"type": "item.completed", "item": _completed_command(raw, "ok")}),
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
        ]
        seen: dict[str, str] = {}
        def supervisor(tool_name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase) -> ToolDecision:
            del phase
            seen["name"] = tool_name
            seen["command"] = str(tool_input.get("command", ""))
            return ToolDecision(veto=False)
        registry = Registry()
        registry.register_tool_supervisor(supervisor)
        set_registry(registry)
        mock_proc = make_mock_process(lines)
        with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
            await run_agent(CodexBackend(model="fixture-model"), Path("/tmp"), "run", phase=DaydreamPhase.REVIEW)
        assert seen["name"] == "shell"
        assert len(command) > 200
        assert seen["command"] == command  # prefix stripped, full secret and tail intact
        assert "cd /srv/app" not in seen["command"]
        assert "[REDACTED" not in seen["command"]


# Parser correlation and diagnostics.


async def test_malformed_assistant_json_is_forwarded_as_text(caplog: pytest.LogCaptureFixture) -> None:
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_fixture(backend, "Parse", "malformed_structured_output.jsonl", output_schema=schema)
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("structured output parse failed" in w for w in warnings)
    assert [event.text for event in events if isinstance(event, TextEvent)] == [
        "this is not valid JSON"]


async def test_parser_coverage_is_bounded_redacted_and_precedes_result(caplog: pytest.LogCaptureFixture) -> None:
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_fixture(backend, "Parse gaps", "parser_coverage_gaps.jsonl", output_schema=schema)
    diagnostics = [event for event in events if isinstance(event, DiagnosticEvent)]
    assert [event.code for event in diagnostics] == [
        "codex_transport_coverage", "codex_parser_coverage", "codex_parser_coverage",
    ]
    assert events.index(diagnostics[-1]) < next(
        index for index, event in enumerate(events) if isinstance(event, ResultEvent)
    )
    transport = diagnostics[0]
    assert transport.metadata == {
        "coverage": "incomplete", "reason": "uncorrelated_public_error_item", "occurrences": 1,
        "contract": "codex-cli-0.153.4-json-code-mode",
    }
    assert diagnostics[1].metadata["unknown_event_types"]["total"] == 1
    parser = diagnostics[-1].metadata
    assert parser["unknown_event_types"] == {
        "total": 35, "labels": {f"unknown.{index:02d}": (2 if index == 0 else 1) for index in range(32)}, "overflow": 2,
    }
    assert parser["unknown_item_types"] == {
        "total": 3, "labels": {"mystery.item": 2, 'token="[REDACTED_CREDENTIAL]"': 1}, "overflow": 0,
    }
    assert parser["malformed_shapes"] == {"event_not_object": 2, "event_type_not_scalar": 1, "item_not_object": 1}
    assert parser["non_json_lines"] == 1
    assert parser["warnings"] == {
        "total": 1, "reasons": {"unmatched_tool_result": 1},
    }
    combined = json.dumps([event.metadata for event in diagnostics]) + "\n" + caplog.text
    assert "opaque-parser-secret" not in combined
    assert "/Users/private-person" not in combined
    assert "printf hidden-command" not in combined


def test_parser_label_redacts_complete_value_before_64_character_cap() -> None:
    label = "x" * 54 + " ghp_" + "y" * 12
    bounded = codex._bounded_diagnostic_label(label)
    assert len(bounded) <= 64
    assert "ghp_" not in bounded
    assert "[REDACTED" in bounded


@pytest.mark.parametrize(
    ("error", "expected_retry"),
    [
        pytest.param(
            {"message": "Selected model is at capacity. Please try a different model."}, True, id="observed-text",
        ),
        pytest.param(
            {"message": "Try again later", "codex_error_info": "server_overloaded"}, True, id="structured-info",
        ),
        pytest.param(
            {"message": "Try again later", "code": "server_overloaded"}, True, id="structured-code",
        ),
        pytest.param(
            {"message": "Invalid API key", "codex_error_info": "server_overloaded"}, False, id="permanent-message",
        ),
        *[
            pytest.param(
                {"message": "Selected model is at capacity. Please try a different model.", "code": code},
                False, id=f"explicit-{code}",
            )
            for code in ("unauthorized", "schema_validation", "tool_policy", "unknown_error")
        ],
        pytest.param({"message": "The model has capacity planning advice."}, False, id="unknown-text"),
        pytest.param({"message": "Try again later", "code": {"server_overloaded": True}}, False, id="malformed-code"),
    ],
)
async def test_public_turn_failure_preserves_metadata_and_normalizes_only_overload(
    tmp_path: Path, error: dict[str, Any], expected_retry: bool,
) -> None:
    """Public JSONL decoding must admit capacity retries without making unknown failures retryable."""
    process = make_mock_process([json.dumps({"type": "turn.failed", "error": error})])
    backend = CodexBackend(model="fixture-model")
    with pytest.raises(CodexError) as failure:
        await replay_process(backend, process, tmp_path, "review")
    decision = classify_failure(failure.value)
    assert decision.retries_allowed is expected_retry
    if expected_retry:
        assert decision.failure_class is FailureClass.SERVER_ERROR
    assert failure.value.error_metadata == error
    assert str(failure.value) == error["message"]
    assert backend._transports == []


async def test_public_authentication_failure_is_known_permanent(tmp_path: Path) -> None:
    """The public exec error has message only; auth rejection must stop staged continuation."""
    message = "authentication required. Run `codex login` to authenticate."
    process = make_mock_process([json.dumps({"type": "turn.failed", "error": {"message": message}})])
    backend = CodexBackend(model="fixture-model")
    with pytest.raises(CodexError) as failure:
        await replay_process(backend, process, tmp_path, "review")
    decision = classify_failure(failure.value)
    assert decision.failure_class is FailureClass.AUTH_CONFIG
    assert decision.retries_allowed is False
    assert str(failure.value) == message
    assert failure.value.error_metadata == {"message": message}
    assert backend._transports == []


def test_codex_authentication_diagnostic_precedes_transient_retry_hints() -> None:
    error = CodexError(
        "authentication required. Run codex login to authenticate", retryable=True,
        category="SERVER_ERROR", error_metadata={"codex_error_info": "server_overloaded"},
    )
    decision = classify_failure(error)
    assert decision.failure_class is FailureClass.AUTH_CONFIG
    assert decision.retries_allowed is False


async def test_public_overload_after_completion_cannot_replay_completed_work(tmp_path: Path) -> None:
    process = make_mock_process([
        json.dumps({"type": "turn.completed", "usage": {}}),
        json.dumps({"type": "turn.failed", "error": {"message": "busy", "code": "server_overloaded"}}),
    ])
    with pytest.raises(CodexError) as failure:
        await replay_process(CodexBackend(model="fixture-model"), process, tmp_path, "apply the fix")
    assert classify_failure(failure.value).retries_allowed is False


@pytest.mark.parametrize("error_code", [None, "model_error"])
async def test_turn_failed_flushes_current_aggregate_without_waiting_for_stdout(error_code: str | None) -> None:
    backend = CodexBackend(model="fixture-model")
    class _FailureThenBlockingStdout:
        def __init__(self) -> None:
            self._lines = iter(
                [
                    json.dumps({"type": "future.one"}), json.dumps({"type": "future.two"}),
                    json.dumps({"type": "turn.failed", "error": {"message": "terminal failure", "code": error_code}}),
                ]
            )
            self.blocking_read_started = False
        async def readline(self) -> bytes:
            try:
                return (next(self._lines) + "\n").encode()
            except StopIteration:
                self.blocking_read_started = True
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
    stdout = _FailureThenBlockingStdout()
    mock_proc = make_mock_process([])
    mock_proc.stdout = stdout
    observed: list[Any] = []
    async def consume() -> None:
        async for event in backend.execute(Path("/tmp"), "Fail now"):
            observed.append(event)
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="terminal failure") as failure:
            await asyncio.wait_for(consume(), timeout=0.2)
    assert failure.value.category is None
    assert isinstance(observed[-1], DiagnosticEvent)
    assert observed[-1].code == "codex_parser_coverage"
    parser_diagnostics = [
        event
        for event in observed
        if isinstance(event, DiagnosticEvent) and event.code == "codex_parser_coverage"
    ]
    assert [event.metadata["unknown_event_types"]["total"] for event in parser_diagnostics] == [1, 2]
    assert stdout.blocking_read_started is False


async def test_parser_diagnostic_precedes_nonzero_process_exit() -> None:
    backend = CodexBackend(model="fixture-model")
    secret_line = (
        "not-json token=opaque-parser-secret /Users/private-person/.codex/config.toml "
        + "x" * 500
    )
    mock_proc = make_mock_process([secret_line] * 30)
    mock_proc.returncode = 9
    observed: list[Any] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="return code 9") as exc_info:
            async for event in backend.execute(Path("/tmp"), "Fail"):
                observed.append(event)
    assert isinstance(observed[-1], DiagnosticEvent)
    assert observed[-1].metadata["non_json_lines"] == 30
    assert "opaque-parser-secret" not in json.dumps(observed[-1].metadata)
    message = str(exc_info.value)
    assert "opaque-parser-secret" not in message
    assert "/Users/private-person" not in message
    assert len(message) <= 3_000


# Use the shared fan-out setting so changing backends does not change endpoint concurrency.


@pytest.fixture(autouse=True)
def _reset_real_git_resolution() -> Iterator[Any]:
    """Clear the real-git resolver cache before AND after every test (S1 cache)."""
    codex._REAL_GIT_DIR = None
    codex._REAL_GIT_RESOLVED = False
    yield
    codex._REAL_GIT_DIR = None
    codex._REAL_GIT_RESOLVED = False


class TestResolveRealGitDir:
    """Darwin real-git resolver (issue #1122): resolve once, validate, fail open."""


    def test_concurrent_first_wave_callers_never_see_unresolved_flag(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        git_bin = tmp_path / "real" / "usr" / "bin"
        git_file = _stage_executable(git_bin / "git")
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        calls: list[int] = []
        proc = subprocess.CompletedProcess[str](args=[], returncode=0, stdout=f"{git_file}\n", stderr="")
        def slow_run(*a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
            calls.append(1)
            time.sleep(0.1)  # widen the flag-set / dir-assign window for the racers
            return proc
        monkeypatch.setattr(codex.subprocess, "run", slow_run)
        barrier = threading.Barrier(8)
        results: list[str | None] = []
        results_lock = threading.Lock()
        def racer() -> None:
            barrier.wait()
            resolved = codex._resolve_real_git_dir()
            with results_lock:
                results.append(resolved)
        threads = [threading.Thread(target=racer) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert results == [str(git_bin)] * 8  # S1: no caller observes the early flag and gets None
        assert codex._resolve_real_git_dir() == str(git_bin)  # the next warm call reuses the resolved cache
        assert len(calls) == 1  # S1: the cache was not thrashed into re-shelling to xcrun

    def test_nonzero_xcrun_exit_falls_back_to_none_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        proc = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="xcrun: error")
        monkeypatch.setattr(codex.subprocess, "run", lambda *a, **k: proc)
        with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
            result = codex._resolve_real_git_dir()
        assert result is None  # M2: never raises, never a hard failure
        assert any("xcrun" in r.message.lower() or "git" in r.message.lower() for r in caplog.records)

    def test_non_executable_target_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="/nonexistent/xx/git\n", stderr="")
        monkeypatch.setattr(codex.subprocess, "run", lambda *a, **k: proc)
        assert codex._resolve_real_git_dir() is None  # M2: must be an existing executable file

    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason="darwin behavior is covered by TestResolveRealGitDir (issue #1122)",
    )
    def test_non_darwin_never_invokes_xcrun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            codex.subprocess, "run",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("xcrun invoked on non-Darwin")),
        )
        assert codex._resolve_real_git_dir() is None


class TestIsolatedChildEnvDarwinPath:
    """Darwin PATH-prepend in _isolated_child_env (issue #1122 M1/M4/M6/M7)."""

    @staticmethod
    def _install_xcrun(
        root: Path, *, label: str, exit_code: int = 0,
    ) -> tuple[dict[str, str], Path, Path]:
        bin_dir = root / label / "shim-bin"
        git_dir = root / label / "real-git-bin"
        bin_dir.mkdir(parents=True)
        git_dir.mkdir(parents=True)
        git = _stage_executable(git_dir / "git")
        log = root / f"{label}-probe.jsonl"
        xcrun = bin_dir / "xcrun"
        xcrun.write_text(
            f"#!{sys.executable}\n"
            "import json, os\n"
            "with open(os.environ['PROBE_LOG'], 'a', encoding='utf-8') as out:\n"
            "    out.write(json.dumps(dict(os.environ), sort_keys=True) + '\\n')\n"
            f"print({str(git)!r})\n"
            f"raise SystemExit({exit_code})\n",
            encoding="utf-8",
        )
        xcrun.chmod(0o755)
        environment = {
            "PATH": str(bin_dir), "HOME": str(root / label / "home"), "DEVELOPER_DIR": str(root / label / "developer"),
            "PROBE_LOG": str(log),
        }
        return environment, git_dir, log

    def test_injected_environments_probe_their_own_paths_despite_ambient_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        ambient, ambient_git_dir, _ = self._install_xcrun(tmp_path, label="ambient")
        first, first_git_dir, first_log = self._install_xcrun(tmp_path, label="first")
        second, second_git_dir, second_log = self._install_xcrun(tmp_path, label="second")
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        ambient["PATH"] += f"{os.pathsep}/usr/bin{os.pathsep}/usr/local/bin"
        for key, value in ambient.items():
            monkeypatch.setenv(key, value)
        for key in ("PWD", "OLDPWD", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            monkeypatch.setenv(key, "/leak")
        assert codex._resolve_real_git_dir() == str(ambient_git_dir)
        ambient_child = codex._isolated_child_env(Path("/source"), Path("/ambient-clone"))
        assert ambient_child is not None
        assert ambient_child["PATH"] == f"{ambient_git_dir}{os.pathsep}{ambient['PATH']}"
        assert not {"PWD", "OLDPWD", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"} & ambient_child.keys()
        real_run = subprocess.run
        probe_environments: list[dict[str, str] | None] = []
        def recording_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            probe_environments.append(kwargs.get("env"))
            return real_run(*args, **kwargs)
        monkeypatch.setattr(codex.subprocess, "run", recording_run)
        first["GIT_DIR"] = "/must/be/stripped"
        first_child = codex._isolated_child_env(Path("/source"), Path("/first-clone"), base_environment=first)
        second_child = codex._isolated_child_env(Path("/source"), Path("/second-clone"), base_environment=second)
        assert first_child is not None and second_child is not None
        assert first_child["PATH"].startswith(f"{first_git_dir}{os.pathsep}")
        assert second_child["PATH"].startswith(f"{second_git_dir}{os.pathsep}")
        first_probe = json.loads(first_log.read_text().strip())
        second_probe = json.loads(second_log.read_text().strip())
        expected_first = {
            key: value for key, value in first.items() if key != "GIT_DIR"
        }
        assert probe_environments == [expected_first, second]
        assert expected_first.items() <= first_probe.items()
        assert second.items() <= second_probe.items()
        assert "GIT_DIR" not in first_probe

    def test_failed_injected_probe_leaves_path_and_ordinary_cache_unpoisoned(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        failed, _, failed_log = self._install_xcrun(tmp_path, label="failed", exit_code=1)
        ambient, ambient_git_dir, ambient_log = self._install_xcrun(tmp_path, label="ambient")
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        original_path = failed["PATH"]
        child = codex._isolated_child_env(Path("/source"), Path("/clone"), base_environment=failed)
        assert child is not None
        assert child["PATH"] == original_path
        assert failed_log.exists()
        assert codex._REAL_GIT_RESOLVED is False
        for key, value in ambient.items():
            monkeypatch.setenv(key, value)
        assert codex._resolve_real_git_dir() == str(ambient_git_dir)
        assert ambient_log.exists()

    def test_darwin_resolver_failure_leaves_path_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        monkeypatch.setattr(codex, "_resolve_real_git_dir", lambda: None)  # M2 fail-open
        monkeypatch.setenv("PATH", "/usr/bin")
        env = codex._isolated_child_env(Path("/work"), Path("/tmp/clone/repo"))
        assert env is not None
        assert env["PATH"] == "/usr/bin"  # fallback: unchanged PATH, no exception

    def test_non_isolated_path_returns_none_even_on_darwin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        called = []

        def counting_resolver() -> str:
            called.append(1)
            return "/real/bin"

        monkeypatch.setattr(codex, "_resolve_real_git_dir", counting_resolver)
        assert codex._isolated_child_env(Path("/work"), Path("/work")) is None  # M7
        assert called == []  # resolver never invoked on the non-clone path


async def test_issue1124_stored_commands_are_replayable() -> None:
    """M1-M3: ToolStartEvent.input['command'] is the exact -lc argument (or raw fallback)."""
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Run the commands", "issue1124_unwrap.jsonl")
    starts = {e.id: e.input["command"] for e in events if isinstance(e, ToolStartEvent)}
    assert starts["cmd_1"] == "awk '{print $1}' file.txt"
    assert starts["cmd_2"] == 'echo "hi" and $HOME'
    assert starts["cmd_3"] == "echo $(date)"
    assert starts["cmd_4"] == "echo line1\nline2"
    assert starts["cmd_5"] == "cat <<EOF\nhello\nEOF"
    assert starts["cmd_6"] == 'python -c "print(1)"'
    assert starts["cmd_7"] == "/bin/zsh -lc 'unbalanced"
    assert starts["cmd_8"] == "make test"
    assert starts["cmd_9"] == '/bin/zsh -lc "cmd" extra_arg'
    assert starts["cmd_10"] == "/bin/zsh -lc"
    assert starts["cmd_11"] == "echo 'hello world'"


# --- P18 Task 1: effective request-config admission at the Codex argv seam ---


async def test_native_codex_conversation_records_delivery_shapes_and_correlated_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One actual native conversation has literal projections rather than cross-adapter equality."""
    monkeypatch.delenv("DAYDREAM_FANOUT_CONCURRENCY", raising=False)
    rows = [
        {"type": "thread.started", "thread_id": "native-thread"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "reasoning", "id": "reason", "text": "I will inspect"}},
        {"type": "item.started", "item": {"type": "command_execution", "id": "shell", "command": "ls -la"}},
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "id": "shell",
                "command": "ls -la",
                "exit_code": 0,
                "aggregated_output": "file.py\ntest.py",
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "file_change", "id": "patch", "file_path": "main.py", "action": "modified"},
        },
        {
            "type": "item.started",
            "item": {"type": "mcp_tool_call", "id": "read", "tool": "Read", "arguments": {"path": "/x"}},
        },
        {
            "type": "item.completed",
            "item": {"type": "mcp_tool_call", "id": "read", "tool": "Read", "result": {"content": "file contents"}},
        },
        {
            "type": "item.started",
            "item": {
                "type": "mcp_tool_call",
                "id": "write",
                "tool": "Write",
                "arguments": {"path": "/y", "content": "hello"},
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "id": "write",
                "tool": "Write",
                "result": {"content": "permission denied"},
                "error": "permission denied",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "id": "top",
                "text": '{"kind":"top"}',
                "content": [{"type": "text", "text": "WRONG"}, {"type": "output_text", "text": "ALSO WRONG"}],
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "id": "blocks",
                "content": [{"type": "text", "text": '{"kind":"blocks"}'}],
            },
        },
        {"type": "item.started", "item": {"type": "agent_message", "id": "stream", "content": []}},
        {
            "type": "item.updated",
            "item": {"type": "agent_message", "id": "stream", "content": [{"type": "text", "text": '{"kind":'}]},
        },
        {
            "type": "item.updated",
            "item": {"type": "agent_message", "id": "stream", "content": [{"type": "text", "text": '"streamed"}'}]},
        },
        {"type": "item.completed", "item": {"type": "agent_message", "id": "stream", "content": []}},
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "id": "output",
                "content": [{"type": "output_text", "text": '{"kind":"output"}'}],
            },
        },
        {"type": "turn.completed", "usage": {
            "input_tokens": 200, "output_tokens": 90, "cached_input_tokens": 50, "reasoning_output_tokens": 20,
        }},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "id": "verified", "text": "Verified"}},
        {"type": "turn.completed", "model": "gpt-5.5-pro", "provider": "native-provider", "usage": {
            "input_tokens": 100, "output_tokens": 30, "cached_input_tokens": 20, "reasoning_output_tokens": 10,
        }, "result": {"issues": []}},
    ]
    process = make_mock_process([json.dumps(row) for row in rows])
    backend = CodexBackend(model="gpt-5.5")
    assert backend.fanout_concurrency == 8
    recorder = make_recorder(tmp_path, agent_model_name="codex")
    events: list[Any] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=process):
        async with recorder:
            async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
                inv.observe_user_step(prompt="Inspect")
                async for event in backend.execute(tmp_path, "Inspect", output_schema={"type": "object"}):
                    events.append(event)
                    inv.observe(event)
    expected_text = ['{"kind":"top"}', '{"kind":"blocks"}', '{"kind":"streamed"}', '{"kind":"output"}',
                     "Verified"]
    assert [event.text for event in events if isinstance(event, TextEvent)] == expected_text
    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == ["I will inspect"]
    assert [(event.id, event.name) for event in events if isinstance(event, ToolStartEvent)] == [
        ("shell", "shell"),
        ("patch", "patch"),
        ("read", "Read"),
        ("write", "Write"),
    ]
    results = {event.id: event for event in events if isinstance(event, ToolResultEvent)}
    assert results["shell"].output == "file.py\ntest.py" and not results["shell"].is_error
    patch_start = next(event for event in events if isinstance(event, ToolStartEvent) and event.id == "patch")
    assert patch_start.input == {"file": "main.py", "action": "modified"}
    assert results["patch"].output == "modified: main.py" and not results["patch"].is_error
    assert results["read"].output == "file contents" and results["write"].is_error
    terminal = [event for event in events if isinstance(event, ResultEvent)]
    assert len(terminal) == 1 and terminal[0].structured_output == {"issues": []}
    assert (terminal[0].session_id, terminal[0].model_name, terminal[0].provider_name) == (
        "native-thread", "gpt-5.5-pro", "native-provider",
    )
    assert sum(isinstance(event, TurnEndEvent) for event in events) == 5
    metrics = [event for event in events if isinstance(event, MetricsEvent)]
    assert [(m.message_id, m.prompt_tokens, m.completion_tokens, m.cached_tokens, m.reasoning_tokens)
            for m in metrics] == [("", 200, 90, 50, 20), ("", 100, 30, 20, 10)]
    costs = [event for event in events if isinstance(event, CostEvent)]
    assert [event.cost_usd for event in costs] == pytest.approx([0.003475, 0.0084])
    assert [(event.model_name, event.provider_name) for event in metrics] == [
        ("gpt-5.5", None), ("gpt-5.5-pro", "native-provider"),
    ]
    assert [(event.model_name, event.provider_name) for event in costs] == [
        ("gpt-5.5", None), ("gpt-5.5-pro", "native-provider"),
    ]
    assert [event.usage_scope for event in metrics] == ["invocation", "invocation"]
    request = next(event for event in events if isinstance(event, RequestEvent))
    assert isinstance(request.config, CodexRequestConfig)
    assert request.model_name == "gpt-5.5" and request.provider_name is None
    assert not any(isinstance(event, (GenerationStartEvent, GenerationEndEvent)) for event in events)
    document = json.loads(recorder.path.read_text())
    steps = [step for step in document["steps"] if step["source"] == "agent"]
    assert [step["message"] for step in steps] == expected_text
    assert steps[0]["reasoning_content"] == "I will inspect"
    assert [call["tool_call_id"] for call in steps[0]["tool_calls"]] == ["shell", "patch", "read", "write"]
    assert {result["source_call_id"] for result in steps[0]["observation"]["results"]} == set(results)
    assert process.returncode == 0 and backend._transports == []
    process.wait.assert_awaited()
    process.stdin.close.assert_called()
    process._transport.close.assert_called_once()
    billed = [step["metrics"] for step in steps if step.get("metrics")]
    assert [(m["prompt_tokens"], m["completion_tokens"], m["cached_tokens"], m["extra"]["reasoning_tokens"])
            for m in billed] == [(200, 90, 50, 20), (100, 30, 20, 10)]
    assert [m["cost_usd"] for m in billed] == pytest.approx([0.003475, 0.0084])
    assert document["final_metrics"]["total_prompt_tokens"] == 300
    assert document["final_metrics"]["total_completion_tokens"] == 120
    assert document["final_metrics"]["total_cached_tokens"] == 70
    assert document["final_metrics"]["total_cost_usd"] == pytest.approx(0.011875)
    assert validate(recorder.path, validate_images=False)
