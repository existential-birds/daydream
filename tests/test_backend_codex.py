"""Tests for CodexBackend with canned JSONL fixtures."""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from daydream import git_ops
from daydream.agent import run_agent
from daydream.backends import (
    BackendExecutionInput,
    CodexRequestConfig,
    ContinuationToken,
    CostEvent,
    DiagnosticEvent,
    MetricsEvent,
    RequestEvent,
    ResultEvent,
    RetryPolicy,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    codex,
    effective_fanout_concurrency,
)
from daydream.backends.codex import (
    _CODEX_STDOUT_LIMIT_BYTES,
    CodexBackend,
    CodexError,
    _isolated_child_env,
    _prepare_read_only_checkout,
    _rebind_source_paths,
    _unwrap_shell_command,
    display_shell_command,
)
from daydream.extensions import Registry, ToolDecision, set_registry
from daydream.trajectory import DaydreamPhase
from tests.harness.codex_replay import (
    make_mock_process,
    make_mock_process_from_fixture,
)
from tests.harness.fake_cli_process import (
    FakeCliProcess,
    LimitAwareStdout,
    assert_concurrent_streams_isolated,
)
from tests.harness.git_helpers import git as _git
from tests.harness.process_replay import replay_process
from tests.harness.protocol_cli import install_protocol_cli

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


async def test_tool_use_events() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Run ls", "tool_use.jsonl")
    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    tool_starts = [e for e in events if isinstance(e, ToolStartEvent)]
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    texts = [e for e in events if isinstance(e, TextEvent)]
    assert len(thinking) == 1
    assert thinking[0].text == "Let me run a command"
    assert any(ts.name == "shell" and ts.input == {"command": "ls -la"} for ts in tool_starts)
    assert any(tr.output == "file.py\ntest.py" and not tr.is_error for tr in tool_results)
    # file_change → synthetic ToolStart("patch") + ToolResult
    patches = [ts for ts in tool_starts if ts.name == "patch"]
    assert len(patches) == 1
    assert patches[0].input == {"file": "main.py", "action": "modified"}
    assert any("main.py" in tr.output for tr in tool_results)
    assert any(t.text == "Done!" for t in texts)

@pytest.mark.parametrize("tool_name", [None, 42, False, [], {"token": "private-value"}])
async def test_malformed_mcp_tool_name_is_string_safe_and_diagnosed(tmp_path: Path, tool_name: Any) -> None:
    item = {"id": "mcp-1", "type": "mcp_tool_call", "tool": tool_name, "arguments": {"path": "api.py"}}
    process = make_mock_process([
        json.dumps({"type": "item.started", "item": item}),
        json.dumps({"type": "item.completed", "item": {**item, "result": {"content": []}}}),
        json.dumps({"type": "turn.completed", "usage": {}}),
    ])
    events, _ = await replay_process(CodexBackend(model="fixture-model"), process, tmp_path, "review")
    starts = [event for event in events if isinstance(event, ToolStartEvent)]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    diagnostics = [event for event in events if isinstance(event, DiagnosticEvent)]
    assert len(starts) == len(results) == 1
    assert starts[0].name == "unknown"
    assert starts[0].id == results[0].id == "mcp-1"
    assert diagnostics[0].code == "codex_parser_coverage"
    assert diagnostics[0].metadata["warnings"]["reasons"] == {"tool_not_string": 1}
    assert events.index(diagnostics[0]) < events.index(starts[0])
    assert "private-value" not in repr(diagnostics)


async def test_file_change_changes_map_single_path() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_add.jsonl")
    starts = [e for e in events if isinstance(e, ToolStartEvent) and e.name == "patch"]
    assert len(starts) == 1
    assert starts[0].input["changes"] == [{"path": "spike-repo/a.py", "kind": "add"}]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1 and not results[0].is_error

async def test_file_change_changes_map_multi_path() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_multi.jsonl")
    starts = [e for e in events if isinstance(e, ToolStartEvent) and e.name == "patch"]
    assert len(starts) == 1  # exactly ONE pair for the item — no id collisions
    # Legacy {"file", "action"} keys stay present for multi-file too (pre-diff
    # fallback values), so tool supervisors keyed on them still see the event.
    assert starts[0].input["file"] == "unknown"
    assert starts[0].input["action"] == "modified"
    assert sorted((c["path"], c["kind"]) for c in starts[0].input["changes"]) == [
        ("spike-repo/a.py", "add"), ("spike-repo/b.py", "update"), ("spike-repo/c.py", "delete"),
        ("spike-repo/d.py", "move"),
    ]
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1 and not results[0].is_error

@pytest.mark.parametrize("status", ["declined", "failed"])
async def test_file_change_unsuccessful_status_is_preserved(status: str) -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", f"file_change_{status}.jsonl")
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].status == status
    if status == "failed":
        assert "failed to apply hunk" in results[0].output

async def test_file_change_pathless_payload_diagnostic() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_fixture(backend, "Edit", "file_change_pathless.jsonl")
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "unparseable" in results[0].output
    assert "file_change" in results[0].output  # echoes available fields
    assert "unknown" not in results[0].output

async def _run_inline_lines(backend: Any, prompt: Any, lines: list[str]) -> list[Any]:
    mock_proc = make_mock_process(lines)
    events, _ = await replay_process(backend, mock_proc, Path("/tmp"), prompt)
    return events

async def test_file_change_pathless_echo_is_bounded() -> None:
    # Cap the ToolResult echo for a pathless item, including oversized fields.
    backend = CodexBackend(model="fixture-model")
    big_field = "x" * 100_000
    pathless_line = (
        '{"type":"item.completed","item":{"type":"file_change",'
        '"id":"x1","stderr":"%s"}}' % big_field
    )
    events = await _run_inline_lines(backend, "Edit", [
        '{"type":"thread.started","thread_id":"t"}', pathless_line,
        '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
    ])
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "unparseable" in results[0].output
    assert len(results[0].output) < 600

async def test_file_change_missing_status_is_success() -> None:
    # Missing or null status retains the successful default.
    backend = CodexBackend(model="fixture-model")
    events = await _run_inline_lines(backend, "Edit", [
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"item.completed","item":{"type":"file_change","id":"x1",'
        '"changes":{"/tmp/spike-repo/a.py":{"type":"add"}}}}',
        '{"type":"item.completed","item":{"type":"file_change","id":"x2",'
        '"status":null,"changes":{"/tmp/spike-repo/b.py":{"type":"update"}}}}',
        '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
    ])
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 2
    assert all(r.is_error is False for r in results)
    assert all(r.status == "completed" for r in results)

async def test_file_change_declined_output_names_paths() -> None:
    backend = CodexBackend(model="fixture-model")
    events = await _run_inline_lines(backend, "Edit", [
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"item.completed","item":{"type":"file_change","id":"x1",'
        '"status":"declined","stderr":"sandbox denied write",'
        '"changes":{"/tmp/spike-repo/b.py":{"type":"update"}}}}',
        '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
    ])
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert results[0].status == "declined"
    assert "File change declined by sandbox" in results[0].output
    assert "b.py" in results[0].output  # names the affected path
    assert "sandbox denied write" in results[0].output  # keeps stderr

async def test_file_change_list_changes_parsed() -> None:
    # A list of changes maps to file events; an empty list is a no-op.
    backend = CodexBackend(model="fixture-model")
    events = await _run_inline_lines(backend, "Edit", [
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"item.completed","item":{"type":"file_change","id":"x1",'
        '"status":"completed","changes":[{"path":"/tmp/spike-repo/a.py","type":"add"}]}}',
        '{"type":"item.completed","item":{"type":"file_change","id":"x2",'
        '"status":"completed","changes":[]}}',
        '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
    ])
    starts = [e for e in events if isinstance(e, ToolStartEvent) and e.name == "patch"]
    assert len(starts) == 2
    assert starts[0].input["changes"] == [{"path": "spike-repo/a.py", "kind": "add"}]
    assert starts[1].input["changes"] == []
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert all(r.is_error is False for r in results)

async def test_file_change_idless_pathless_no_unmatched_warning(caplog: pytest.LogCaptureFixture) -> None:
    # An idless file change cannot claim a pending call; assign a new ID without a warning.
    backend = CodexBackend(model="fixture-model")
    with caplog.at_level("WARNING"):
        events = await _run_inline_lines(backend, "Edit", [
            '{"type":"thread.started","thread_id":"t"}', '{"type":"item.completed","item":{"type":"file_change"}}',
            '{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":50}}',
        ])
    results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(results) == 1
    assert "unparseable" in results[0].output
    assert "unmatched tool result" not in caplog.text

@pytest.mark.parametrize(
    ("fixture", "expected_output", "expected_text_count"),
    [
        (
            "structured_output.jsonl",
            {"issues": [{"id": 1, "description": "Fix type hints", "file": "app.py", "line": 5}]}, None,
        ),
        (
            "streamed_structured_output.jsonl",
            {"issues": [{"id": 1, "description": "Missing type hint", "file": "app.py", "line": 10}]}, 1,
        ),
        (
            "output_text_blocks.jsonl",
            {"issues": [{"id": 1, "description": "Bad import", "file": "main.py", "line": 3}]}, None,
        ),
        (
            "turn_completed_result.jsonl",
            {"issues": [{"id": 1, "description": "Unused variable", "file": "utils.py", "line": 22}]}, None,
        ),
    ], ids=["item-completed", "streamed-item-updated", "output-text-blocks", "turn-completed-result"],
)
async def test_structured_output(fixture: Any, expected_output: Any, expected_text_count: Any) -> None:
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    events = await _run_fixture(backend, "Parse", fixture, output_schema=schema)
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == expected_output
    if expected_text_count is not None:
        assert len([e for e in events if isinstance(e, TextEvent)]) == expected_text_count

async def test_turn_failed_raises() -> None:
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("turn_failed.jsonl")
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="Model returned an error") as exc_info:
            async for _ in backend.execute(Path("/tmp"), "Fail"):
                pass
    assert exc_info.value.category is None

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
    assert exc_info.value.category == "PROCESS_EXIT"

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
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        events = [event async for event in backend.execute(
            tmp_path, "Continue", continuation=token, read_only=read_only,
        )]
        call_args = mock_exec.call_args
        flat_args = list(call_args.args) if call_args.args else []
        assert "resume" in flat_args
        assert "th_prev" in flat_args
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

@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="darwin isolated-env PATH behavior is covered by TestIsolatedChildEnvDarwinPath (issue #1122)",
)
def test_isolated_child_env_strips_redirect_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """_isolated_child_env returns None when no isolation and strips the
    repo-redirect env vars (PWD/$GIT_*) when running in the disposable clone."""
    monkeypatch.setenv("PWD", "/home/exedev/work")
    monkeypatch.setenv("OLDPWD", "/home/exedev")
    monkeypatch.setenv("GIT_DIR", "/home/exedev/work/.git")
    monkeypatch.setenv("HOME", "/home/exedev")
    monkeypatch.setenv("PATH", "/usr/bin")
    assert _isolated_child_env(Path("/home/exedev/work"), Path("/home/exedev/work")) is None
    env = _isolated_child_env(Path("/home/exedev/work"), Path("/tmp/clone/repo"))
    assert env is not None
    assert "PWD" not in env
    assert "OLDPWD" not in env
    assert "GIT_DIR" not in env
    # Non-redirect vars are preserved (isolation is path-hiding only).
    assert env["HOME"] == "/home/exedev"
    assert env["PATH"] == "/usr/bin"

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
    monkeypatch.setattr(codex.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("xcrun on Linux")))  # noqa: E501
    env = codex._isolated_child_env(Path("/work"), Path("/tmp/clone/repo"))
    assert env is not None
    assert env["PATH"] == "/usr/bin:/opt/bin"  # byte-for-byte, no prepend (M3)
    assert "GIT_DIR" not in env

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

async def test_toplevel_text_field() -> None:
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    events = await _run_fixture(backend, "Parse", "toplevel_text.jsonl", output_schema=schema)
    thinking = [e for e in events if isinstance(e, ThinkingEvent)]
    assert len(thinking) == 1
    assert "read the review" in thinking[0].text
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output == {
        "issues": [
            {"id": 1, "description": "Missing yield for non-result events", "file": "agents/architect.py", "line": 134}
        ]
    }
    text_events = [e for e in events if isinstance(e, TextEvent)]
    assert len(text_events) == 1


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

def _codex_price_process() -> Any:
    return make_mock_process([
        '{"type":"thread.started","thread_id":"th_price"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        '{"type":"turn.completed","usage":{"input_tokens":1000000,'
        '"cached_input_tokens":0,"output_tokens":1000000}}',
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
        side_effect=[_codex_price_process(), _codex_price_process()],
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
    )

class TestUnwrapShellCommand:

    @pytest.mark.parametrize("command,expected", [
        pytest.param(
            '/bin/zsh -lc "cd /home/user/project && make test"',
            'cd /home/user/project && make test',
            id='zsh-cd',
        ),
        pytest.param(
            "/bin/zsh -lc 'awk '\\''{print $1}'\\'' file.txt'",
            "awk '{print $1}' file.txt",
            id='nested-quotes',
        ),
        pytest.param('/bin/zsh -lc "echo \\"hi\\" and $HOME"', 'echo "hi" and $HOME', id='escaped-quotes'),
        pytest.param('/bin/zsh -lc "echo $(date)"', 'echo $(date)', id='substitution'),
        pytest.param("/bin/zsh -lc 'echo line1\nline2'", 'echo line1\nline2', id='multiline'),
        pytest.param("/bin/zsh -lc 'cat <<EOF\nhello\nEOF'", 'cat <<EOF\nhello\nEOF', id='heredoc'),
        pytest.param('python -c "print(1)"', 'python -c "print(1)"', id='non-shell'),
        pytest.param('/bin/zsh -lc "cmd" extra_arg', '/bin/zsh -lc "cmd" extra_arg', id='trailing-args'),
        pytest.param("/bin/zsh -lc 'unbalanced", "/bin/zsh -lc 'unbalanced", id='bad-quotes'),
        pytest.param('/bin/zsh -lc', '/bin/zsh -lc', id='missing-payload'),
        pytest.param('/bin/zsh -lc "ls -la"', 'ls -la', id='no-cd'),
        pytest.param('', '', id='empty'),
        pytest.param('/bin/zsh -lc ls', 'ls', id='bare-word'),
        pytest.param('/bin/zsh -lc make test', 'make test', id='bare-make'),
        pytest.param('/bin/zsh -lc ls -la', 'ls -la', id='bare-ls'),
        pytest.param('/bin/bash -lc git status --short', 'git status --short', id='bare-bash'),
        pytest.param("/bin/zsh -lc echo 'hello world'", "echo 'hello world'", id='bare-quotes'),
        pytest.param("/bin/zsh -lc 'git diff main...HEAD'", 'git diff main...HEAD', id='git-diff'),
        pytest.param(
            '/bin/zsh -lc "sed -n \'1,260p\' amelia/agents/architect.py"',
            "sed -n '1,260p' amelia/agents/architect.py",
            id='sed',
        ),
    ])
    def test_unwrap_payload(self, command: str, expected: str) -> None:
        assert _unwrap_shell_command(command) == expected

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
        raw = '/bin/zsh -lc "cd /home/user/project && make test"'
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

    def test_unquoted_multi_word_cd_display(self) -> None:
        """Unquoted multi-word cd chains stay replayable stored, cd-stripped on display."""
        raw = "/bin/zsh -lc cd /app && make test"
        assert _unwrap_shell_command(raw) == "cd /app && make test"
        assert display_shell_command(raw) == "make test"


# Parser correlation and diagnostics.


async def test_orphaned_tool_result_is_observable(caplog: pytest.LogCaptureFixture) -> None:
    """Unmatched completions warn and carry deterministic codex-unmatched ids.

    They remain available to the trajectory's unmatched-result bucket.
    """
    backend = CodexBackend(model="fixture-model")
    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_fixture(backend, "Orphan", "orphaned_tool_result.jsonl")
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].id == "codex-unmatched-0", (
        f"orphan should receive a deterministic sequence id; got {tool_results[0].id!r}"
    )
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("unmatched tool result" in w for w in warnings), (
        f"expected an 'unmatched tool result' WARNING; got {warnings}"
    )

async def test_malformed_structured_output_warns(caplog: pytest.LogCaptureFixture) -> None:
    backend = CodexBackend(model="fixture-model")
    schema = {"type": "object", "properties": {"issues": {"type": "array"}}}
    with caplog.at_level(logging.WARNING, logger="daydream.backends.codex"):
        events = await _run_fixture(backend, "Parse", "malformed_structured_output.jsonl", output_schema=schema)
    result_events = [e for e in events if isinstance(e, ResultEvent)]
    assert len(result_events) == 1
    assert result_events[0].structured_output is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("structured output parse failed" in w for w in warnings), (
        f"expected a 'structured output parse failed' WARNING; got {warnings}"
    )

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
        "total": 2, "reasons": {"structured_output_parse_failed": 1, "unmatched_tool_result": 1},
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

async def test_parser_diagnostic_precedes_structured_turn_failure() -> None:
    backend = CodexBackend(model="fixture-model")
    lines = [
        json.dumps({"type": "future.event"}),
        json.dumps({"type": "turn.failed", "error": {"message": "Model returned an error"}}),
    ]
    mock_proc = make_mock_process(lines)
    observed: list[Any] = []
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(CodexError, match="Model returned an error"):
            async for event in backend.execute(Path("/tmp"), "Fail"):
                observed.append(event)
    assert isinstance(observed[-1], DiagnosticEvent)
    assert observed[-1].code == "codex_parser_coverage"

async def test_turn_failed_flushes_current_aggregate_without_waiting_for_stdout() -> None:
    backend = CodexBackend(model="fixture-model")
    class _FailureThenBlockingStdout:
        def __init__(self) -> None:
            self._lines = iter(
                [
                    json.dumps({"type": "future.one"}), json.dumps({"type": "future.two"}),
                    json.dumps({"type": "turn.failed", "error": {"message": "terminal failure"}}),
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
        with pytest.raises(CodexError, match="terminal failure"):
            await asyncio.wait_for(consume(), timeout=0.2)
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


def test_text_extraction_top_level_wins_over_content_blocks() -> None:
    """When both delivery shapes exist, top-level text takes precedence over content blocks."""
    item = {
        "text": "TOP-LEVEL", "content": [{"type": "text", "text": "BLOCK"}, {"type": "output_text", "text": "OUTPUT"}],
    }
    assert CodexBackend._extract_text(item) == "TOP-LEVEL"

# Use the shared fan-out setting so changing backends does not change endpoint concurrency.

@pytest.mark.parametrize(
    ("raw", "ceiling", "expected"),
    [(None, 10, 8), ("3", 10, 3), ("8", 2, 2), ("0", 10, 8), ("-1", 10, 8), ("notanint", 10, 8)],
)
def test_codex_fanout_concurrency_honours_the_shared_env_override(
    monkeypatch: pytest.MonkeyPatch, raw: str | None, ceiling: int, expected: int,
) -> None:
    if raw is None:
        monkeypatch.delenv("DAYDREAM_FANOUT_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("DAYDREAM_FANOUT_CONCURRENCY", raw)
    assert effective_fanout_concurrency(ceiling, CodexBackend("gpt-test")) == expected


class TestDisplayShellCommand:
    """S1/M5: display variant decodes AND strips the leading cd prefix."""


    def test_raw_and_display_distinct_for_cd_command(self) -> None:
        raw = '/bin/zsh -lc "cd /app && echo hello"'
        assert _unwrap_shell_command(raw) == "cd /app && echo hello"
        assert display_shell_command(raw) == "echo hello"

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


    def test_caches_at_most_once_per_process(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        git_bin = tmp_path / "real" / "usr" / "bin"
        git_file = _stage_executable(git_bin / "git")
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        calls: list[int] = []
        proc = subprocess.CompletedProcess[str](args=[], returncode=0, stdout=f"{git_file}\n", stderr="")
        def counting_run(*a: Any, **k: Any) -> subprocess.CompletedProcess[str]:
            calls.append(1)
            return proc
        monkeypatch.setattr(codex.subprocess, "run", counting_run)
        first = codex._resolve_real_git_dir()
        second = codex._resolve_real_git_dir()
        assert first == second == str(git_bin)
        assert len(calls) == 1  # S1: repeated execute() calls never re-shell out to xcrun

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
        for key, value in ambient.items():
            monkeypatch.setenv(key, value)
        assert codex._resolve_real_git_dir() == str(ambient_git_dir)
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

    def test_darwin_prepends_real_git_dir_preserving_rest_of_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        monkeypatch.setattr(codex, "_resolve_real_git_dir", lambda: "/Library/Developer/CommandLineTools/usr/bin")
        monkeypatch.setenv("PATH", "/usr/bin:/usr/local/bin")
        env = codex._isolated_child_env(Path("/work"), Path("/tmp/clone/repo"))
        assert env is not None
        assert env["PATH"].startswith("/Library/Developer/CommandLineTools/usr/bin:")  # M1: precedes
        assert env["PATH"].endswith("/usr/bin:/usr/local/bin")  # M1: remainder + order preserved

    def test_darwin_still_strips_redirect_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        monkeypatch.setattr(codex, "_resolve_real_git_dir", lambda: "/real/bin")
        for var in ("PWD", "OLDPWD", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            monkeypatch.setenv(var, "/leak")
        env = codex._isolated_child_env(Path("/work"), Path("/tmp/clone/repo"))
        assert env is not None
        assert env["PATH"].startswith("/real/bin")
        for var in ("PWD", "OLDPWD", "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):  # M4
            assert var not in env

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

# --- P18 Task 1: effective request-config admission at the Codex argv seam ---


async def test_request_event_config_read_only_sandbox_and_isolation() -> None:
    worktree = Path(tempfile.mkdtemp(prefix="codex-p18-ro-"))
    subprocess_run = subprocess.run
    # Make the cwd a genuine git worktree root via the real git binary.
    env = dict(os.environ)
    env.pop("GIT_DIR", None)
    init = subprocess_run(["git", "init", "-q", str(worktree)], env=env, capture_output=True, check=True)
    assert init.returncode == 0
    (worktree / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess_run(["git", "-C", str(worktree), "add", "seed.txt"], env=env, check=True)
    subprocess_run(
        ["git", "-C", str(worktree), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        env=env, check=True,
    )
    captured_argv: dict[str, Any] = {}
    ro_item = {"type": "agent_message", "id": "m1", "content": [{"type": "text", "text": "ro"}]}
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "th_ro"}),
        json.dumps({"type": "item.completed", "item": ro_item}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
    ]
    def _spawner(argv: list[str], **kwargs: Any) -> FakeCliProcess:
        captured_argv["argv"] = list(argv)
        return FakeCliProcess(lines)
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=lambda *a, **k: _spawner(list(a)),
    ):
        backend = CodexBackend(model="gpt-5.3-codex")
        events = [event async for event in backend.execute(worktree, "hello", read_only=True)]
    argv = captured_argv["argv"]
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, CodexRequestConfig)
    assert config.sandbox_mode == "read-only"
    assert config.read_only_isolation is True  # disposable clone (execution_cwd != cwd)
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert config.native_output_schema is False
    shutil.rmtree(worktree, ignore_errors=True)

async def test_request_event_config_resume_and_schema() -> None:
    captured_argv: dict[str, Any] = {}
    def _spawner(argv: list[str], **kwargs: Any) -> FakeCliProcess:
        captured_argv["argv"] = list(argv)
        return FakeCliProcess(lines)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    res_item = {"type": "agent_message", "id": "m1", "content": [{"type": "text", "text": "{\"ok\": true}"}]}
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "th_res"}),
        json.dumps({"type": "item.completed", "item": res_item}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}),
    ]
    token = ContinuationToken(backend="codex", data={"thread_id": "th_old"})
    with patch(
        "daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=lambda *a, **k: _spawner(list(a)),
    ):
        backend = CodexBackend(model="gpt-5.3-codex")
        events = [
            event
            async for event in backend.execute(Path("/tmp"), "hello", output_schema=schema, continuation=token)
        ]
    argv = captured_argv["argv"]
    request = next(e for e in events if isinstance(e, RequestEvent))
    config = request.config
    assert isinstance(config, CodexRequestConfig)
    assert config.continuation_mode == "resume"
    assert config.native_output_schema is True
    assert "--output-schema" in argv
    assert "resume" in argv
    assert request.session_id == "th_old"
    assert request.session_source == "configured"
    assert "hello" not in argv
