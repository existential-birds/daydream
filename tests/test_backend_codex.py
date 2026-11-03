"""Tests for CodexBackend with canned JSONL fixtures."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    codex,
)
from daydream.backends.codex import (
    CodexBackend,
    CodexError,
)
from daydream.extensions import Registry, ToolDecision, set_registry
from daydream.phases.schemas import review_stage_schema, review_stage_submission_schema
from daydream.retry_policy import FailureClass, classify_failure
from daydream.review_budget import ReviewInvestigationBudget, ReviewLimits
from daydream.trajectory import DaydreamPhase
from tests.harness.codex_replay import (
    make_mock_process,
    make_mock_process_from_fixture,
)
from tests.harness.git_helpers import git as _git
from tests.harness.process_replay import replay_process
from tests.harness.trajectory import make_recorder

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "codex_jsonl"



def _case_dir(tmp_path: Path, index: int) -> Path:
    case = tmp_path / f"case-{index}"
    case.mkdir()
    return case


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


async def test_repeated_turn_completion_charges_usage_once() -> None:
    """Each public turn is charged once: repeated completions and partial usage never invent billing."""
    lines = [json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "repeated-terminal"}, {"type": "turn.started"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}},
        {"type": "turn.completed", "usage": {"input_tokens": 99, "output_tokens": 99}},
    ]]
    backend = CodexBackend(model="fixture-model")
    events = await _run_inline_lines(backend, "review", lines)
    await backend.cancel()  # no transport remains after a settled turn
    costs = [event for event in events if isinstance(event, CostEvent)]
    assert [(cost.input_tokens, cost.output_tokens) for cost in costs] == [(10, 5)]
    assert len([event for event in events if isinstance(event, ResultEvent)]) == 1
    # Partial terminal usage keeps its known input and never invents output tokens or per-turn metrics.
    partial = await _run_fixture(CodexBackend(model="fixture-model"), "p", "turn_completed_partial_usage.jsonl")
    assert not any(isinstance(event, MetricsEvent) for event in partial)
    cost = next(event for event in partial if isinstance(event, CostEvent))
    assert (cost.input_tokens, cost.output_tokens) == (200, None)


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
    # Parser diagnostic labels redact a complete credential before applying their 64-character cap.
    bounded = codex._bounded_diagnostic_label("x" * 54 + " ghp_" + "y" * 12)
    assert len(bounded) <= 64 and "ghp_" not in bounded and "[REDACTED" in bounded


async def _continuation_token_resumes_case(tmp_path: Path, read_only: bool) -> None:
    """A resumed thread persists the staged selector transport schema as native output."""
    backend = CodexBackend(model="fixture-model")
    mock_proc = make_mock_process_from_fixture("simple_text.jsonl")
    token = ContinuationToken(backend="codex", data={"thread_id": "th_prev"})
    schema = review_stage_submission_schema(review_stage_schema(['api.py'], [], triage=False))
    persisted: dict[str, Any] = {}

    async def spawn(*args: str, **kwargs: Any) -> Any:
        persisted.update(json.loads(Path(args[args.index('--output-schema') + 1]).read_text()))
        return mock_proc

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", side_effect=spawn) as mock_exec:
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
        assert request.output_schema == schema == persisted
        assert (request.session_id, request.session_source) == ("th_prev", "configured")
        result = next(event for event in events if isinstance(event, ResultEvent))
        assert result.continuation is not None
        assert result.continuation.data["thread_id"] == "th_abc123"
        assert result.session_id == "th_abc123"


async def test_continuation_token_resumes(tmp_path: Path) -> None:
    for index, read_only in enumerate([False, True]):
        try:
            await _continuation_token_resumes_case(tmp_path=_case_dir(tmp_path, index), read_only=read_only)
        except BaseException as exc:
            exc.add_note(f"case: read_only={read_only}")
            raise


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
        # Ref writes inside the disposable clone never reach the source repository.
        git_ops.update_refs(isolated, {"refs/heads/clone-only": captured["head"]})
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

    # Overlapping read-only calls share one disposable checkout, removed only after the last call ends.
    arrived: list[Path] = []
    both_started = asyncio.Event()

    async def overlapping_exec(*args: Any, **kwargs: Any) -> Any:
        flat = list(args)
        arrived.append(Path(flat[flat.index("--cd") + 1]))
        if len(arrived) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=3)
        return make_mock_process_from_fixture("simple_text.jsonl")

    backend = CodexBackend(model="fixture-model")

    async def audit() -> list[Any]:
        return [event async for event in backend.execute(source, "Audit", read_only=True)]

    with patch("daydream.backends._transport.asyncio.create_subprocess_exec", overlapping_exec):
        await asyncio.gather(audit(), audit())
    assert len(arrived) == 2 and arrived[0] == arrived[1] and not arrived[0].exists()
    assert "clone-only" not in git_ops.list_local_branches(source)
    assert git_ops.head_sha(source) == source_head

    # Without read_only, Codex runs at the worktree itself with full access and no isolation.
    with patch("daydream.backends._transport.asyncio.create_subprocess_exec",
               return_value=make_mock_process_from_fixture("simple_text.jsonl")) as full_access:
        async for _ in CodexBackend(model="fixture-model").execute(source, "p"):
            pass
    flat = list(full_access.call_args.args)
    assert flat[flat.index("--sandbox") + 1] == "danger-full-access" and "read-only" not in flat
    assert flat[flat.index("--cd") + 1] == str(source)


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

# Parser correlation and diagnostics.


async def test_public_overload_after_completion_cannot_replay_completed_work(tmp_path: Path) -> None:
    process = make_mock_process([
        json.dumps({"type": "turn.completed", "usage": {}}),
        json.dumps({"type": "turn.failed", "error": {"message": "busy", "code": "server_overloaded"}}),
    ])
    with pytest.raises(CodexError) as failure:
        await replay_process(CodexBackend(model="fixture-model"), process, tmp_path, "apply the fix")
    assert classify_failure(failure.value).retries_allowed is False


async def test_public_turn_failure_preserves_metadata_and_normalizes_only_overload(tmp_path: Path) -> None:
    """Public turn.failed admits capacity retries only; unknown and permanent failures never retry."""
    capacity = "Selected model is at capacity. Please try a different model."
    cases: list[tuple[dict[str, Any], FailureClass | None]] = [
        ({"message": capacity}, FailureClass.SERVER_ERROR),
        ({"message": "Try again later", "codex_error_info": "server_overloaded"}, FailureClass.SERVER_ERROR),
        ({"message": "Try again later", "code": "server_overloaded"}, FailureClass.SERVER_ERROR),
        ({"message": "Invalid API key", "codex_error_info": "server_overloaded"}, None),
        *[({"message": capacity, "code": code}, None)
          for code in ("unauthorized", "schema_validation", "tool_policy", "unknown_error")],
        ({"message": "The model has capacity planning advice."}, None),
        ({"message": "Try again later", "code": {"server_overloaded": True}}, None),
        ({"message": "authentication required. Run `codex login` to authenticate."}, FailureClass.AUTH_CONFIG),
    ]
    for error, expected_class in cases:
        try:
            process = make_mock_process([json.dumps({"type": "turn.failed", "error": error})])
            backend = CodexBackend(model="fixture-model")
            with pytest.raises(CodexError) as failure:
                await replay_process(backend, process, tmp_path, "review")
            decision = classify_failure(failure.value)
            assert decision.retries_allowed is (expected_class is FailureClass.SERVER_ERROR)
            if expected_class is not None:
                assert decision.failure_class is expected_class
            assert failure.value.error_metadata == error
            assert str(failure.value) == error["message"]
            assert backend._transports == []
        except BaseException as exc:
            exc.add_note(f"case: {error!r}")
            raise


def test_codex_authentication_diagnostic_precedes_transient_retry_hints() -> None:
    decision = classify_failure(CodexError(
        "authentication required. Run codex login to authenticate", retryable=True,
        category="SERVER_ERROR", error_metadata={"codex_error_info": "server_overloaded"},
    ))
    assert decision.failure_class is FailureClass.AUTH_CONFIG and decision.retries_allowed is False


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


    def test_non_executable_target_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(codex.sys, "platform", "darwin")
        proc = subprocess.CompletedProcess(args=[], returncode=0, stdout="/nonexistent/xx/git\n", stderr="")
        monkeypatch.setattr(codex.subprocess, "run", lambda *a, **k: proc)
        assert codex._resolve_real_git_dir() is None  # M2: must be an existing executable file

    def test_non_darwin_never_invokes_xcrun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(codex.sys, "platform", "linux")
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
    assert [json.loads(step["message"]) for step in steps[:-1]] == [json.loads(text) for text in expected_text[:-1]]
    assert steps[-1]["message"] == "Verified"
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
