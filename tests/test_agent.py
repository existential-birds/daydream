"""Tests for agent execution and supporting helpers."""

import os
import sys
from collections.abc import Callable
from dataclasses import replace as dataclass_replace
from io import StringIO
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from rich.console import Console

from daydream import prompt_budget
from daydream.agent import (
    _RedactedSupervisorError,
    _run_agent,
    _scrubbed_supervisor_error,
    _ToolSupervisorFailure,
    is_environmental_failure,
    run_agent,
)
from daydream.backends import DiagnosticEvent, ResultEvent, ToolStartEvent
from daydream.extensions import ToolDecision, get_registry, set_registry
from daydream.extensions.registry import Registry
from daydream.prompt_budget import (
    SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES,
    SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES,
    SANCTIONED_EXACT_INPUT_MAX_FILES,
    SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    SanctionedInputUnavailable,
    prepare_sanctioned_inputs,
)
from daydream.run_context import (
    InteractionPolicy,
    RunContext,
    active_backends,
    current_run_context,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import make_recorder, read_trajectory


def _strict_backend(audit_root: Path) -> ScriptedBackend:
    """A backend whose strict PreToolUse isolation forces the inline transport."""
    return ScriptedBackend(audit_root_isolation="claude-pretooluse", audit_root=audit_root)

def _count_prompt_budget_reads(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count the bytes ``prompt_budget`` actually streams off disk."""
    total = 0
    real_read = os.read
    def counted_read(fd: int, size: int) -> bytes:
        nonlocal total
        chunk = real_read(fd, size)
        total += len(chunk)
        return chunk
    monkeypatch.setattr("daydream.prompt_budget.os.read", counted_read)
    return lambda: total

def _captured_artifact(
    tmp_path: Path, *, backend: ScriptedBackend | None = None
) -> tuple[Path, ScriptedBackend, PreparedSanctionedInputs]:
    """A captured exact input file with its backend and prepared input set."""
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("captured", encoding="utf-8")
    backend = backend or ScriptedBackend()
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"artifact": artifact}, read_only=False)
    return artifact, backend, prepared

def _count_capture_calls(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count how often ``prompt_budget`` captures (re-reads/hashes) an input."""
    total = 0
    real_capture = prompt_budget._capture_input
    def counted_capture(*args: object, **kwargs: object) -> object:
        nonlocal total
        total += 1
        return real_capture(*args, **kwargs)  # type: ignore[arg-type]
    monkeypatch.setattr(prompt_budget, "_capture_input", counted_capture)
    return lambda: total

async def _run_with_inputs(backend: Any, root: Path, prepared: Any) -> Any:
    """Enter ``run_agent`` with *prepared* sanctioned inputs attached."""
    return await run_agent(backend, root, "inspect", phase=DaydreamPhase.REVIEW, sanctioned_inputs=prepared)

@pytest.mark.anyio
async def test_run_agent_binds_context_and_keeps_backend_registered_across_retry(tmp_path: Path) -> None:
    context = RunContext(InteractionPolicy(quiet=True))
    observations: list[tuple[RunContext | None, tuple[object, ...], tuple[object, ...]]] = []
    class RetryableFailure(RuntimeError):
        retryable = True
    def respond(*_args: Any, **_kwargs: Any) -> Any:
        observations.append((current_run_context(), context.active_backends(), active_backends()))
        if backend.call_count == 1:
            return [RetryableFailure("retry")]
        return None
    backend = ScriptedBackend(
        responder=respond, retry_attempts=1, retry_base_delay_s=0, retry_max_delay_s=0,
    )
    assert current_run_context() is None
    result = await run_agent(backend, tmp_path, "inspect", phase=DaydreamPhase.REVIEW, run_context=context)
    assert result == ("", None, None)
    assert observations == [(context, (backend,), (backend,)), (context, (backend,), (backend,))]
    assert context.active_backends() == ()
    assert active_backends() == ()
    assert current_run_context() is None

@pytest.mark.anyio
async def test_run_agent_unregisters_backend_after_exception(tmp_path: Path) -> None:
    context = RunContext(InteractionPolicy(quiet=True))
    backend = ScriptedBackend(events=[RuntimeError("boom")], retry_attempts=0)
    with pytest.raises(RuntimeError, match="boom"):
        await run_agent(backend, tmp_path, "inspect", phase=DaydreamPhase.REVIEW, run_context=context)
    assert context.active_backends() == ()
    assert active_backends() == ()

async def test_run_agent_interrupt_after_registration_cleans_up(tmp_path: Path) -> None:
    class InjectedInterrupt(BaseException):
        pass
    context = RunContext(InteractionPolicy())
    backend = ScriptedBackend()
    previous_trace = sys.gettrace()
    def interrupt(frame: FrameType, event: str, _arg: Any) -> Any:
        if (
            event == "line"
            and frame.f_code is _run_agent.__code__
            and context.active_backends()
        ):
            raise InjectedInterrupt("after registration")
        return interrupt
    try:
        sys.settrace(interrupt)
        with pytest.raises(InjectedInterrupt) as caught:
            await run_agent(backend, tmp_path, "inspect", phase=DaydreamPhase.REVIEW, run_context=context)
    finally:
        sys.settrace(previous_trace)
    # Keep the traceback alive: cleanup must not depend on collecting frames.
    assert caught.value.__traceback__ is not None
    assert context.active_backends() == ()
    assert active_backends() == ()
    assert current_run_context() is None

def _sized_inputs(tmp_path: Path, count: int, size: int) -> dict[str, Path]:
    """*count* real files of exactly *size* bytes, labelled ``input-<n>``."""
    inputs: dict[str, Path] = {}
    for index in range(count):
        path = tmp_path / f"input-{index}.txt"
        path.write_bytes(b"x" * size)
        inputs[f"input-{index}"] = path
    return inputs

def test_prepare_sanctioned_inputs_selects_transport_and_enforces_aggregate_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("x" * (SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES - 2), encoding="utf-8")
    second.write_text("¢", encoding="utf-8")
    ordinary = ScriptedBackend()
    prepared = prepare_sanctioned_inputs(ordinary, tmp_path, {"zeta": second, "alpha": first}, read_only=False)
    assert prepared.transport is SanctionedInputTransport.EXACT_PATHS
    assert [item.label for item in prepared.inputs] == ["alpha", "zeta"]
    assert all(item.text is None for item in prepared.inputs)
    assert str(first.resolve()) in prepared.render()
    assert "x" * 100 not in prepared.render()
    strict = _strict_backend(tmp_path.resolve())
    inline = prepare_sanctioned_inputs(strict, tmp_path, {"alpha": first, "zeta": second}, read_only=True)
    assert inline.transport is SanctionedInputTransport.INLINE
    assert "x" * 100 in inline.render()
    assert str(first.resolve()) not in inline.render()
    assert str(first.resolve()) not in inline.render_prompt(f"Read {first.resolve()}")
    assert "sanctioned input 'alpha'" in inline.render_prompt(f"Read {first.resolve()}")
    # One byte over the inline aggregate: exact paths still admit it, inline refuses.
    second.write_text("¢x", encoding="utf-8")
    over = prepare_sanctioned_inputs(ordinary, tmp_path, {"alpha": first, "zeta": second}, read_only=False)
    assert over.transport is SanctionedInputTransport.EXACT_PATHS
    with pytest.raises(SanctionedInputUnavailable, match="byte budget"):
        prepare_sanctioned_inputs(strict, tmp_path, {"alpha": first, "zeta": second}, read_only=True)

def test_prepare_sanctioned_inputs_rejects_mismatched_strict_audit_root(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("captured", encoding="utf-8")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SanctionedInputUnavailable, match="audit root"):
        prepare_sanctioned_inputs(_strict_backend(other), tmp_path, {"artifact": artifact}, read_only=True)

def test_prepare_sanctioned_exact_inputs_enforces_resource_caps(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    maximum = tmp_path / "maximum.txt"
    maximum.write_bytes(b"x" * SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES)
    admitted = prepare_sanctioned_inputs(backend, tmp_path, {"maximum": maximum}, read_only=False)
    assert admitted.inputs[0].size == SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES
    maximum.write_bytes(b"x" * (SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES + 1))
    with pytest.raises(SanctionedInputUnavailable, match="file byte limit"):
        prepare_sanctioned_inputs(backend, tmp_path, {"maximum": maximum}, read_only=False)
    count = SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES // SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES + 1
    size, remainder = divmod(SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES, count)
    files = _sized_inputs(tmp_path, count, size)
    files[f"input-{count - 1}"].write_bytes(b"x" * (size + remainder))
    boundary = prepare_sanctioned_inputs(backend, tmp_path, files, read_only=False)
    assert len(boundary.inputs) == count
    assert sum(item.size for item in boundary.inputs) == SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES
    assert all(item.size < SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES for item in boundary.inputs)
    files["overflow"] = tmp_path / "overflow.txt"
    files["overflow"].write_text("x", encoding="utf-8")
    with pytest.raises(SanctionedInputUnavailable, match="aggregate byte limit"):
        prepare_sanctioned_inputs(backend, tmp_path, files, read_only=False)
    same = tmp_path / "same.txt"
    same.write_text("x", encoding="utf-8")
    too_many = {f"input-{i:03d}": same for i in range(SANCTIONED_EXACT_INPUT_MAX_FILES + 1)}
    with pytest.raises(SanctionedInputUnavailable, match="file count"):
        prepare_sanctioned_inputs(backend, tmp_path, too_many, read_only=False)

def test_prepare_sanctioned_exact_inputs_bounds_aggregate_streaming_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aggregate refusal reads only the admitted bytes plus one look-ahead."""
    backend = ScriptedBackend()
    count = SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES // SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES + 1
    inputs = _sized_inputs(tmp_path, count, SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES)
    delegated_bytes = _count_prompt_budget_reads(monkeypatch)
    with pytest.raises(SanctionedInputUnavailable, match="aggregate byte limit"):
        prepare_sanctioned_inputs(backend, tmp_path, inputs, read_only=False)
    assert delegated_bytes() <= SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES + 1
    assert backend.call_count == 0

def test_revalidate_sanctioned_exact_inputs_keeps_aggregate_read_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = ScriptedBackend()
    count = SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES // SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES + 1
    inputs = _sized_inputs(tmp_path, count, SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES // count)
    prepared = prepare_sanctioned_inputs(backend, tmp_path, inputs, read_only=False)
    inputs[f"input-{count - 1}"].write_bytes(b"x" * SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES)
    delegated_bytes = _count_prompt_budget_reads(monkeypatch)
    with pytest.raises(SanctionedInputUnavailable, match="aggregate byte limit"):
        prepared.revalidate(backend, tmp_path, read_only=False)
    assert delegated_bytes() <= SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES + 1
    assert backend.call_count == 0

def test_revalidate_skips_rehash_when_identity_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Matching (dev, ino, size, mtime_ns) reuses already captured bytes without rereading or hashing."""
    artifact, backend, prepared = _captured_artifact(tmp_path)
    capture_calls = _count_capture_calls(monkeypatch)
    # Unchanged file: identity matches, so no re-read/re-hash happens.
    prepared.revalidate(backend, tmp_path, read_only=False)
    assert capture_calls() == 0
    # Touched file (size changed): the full capture runs and fails closed.
    artifact.write_text("captured but longer now", encoding="utf-8")
    with pytest.raises(SanctionedInputUnavailable, match="changed"):
        prepared.revalidate(backend, tmp_path, read_only=False)
    assert capture_calls() == 1
    # Same-length rewrite keeps size but advances mtime: still re-captured.
    baseline = capture_calls()
    os.utime(artifact, ns=(1_000_000_000, 1_000_000_000))
    with pytest.raises(SanctionedInputUnavailable, match="changed"):
        prepared.revalidate(backend, tmp_path, read_only=False)
    assert capture_calls() == baseline + 1

def test_revalidate_fails_closed_when_captured_file_vanishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing captured input is re-captured and surfaces the real error."""
    artifact, backend, prepared = _captured_artifact(tmp_path)
    artifact.unlink()
    capture_calls = _count_capture_calls(monkeypatch)
    with pytest.raises(SanctionedInputUnavailable, match="unavailable"):
        prepared.revalidate(backend, tmp_path, read_only=False)
    assert capture_calls() == 1

def test_revalidate_unchanged_item_exceeding_remaining_budget_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unchanged-identity fast path enforces aggregate limits without rereading the file."""
    artifact, backend, prepared = _captured_artifact(tmp_path)
    captured = prepared.inputs[0]
    def failing_capture(*args: object, **kwargs: object) -> object:
        raise AssertionError("over-budget item must fail closed without re-capture")
    def always_unchanged(item: object) -> bool:
        return True
    monkeypatch.setattr(prompt_budget, "_capture_input", failing_capture)
    monkeypatch.setattr(prompt_budget, "_unchanged_since_capture", always_unchanged)
    over_budget = dataclass_replace(captured, size=prompt_budget.SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES + 1)
    exhausted = dataclass_replace(prepared, inputs=(over_budget,))
    with pytest.raises(SanctionedInputUnavailable, match="aggregate input budget"):
        exhausted.revalidate(backend, tmp_path, read_only=False)

def test_prepare_sanctioned_inputs_rejects_invalid_utf8_and_symlinks(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    regular = tmp_path / "regular.txt"
    regular.write_text("safe", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(regular)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for path, reason in ((invalid, "UTF-8"), (link, "regular file"), (fifo, "regular file")):
        with pytest.raises(SanctionedInputUnavailable, match=reason):
            prepare_sanctioned_inputs(ScriptedBackend(), tmp_path, {"input": path}, read_only=False)

@pytest.mark.anyio
async def test_run_agent_revalidates_sanctioned_inputs_before_backend_entry(tmp_path: Path) -> None:
    artifact, backend, prepared = _captured_artifact(tmp_path)
    artifact.write_text("mutated", encoding="utf-8")
    with pytest.raises(SanctionedInputUnavailable, match="changed"):
        await _run_with_inputs(backend, tmp_path, prepared)
    assert backend.call_count == 0

@pytest.mark.anyio
async def test_run_agent_revalidates_sanctioned_input_before_retry(tmp_path: Path) -> None:
    class RetryableFailure(RuntimeError):
        retryable = True
    class MutatingBackend(ScriptedBackend):
        async def execute(self, *args: Any, **kwargs: Any) -> Any:
            async for event in super().execute(*args, **kwargs):
                if self.call_count == 1:
                    artifact.write_text("mutated", encoding="utf-8")
                    raise RetryableFailure("retry")
                yield event
    artifact, backend, prepared = _captured_artifact(
        tmp_path, backend=MutatingBackend(retry_attempts=1, retry_base_delay_s=0, retry_max_delay_s=0),
    )
    with pytest.raises(SanctionedInputUnavailable, match="changed"):
        await _run_with_inputs(backend, tmp_path, prepared)
    assert backend.call_count == 1

@pytest.mark.anyio
async def test_run_agent_rejects_same_backend_object_when_transport_mode_changes(tmp_path: Path) -> None:
    artifact, backend, prepared = _captured_artifact(tmp_path, backend=ScriptedBackend(sandbox=False))
    setattr(backend, "sandbox", True)
    with pytest.raises(SanctionedInputUnavailable, match="transport mode changed"):
        await _run_with_inputs(backend, tmp_path, prepared)
    assert backend.call_count == 0

def test_is_environmental_failure_both_directions() -> None:
    environmental = [
        "The dev Postgres container is not running", "could not connect to server: Connection refused",
        "localhost:5432", "ECONNREFUSED",
    ]
    for output in environmental:
        assert is_environmental_failure(output) is True, output
    ordinary = ["AssertionError: assert 1 == 2", "1 failed, 3 passed", "ValueError: bad input"]
    for output in ordinary:
        assert is_environmental_failure(output) is False, output

def test_scrubbed_supervisor_error_scrubs_all_str_surfaces() -> None:
    """Scrub OSError fields and custom str/repr surfaces while preserving retryability."""
    credential = "ZAI_API_KEY=credential-shaped-supervisor-value"
    # OSError-family: real (sub)type preserved, str() scrubbed
    err = OSError(2, f"failed auth with {credential}")
    rebuilt = _scrubbed_supervisor_error(err)
    assert type(rebuilt) is type(err)
    assert isinstance(rebuilt, OSError)
    assert credential not in str(rebuilt)
    assert "[REDACTED_ENV_VAR]" in str(rebuilt)
    # custom __str__ override: fail closed to the already-redacted stand-in
    class CustomLeak(Exception):
        def __str__(self) -> str:
            return f"boom {self.args}"
    custom = CustomLeak((credential,))
    stand_in = _scrubbed_supervisor_error(custom)
    assert type(stand_in) is _RedactedSupervisorError
    assert credential not in str(stand_in)
    assert stand_in.original_type_name == "CustomLeak"
    # reconstruction path must preserve retryable even when it is an
    # instance attribute set from a non-args kwarg (e.g. BackendError).
    class RetryableBackendError(RuntimeError):
        def __init__(self, message: str, *, retryable: bool = False) -> None:
            super().__init__(message)
            self.retryable = retryable
    backend = RetryableBackendError(f"boom {credential}", retryable=True)
    rebuilt_retryable = _scrubbed_supervisor_error(backend)
    assert type(rebuilt_retryable) is RetryableBackendError
    assert credential not in str(rebuilt_retryable)
    assert getattr(rebuilt_retryable, "retryable", False) is True

class ExplodingStrError(RuntimeError):
    """A hostile supervisor exception: its ``__str__`` itself raises."""

    def __str__(self) -> str:
        raise RuntimeError("boom in str")

def test_scrubbed_supervisor_error_hostile_str_fails_closed() -> None:
    """A hostile __str__ must not escape during either wrapper construction or scrubbing."""
    # Scrubber: no raise; fail-closed stand-in carrying only the type name.
    stand_in = _scrubbed_supervisor_error(ExplodingStrError("secret-shaped-payload"))
    assert type(stand_in) is _RedactedSupervisorError
    assert stand_in.original_type_name == "ExplodingStrError"
    assert "secret-shaped-payload" not in str(stand_in)
    assert "boom in str" not in str(stand_in)
    # Wrapper: construction never evaluates the hostile __str__.
    original = ExplodingStrError("secret-shaped-payload")
    failure = _ToolSupervisorFailure(original)
    assert failure.original is original
    assert str(failure) == ""

@pytest.mark.anyio
async def test_hostile_supervisor_str_surfaces_as_scrubbed_extension_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = StringIO()
    monkeypatch.setattr("daydream.agent.console", Console(file=output, force_terminal=False))
    def hostile_supervisor(
        tool_name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase
    ) -> ToolDecision:
        raise ExplodingStrError("secret-shaped-payload")
    registry = Registry()
    registry.register_tool_supervisor(hostile_supervisor)
    previous_registry = get_registry()
    set_registry(registry)
    try:
        with pytest.raises(_RedactedSupervisorError) as excinfo:
            await run_agent(
                ScriptedBackend(
                    events=[
                        ToolStartEvent(id="t1", name="bash", input={"command": "opaque-not-included"}),
                        ResultEvent(structured_output=None, continuation=None),
                    ]
                ), tmp_path, "inspect", phase=DaydreamPhase.REVIEW,
            )
        raised = excinfo.value
    finally:
        set_registry(previous_registry)
    assert type(raised) is _RedactedSupervisorError
    assert raised.original_type_name == "ExplodingStrError"
    rendered = output.getvalue()
    assert "Extension Failure" in rendered
    assert "secret-shaped-payload" not in rendered
    assert "boom in str" not in rendered

@pytest.mark.anyio
async def test_diagnostic_event_is_recorder_only_and_has_no_agent_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_agent forwards diagnostics without UI, callback, supervision, or budget effects."""
    output = StringIO()
    monkeypatch.setattr("daydream.agent.console", Console(file=output, force_terminal=False))
    callback_events: list[object] = []
    supervisor_events: list[tuple[str, dict[str, Any]]] = []
    def callback(value: object) -> None:
        callback_events.append(value)
    def supervisor(
        tool_name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase
    ) -> ToolDecision:
        supervisor_events.append((tool_name, tool_input))
        return ToolDecision(False, "")
    registry = Registry()
    registry.register_tool_supervisor(supervisor)
    previous_registry = get_registry()
    set_registry(registry)
    recorder = make_recorder(tmp_path)
    try:
        async with recorder:
            result = await run_agent(
                ScriptedBackend(
                    events=[
                        DiagnosticEvent(
                            code="codex_parser_coverage", message="bounded parser evidence", metadata={"count": 1},
                        ), ResultEvent(structured_output=None, continuation=None),
                    ]
                ), tmp_path, "inspect", phase=DaydreamPhase.REVIEW, progress_callback=callback, tool_call_budget=0,
            )
    finally:
        set_registry(previous_registry)
    assert result == ("", None, None)
    assert callback_events == []
    assert supervisor_events == []
    assert output.getvalue() == ""
    trajectory = read_trajectory(recorder.path)
    agent_steps = [step for step in trajectory["steps"] if step["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == ""
    assert agent_steps[0]["extra"]["backend_diagnostics"] == [
        {"code": "codex_parser_coverage", "message": "bounded parser evidence", "metadata": {"count": 1}}
    ]
