"""Tests for agent execution and supporting helpers."""

import os
from collections.abc import Callable
from dataclasses import replace as dataclass_replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream import prompt_budget
from daydream.agent import (
    _RedactedSupervisorError,
    _scrubbed_supervisor_error,
    run_agent,
)
from daydream.backends import ResultEvent, ToolStartEvent
from daydream.extensions import ToolDecision, get_registry, set_registry
from daydream.extensions.registry import Registry
from daydream.prompt_budget import (
    SANCTIONED_EXACT_INPUT_AGGREGATE_MAX_BYTES,
    SANCTIONED_EXACT_INPUT_FILE_MAX_BYTES,
    SANCTIONED_EXACT_INPUT_MAX_FILES,
    SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES,
    AdvisoryCandidate,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    SanctionedInputUnavailable,
    prepare_sanctioned_inputs,
    select_advisory_inputs,
)
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend


def _strict_backend(audit_root: Path) -> ScriptedBackend:
    """A backend whose strict PreToolUse isolation forces the inline transport."""
    return ScriptedBackend(audit_root_isolation="claude-pretooluse", audit_root=audit_root)


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
    prompt = prepared.render_prompt("Review src/app.py")
    assert prompt.startswith("Review src/app.py") and prompt.endswith(f"- zeta: {second.resolve()}")
    assert prepared.render_prompt(prompt) == prompt
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
    # Staged review may opt into a bounded larger capture; ordinary callers keep 12 KiB.
    first.write_text('界' * 21000, encoding='utf-8')
    with pytest.raises(SanctionedInputUnavailable, match='byte budget'):
        prepare_sanctioned_inputs(strict, tmp_path, {'required': first}, read_only=True)
    staged = prepare_sanctioned_inputs(strict, tmp_path, {'required': first}, read_only=True,
                                      staged_review=True)
    assert staged.inputs[0].text == '界' * 21000
    staged.revalidate(strict, tmp_path, True)
    assert 12 * 1024 < len(staged.render().encode()) <= 64 * 1024
    first.write_bytes(b'x' * (64 * 1024 + 1))
    with pytest.raises(SanctionedInputUnavailable, match='byte budget'):
        prepare_sanctioned_inputs(strict, tmp_path, {'required': first}, read_only=True,
                                  staged_review=True)


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
    advisory = select_advisory_inputs(backend, tmp_path, [
        AdvisoryCandidate(f"advisory-{i:03d}", same) for i in range(SANCTIONED_EXACT_INPUT_MAX_FILES + 1)],
        read_only=False)
    assert len(advisory.admitted) == SANCTIONED_EXACT_INPUT_MAX_FILES
    assert [item.reason for item in advisory.omitted] == ["exceeds-file-count"]
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SanctionedInputUnavailable, match="audit root"):
        prepare_sanctioned_inputs(_strict_backend(other), tmp_path, {"same": same}, read_only=True)
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    link = tmp_path / "link.txt"
    link.symlink_to(same)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for path, reason in ((invalid, "UTF-8"), (link, "regular file"), (fifo, "regular file")):
        with pytest.raises(SanctionedInputUnavailable, match=reason):
            prepare_sanctioned_inputs(backend, tmp_path, {"input": path}, read_only=False)


def test_revalidate_fails_closed_on_vanished_or_over_budget_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revalidation fails closed: a vanished input is re-captured once and surfaces the real error;
    the unchanged-identity fast path enforces aggregate limits without rereading the file."""
    artifact, backend, prepared = _captured_artifact(tmp_path)
    captured = prepared.inputs[0]
    artifact.unlink()
    capture_calls = _count_capture_calls(monkeypatch)
    with pytest.raises(SanctionedInputUnavailable, match="unavailable"):
        prepared.revalidate(backend, tmp_path, read_only=False)
    assert capture_calls() == 1
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


@pytest.mark.anyio
async def test_run_agent_rejects_same_backend_object_when_transport_mode_changes(tmp_path: Path) -> None:
    artifact, backend, prepared = _captured_artifact(tmp_path, backend=ScriptedBackend(sandbox=False))
    setattr(backend, "sandbox", True)
    with pytest.raises(SanctionedInputUnavailable, match="transport mode changed"):
        await _run_with_inputs(backend, tmp_path, prepared)
    assert backend.call_count == 0


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
    assert "secret-shaped-payload" not in str(raised)
    assert "boom in str" not in str(raised)
    rendered = output.getvalue()
    assert "Extension Failure" in rendered
    assert "secret-shaped-payload" not in rendered
    assert "boom in str" not in rendered
