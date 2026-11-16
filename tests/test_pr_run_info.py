"""Tests for live, session-owned PR run-info acquisition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import anyio
import pytest

import daydream.pr_run_info as provider
from daydream.artifact_visibility import (
    ArtifactSession,
)
from daydream.backends import MetricsEvent
from daydream.pr_comment_renderer import FALLBACK_NOTE
from daydream.pr_run_info import (
    LiveRunInfoSource,
    render_live_run_info,
)
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    DaydreamPhase,
    EvidenceRecorder,
    TrajectoryDocumentReference,
)
from daydream.trajectory.recorder import _SignalFlushRegistry
from tests.harness.trajectory import evidence_document, make_recorder, observe_text_and_result


def _steps(
    *, phase: str = "review", model: str = "gpt-5.5", prompt_tokens: int = 100, cost_usd: float | None = 0.01,
) -> tuple[dict[str, Any], ...]:
    metrics: dict[str, Any] = {"prompt_tokens": prompt_tokens, "completion_tokens": 10, "cached_tokens": 20}
    if cost_usd is not None:
        metrics["cost_usd"] = cost_usd
    return ({"step_id": 1, "timestamp": "2026-09-11T00:00:00Z", "source": "user", "message": "review",
             "extra": {"daydream_phase": phase}},
            {"step_id": 2, "timestamp": "2026-09-11T00:00:01Z", "source": "agent", "message": "done",
             "model_name": model, "metrics": metrics, "extra": {"daydream_phase": phase}})


def _document(session_id: str, trajectory_id: str, *, phase: str = "review") -> bytes:
    return evidence_document(trajectory_id, session_id=session_id, steps=_steps(phase=phase))


def _recorder(tmp_path: Path, session_id: str = "run-info", *, model: str = "gpt-5.5", prompt_tokens: int = 100,
    cost_usd: float | None = 0.01,
) -> EvidenceRecorder:
    """A root recorder whose evidence document holds one recorded review turn."""
    recorder = make_recorder(tmp_path, path=tmp_path / RUN_DOCUMENT_NAME, agent_model_name=model, session_id=session_id,
    )

    async def record() -> None:
        async with recorder, recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step("review")
            invocation.observe(MetricsEvent(prompt_tokens=prompt_tokens, completion_tokens=10, cached_tokens=20,
                                            cost_usd=cost_usd, message_id="m1", model_name=model))
            observe_text_and_result(invocation, "done")

    anyio.run(record)
    return recorder

@pytest.mark.parametrize(("source", "diagnostic"),
    [
        (LiveRunInfoSource(recorder=make_recorder(Path(".")), artifacts=None),
         "run info: artifact session unavailable"),
    ],
)
def test_missing_live_source_returns_bounded_unavailable_result(source: LiveRunInfoSource, diagnostic: str,) -> None:
    result = render_live_run_info(source)

    assert FALLBACK_NOTE in result.markdown
    assert result.diagnostic == diagnostic

def _source(
    tmp_path: Path, children: tuple[tuple[str, bytes], ...] = (), *, recorder: EvidenceRecorder | None = None,
) -> LiveRunInfoSource:
    """Retain real written child files as the active recorder registry's references."""
    recorder = recorder or _recorder(tmp_path)
    registry = _SignalFlushRegistry()
    for index, (trajectory_id, content) in enumerate(children):
        path = tmp_path / f"child-{index}.json"
        path.write_bytes(content)
        registry.retain(TrajectoryDocumentReference.written(trajectory_id, path, path, content))
    recorder._signal_registry = registry
    return LiveRunInfoSource(recorder=recorder, artifacts=cast(ArtifactSession, object()))

def test_snapshot_failure_returns_fixed_diagnostic_without_exception_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "private/session/path/and-bad-json"
    source = _source(tmp_path)

    def unavailable() -> None:
        raise RuntimeError(secret)

    monkeypatch.setattr(source.recorder, "completed_child_documents", unavailable)
    result = render_live_run_info(source)

    assert result.diagnostic == "run info: trajectory snapshot unavailable"
    assert secret not in result.markdown
    assert secret not in (result.diagnostic or "")

def test_non_root_recorder_is_rejected_before_acquisition(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path)
    recorder._trajectory_id = "child-recorder"
    result = render_live_run_info(_source(tmp_path, recorder=recorder))

    assert result.diagnostic == "run info: trajectory identity invalid"


@pytest.mark.parametrize(("snapshot_id", "document_session", "document_id"),
    [("child", "other-session", "child"), ("child", "run-info", "other-child"), ],
)
def test_invalid_sibling_identity_returns_unavailable(
    tmp_path: Path, snapshot_id: str, document_session: str, document_id: str,
) -> None:
    document = _document(document_session, document_id)
    result = render_live_run_info(_source(tmp_path, ((snapshot_id, document),)))

    assert result.diagnostic == "run info: trajectory identity invalid"
    assert FALLBACK_NOTE in result.markdown

def test_malformed_sibling_and_duplicate_ids_return_unavailable(tmp_path: Path) -> None:
    malformed_result = render_live_run_info(_source(tmp_path, (("child", b"{bad json"),)))
    # A retained child claiming the root's identity duplicates the parent.
    root_copy = _document("run-info", "run-info")
    duplicate_result = render_live_run_info(_source(tmp_path, (("run-info", root_copy),)))

    assert malformed_result.diagnostic == "run info: trajectory document invalid"
    assert duplicate_result.diagnostic == "run info: trajectory identity invalid"

@pytest.mark.parametrize(("operation", "message"), [
    ("render_run_info", "secret renderer failure"),
    ("load_user_prices", "private prices path"),
])
def test_pricing_or_render_failure_returns_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str, message: str,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(message)

    monkeypatch.setattr(provider, operation, fail)
    source = _source(tmp_path)
    result = render_live_run_info(source)

    assert result.diagnostic == "run info: rendering unavailable"
    assert FALLBACK_NOTE in result.markdown


def test_live_provider_renders_parent_and_completed_sibling_with_user_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A completed fix-phase sibling joins the parent's rows, and user price overrides synthesize missing cost.

    A child whose file changed after it was written is omitted with a warning rather than blocking the post."""
    custom_model = "custom-live-model"
    prices_file = tmp_path / "prices.toml"
    prices_file.write_text(f'[prices."{custom_model}"]\ninput = 2.0\ncached_input = 0.5\noutput = 8.0\n',
                           encoding="utf-8")
    monkeypatch.setenv("DAYDREAM_PRICES_FILE", str(prices_file))
    child = _document("run-info", "child", phase="fix")
    changed = _document("run-info", "changed", phase="fix")
    recorder = _recorder(tmp_path, model=custom_model, prompt_tokens=1_000_000, cost_usd=None)
    source = _source(tmp_path, (("child", child), ("changed", changed)), recorder=recorder)
    (tmp_path / "child-1.json").write_bytes(changed.replace(b'"fix"', b'"FIX"'))
    result = render_live_run_info(source)
    assert "TrajectoryDocumentIntegrityError" in capsys.readouterr().out
    assert result.diagnostic is None and FALLBACK_NOTE not in result.markdown
    assert "| Review |" in result.markdown and "| Fix |" in result.markdown
    assert "- **Steps / tool calls:** 2 / 0" in result.markdown
    assert "- **Cost:** $2.01" in result.markdown
