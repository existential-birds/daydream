"""Tests for the shared ``ScriptedBackend`` harness.

The harness is depended on by ~30 test modules, so its own semantics — turn
sequencing, last-turn repeat, mid-stream raise, argument recording — are pinned
here rather than left to be inferred from its consumers.
"""

from __future__ import annotations

import ast
import inspect
from importlib import import_module
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.agent import run_agent
from daydream.backends import AgentEvent, Backend, ResultEvent, TextEvent
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend

# The shared ``tests/harness`` backend fakes, by (label, module, class). Every
# fake the suite constructs directly or installs through ``create_backend`` is
# listed here so the protocol-parity check below covers the whole harness.
_SHARED_HARNESS_BACKENDS = (
    ("ScriptedBackend", "tests.harness.backend", "ScriptedBackend"),
    ("MockBackend", "tests.harness.stub_backend", "MockBackend"),
    ("StubBackend", "tests.harness.stub_backend", "StubBackend"),
    ("PhaseDispatchBackend", "tests.harness.phase_backend", "PhaseDispatchBackend"),
    ("ImproveStubBackend", "tests.harness.improve_backend", "ImproveStubBackend"),
    ("AuditAbsoluteWorkingDirectoryBackend", "tests.harness.improve_backend", "AuditAbsoluteWorkingDirectoryBackend"),
    ("ProductionPathBackend", "tests.harness.improve_backend", "ProductionPathBackend"),
    ("IncrementalPlanBackend", "tests.harness.improve_backend", "IncrementalPlanBackend"),
    ("OutOfOrderPlanBackend", "tests.harness.improve_backend", "OutOfOrderPlanBackend"),
)


_PROTOCOL_PARAMS = (
    "cwd",
    "prompt",
    "output_schema",
    "continuation",
    "agents",
    "max_turns",
    "read_only",
    "persist_session",
)
_HARNESS_DIR = Path(__file__).resolve().parent
_TESTS_ROOT = _HARNESS_DIR.parent

# Ratchet: the protocol-shaped execute() declarations measured outside tests/harness/
# at 0030596a (58 entries; the plan's 60 predates the #1247 dead-code sweep). Every
# migration task deletes its own entries; the guard fails on a stale entry as well as
# an unmigrated one, so this set must end up exactly empty.
_ALLOWED: frozenset[str] = frozenset({
    "deep_orchestrator/support.py::_RejectingArbiterBackend",
    "deep_orchestrator/test_related_regression_commit.py::FootprintBackend",
    "test_agent_budget.py::_BurstBackend",
    "test_agent_budget.py::_RaisingCloseBackend",
    "test_agent_budget.py::_RetryableFailingBackend",
    "test_agent_budget.py::_RetryableThenSucceedingBackend",
    "test_agent_budget.py::_SharedBackend",
    "test_agent_recorder_integration.py::MaxTurnsBackend",
    "test_agent_retry.py::_SharedBackend",
    "test_arbiter_prose_extraction.py::_PiLikeBackend",
    "test_arbiter_prose_extraction.py::_SplitTextBackend",
    "test_archive_data_capture.py::MalformedToolBackend",
    "test_archive_data_capture.py::_ArchiveCaptureBackend",
    "test_archive_data_capture.py::_CodexEvidenceBackend",
    "test_archive_data_capture.py::_FixEditingBackend",
    "test_archive_data_capture.py::_JoinedArtifactEvidenceBackend",
    "test_archive_integration.py::_SecretFailureBackend",
    "test_deep_fanout.py::_FlakyBackend",
    "test_deep_integration.py::_DeepMockBackend",
    "test_deep_merge_recovery.py::_MergeTextBackend",
    "test_deep_orchestrator.py::_ExtraEditBackend",
    "test_deep_orchestrator.py::_PromptHookStub",
    "test_deep_wonder_concurrency.py::_WonderRendezvousStub",
    "test_exploration_runner.py::_FailingPatternScanner",
    "test_exploration_runner.py::_FailingSurveyBackend",
    "test_exploration_runner.py::_PartlyBlockedBackend",
    "test_exploration_runner.py::_SpecialistMockBackend",
    "test_extension_seam_integration.py::DeferredWriteBackend",
    "test_extension_seam_integration.py::ShallowRecordingBackend",
    "test_findings.py::ErroringBackend",
    "test_fix_footprint.py::BarrierBackend",
    "test_fix_footprint.py::CancelBackend",
    "test_github_app_integration.py::_MinimalBackend",
    "test_improve_flow.py::BlockingAuditBackend",
    "test_improve_flow.py::BlockingBackend",
    "test_improve_flow.py::_AuditCommittingBackend",
    "test_improve_flow.py::_AuditGitBoundaryBackend",
    "test_improve_flow.py::_DirectoryToFileBackend",
    "test_improve_flow.py::_SymlinkGuardBackend",
    "test_improve_flow.py::_UnbornAuditBackend",
    "test_integration.py::RelatedOnlyBackend",
    "test_integration.py::ReviewStagingBackend",
    "test_integration.py::_WorktreeMutatingBackend",
    "test_integration_workspace.py::MockBackend",
    "test_multi_turn_tokens.py::_CostOnlyBackend",
    "test_multi_turn_tokens.py::_MetricsAndCostBackend",
    "test_multi_turn_tokens.py::_MetricsOnlyBackend",
    "test_multi_turn_tokens.py::_PiShapedBackend",
    "test_phase_parse_input_path.py::_SpyBackend",
    "test_phases.py::_FallbackThenFailureBackend",
    "test_phases.py::_HostCommitBackend",
    "test_phases.py::_OneFixFailsBackend",
    "test_phases_render.py::_PerStackBackend",
    "test_runner.py::InitialExplorationBarrierBackend",
    "test_runner.py::_CommitWritingBackend",
    "test_summarize.py::_MultiTurn",
    "test_trajectory_phase_events.py::_OverlappingReviewBackend",
    "test_worktree_cwd_grounding.py::_PromptCapturingBackend",
})


def _protocol_shaped_declarations() -> set[str]:
    """Every class outside tests/harness/ whose execute() re-types >=5 protocol parameters."""
    found: set[str] = set()
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        if _HARNESS_DIR in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name == "execute":
                    names = {a.arg for a in (*member.args.posonlyargs, *member.args.args, *member.args.kwonlyargs)}
                    if len(names & set(_PROTOCOL_PARAMS)) >= 5:
                        found.add(f"{path.relative_to(_TESTS_ROOT)}::{node.name}")
    return found


def test_no_module_outside_the_harness_declares_the_protocol_execute() -> None:
    """The acceptance criterion: one declaration of the protocol signature, in the harness."""
    violations = _protocol_shaped_declarations()

    assert violations - _ALLOWED == set(), (
        "these declarations re-type the Backend protocol's execute() outside tests/harness/ — "
        f"migrate them or narrow them to a forwarding override: {sorted(violations - _ALLOWED)}"
    )
    assert _ALLOWED - violations == set(), (
        f"stale ratchet allowlist entries (already migrated) — delete them: {sorted(_ALLOWED - violations)}"
    )


def test_shared_harness_backends_satisfy_the_backend_protocol_signature() -> None:
    """A harness fake that drops or renames a protocol parameter silently under-exercises it."""
    def _params(func: Any) -> dict[str, tuple[Any, Any]]:
        return {
            name: (param.kind, param.default)
            for name, param in inspect.signature(func).parameters.items()
            if name != "self"
        }

    expected = _params(Backend.execute)
    for label, module_name, class_name in _SHARED_HARNESS_BACKENDS:
        declared = _params(getattr(import_module(module_name), class_name).execute)
        missing = [name for name in expected if name not in declared]
        extra = [name for name in declared if name not in expected]
        drift = [name for name in expected if name in declared and declared[name] != expected[name]]
        order = [name for name in declared if name in expected]
        assert order == [name for name in expected if name in declared], (
            f"{label}.execute declares the protocol parameters out of order"
        )
        assert not (missing or extra or drift), (
            f"{label}.execute drifted from daydream.backends.Backend.execute: "
            f"missing={missing} extra={extra} kind/default drift={drift}"
        )


async def _drain(backend: ScriptedBackend, prompt: str = "go", **kwargs: Any) -> list[AgentEvent]:
    return [event async for event in backend.execute(Path("/tmp"), prompt, **kwargs)]


def _texts(events: list[AgentEvent]) -> list[str]:
    return [event.text for event in events if isinstance(event, TextEvent)]


@pytest.mark.asyncio
async def test_turns_advance_then_the_last_turn_repeats() -> None:
    """Each call consumes the next turn; calls past the script re-serve the final one."""
    backend = ScriptedBackend(
        script=[
            [TextEvent(text="first")],
            [TextEvent(text="second")],
        ]
    )

    assert _texts(await _drain(backend)) == ["first"]
    assert _texts(await _drain(backend)) == ["second"]
    assert _texts(await _drain(backend)) == ["second"]
    assert backend.call_count == 3


@pytest.mark.asyncio
async def test_an_exception_in_a_turn_raises_after_earlier_events_are_yielded() -> None:
    """A turn can emit partial output and then fail — the retry path's real shape."""
    backend = ScriptedBackend(
        script=[
            [TextEvent(text="partial"), RuntimeError("boom"), TextEvent(text="unreached")],
            [TextEvent(text="recovered")],
        ]
    )

    seen: list[AgentEvent] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for event in backend.execute(Path("/tmp"), "go"):
            seen.append(event)

    assert _texts(seen) == ["partial"], "events before the exception must still reach the consumer"
    assert _texts(await _drain(backend)) == ["recovered"]


def test_script_and_events_together_is_rejected() -> None:
    """Two sources of truth for the script would silently drop one."""
    with pytest.raises(ValueError, match="not both"):
        ScriptedBackend(script=[[TextEvent(text="a")]], events=[TextEvent(text="b")])


@pytest.mark.asyncio
async def test_responses_are_selected_by_output_schema_regardless_of_call_order() -> None:
    """A parallel fan-out completes in any order; the schema key still picks the right turn."""
    schema_a = {"type": "object", "title": "a"}
    schema_b = {"type": "object", "title": "b"}
    backend = ScriptedBackend(
        responses_by_schema=[(schema_a, [TextEvent(text="A")]), (schema_b, [TextEvent(text="B")])],
        events=[TextEvent(text="script")],
    )

    assert _texts(await _drain(backend, output_schema=schema_b)) == ["B"]
    assert _texts(await _drain(backend, output_schema=schema_a)) == ["A"]
    assert _texts(await _drain(backend, output_schema={"type": "object", "title": "z"})) == ["script"]


@pytest.mark.asyncio
async def test_a_none_schema_pair_is_the_fallback_for_an_unmatched_schema() -> None:
    backend = ScriptedBackend(
        responses_by_schema=[({"title": "a"}, [TextEvent(text="A")]), (None, [TextEvent(text="fallback")])],
        events=[TextEvent(text="script")],
    )

    assert _texts(await _drain(backend, output_schema={"title": "a"})) == ["A"]
    assert _texts(await _drain(backend, output_schema=None)) == ["fallback"]
    assert _texts(await _drain(backend, output_schema={"title": "z"})) == ["fallback"]


@pytest.mark.asyncio
async def test_a_responder_turn_replaces_the_script_and_raises_mid_stream() -> None:
    """A prompt-conditional failure keeps its old timing: after earlier events are yielded."""
    def responder(cwd: Any, prompt: str, output_schema: Any = None, continuation: Any = None, agents: Any = None,
                  max_turns: Any = None, read_only: Any = False, persist_session: Any = True) -> Any:
        if "react" in prompt:
            return [TextEvent(text="partial"), RuntimeError("react failed")]
        return None

    backend = ScriptedBackend(events=[TextEvent(text="scripted")], responder=responder)

    seen: list[AgentEvent] = []
    with pytest.raises(RuntimeError, match="react failed"):
        async for event in backend.execute(Path("/tmp"), "react please"):
            seen.append(event)
    assert _texts(seen) == ["partial"]
    assert _texts(await _drain(backend, "python please")) == ["scripted"]


@pytest.mark.asyncio
async def test_an_async_iterator_responder_streams_between_awaits_and_closes_with_the_consumer() -> None:
    release = anyio.Event()
    closed = anyio.Event()

    async def responder(*args: Any, **kwargs: Any) -> Any:
        async def _gen() -> Any:
            try:
                yield TextEvent(text="first")
                await release.wait()
                yield TextEvent(text="second")
            finally:
                closed.set()
        return _gen()

    backend = ScriptedBackend(responder=responder)
    stream = backend.execute(Path("/tmp"), "go")
    assert isinstance(await stream.__anext__(), TextEvent)
    await stream.aclose()

    assert closed.is_set(), "the consumer closing the stream must close the responder iterator"
    assert not release.is_set()


@pytest.mark.asyncio
async def test_every_execute_argument_is_recorded() -> None:
    """Each execute argument is retained for inspection."""
    backend = ScriptedBackend()

    await _drain(backend, "first", max_turns=7, output_schema={"type": "object"}, read_only=True)
    await _drain(backend, "second")

    assert backend.prompts == ["first", "second"]
    assert backend.last_prompt == "second"
    assert backend.max_turns == [7, None]
    assert backend.schemas == [{"type": "object"}, None]
    assert backend.continuations == [None, None]
    assert backend.calls[0]["read_only"] is True
    assert backend.calls[0]["persist_session"] is True


def test_extra_attrs_are_set_for_the_optional_protocol_extensions() -> None:
    """Backends advertise opt-in hints as plain attributes, read via ``getattr``."""
    backend = ScriptedBackend(model="pi-glm", retry_attempts=1, reasoning_effort="high")

    assert backend.model == "pi-glm"
    assert getattr(backend, "retry_attempts", None) == 1
    assert getattr(backend, "reasoning_effort", None) == "high"
    assert backend.fanout_concurrency == 4


@pytest.mark.asyncio
async def test_scripted_backend_drives_run_agent_end_to_end(tmp_path: Path) -> None:
    """The harness satisfies the real ``run_agent`` seam, not just its own drain helper."""
    backend = ScriptedBackend(
        events=[
            TextEvent(text="Review complete"),
            ResultEvent(structured_output=None, continuation=None),
        ]
    )

    output, _, _ = await run_agent(backend, tmp_path, "review this", phase=DaydreamPhase.REVIEW)

    assert output == "Review complete"
    assert backend.last_prompt == "review this"
