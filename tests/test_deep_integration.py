"""Deep-mode backend parity + primitive-preservation tests (D-38, D-39, D-40)."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest

import daydream.phases as phases_mod
from daydream.backends import CostEvent, ResultEvent, TextEvent
from daydream.config import REVIEW_OUTPUT_FILE, STRUCTURE_STACK_NAME
from daydream.deep import detection as _detection
from daydream.deep.detection import StackAssignment
from daydream.deep.prompts import (
    CONFIG_FLOW_TRACE_INSTRUCTION,
    CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION,
    TRUST_MODEL_INSTRUCTION,
)
from daydream.exploration import ExplorationContext
from daydream.phases import (
    phase_alternative_review,
    phase_commit_push,
    phase_fix,
    phase_test_and_heal,
    phase_understand_intent,
)
from daydream.prompts.wire_contract import WIRE_CONTRACT_GENERIC_INSTRUCTION, WIRE_CONTRACT_RUST_INSTRUCTION
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.conftest import silence_module_console
from tests.deep_orchestrator.test_review_completion import record
from tests.deep_orchestrator.test_review_investigation import candidate
from tests.harness.backend import ScriptedBackend
from tests.harness.review_result import merge_result
from tests.harness.stub_backend import review_stage_state, stage_result


class _DeepMockBackend(ScriptedBackend):
    """Review schema dispatch over real fixture source; retain downstream parity.

    Stage labels live in stages to avoid colliding with the harness's calls.
    """

    def __init__(
        self, *, cost_usd: float | None = 0.01, raise_on_agents: bool = False, language_finding: bool = False,
    ) -> None:
        super().__init__(model="mock-model", cost_usd=cost_usd, responder=self._dispatch)
        self.cost_usd = cost_usd
        self.raise_on_agents = raise_on_agents
        self.language_finding = language_finding
        self.stages: list[str] = []

    def _dispatch(
        self, cwd: Any, prompt: str, _output_schema: Any = None, _continuation: Any = None, agents: Any = None,
        _max_turns: Any = None, _read_only: Any = False, _persist_session: Any = True,
    ) -> list[Any]:
        """Return this turn's events for *prompt*, recording its stage tag."""
        # Record parity evidence -- D-38: no stage may pass `agents=`.
        if agents and self.raise_on_agents:
            return [NotImplementedError("Mock Codex: agents kwarg not supported")]

        events: list[Any] = [CostEvent(cost_usd=self.cost_usd, input_tokens=None, output_tokens=None)]
        pl = prompt.lower()

        stage = review_stage_state(prompt)
        if stage is not None:
            events.extend(self._review_stage(Path(cwd), stage))
            return events

        # Checked before the alt branch: the alt-review prompt also contains "intent".
        if "understand" in pl and "intent" in pl:
            self.stages.append("intent")
            events += [TextEvent(text="Intent summary stub."), ResultEvent(structured_output=None, continuation=None)]
            return events

        # Alternative-review prompt contains "architectural alternatives".
        if "architectural alternatives" in pl or ("alternative" in pl and "given this intent" in pl):
            self.stages.append("alternatives")
            events += [TextEvent(text=""), ResultEvent(structured_output={"issues": []}, continuation=None)]
            return events

        # Backend parity fixtures exercise the model merge with language findings;
        # structural-only fixtures retain the host's deterministic merge path.
        if "cross-stack merge agent" in pl:
            self.stages.append("merge")
            items = [{"id": 1, "lens": "per-stack", "description": "Language review finding", "file": "api.py",
                      "line": 2, "severity": "medium", "confidence": "MEDIUM",
                      "rationale": "The changed hello() return is verified in api.py:2.",
                      "evidence": "api.py:2 return 'universe'", "source_uids": ["python:1"]}
                     ] if self.language_finding else []
            events += [TextEvent(text=""), ResultEvent(structured_output=merge_result(items), continuation=None),]
            return events

        # Fallback -- unexpected prompt, but keep the pipeline alive.
        self.stages.append("other")
        events += [TextEvent(text=""), ResultEvent(structured_output=None, continuation=None)]
        return events

    def _review_stage(self, cwd: Path, stage: dict[str, Any]) -> list[Any]:
        """Supply provider findings for the parity runner's assigned scopes."""
        structural = stage["scope_id"] == STRUCTURE_STACK_NAME
        assert stage["stage"] == ("integration" if structural else "first_pass")
        self.stages.append("structure" if structural else "per-stack")
        assert stage["assigned_target_ids"] == (
            ["integration:structure"] if structural else stage["assigned_files"]
        )
        findings = []
        if ("api.py" in stage["assigned_files"]
                and (structural or self.language_finding and stage["scope_id"] == "python")):
            assert "return 'universe'" in (cwd / "api.py").read_text()
            description = "Greeting contract changed across modules" if structural else "Language review finding"
            finding = dict(record(), description=description,
                           rationale="The changed hello() return is verified in api.py:2.",
                           evidence="api.py:2 return 'universe'")
            findings.append(dict(candidate(disposition='confirmed', finding=finding),
                                 trigger="Calling hello() after the changed greeting contract",
                                 consequence="The returned greeting changes from world to universe.",
                                 grounds="api.py:2 return 'universe' in the complete hello() symbol"))
        return [ResultEvent(structured_output=stage_result(stage, candidates=findings), continuation=None)]

def _silence_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence noisy UI helpers at their current production owners."""
    for module in ("daydream.deep.orchestrator", "daydream.deep.review_steps", "daydream.deep.merge_steps",
        "daydream.deep.diagram_steps", "daydream.deep.fix_steps", "daydream.phases", "daydream.runner",
    ):
        silence_module_console(monkeypatch, module)

def _wire_mocks(monkeypatch: pytest.MonkeyPatch, backend: _DeepMockBackend) -> None:
    """Install the backend, accept intent, and decline other interactive gates."""
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend,)
    def answer(_console: Any, message: str, default: str = "") -> str:
        return "y" if "understanding correct" in message.lower() else "n"
    monkeypatch.setattr("daydream.run_context._prompt_user", answer)
    _silence_ui(monkeypatch)

async def _run_deep(target: Path, backend: _DeepMockBackend, monkeypatch: pytest.MonkeyPatch,
    shallow_fanout_threshold: int | None = None,
) -> int:
    """Run the deep pipeline with mocks and an explicit tiny-diff collapse threshold."""
    _wire_mocks(monkeypatch, backend)

    # Pre-populate exploration context to skip the safe_explore backend call.
    # The orchestrator only runs pre-scan when `exploration_context is None`.
    config = RunConfig(target=str(target), cleanup=False, exploration_context=ExplorationContext(),
        shallow_fanout_threshold=shallow_fanout_threshold,
    )
    return await run(config)

async def test_claude_shape_backend(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-38: run_deep completes end-to-end on a Claude-shaped backend (cost_usd populated)."""
    backend = _DeepMockBackend(cost_usd=0.0123, language_finding=True)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists(), ("merged report missing after Claude-shape run")
    # Design uses structural review; reviewers emit records directly.
    required = {"intent", "structure", "per-stack", "merge"}
    assert required.issubset(set(backend.stages)), (f"missing stages; saw only: {sorted(set(backend.stages))}")
    assert "alternatives" not in backend.stages
    report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "Language review finding" in report
    assert "Greeting contract changed across modules" in report

async def test_codex_shape_backend(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-38: run_deep completes on Codex-shape (cost_usd=None, no agents= ever passed)."""
    backend = _DeepMockBackend(cost_usd=None, raise_on_agents=True, language_finding=True)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists(), ("merged report missing after Codex-shape run")
    assert {"intent", "structure", "per-stack", "merge"}.issubset(backend.stages)
    report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "Language review finding" in report
    assert "Greeting contract changed across modules" in report
    # Parity guarantee: any stage passing agents= would have raised
    # NotImplementedError above; this asserts it directly too.
    agents_kwargs_seen = [call["agents"] for call in backend.calls]
    assert all(a in (None, False, [], {}, 0, "") for a in agents_kwargs_seen), (
        f"agents kwarg was passed somewhere: {agents_kwargs_seen}"
    )

def test_phase_primitives_unmodified() -> None:
    """Phase primitives accept backend then WorkContext; workspace resolves the base."""
    # Other primitives: first two params are (backend, work).
    for fn in (phase_understand_intent, phase_alternative_review, phase_fix, phase_test_and_heal, phase_commit_push,):
        params = list(inspect.signature(fn).parameters.values())
        assert params[0].name == "backend", (f"{fn.__name__} first param: {params[0].name}")
        assert params[1].name == "work", (f"{fn.__name__} second param: {params[1].name}")

    # D-39 negative guard: no "v2" or "_deep_" wrappers crept in.

    leaked = [name
        for name in dir(phases_mod)
        if ("v2" in name.lower() or "_deep_" in name.lower())
        and name.startswith("phase_")
    ]
    assert not leaked, f"forbidden phase wrappers present: {leaked}"

async def test_deep_default_backend_line_is_phase_agnostic(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:

    backend = _DeepMockBackend(cost_usd=0.0123)
    _wire_mocks(monkeypatch, backend)
    # Capture orchestrator print_info messages (override the _silence_ui noop).
    captured: list[str] = []
    monkeypatch.setattr("daydream.deep.orchestrator.print_info",
        lambda *a, **kw: captured.append(str(a[1]) if len(a) > 1 else kw.get("msg", "")),
    )
    config = RunConfig(
        target=str(multi_stack_target), cleanup=False, exploration_context=ExplorationContext(), backend="claude",
        review_backend="codex",
    )
    exit_code = await run(config)
    assert exit_code == 0, f"run returned {exit_code} (expected 0)"
    assert "Default backend: claude" in captured, captured

async def test_structural_meta_stack_flows_end_to_end(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-language run detects structure, writes its review, and renders its section."""
    detected_stacks: list[StackAssignment] = []
    real_detect = _detection.detect_stacks

    def _spy_detect(changed_files: Any, **kwargs: Any) -> Any:
        result = real_detect(changed_files, **kwargs)
        detected_stacks.extend(result)
        return result

    # ``detect_stacks`` is imported into the orchestrator namespace; patch there.
    monkeypatch.setattr("daydream.deep.orchestrator.detect_stacks", _spy_detect)

    backend = _DeepMockBackend(cost_usd=0.0123)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"

    # (1) detect_stacks emitted the structure meta-stack for this code diff.
    structure = next((a for a in detected_stacks if a.stack_name == STRUCTURE_STACK_NAME), None)
    assert structure is not None, (f"structure stack not emitted; saw: {[a.stack_name for a in detected_stacks]}")

    deep_dir = multi_stack_target / ".daydream" / "deep"

    # (2) phase_per_stack_reviews produced the structural review artifact.
    structural_review = deep_dir / "stack-structure-review.md"
    assert structural_review.is_file(), ("stack-structure-review.md missing -- structure stack was not routed "
        "through build_structural_prompt or its agent never wrote the artifact"
    )

    # (3) The merged report on disk carries the dedicated structural section.
    merged_report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "## Structural Review" in merged_report, (
        "merged report is missing the ## Structural Review header"
    )

async def test_310_prompt_gates_reach_built_prompts_in_real_run(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect prompt gates delivered through runner.run to the backend.

    The Python/TSX/Markdown fixture exercises language, structural, and generic
    builders. Structural prompts carry cross-file/trust gates; language and
    generic prompts carry config-trace/trust gates.
    """
    backend = _DeepMockBackend(cost_usd=0.0123)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"

    # Exclude generic fallback from the per-stack predicate it also matches.
    structural = [p for p in backend.prompts if "structural reviewer" in p]
    generic = [p for p in backend.prompts if "generic-fallback" in p]
    per_stack = [p
        for p in backend.prompts
        if "You are reviewing the" in p
        and "stack" in p
        and "structural reviewer" not in p
        and "generic-fallback" not in p
    ]

    assert structural, "structural prompt never reached the backend"
    assert per_stack, "per-stack language prompt never reached the backend"
    assert generic, "generic-fallback prompt never reached the backend"

    for prompt in structural:
        assert CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION in prompt, (
            "structural prompt missing the cross-file symbol-existence gate"
        )
        assert TRUST_MODEL_INSTRUCTION in prompt, ("structural prompt missing the trust-model gate")
        assert CONFIG_FLOW_TRACE_INSTRUCTION not in prompt, ("config-trace gate leaked into the structural prompt")

    for prompt in per_stack:
        assert "Config/env flow trace (apply only to changed fields in this assignment" in prompt
        assert "config struct -> driver config -> request construction" in prompt
        assert "Flag silent drops" in prompt and "Flag double-resolves" in prompt
        assert CONFIG_FLOW_TRACE_INSTRUCTION not in prompt, "whole-scope config audit leaked into a file stage"
        assert TRUST_MODEL_INSTRUCTION in prompt, ("per-stack prompt missing the trust-model gate")
        assert CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION not in prompt, (
            "cross-file gate leaked into the per-stack prompt"
        )

    for prompt in generic:
        assert "Config/env flow trace (apply only to changed fields in this assignment" in prompt
        assert "config struct -> driver config -> request construction" in prompt
        assert "Flag silent drops" in prompt and "Flag double-resolves" in prompt
        assert CONFIG_FLOW_TRACE_INSTRUCTION not in prompt, "whole-scope config audit leaked into a generic stage"
        assert TRUST_MODEL_INSTRUCTION in prompt, ("generic-fallback prompt missing the trust-model gate")
        assert CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION not in prompt, (
            "cross-file gate leaked into the generic-fallback prompt"
        )

async def test_311_wire_contract_reaches_delivered_prompts_in_real_run(
    rust_wire_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delivered Rust/generic prompts receive their distinct wire-contract rules.

    Rust gets serde/routing, generic gets URL/quoting, and structural gets
    neither. Threshold zero prevents the two-file fixture's generic bucket
    from collapsing into Rust before its prompt reaches the backend.
    """
    backend = _DeepMockBackend(cost_usd=0.0123)
    exit_code = await _run_deep(rust_wire_target, backend, monkeypatch, shallow_fanout_threshold=0)
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"

    rust_per_stack = [p for p in backend.prompts if "You are reviewing the rust stack" in p]
    generic = [p for p in backend.prompts if "generic-fallback" in p]
    structural = [p for p in backend.prompts if "structural reviewer" in p]

    assert rust_per_stack, "rust per-stack prompt never reached the backend"
    assert generic, "generic-fallback prompt never reached the backend"
    assert structural, "structural prompt never reached the backend"

    for prompt in rust_per_stack:
        assert WIRE_CONTRACT_RUST_INSTRUCTION in prompt, (
            "rust per-stack prompt missing the serde/routing wire-contract instruction"
        )
        assert WIRE_CONTRACT_GENERIC_INSTRUCTION not in prompt, (
            "URL/quoting wire-contract instruction leaked into the rust per-stack prompt"
        )

    for prompt in generic:
        assert WIRE_CONTRACT_GENERIC_INSTRUCTION in prompt, (
            "generic-fallback prompt missing the URL/quoting wire-contract instruction"
        )
        assert WIRE_CONTRACT_RUST_INSTRUCTION not in prompt, (
            "serde wire-contract instruction leaked into the generic-fallback prompt"
        )

    for prompt in structural:
        assert WIRE_CONTRACT_RUST_INSTRUCTION not in prompt, (
            "serde wire-contract instruction leaked into the structural prompt"
        )
        assert WIRE_CONTRACT_GENERIC_INSTRUCTION not in prompt, (
            "URL/quoting wire-contract instruction leaked into the structural prompt"
        )
