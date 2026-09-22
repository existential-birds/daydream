"""Deep-mode backend parity + primitive-preservation tests (D-38, D-39, D-40)."""
from __future__ import annotations

import inspect
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, CostEvent, ResultEvent, TextEvent
from daydream.config import REVIEW_OUTPUT_FILE
from tests.harness.backend import ScriptedBackend


class _DeepMockBackend(ScriptedBackend):
    """Prompt-dispatching mock backend keyed on the prompt's stage wording.

    The dispatch itself lives in the ``ScriptedBackend`` responder seam, so the
    harness owns call recording: ``calls`` (one dict per call, including
    ``agents``) and the ``prompts`` observable come from the shared fake. This
    subclass only records the stage tag of each dispatched prompt in ``stages``
    -- naming it ``calls`` would collide with the harness's dict list.
    """

    def __init__(
        self,
        target_dir: Path,
        *,
        cost_usd: float | None = 0.01,
        raise_on_agents: bool = False,
    ) -> None:
        super().__init__(model="mock-model", cost_usd=cost_usd, responder=self._dispatch)
        self.target_dir = target_dir
        self.cost_usd = cost_usd
        self.raise_on_agents = raise_on_agents
        self.stages: list[str] = []

    async def execute(
        self,
        cwd: Any,
        prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            yield event

    def _dispatch(
        self,
        cwd: Any,
        prompt: str,
        _output_schema: Any = None,
        _continuation: Any = None,
        agents: Any = None,
        _max_turns: Any = None,
        _read_only: Any = False,
        _persist_session: Any = True,
    ) -> list[Any]:
        """Return this turn's events for *prompt*, recording its stage tag."""
        # Record parity evidence -- D-38: no stage may pass `agents=`.
        if agents and self.raise_on_agents:
            return [NotImplementedError("Mock Codex: agents kwarg not supported")]

        events: list[Any] = [CostEvent(cost_usd=self.cost_usd, input_tokens=None, output_tokens=None)]
        pl = prompt.lower()

        # Checked before the alt branch: the alt-review prompt also contains "intent".
        if "understand" in pl and "intent" in pl:
            self.stages.append("intent")
            events += [TextEvent(text="Intent summary stub."), ResultEvent(structured_output=None, continuation=None)]
            return events

        # Alternative-review prompt contains "architectural alternatives".
        if "architectural alternatives" in pl or (
            "alternative" in pl and "given this intent" in pl
        ):
            self.stages.append("alternatives")
            events += [TextEvent(text=""), ResultEvent(structured_output={"issues": []}, continuation=None)]
            return events

        # Checked before per-stack: the structural prompt lacks "you are reviewing the
        # ... stack" but embeds the stack-structure-review.md path, so it would
        # otherwise fall through to the "other" fallback and write no artifact.
        if "structural reviewer" in pl:
            self.stages.append("structure")
            m = re.search(r"stack-(\S+?)-review\.md", prompt)
            if m:
                name = m.group(1)
                out = self._review_output_path(prompt)
                if out is not None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(
                        f"# Structural Review ({name})\n\n## Issues\n"
                        "1. [api.py:1] hello() leaks a god-object boundary\n"
                    )
            # Issue #745: the structural reviewer emits PER_STACK_RECORD_SCHEMA
            # structured output directly (its finding lands lens="structural").
            events += [
                TextEvent(text=""),
                ResultEvent(
                    structured_output={
                        "issues": [
                            {
                                "id": 1,
                                "description": "hello() leaks a god-object boundary",
                                "file": "api.py",
                                "line": 1,
                                "severity": "medium",
                                "confidence": "MEDIUM",
                                "rationale": "stub",
                                "evidence": "api.py:1",
                            }
                        ],
                        "verdicts": [],
                    },
                    continuation=None,
                ),
            ]
            return events

        # Per-stack review prompt contains "You are reviewing the ... stack".
        if "you are reviewing the" in pl and "stack" in pl:
            self.stages.append("per-stack")
            # Write the per-stack review file to the path embedded in the prompt
            # (the session's live artifact tree).
            m = re.search(r"stack-(\S+?)-review\.md", prompt)
            if m:
                name = m.group(1)
                out = self._review_output_path(prompt)
                if out is not None:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(f"# Review ({name})\n\n## Issues\n1. [a.py:1] stub\n")
            # Issue #745: per-stack reviewer emits structured output directly.
            events += [
                TextEvent(text=""),
                ResultEvent(structured_output={"issues": [], "verdicts": []}, continuation=None),
            ]
            return events

        # Parse-feedback prompt contains "Read the review output file at".
        if "read the review output file" in pl or "extract only actionable issues" in pl:
            self.stages.append("parse")
            # Parsing the structural review yields a finding (tagged lens="structural"
            # in merge, rendering the ## Structural Review section); other stacks yield none.
            # Issue #742: the per-stack parse schema requires a ``verdicts`` property.
            if "stack-structure-review.md" in prompt:
                events += [
                    TextEvent(text=""),
                    ResultEvent(
                        structured_output={
                            "issues": [
                                {
                                    "id": 1,
                                    "description": "hello() leaks a god-object boundary",
                                    "file": "api.py",
                                    "line": 1,
                                    "evidence": "api.py:1",
                                }
                            ],
                            "verdicts": [],
                        },
                        continuation=None,
                    ),
                ]
            else:
                events += [
                    TextEvent(text=""),
                    ResultEvent(structured_output={"issues": [], "verdicts": []}, continuation=None),
                ]
            return events

        # Merge: return an empty item list (no language-stack issues in this fixture),
        # so the host's canonical report carries only the appended structural section.
        if "cross-stack merge agent" in pl:
            self.stages.append("merge")
            events += [
                TextEvent(text=""),
                ResultEvent(structured_output={"items": []}, continuation=None),
            ]
            return events

        # Fallback -- unexpected prompt, but keep the pipeline alive.
        self.stages.append("other")
        events += [TextEvent(text=""), ResultEvent(structured_output=None, continuation=None)]
        return events

    def _review_output_path(self, prompt: str) -> Path | None:
        """The review-file path the delivered prompt names.

        The prompt names the session's live artifact tree. Writing there rather
        than reconstructing a path under ``target_dir`` matches the sanctioned
        adapter contract: the public ``.daydream`` tree is detached for the
        run's duration, so a mid-run write to a reconstructed public path trips
        the model-cwd artifact gate and fails the run.
        """
        m = re.search(r"Write your full review to (\S+\.md)\.", prompt)
        return Path(m.group(1)) if m else None


def _silence_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence noisy UI helpers at their current production owners."""
    noop = lambda *a, **kw: None  # noqa: E731 -- terse silencer
    targets = {
        "daydream.deep.orchestrator": (
            "print_preflight_notice",
            "print_info",
            "print_warning",
            "print_error",
        ),
        "daydream.deep.review_steps": (
            "print_stage_progress",
            "print_phase_hero",
            "print_warning",
            "print_error",
            "print_dim",
        ),
        "daydream.deep.merge_steps": (
            "print_stage_progress",
            "print_info",
            "print_warning",
            "print_error",
        ),
        "daydream.deep.diagram_steps": (
            "print_info",
            "print_success",
            "print_warning",
            "print_error",
        ),
        "daydream.deep.fix_steps": (
            "print_info",
            "print_success",
            "print_warning",
            "print_error",
            "print_verification_summary",
        ),
        "daydream.phases": (
            "print_phase_hero",
            "print_info",
            "print_success",
            "print_warning",
            "print_error",
            "print_dim",
            "print_issues_table",
        ),
        "daydream.runner": (
            "print_phase_hero",
            "print_info",
            "print_success",
            "print_error",
            "print_dim",
        ),
    }
    for module, names in targets.items():
        for name in names:
            monkeypatch.setattr(f"{module}.{name}", noop)


def _wire_mocks(monkeypatch: pytest.MonkeyPatch, backend: _DeepMockBackend) -> None:
    """Install the backend, accept intent, and decline other interactive gates."""
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: backend,
    )

    def answer(_console: Any, message: str, default: str = "") -> str:
        return "y" if "understanding correct" in message.lower() else "n"

    monkeypatch.setattr("daydream.run_context._prompt_user", answer)
    _silence_ui(monkeypatch)


async def _run_deep(
    target: Path,
    backend: _DeepMockBackend,
    monkeypatch: pytest.MonkeyPatch,
    shallow_fanout_threshold: int | None = None,
) -> int:
    """Common driver: wire mocks and execute the full deep pipeline.

    ``shallow_fanout_threshold`` is forwarded to :class:`RunConfig`. The #311
    wire-contract test passes ``0`` so its 2-file fixture does not trigger the
    tiny-diff collapse (which would absorb the generic bucket into the single
    rust assignment and suppress the generic-fallback prompt).
    """
    from daydream.exploration import ExplorationContext
    from daydream.runner import RunConfig, run

    _wire_mocks(monkeypatch, backend)

    # Pre-populate exploration context to skip the safe_explore backend call.
    # The orchestrator only runs pre-scan when `exploration_context is None`.
    config = RunConfig(
        target=str(target),
        cleanup=False,
        exploration_context=ExplorationContext(),
        shallow_fanout_threshold=shallow_fanout_threshold,
    )
    return await run(config)


async def test_claude_shape_backend(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-38: run_deep completes end-to-end on a Claude-shaped backend (cost_usd populated)."""
    backend = _DeepMockBackend(multi_stack_target, cost_usd=0.0123)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)

    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists(), (
        "merged report missing after Claude-shape run"
    )
    # The default design lens shares structural review instead of a separate
    # alternatives stage. Intent, language review and merge still run. The
    # parse-<stack> stage was removed (issue #745) -- reviewers emit records
    # directly.
    required = {"intent", "structure", "per-stack", "merge"}
    assert required.issubset(set(backend.stages)), (
        f"missing stages; saw only: {sorted(set(backend.stages))}"
    )
    assert "alternatives" not in backend.stages


async def test_codex_shape_backend(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-38: run_deep completes on Codex-shape (cost_usd=None, no agents= ever passed)."""
    backend = _DeepMockBackend(multi_stack_target, cost_usd=None, raise_on_agents=True)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)

    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists(), (
        "merged report missing after Codex-shape run"
    )
    # Parity guarantee: any stage passing agents= would have raised
    # NotImplementedError above; this asserts it directly too.
    agents_kwargs_seen = [call["agents"] for call in backend.calls]
    assert all(a in (None, False, [], {}, 0, "") for a in agents_kwargs_seen), (
        f"agents kwarg was passed somewhere: {agents_kwargs_seen}"
    )


def test_phase_primitives_unmodified() -> None:
    """D-39: existing phase primitives imported unchanged by run_deep.

    Stage 3 renames the second positional parameter from ``cwd`` to ``work``
    (a :class:`WorkContext`). The contract this test enforces is now:
    ``backend`` first, ``work`` second — base resolution happens once at
    workspace open time and is threaded through every phase.
    """
    from daydream.phases import (
        phase_alternative_review,
        phase_commit_push,
        phase_fix,
        phase_test_and_heal,
        phase_understand_intent,
    )

    # Other primitives: first two params are (backend, work).
    for fn in (
        phase_understand_intent,
        phase_alternative_review,
        phase_fix,
        phase_test_and_heal,
        phase_commit_push,
    ):
        params = list(inspect.signature(fn).parameters.values())
        assert params[0].name == "backend", (
            f"{fn.__name__} first param: {params[0].name}"
        )
        assert params[1].name == "work", (
            f"{fn.__name__} second param: {params[1].name}"
        )

    # D-39 negative guard: no "v2" or "_deep_" wrappers crept in.
    import daydream.phases as phases_mod

    leaked = [
        name
        for name in dir(phases_mod)
        if ("v2" in name.lower() or "_deep_" in name.lower())
        and name.startswith("phase_")
    ]
    assert not leaked, f"forbidden phase wrappers present: {leaked}"


async def test_deep_default_backend_line_is_phase_agnostic(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#647: the 'Default backend' status line never shows a review override."""
    from daydream.exploration import ExplorationContext
    from daydream.runner import RunConfig, run

    backend = _DeepMockBackend(multi_stack_target, cost_usd=0.0123)
    _wire_mocks(monkeypatch, backend)
    # Capture orchestrator print_info messages (override the _silence_ui noop).
    captured: list[str] = []
    monkeypatch.setattr(
        "daydream.deep.orchestrator.print_info",
        lambda *a, **kw: captured.append(str(a[1]) if len(a) > 1 else kw.get("msg", "")),
    )
    config = RunConfig(
        target=str(multi_stack_target),
        cleanup=False,
        exploration_context=ExplorationContext(),
        backend="claude",
        review_backend="codex",
    )
    exit_code = await run(config)
    assert exit_code == 0, f"run returned {exit_code} (expected 0)"
    assert "Default backend: claude" in captured, captured


async def test_structural_meta_stack_flows_end_to_end(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end smoke for the structural meta-stack pipeline (Tasks 2-7 composed).

    Drives the deep orchestrator on a real multi-language diff (Python + TSX +
    Markdown — a code diff, so NOT docs-only) and asserts the four observable
    states the structural pipeline must produce:

      1. ``detect_stacks`` returned ``structure`` as one of the stacks.
      2. ``phase_per_stack_reviews`` produced a ``stack-structure-review.md``
         artifact on disk.
      3. The final merged report on disk carries a ``## Structural Review``
         header.

    These are real observable side effects (files on disk, the rendered report
    header) — not dispatch bookkeeping. If any wire in Tasks 2-7 were broken
    (structure stack not emitted, records not partitioned out and appended,
    section not rendered) the corresponding assertion below fails.
    """
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep import detection as _detection
    from daydream.deep.detection import StackAssignment

    detected_stacks: list[StackAssignment] = []
    real_detect = _detection.detect_stacks

    def _spy_detect(changed_files: Any, **kwargs: Any) -> Any:
        result = real_detect(changed_files, **kwargs)
        detected_stacks.extend(result)
        return result

    # ``detect_stacks`` is imported into the orchestrator namespace; patch there.
    monkeypatch.setattr("daydream.deep.orchestrator.detect_stacks", _spy_detect)

    backend = _DeepMockBackend(multi_stack_target, cost_usd=0.0123)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"

    # (1) detect_stacks emitted the structure meta-stack for this code diff.
    structure = next(
        (a for a in detected_stacks if a.stack_name == STRUCTURE_STACK_NAME), None
    )
    assert structure is not None, (
        f"structure stack not emitted; saw: {[a.stack_name for a in detected_stacks]}"
    )

    deep_dir = multi_stack_target / ".daydream" / "deep"

    # (2) phase_per_stack_reviews produced the structural review artifact.
    structural_review = deep_dir / "stack-structure-review.md"
    assert structural_review.is_file(), (
        "stack-structure-review.md missing -- structure stack was not routed "
        "through build_structural_prompt or its agent never wrote the artifact"
    )

    # (3) The merged report on disk carries the dedicated structural section.
    merged_report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "## Structural Review" in merged_report, (
        "merged report is missing the ## Structural Review header"
    )


async def test_310_prompt_gates_reach_built_prompts_in_real_run(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path coverage for the #310 prompt gates (PR #328, Finding 2).

    The #310 unit tests call the prompt builders directly. This one drives a
    deep run through ``runner.run`` with the stub backend and observes at the
    backend seam: ``_DeepMockBackend.execute`` records the full ``prompt`` it
    receives on every call, so the prompts actually delivered to the external
    backend are classified by content -- no builder spies, no assumption that a
    built prompt survives the trip from builder to ``Backend.execute``. The
    gates are asserted on what the backend received:

      - structural: cross-file symbol-existence + trust-model, NOT config-trace;
      - per-stack (language): config-trace + trust-model, NOT cross-file;
      - generic-fallback: config-trace + trust-model, NOT cross-file.

    ``multi_stack_target`` (api.py + App.tsx + README.md) routes one file to
    each of the three built-in stack builders, so
    all three gate assignments are exercised in a single real run.
    """
    from daydream.deep.prompts import (
        CONFIG_FLOW_TRACE_INSTRUCTION,
        CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION,
        TRUST_MODEL_INSTRUCTION,
    )

    backend = _DeepMockBackend(multi_stack_target, cost_usd=0.0123)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"

    # Classify the prompts the backend actually received by content. The
    # per-stack predicate also matches the generic-fallback scope line ("You are
    # reviewing the generic-fallback stack"), so structural + generic-fallback
    # are excluded from the per-stack class to keep each class meaningful.
    structural = [p for p in backend.prompts if "structural reviewer" in p]
    generic = [p for p in backend.prompts if "generic-fallback" in p]
    per_stack = [
        p
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
        assert TRUST_MODEL_INSTRUCTION in prompt, (
            "structural prompt missing the trust-model gate"
        )
        assert CONFIG_FLOW_TRACE_INSTRUCTION not in prompt, (
            "config-trace gate leaked into the structural prompt"
        )

    for prompt in per_stack:
        assert CONFIG_FLOW_TRACE_INSTRUCTION in prompt, (
            "per-stack prompt missing the config-flow trace gate"
        )
        assert TRUST_MODEL_INSTRUCTION in prompt, (
            "per-stack prompt missing the trust-model gate"
        )
        assert CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION not in prompt, (
            "cross-file gate leaked into the per-stack prompt"
        )

    for prompt in generic:
        assert CONFIG_FLOW_TRACE_INSTRUCTION in prompt, (
            "generic-fallback prompt missing the config-flow trace gate"
        )
        assert TRUST_MODEL_INSTRUCTION in prompt, (
            "generic-fallback prompt missing the trust-model gate"
        )
        assert CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION not in prompt, (
            "cross-file gate leaked into the generic-fallback prompt"
        )


async def test_311_wire_contract_reaches_delivered_prompts_in_real_run(
    rust_wire_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real-path coverage for the #311 wire-contract instructions (R3/R4).

    Mirrors ``test_310_prompt_gates_reach_built_prompts_in_real_run``: drives a
    deep run through ``runner.run`` with the stub backend and classifies the
    prompts actually delivered to ``_DeepMockBackend.execute`` by content -- no
    builder spies, no assumption that a built prompt survives the trip from
    builder to backend. ``rust_wire_target`` (src/main.rs + README.md) routes
    the rust file to the per-stack builder and README.md to the
    generic-fallback bucket, so both wire-contract instructions are pinned on
    the delivered prompts in one run:

      - rust per-stack: serde/routing instruction present, URL instruction NOT;
      - generic-fallback: URL/quoting instruction present, serde instruction NOT;
      - structural: neither (out of scope per spec).

    ``shallow_fanout_threshold=0`` disables the tiny-diff short-circuit
    (``DEFAULT_SHALLOW_FANOUT_THRESHOLD`` = 2): with the default threshold a
    2-file diff containing exactly one real language stack absorbs the generic
    bucket into that stack, and the generic-fallback prompt would never be
    delivered. Disabling the collapse restores the full pipeline so the rust
    per-stack and generic-fallback prompts both reach the backend seam.
    """
    from daydream.prompts.wire_contract import (
        WIRE_CONTRACT_GENERIC_INSTRUCTION,
        WIRE_CONTRACT_RUST_INSTRUCTION,
    )

    backend = _DeepMockBackend(rust_wire_target, cost_usd=0.0123)
    exit_code = await _run_deep(
        rust_wire_target, backend, monkeypatch, shallow_fanout_threshold=0
    )
    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"

    rust_per_stack = [
        p for p in backend.prompts if "You are reviewing the rust stack" in p
    ]
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
