"""Deep-mode backend parity + primitive-preservation tests (D-38, D-39, D-40)."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from daydream.backends import CostEvent, ResultEvent, TextEvent
from daydream.config import REVIEW_OUTPUT_FILE


class _DeepMockBackend:
    """Prompt-dispatching mock backend. Subclasses tune cost + agents behavior."""

    model = "mock-model"
    cost_usd: float | None = 0.01
    raise_on_agents: bool = False

    def __init__(self, target_dir: Path) -> None:
        self.target_dir = target_dir
        self.calls: list[str] = []
        self.agents_kwargs_seen: list[object] = []

    async def execute(
        self,
        cwd,
        prompt,
        output_schema=None,
        continuation=None,
        *,
        agents=None,
        max_turns=None,
        read_only=False,
    ):
        # Record parity evidence -- D-38: no stage may pass `agents=`.
        self.agents_kwargs_seen.append(agents)
        if agents and self.raise_on_agents:
            raise NotImplementedError("Mock Codex: agents kwarg not supported")

        yield CostEvent(
            cost_usd=self.cost_usd, input_tokens=None, output_tokens=None
        )

        pl = prompt.lower()

        # Checked before the alt branch: the alt-review prompt also contains "intent".
        if "understand" in pl and "intent" in pl:
            self.calls.append("intent")
            yield TextEvent(text="Intent summary stub.")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Alternative-review prompt contains "architectural alternatives".
        if "architectural alternatives" in pl or (
            "alternative" in pl and "given this intent" in pl
        ):
            self.calls.append("alternatives")
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"issues": []}, continuation=None)
            return

        # Checked before per-stack: the structural prompt lacks "you are reviewing the
        # ... stack" but embeds the stack-structure-review.md path, so it would
        # otherwise fall through to the "other" fallback and write no artifact.
        if "structural reviewer" in pl:
            self.calls.append("structure")
            m = re.search(r"stack-(\S+?)-review\.md", prompt)
            if m:
                name = m.group(1)
                out = self.target_dir / ".daydream" / "deep" / f"stack-{name}-review.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    f"# Structural Review ({name})\n\n## Issues\n"
                    "1. [api.py:1] hello() leaks a god-object boundary\n"
                )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Per-stack review prompt contains "You are reviewing the ... stack".
        if "you are reviewing the" in pl and "stack" in pl:
            self.calls.append("per-stack")
            # Write the per-stack review file to the path embedded in the prompt.
            m = re.search(r"stack-(\S+?)-review\.md", prompt)
            if m:
                name = m.group(1)
                out = self.target_dir / ".daydream" / "deep" / f"stack-{name}-review.md"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(f"# Review ({name})\n\n## Issues\n1. [a.py:1] stub\n")
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Parse-feedback prompt contains "Read the review output file at".
        if "read the review output file" in pl or "extract only actionable issues" in pl:
            self.calls.append("parse")
            yield TextEvent(text="")
            # Parsing the structural review yields a finding (tagged lens="structural"
            # in merge, rendering the ## Structural Review section); other stacks yield none.
            if "stack-structure-review.md" in prompt:
                yield ResultEvent(
                    structured_output={
                        "issues": [
                            {
                                "id": 1,
                                "description": "hello() leaks a god-object boundary",
                                "file": "api.py",
                                "line": 1,
                                "evidence": "api.py:1",
                            }
                        ]
                    },
                    continuation=None,
                )
            else:
                yield ResultEvent(structured_output={"issues": []}, continuation=None)
            return

        # Merge: return an empty item list (no language-stack issues in this fixture),
        # so the host's canonical report carries only the appended structural section.
        if "cross-stack merge agent" in pl:
            self.calls.append("merge")
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"items": []}, continuation=None)
            return

        # Fallback -- unexpected prompt, but keep the pipeline alive.
        self.calls.append("other")
        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


class _ClaudeShape(_DeepMockBackend):
    cost_usd = 0.0123
    raise_on_agents = False  # Claude accepts agents kwarg silently


class _CodexShape(_DeepMockBackend):
    cost_usd = None          # Codex does not report cost
    raise_on_agents = True   # Codex rejects agents kwarg (parity contract)


def _silence_ui(monkeypatch) -> None:
    """Silence noisy UI helpers across orchestrator, phases, and runner."""
    noop = lambda *a, **kw: None  # noqa: E731 -- terse silencer
    for module in (
        "daydream.deep.orchestrator",
        "daydream.phases",
        "daydream.runner",
    ):
        for name in (
            "print_stage_progress",
            "print_preflight_notice",
            "print_phase_hero",
            "print_info",
            "print_success",
            "print_warning",
            "print_error",
            "print_dim",
            "print_issues_table",
        ):
            monkeypatch.setattr(f"{module}.{name}", noop, raising=False)


def _wire_mocks(monkeypatch, backend: _DeepMockBackend) -> None:
    """Install the mock backend + silence prompts and UI.

    ``phase_understand_intent`` loops on ``prompt_user`` until the user confirms
    with ``y`` -- so ``daydream.phases.prompt_user`` must return ``y``.
    The orchestrator's fix gate must return ``n`` so we skip the fix pass.
    """
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: backend,
    )
    # Orchestrator fix gate: decline to apply fixes.
    monkeypatch.setattr(
        "daydream.deep.orchestrator.prompt_user",
        lambda *a, **kw: "n",
        raising=False,
    )
    # Intent confirmation loop: accept the first answer.
    monkeypatch.setattr(
        "daydream.phases.prompt_user",
        lambda *a, **kw: "y",
        raising=False,
    )
    # Runner-level prompts (target dir, cleanup): never reached when `target`
    # and `cleanup` are explicit on RunConfig, but be defensive.
    monkeypatch.setattr(
        "daydream.runner.prompt_user",
        lambda *a, **kw: "n",
        raising=False,
    )
    _silence_ui(monkeypatch)


async def _run_deep(target: Path, backend: _DeepMockBackend, monkeypatch) -> int:
    """Common driver: wire mocks and execute the full deep pipeline."""
    from daydream.exploration import ExplorationContext
    from daydream.runner import RunConfig, run

    _wire_mocks(monkeypatch, backend)

    # Pre-populate exploration context to skip the safe_explore backend call.
    # The orchestrator only runs pre-scan when `exploration_context is None`.
    config = RunConfig(
        target=str(target),
        cleanup=False,
        exploration_context=ExplorationContext(),
    )
    return await run(config)


async def test_claude_shape_backend(multi_stack_target: Path, monkeypatch) -> None:
    """D-38: run_deep completes end-to-end on a Claude-shaped backend (cost_usd populated)."""
    backend = _ClaudeShape(multi_stack_target)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)

    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists(), (
        "merged report missing after Claude-shape run"
    )
    # Stages fired: intent, alternatives, at least one per-stack, parse, merge.
    required = {"intent", "alternatives", "per-stack", "parse", "merge"}
    assert required.issubset(set(backend.calls)), (
        f"missing stages; saw only: {sorted(set(backend.calls))}"
    )


async def test_codex_shape_backend(multi_stack_target: Path, monkeypatch) -> None:
    """D-38: run_deep completes on Codex-shape (cost_usd=None, no agents= ever passed)."""
    backend = _CodexShape(multi_stack_target)
    exit_code = await _run_deep(multi_stack_target, backend, monkeypatch)

    assert exit_code == 0, f"run_deep returned {exit_code} (expected 0)"
    assert (multi_stack_target / REVIEW_OUTPUT_FILE).exists(), (
        "merged report missing after Codex-shape run"
    )
    # Parity guarantee: any stage passing agents= would have raised
    # NotImplementedError above; this asserts it directly too.
    assert all(a in (None, False, [], {}, 0, "") for a in backend.agents_kwargs_seen), (
        f"agents kwarg was passed somewhere: {backend.agents_kwargs_seen}"
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
        phase_parse_feedback,
        phase_test_and_heal,
        phase_understand_intent,
    )

    # phase_parse_feedback gained an OPTIONAL keyword-only kwarg per plan 05-06.
    sig = inspect.signature(phase_parse_feedback)
    params = list(sig.parameters.values())
    assert params[0].name == "backend", (
        f"phase_parse_feedback first param: {params[0].name}"
    )
    assert params[1].name == "work", (
        f"phase_parse_feedback second param: {params[1].name}"
    )
    input_path = sig.parameters.get("input_path")
    assert input_path is not None, "phase_parse_feedback missing input_path kwarg"
    assert input_path.kind == inspect.Parameter.KEYWORD_ONLY, (
        f"input_path kind: {input_path.kind}"
    )
    assert input_path.default is None, (
        f"input_path default: {input_path.default!r}"
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


def test_existing_tests_still_collect() -> None:
    """D-40: existing tests still import and collect.

    Loads the target test modules by absolute file path via ``importlib``
    so the check doesn't depend on ``tests`` being resolvable as a package
    on ``sys.path`` — a sibling repository can shadow that name when
    multiple projects share a ``PYTHONPATH`` root.
    """
    import importlib.util

    tests_dir = Path(__file__).parent
    for name in ("test_cli", "test_integration", "test_loop", "test_phases"):
        spec = importlib.util.spec_from_file_location(
            f"_d40_probe_{name}", tests_dir / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


async def test_structural_meta_stack_flows_end_to_end(
    multi_stack_target: Path, monkeypatch
) -> None:
    """End-to-end smoke for the structural meta-stack pipeline (Tasks 2-7 composed).

    Drives the deep orchestrator on a real multi-language diff (Python + TSX +
    Markdown — a code diff, so NOT docs-only) and asserts the four observable
    states the structural pipeline must produce:

      1. ``detect_stacks`` returned ``structure`` as one of the stacks.
      2. ``phase_per_stack_reviews`` produced a ``stack-structure-review.md``
         artifact on disk.
      3. The merge prompt received the ``stack-structure-records.json`` path
         via ``structural_records_path``.
      4. The final merged report on disk carries a ``## Structural Review``
         header.

    These are real observable side effects (files on disk, the path threaded
    into the merge call, the rendered report header) — not dispatch bookkeeping.
    If any wire in Tasks 2-7 were broken (structure stack not emitted, prompt
    not routed, records not partitioned out and forwarded, section not
    rendered) the corresponding assertion below fails.
    """
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep import detection as _detection
    from daydream.deep import prompts as _prompts
    from daydream.deep.detection import StackAssignment

    detected_stacks: list[StackAssignment] = []
    real_detect = _detection.detect_stacks

    def _spy_detect(changed_files, *, skill_availability):
        result = real_detect(changed_files, skill_availability=skill_availability)
        detected_stacks.extend(result)
        return result

    merge_kwargs: dict = {}
    real_build_merge = _prompts.build_merge_prompt

    def _spy_merge(**kwargs):
        merge_kwargs.update(kwargs)
        return real_build_merge(**kwargs)

    # ``detect_stacks`` is imported into the orchestrator namespace; patch there.
    monkeypatch.setattr("daydream.deep.orchestrator.detect_stacks", _spy_detect)
    # ``build_merge_prompt`` is imported lazily inside phase_cross_stack_merge;
    # patch the source module so the late import resolves to the spy.
    monkeypatch.setattr("daydream.deep.prompts.build_merge_prompt", _spy_merge)

    backend = _ClaudeShape(multi_stack_target)
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

    # (3) The merge prompt received the structural records path.
    structural_records_path = merge_kwargs.get("structural_records_path")
    assert structural_records_path is not None, (
        "merge prompt did not receive structural_records_path -- orchestrator "
        "failed to partition + forward the structural records"
    )
    assert structural_records_path.name == "stack-structure-records.json", (
        f"unexpected structural records filename: {structural_records_path.name}"
    )

    # (4) The merged report on disk carries the dedicated structural section.
    merged_report = (multi_stack_target / REVIEW_OUTPUT_FILE).read_text()
    assert "## Structural Review" in merged_report, (
        "merged report is missing the ## Structural Review header"
    )
