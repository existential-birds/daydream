"""Deep-mode backend parity + primitive-preservation tests (D-38, D-39, D-40)."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from daydream.backends import CostEvent, ResultEvent, TextEvent
from daydream.config import REVIEW_OUTPUT_FILE


class _DeepMockBackend:
    """Prompt-dispatching mock backend. Subclasses tune cost + agents behavior."""

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
    ):
        # Record parity evidence -- D-38: no stage may pass `agents=`.
        self.agents_kwargs_seen.append(agents)
        if agents and self.raise_on_agents:
            raise NotImplementedError("Mock Codex: agents kwarg not supported")

        yield CostEvent(
            cost_usd=self.cost_usd, input_tokens=None, output_tokens=None
        )

        pl = prompt.lower()

        # Intent prompt contains "intent of these changes" + "understand".
        # Must be checked BEFORE the alternative branch because the alt-review
        # prompt also contains "intent".
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
            yield ResultEvent(structured_output={"issues": []}, continuation=None)
            return

        # Cross-stack merge prompt contains "cross-stack merge agent".
        if "cross-stack merge agent" in pl:
            self.calls.append("merge")
            (self.target_dir / REVIEW_OUTPUT_FILE).write_text(
                "# Review\n\n## Issues\n\n## Cross-Stack Issues\n"
            )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
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
        lambda name, model=None: backend,
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
        deep=True,
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
    # Parity guarantee: if any stage had passed agents=, the mock would have
    # raised NotImplementedError and run_deep would NOT have reached exit 0.
    # Extra belt-and-braces assertion:
    assert all(a in (None, False, [], {}, 0, "") for a in backend.agents_kwargs_seen), (
        f"agents kwarg was passed somewhere: {backend.agents_kwargs_seen}"
    )


def test_phase_primitives_unmodified() -> None:
    """D-39: existing phase primitives imported unchanged by run_deep."""
    from daydream.phases import (
        phase_alternative_review,
        phase_commit_push,
        phase_fix,
        phase_parse_feedback,
        phase_test_and_heal,
        phase_understand_intent,
    )

    # phase_parse_feedback gained an OPTIONAL keyword-only kwarg per plan 05-06.
    # Positional callers (runner.py:231, 580, 704) still work because
    # `input_path` is keyword-only with default None.
    sig = inspect.signature(phase_parse_feedback)
    params = list(sig.parameters.values())
    assert params[0].name == "backend", (
        f"phase_parse_feedback first param: {params[0].name}"
    )
    assert params[1].name == "cwd", (
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

    # Other primitives: first two params are (backend, cwd).
    # Don't over-specify beyond that -- the tail may grow with kwargs in future
    # phases without violating the D-39 contract.
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
        assert params[1].name in ("cwd", "target_dir"), (
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
    """D-40: existing tests still import and collect."""
    import tests.test_cli  # noqa: F401
    import tests.test_integration  # noqa: F401
    import tests.test_loop  # noqa: F401
    import tests.test_phases  # noqa: F401
