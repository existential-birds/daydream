"""Pi discovery uses admitted diff references even for small changes."""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep.detection import StackAssignment
from daydream.phases import phase_per_stack_reviews
from daydream.run_context import InteractionPolicy, RunContext


async def test_small_pi_review_keeps_structural_dispatch_and_empty_inline_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_work: Any,
) -> None:
    (tmp_path / "app.py").write_text("value = 'DIFF_SENTINEL'\n")
    diff = tmp_path / "diff.patch"
    diff.write_text("diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
                    "@@ -1 +1 @@\n-value = 0\n+value = 'DIFF_SENTINEL'\n")
    intent = tmp_path / "intent.md"
    intent.write_text("Change the value")
    calls: list[str] = []

    async def review(*args: Any, **kwargs: Any) -> Any:
        prompt = args[2]
        assert str(diff) in prompt
        assert "DIFF_SENTINEL" not in prompt
        assert kwargs["read_only"] is True
        assert not kwargs.get("tools_disabled")
        inputs = kwargs["sanctioned_inputs"]
        inputs.revalidate(args[0], args[1], True)
        assert "DIFF_SENTINEL" not in inputs.finalization_text(args[0], args[1], True)
        assert "DIFF_SENTINEL" not in repr(kwargs["finalization_context"])
        calls.append(prompt)
        return {"issues": [], "verdicts": []}, None, None

    monkeypatch.setattr("daydream.phases.run_agent", review)
    results, failures = await phase_per_stack_reviews(
        PiBackend(model="fixture"), make_work(tmp_path),
        [StackAssignment("python", ["app.py"]), StackAssignment("structure", ["app.py"])],
        diff_path=diff, diff_text=diff.read_text(), intent_path=intent,
        alternatives_path=tmp_path / "alternatives.json", allow_standalone=True,
        write_coverage_receipts=True, run_context=RunContext(InteractionPolicy(interactive=False)),
    )
    assert failures == {}
    assert set(results) == {"python", "structure"}
    assert len(calls) == 2
    deep = tmp_path / ".daydream/deep"
    assert not (deep / "structural-delegation.json").exists()
    receipts = json.loads((deep / "coverage-receipts.json").read_text())
    assert all(entry["inline_files"] == [] for entry in receipts.values())
    assert all("source_packet_files" not in entry for entry in receipts.values())


@pytest.mark.parametrize("diff_bytes", [4096, 3_690_129])
@pytest.mark.parametrize("source_read", [True, False])
async def test_runner_pi_diff_reference_reaches_sweep_and_preserves_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any,
    diff_bytes: int, source_read: bool,
) -> None:
    import hashlib
    import os
    import re
    from collections.abc import AsyncGenerator

    from daydream.artifact_visibility import artifact_session_active
    from daydream.backends import AgentEvent, BackendExecutionInput, ResultEvent, ToolResultEvent, ToolStartEvent
    from daydream.runner import run
    from tests.deep_orchestrator.support import _uncovered_sweep_target
    from tests.harness.git_helpers import git
    from tests.harness.protocol_cli import install_protocol_cli
    from tests.harness.review_profile import independent_alternatives_profile
    from tests.harness.stub_backend import StubBackend

    target = _uncovered_sweep_target(tmp_path)
    sentinel = "DIFF_CONTENT_1321_SENTINEL"
    notes = target / "notes.txt"
    first_line = f"expiry_ms=5 # minimum_ms=1000; {sentinel}\n"
    notes.write_text(first_line + "line\n" * 5)
    initial_diff = git(target, "diff", "main", "--", "notes.txt") + "\n"
    notes.write_text(first_line + "line\n" * 4 + "x" * (diff_bytes - len(initial_diff.encode())) + "line\n")
    git(target, "add", "notes.txt")
    git(target, "commit", "-m", "add sweep acceptance diff")
    assert len((git(target, "diff", "main", "--", "notes.txt") + "\n").encode()) == diff_bytes
    stub = StubBackend(target)
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"notes.txt"})
    stub.sweep_no_read = not source_read
    stub.merge_echo_records = True
    sweep_prompts: list[str] = []
    referenced_phases: list[str] = []

    class PiDiscovery(PiBackend):
        async def execute(
            self, cwd: Path, prompt: str, output_schema: Any = None, continuation: Any = None,
            *args: Any, **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            kwargs.update(output_schema=output_schema, continuation=continuation)
            assert sentinel not in prompt
            match = re.search(r"^- diff: (.+)$", prompt, re.M)
            if match:
                assert artifact_session_active()
                diff_path = Path(match[1])
                assert diff_path != target / ".daydream/diff.patch"
                assert diff_path.is_file()
                referenced_phases.append(prompt)
            if "uncovered file sweep" in prompt:
                assert match is not None
                assert kwargs["read_only"] is True
                assert not kwargs.get("tools_disabled")
                assert len(prompt.encode()) < 32_768
                sweep_prompts.append(prompt)
                fixture = install_protocol_cli(tmp_path / "protocol", "pi", sanctioned_files=(diff_path,))
                execution = BackendExecutionInput.from_environment(
                    {**os.environ, "PATH": f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}",
                     "PI_CODING_AGENT_DIR": str(tmp_path / "settings")}, backend="pi",
                )
                # Exercise actual Pi transport while the live artifact session is active.
                async for _ in PiBackend(model="fixture", execution_input=execution).execute(
                    cwd, prompt, read_only=True, output_schema=kwargs.get("output_schema"),
                ):
                    pass
                observation = fixture.read_observations()[0]
                assert observation["sanctioned_reads"][str(diff_path)] == hashlib.sha256(
                    diff_path.read_bytes()[:12_289],
                ).hexdigest()
                assert not Path(observation["prompt_attachment"]).exists()
                assert observation["argv_bytes"] < 32_768
                assert observation["prompt_bytes"] < 32_768
                yield ToolStartEvent(id="diff-read", name="Read", input={"file_path": str(diff_path)})
                yield ToolResultEvent(id="diff-read", output=diff_path.read_text()[:4096], is_error=False)
            supported = {key: value for key, value in kwargs.items() if key in {
                "output_schema", "continuation", "agents", "max_turns", "read_only", "persist_session",
            }}
            async for event in stub.execute(cwd, prompt, **supported):
                if isinstance(event, ToolResultEvent) and event.id == "sweep-read-notes.txt":
                    event = ToolResultEvent(id=event.id, output=notes.read_text()[:4096], is_error=False)
                if isinstance(event, ResultEvent) and "uncovered file sweep" in prompt:
                    output = event.structured_output
                    assert isinstance(output, dict)
                    output["issues"][0].update(
                        description="Expiry is below the declared minimum",
                        evidence="notes.txt:1 sets expiry_ms=5 despite minimum_ms=1000",
                        rationale="The configured expiry violates the declared lower bound",
                    )
                yield event

    backend = PiDiscovery(model="fixture")
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    assert await run(make_config(
        target, backend="pi", output_mode="review",
        review_profile=independent_alternatives_profile(),
    )) == 0
    assert len(sweep_prompts) == 1
    assert len(referenced_phases) >= 4  # intent, wonder, discovery, sweep
    deep = target / ".daydream/deep"
    stats = json.loads((deep / "coverage-stats.json").read_text())
    assert stats["attempted_files"] == ["notes.txt"]
    assert stats["sweep_failures"] == {}
    assert stats["covered_files"] == (["notes.txt"] if source_read else [])
    assert stats["post_sweep"]["coverage_ratio"] > stats["pre_sweep"]["coverage_ratio"] if source_read else (
        stats["post_sweep"]["coverage_ratio"] == stats["pre_sweep"]["coverage_ratio"]
    )
    records = json.loads((deep / "stack-uncovered-records.json").read_text())
    assert records[0]["uid"] == "uncovered:1"
    assert records[0]["file"] == "notes.txt"
    merged = json.loads((deep / "merged-items.json").read_text())
    assert any(item["file"] == "notes.txt" and item["description"] == "Expiry is below the declared minimum"
               for item in merged["items"])
    assert "notes.txt" in (target / ".review-output.md").read_text()


async def test_non_pi_sweep_keeps_bounded_file_context_and_existing_access_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any,
) -> None:
    from daydream.runner import run
    from tests.deep_orchestrator.support import _uncovered_sweep_target
    from tests.harness.git_helpers import git
    from tests.harness.stub_backend import StubBackend

    target = _uncovered_sweep_target(tmp_path)
    (target / "notes.txt").write_text("NON_PI_EXCERPT\n" + "line\n" * 5 + "x" * 20_000 + "\n")
    git(target, "add", "notes.txt")
    git(target, "commit", "-m", "large non-Pi sweep scope")
    stub = StubBackend(target)
    monkeypatch.setattr(stub, "read_only_disposable_clone", True, raising=False)
    stub.per_stack_emit_reads = True
    stub.per_stack_unread = frozenset({"notes.txt"})
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: stub)
    assert await run(make_config(target, output_mode="review")) == 0
    sweep = next(call for call in stub.calls if "uncovered file sweep" in call["prompt"])
    assert sweep["read_only"] is False
    assert "NON_PI_EXCERPT" in sweep["prompt"]
    assert "diff excerpt truncated" in sweep["prompt"]
    assert "- diff:" not in sweep["prompt"]
    assert len(sweep["prompt"].encode()) < 32_768
    stats = json.loads((target / ".daydream/deep/coverage-stats.json").read_text())
    assert stats["sweep_failures"] == {}
    assert stats["covered_files"] == ["notes.txt"]
