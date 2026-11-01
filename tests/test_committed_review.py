"""Review admission, revision and failure contracts; native journeys own transport proof."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import tracemalloc
import zlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

from daydream import agent, clock, committed_review, git_ops, json_utils
from daydream.artifact_visibility import ArtifactSession, artifact_session_active, open_artifact_session
from daydream.backends import MetricsEvent, TextEvent
from daydream.backends.pi import PiBackend, PiError
from daydream.committed_review import CommittedReviewRequest, Review, run_committed_review
from daydream.committed_review_model import planned_work_id, validate_committed_result
from daydream.deep.detection import StackAssignment
from daydream.git_ops.captured_source import capture_source
from daydream.host_tools import HostToolBinding, HostToolCallBudgetExceeded, HostToolFatal, HostToolRejected
from daydream.hosted_review import HostedReviewPolicy, UnsupportedHostedPolicy
from daydream.hunk_index import write_hunk_index
from daydream.immutable_review import ImmutableReviewInput
from daydream.pr_review import parse_finding_markers, post_findings_from_artifact
from daydream.review_budget import ReviewLimits
from daydream.review_profile import ResolvedProfile, build_default_profile
from daydream.review_result import validate_terminal_result
from daydream.review_source_refs import source_options
from daydream.review_stage_inputs import StageInputFactory
from daydream.run_config import RunConfig
from daydream.trajectory import get_current_recorder
from tests.harness.fake_clock import FakeClock
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import (
    commit,
    git,
    init_repo,
    seed_feature_branch,
    tracked_source_state,
    work_context,
    write_and_stage,
)
from tests.harness.trajectory import root_trajectory


def _matches_link_destination(destination: Any, target: Path, *, dir_fd: int | None) -> bool:
    """Select the requested output, including installs anchored to its real parent."""
    if Path(destination) == target:
        return True
    if dir_fd is None or Path(destination) != Path(target.name):
        return False
    parent, expected = os.fstat(dir_fd), target.parent.stat()
    return (parent.st_dev, parent.st_ino) == (expected.st_dev, expected.st_ino)


@pytest.fixture
def operator_policy(tmp_path: Path) -> HostedReviewPolicy:
    """Nominal isolated authority; each journey owns its faults and explicit budgets."""
    operator = tmp_path / "operator"
    operator.mkdir()
    return HostedReviewPolicy(
        agent_dir=operator, environment={"PATH": os.environ["PATH"]}, output_dir=tmp_path / "output",
        private_base=tmp_path / "private", profile=ResolvedProfile(build_default_profile(), "operator"),
        trusted_prepared_source=True,
    )


@pytest.fixture
def review_inputs(tmp_path: Path) -> StageInputFactory:
    """Full three-hunk inventories exceed the bounded citations used by judgments."""
    repo = tmp_path / "subject"
    base_contents = {
        "api.py": "VALUE = 1\nOTHER = 1\n" + "# unchanged context\n" * 12 + "SECOND = 1\n"
        + "# unchanged context\n" * 12 + "THIRD = 1\n",
        "caller.py": "VALUE = 1\nOTHER = 1\n",
        "support/helper.py": "SUPPORT = True\n",
    }
    seed_feature_branch(repo, base=base_contents,
                        feature={name: contents.replace("= 1", "= 0") for name, contents in base_contents.items()})
    return _inputs(repo, tmp_path / "inputs", ["api.py", "caller.py"])


def _inputs(repo: Path, root: Path, files: list[str], patch: str | None = None) -> StageInputFactory:
    root.mkdir()
    base, head = git(repo, "rev-parse", "main"), git(repo, "rev-parse", "HEAD")
    patch = patch if patch is not None else git_ops.diff(repo, base, head)
    diff_path = root / "diff.patch"
    diff_path.write_text(patch)
    revision = {"head_sha": head, "merge_base_sha": base, "diff_key": hashlib.sha256(patch.encode()).hexdigest()}
    return StageInputFactory(PiBackend(model="unexecuted-domain-test"), work_context(repo, run_id="domain"),
        StackAssignment("committed", files), diff_path=diff_path, hunk_index_path=write_hunk_index(root, patch),
        shared_paths={}, revision=revision, artifact_session=None, allow_standalone=True, read_only=True)


def _selectors(inputs: StageInputFactory, file: str = "api.py") -> list[dict[str, Any]]:
    part = next(p for batch in inputs.assignment_batches for p in batch if p["file"] == file)
    ref = next(option for option in source_options(part["source_refs"]) if option["side"] == "new")
    return [{"source_id": ref["source_id"], "basis": "supplied_diff", "source_range": None}]


def _proposal(inputs: StageInputFactory) -> dict[str, Any]:
    return {"uid": "", "revision": 0, "source_uids": [], "title": "Zero breaks the caller", "payload": {
        "claim": "The new zero reaches a divisor.", "trigger": "The caller divides by VALUE.",
        "consequence": "A request fails.", "grounds": "The changed assignment feeds the caller.",
        "file": "api.py", "line": 1, "evidence_refs": _selectors(inputs)}}


def _discovery(inputs: StageInputFactory) -> dict[str, Any]:
    return {"targets": [
        {"target_id": part["target_id"], "status": "reviewed", "reason": "Captured change inspected.",
         "evidence_refs": _selectors(inputs, part["file"])}
        for batch in inputs.assignment_batches for part in batch
    ], "claims": [_proposal(inputs)]}


def _verdict(inputs: StageInputFactory, claim: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"uid": claim["uid"], "revision": claim["revision"], "verdict": "confirmed",
            "grounds": "Original source corroborates the trigger after counterevidence search.",
            "evidence_refs": _selectors(inputs), **fields}


def _assignment(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The dispatch's assignment, carried in compaction-stable system instructions."""
    text = kwargs["review_system_instructions"]
    value: dict[str, Any] = json.loads(text.split("\nCommitted review assignment:\n")[1]
                                       .split("\nEnd committed review assignment.\n")[0])
    return value


def _owner(binding: HostToolBinding) -> Review:
    """The Review serving this binding's read-only lookup."""
    owner: Review = getattr(binding.host_tools["review_source"].callback, "__self__")
    return owner


def _looked_up(binding: HostToolBinding) -> list[dict[str, Any]]:
    """Every planned target's mapping through the host lookup, paging the catalog."""
    lookup = binding.host_tools["review_source"].callback
    parts: list[dict[str, Any]] = []
    cursor = None
    while True:
        page = lookup({"target_id": None, "path": None, "cursor": cursor})
        parts.extend(lookup({"target_id": entry["target_id"], "path": None, "cursor": None})["target"]
                     for entry in page["entries"])
        if (cursor := page["next_cursor"]) is None:
            return parts


def _forge_completion(value: dict[str, Any]) -> None:
    terminal = value["terminal_result"]
    for outcome in terminal["stack_outcomes"]:
        outcome.update(status="complete", reason_codes=[], partial_evidence=False)
        for decision in outcome["target_decisions"]:
            decision["status"] = "reviewed"
    terminal["phase_outcomes"][0].update(status="complete", reason_codes=[])
    terminal.update(analysis_state="complete", pipeline_state="completed", reason_codes=[],
                    completed_stacks=[outcome["scope_id"] for outcome in terminal["stack_outcomes"]],
                    uncovered_stacks=[])
    validate_terminal_result(terminal)


async def _submit(review: Review, kind: str, value: dict[str, Any]) -> dict[str, Any]:
    binding, schema = review.assignment(kind, deadline=clock.monotonic() + 30)
    binding.configure_schema(schema)
    async with binding.session():
        binding.admit("terminal", "structured_output", value, terminal=True)
        return binding.submit("terminal", value)


@pytest.mark.parametrize("boundary", ["before-start", "between-attempts"])
async def test_assignment_stays_current_until_explicit_closure(
    review_inputs: StageInputFactory, boundary: str,
) -> None:
    """Unlike in-session admission races, idle transports cannot replace still-authorized challenge work."""
    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    await _submit(review, "discovery", _discovery(review_inputs))
    prior = review.pending()
    binding, _ = review.assignment("challenge", deadline=clock.monotonic() + 30)
    if boundary == "between-attempts":
        with pytest.raises(PiError, match="transient"):
            async with binding.session():
                raise PiError("transient native failure")
    with pytest.raises(ValueError, match="active assignment"):
        review.assignment("challenge", deadline=clock.monotonic() + 30)
    assert review.pending() == prior
    binding.close_authority()
    replacement, _ = review.assignment("challenge", deadline=clock.monotonic() + 30)
    with pytest.raises(HostToolFatal, match="closed"):
        async with binding.session():
            pytest.fail("closed assignment reopened")
    async with replacement.session():
        value = {"verdicts": [_verdict(review_inputs, prior[0])], "claims": []}
        replacement.admit("replacement", "structured_output", value, terminal=True)
        replacement.submit("replacement", value)
    assert len(review.seal({})["findings"]) == 1 and review.pending() == []


@pytest.mark.parametrize(
    "change", ["claim", "trigger", "consequence", "file", "line", "grounds", "evidence_refs", "synthesis", "title"]
)
async def test_challenge_applies_to_semantic_revision_and_synthesis_but_not_title(
    review_inputs: StageInputFactory,
    change: str,
) -> None:
    """Each semantic field renews challenge; legacy phase tests have no revision authority."""
    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    await _submit(review, "discovery", _discovery(review_inputs))
    previous = review.pending()[0]
    revised = {k: deepcopy(previous[k]) for k in ("uid", "revision", "source_uids", "title", "payload")}
    revised["payload"]["evidence_refs"] = _selectors(review_inputs)
    if change == "synthesis":
        revised.update(uid="", revision=0, source_uids=[previous["uid"]])
        revised["payload"]["claim"] = "A combined failure across caller and API."
    elif change == "title":
        revised["title"] = "**Zero** breaks caller"
    elif change == "file":
        revised["payload"].update(file="caller.py", evidence_refs=_selectors(review_inputs, "caller.py"))
    elif change == "line":
        revised["payload"]["line"] = 2
    elif change == "evidence_refs":
        revised["payload"]["evidence_refs"][0]["source_range"] = {"start": 1, "end": 1}
    else:
        revised["payload"][change] += " New source-supported meaning."
    verdict = _verdict(review_inputs, previous)
    await _submit(review, "challenge", {"verdicts": [verdict], "claims": [revised]})
    if change == "title":
        assert review.pending() == []
        value = review.seal({})
        assert value["findings"][0]["title"] == revised["title"]
        assert value["findings"][0]["revision"] == 1
    else:
        pending = review.pending()
        assert len(pending) == 1
        assert pending[0]["revision"] == (1 if change == "synthesis" else 2)
        if change == "synthesis":
            assert pending[0]["uid"] != previous["uid"] and pending[0]["source_uids"] == [previous["uid"]]
        else:
            assert pending[0]["challenge_evidence"][0]["revision"] == 1
            stale_binding, schema = review.assignment("challenge", deadline=clock.monotonic() + 30)
            stale_binding.configure_schema(schema)
            async with stale_binding.session():
                with pytest.raises(HostToolRejected, match="current revision"):
                    stale = {"verdicts": [verdict], "claims": []}
                    stale_binding.admit("stale", "structured_output", stale, terminal=True)
                    stale_binding.submit("stale", stale)
            stale_binding.close_authority()
            assert review.pending() == pending
        rejection = _verdict(review_inputs, pending[0], verdict="rejected",
                             grounds="The captured caller guards zero before division.")
        await _submit(review, "challenge", {"verdicts": [rejection], "claims": []})
        value = review.seal({})
        rejected = next(c for c in value["claims"] if c["uid"] == pending[0]["uid"])
        assert rejected["disposition"] == "rejected"
        assert rejected["challenge_evidence"][-1]["grounds"] == rejection["grounds"]
        assert all(c["uid"] != pending[0]["uid"] for c in value["findings"])
    with pytest.raises(ValueError, match="sealed"):
        review.assignment("challenge", deadline=clock.monotonic() + 30)


@pytest.mark.parametrize("fault", ["invalid-selector", "shared-capacity", "deadline", "closed", "capture-integrity"])
async def test_whole_admission_rejection_preserves_prior_work_and_obligations(
    review_inputs: StageInputFactory,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Late checks cannot partially accept a new claim or consume semantic capacity.

    Existing native correction tests exercise a single stage; they do not protect
    a previously accepted claim/revision while a complete later update is rejected.
    """
    if fault == "capture-integrity":
        review_inputs.captured_source = capture_source(
            review_inputs.work.repo, review_inputs.canonical[0].path.parent / "capture",
            base_sha=review_inputs.revision['merge_base_sha'], head_sha=review_inputs.revision['head_sha'], local=False)
    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    await _submit(review, "discovery", _discovery(review_inputs))
    prior = review.pending()
    update = {"verdicts": [_verdict(review_inputs, prior[0])], "claims": [_proposal(review_inputs)]}
    if fault == "shared-capacity":
        compact = _proposal(review_inputs)
        for key in ("claim", "trigger", "consequence", "grounds"):
            compact["payload"][key] = "source"
        compact["payload"]["evidence_refs"] *= 4
        update["claims"] = [deepcopy(compact) for _ in range(127)]
    if fault == "invalid-selector":
        update["claims"][0]["payload"]["evidence_refs"][0]["source_id"] = "source:" + "f" * 64
    binding, schema = review.assignment("challenge", deadline=clock.monotonic() + 30)
    binding.configure_schema(schema)
    injected = False
    real_open = os.open
    async with binding.session():
        binding.admit("terminal", "structured_output", update, terminal=True)
        with monkeypatch.context() as patching:

            def close_during_source_read(path: Any, *args: Any, **kwargs: Any) -> int:
                nonlocal injected
                if fault == "capture-integrity" and path == "caller.py":
                    injected = True
                    raise OSError("final native integrity read failed")
                fd = real_open(path, *args, **kwargs)
                if Path(path) == review_inputs.canonical[0].path and not injected:
                    injected = True
                    if fault == "deadline":
                        binding.deadline = clock.monotonic() - 1
                    else:
                        binding.close_authority()
                return fd

            if fault in {"deadline", "closed", "capture-integrity"}:
                patching.setattr(os, "open", close_during_source_read)
            with pytest.raises((HostToolRejected, HostToolFatal)) as caught:
                binding.submit("terminal", update)
        if fault == "capture-integrity":
            assert injected and isinstance(caught.value, HostToolFatal) and caught.value.code == 'input_unavailable'
        if fault in {"deadline", "closed"}:
            assert isinstance(caught.value, HostToolFatal)
            assert injected and caught.value.code == ("deadline_expired" if fault == "deadline" else "authority_closed")
    assert review.pending() == prior
    value = review.seal({})
    assert value["terminal_result"]["analysis_state"] == "incomplete"
    assert value["findings"] == [] and len(value["claims"]) == 1


@pytest.mark.parametrize(
    "challenge", [
        "missing", "timeout", "accepted-then-failed", "last-call-quota", "exhausted", "setup", "output-install",
        "cancelled", "preparation",
        "report-stage", "bundle-stage", "beyond-512-parts", "capacity-refusal", "hub-export", "hosted-output-install",
        "hosted-bundle-install",
    ]
)
async def test_program_retains_work_and_reserves_challenge_after_discovery(
    review_inputs: StageInputFactory,
    monkeypatch: pytest.MonkeyPatch,
    challenge: str,
) -> None:
    """Real host/Git/output/recorder composition preserves work across execution failures.

    Distinct rows cover cancellation usage, preparation time, typed terminal quota, protected allowances, pre-dispatch
    refusals, pipeline-only strategy fallback, and independent capture cleanup.
    Native journeys separately own transport/history/accounting.
    """
    output_failure = challenge in {
        "output-install", "report-stage", "bundle-stage", "hosted-output-install", "hosted-bundle-install",
    }
    injected = staging_output = False
    dispatches: list[dict[str, Any]] = []
    challenge_started = asyncio.Event()
    bindings: list[HostToolBinding] = []
    captures: list[Path] = []
    if challenge == "preparation":
        fake = FakeClock(monotonic_value=100).install(monkeypatch)
        real_open = os.open

        def charge_preparation(path: Any, *args: Any, **kwargs: Any) -> int:
            nonlocal injected
            fd = real_open(path, *args, **kwargs)
            if not injected and Path(path).name == "diff.patch":
                injected = True
                fake.advance(10)
            return fd

        monkeypatch.setattr(os, "open", charge_preparation)

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, str | None]:
        assignment = _assignment(kwargs)
        dispatches.append(
            {"kind": assignment["kind"], "allocation": kwargs["tool_call_budget"], "deadline": kwargs["deadline"]}
        )
        if assignment["kind"] == "discovery" and challenge in {"exhausted", "cancelled"}:
            captures.append(cwd)
        binding = kwargs["host_tools"]
        bindings.append(binding)
        binding.configure_schema(kwargs["output_schema"])
        binding.restrict(deadline=kwargs["deadline"], tool_call_budget=kwargs["tool_call_budget"])
        if assignment["kind"] == "challenge" and challenge == "timeout":
            raise TimeoutError("native challenge deadline")
        async with binding.session():
            if assignment["kind"] == "challenge" and challenge == "cancelled":
                recorder = get_current_recorder()
                assert recorder is not None
                async with recorder.invocation(phase=kwargs["phase"]) as invocation:
                    invocation.observe(TextEvent(text="Counterevidence search is in progress."))
                    invocation.observe(MetricsEvent(
                        message_id="observed-challenge", prompt_tokens=17, completion_tokens=3,
                        cached_tokens=None, cost_usd=0.002,
                    ))
                    challenge_started.set()
                    await asyncio.Event().wait()
            if assignment["kind"] == "discovery" and challenge == "beyond-512-parts":
                # More than 512 parts: progressive atomic batches, then a final batch.
                decisions = [{"target_id": part["target_id"], "status": "reviewed", "reason": "",
                              "evidence_refs": [{"source_id": part["source_options"][0]["source_id"],
                                                 "basis": "supplied_diff", "source_range": None}]}
                             for part in _looked_up(binding)]
                assert len(decisions) > 512
                # Four bounded review_admit batches plus the final batch fit the five-call allocation.
                for number, start in enumerate(range(0, 440, 110)):
                    batch = {"targets": decisions[start:start + 110], "claims": []}
                    receipt = _charged(binding, "review_admit", batch)
                    assert receipt["progress"]["decided"] == start + 110 and not receipt["final"]
                    assert receipt["capacity"]["remaining_tool_calls"] == 4 - number
                    assert receipt["capacity"]["protected"]["challenge_tool_calls"] == 5
                proposal = _proposal(review_inputs)
                # Scope-local source handles are assigned over this run's full population.
                proposal["payload"]["evidence_refs"] = decisions[0]["evidence_refs"]
                value = {"targets": decisions[440:], "claims": [proposal]}
            elif assignment["kind"] == "discovery":
                value = _discovery(review_inputs)
                for number in range(kwargs["tool_call_budget"] - 1):
                    binding.admit(f"read-{number}", "read", {})
            else:
                value = {"verdicts": [] if challenge == "missing" else [
                    _verdict(review_inputs, assignment["claims"][0])], "claims": []}
                if challenge == "beyond-512-parts":
                    part = _looked_up(binding)[0]
                    value["verdicts"][0]["evidence_refs"] = [{"source_id": part["source_options"][0]["source_id"],
                                                              "basis": "supplied_diff", "source_range": None}]
                if challenge == "last-call-quota":
                    for number in range(kwargs["tool_call_budget"] - 1):
                        binding.admit(f"challenge-read-{number}", "read", {})
                    value["verdicts"][0]["evidence_refs"][0]["source_id"] = "source:" + "0" * 64
            binding.admit("terminal", "structured_output", value, terminal=True)
            if assignment["kind"] == "challenge" and challenge == "last-call-quota":
                # Native transport can lose exception causes; the real binding's
                # typed terminal outcome must survive into committed coverage.
                with pytest.raises(HostToolCallBudgetExceeded) as rejected:
                    binding.submit("terminal", value)
                assert binding.fatal_error is rejected.value
                assert binding.calls == kwargs["tool_call_budget"]
                assert not binding.accepted and not binding.possibly_accepted
                kwargs["investigation_budget"].observed_tool_starts += binding.calls
                raise PiError("native terminal submission failed") from None
            accepted = binding.submit("terminal", value)
            kwargs["investigation_budget"].observed_tool_starts += binding.calls
            if assignment["kind"] == "challenge" and challenge == "accepted-then-failed":
                raise RuntimeError("native failed after host acceptance")
            binding.native_settled = True
        if assignment["kind"] == "discovery" and challenge == "setup":
            # The actual next canonical-input capture filesystem operation must fail.
            for item in _owner(kwargs["host_tools"]).inputs.canonical:
                item.path.unlink()
        return accepted, None, None

    monkeypatch.setattr(committed_review, "agent", SimpleNamespace(run_agent=execute, console=agent.console))
    repo = review_inputs.work.source
    profile_path: Path | None = None
    if challenge == "exhausted":
        profile_path = repo.parent / "pipeline-only.toml"
        profile_path.write_text('name = "pipeline-only"\n[pipeline]\nreviewer_investigation_budget_s = 60\n')
    if challenge == "beyond-512-parts":
        paths = [f"extra-{number:03}.py" for number in range(511)]
        for path in paths:
            (repo / path).write_text("VALUE = 1\n")
        git(repo, "add", *paths)
        commit(repo, "add files beyond the former 512-part refusal")
    output_root = repo
    hosted_policy = None
    if challenge.startswith("hosted-"):
        operator = repo.parent / "operator"
        operator.mkdir()
        output_root = repo.parent / "hosted-output"
        hosted_policy = HostedReviewPolicy(
            agent_dir=operator, environment={"PATH": os.environ["PATH"]},
            output_dir=output_root, private_base=repo.parent / "private-hosted",
            profile=ResolvedProfile(build_default_profile(), "operator"),
            limits=ReviewLimits(90, 12, 10), trusted_prepared_source=True,
        )
    if challenge in {"output-install", "hosted-output-install", "hosted-bundle-install"}:
        real_link = os.link

        def fail_output_install(source: Any, destination: Any, **kwargs: Any) -> None:
            nonlocal injected
            output_name = (
                "committed-review.json" if challenge == "hosted-bundle-install"
                else "review-output.md" if hosted_policy is not None else ".review-output.md"
            )
            if _matches_link_destination(destination, output_root / output_name, dir_fd=kwargs.get("dst_dir_fd")):
                injected = True
                raise OSError("new current report cannot install")
            real_link(source, destination, **kwargs)

        monkeypatch.setattr(os, "link", fail_output_install)
    if challenge in {"report-stage", "bundle-stage"}:
        stage, allocate = json_utils._stage_bytes, tempfile.mkstemp
        target = ".review-output.md" if challenge == "report-stage" else "committed-review.json"

        def observe_stage(path: Path, content: bytes, **kwargs: Any) -> Path:
            nonlocal staging_output
            staging_output = path.name == target
            try:
                return stage(path, content, **kwargs)
            finally:
                staging_output = False

        def fail_allocation(*args: Any, **kwargs: Any) -> tuple[int, str]:
            nonlocal injected
            if staging_output:
                injected = True
                raise OSError("committed output staging failed")
            return allocate(*args, **kwargs)

        monkeypatch.setattr(json_utils, "_stage_bytes", observe_stage)
        monkeypatch.setattr(tempfile, "mkstemp", fail_allocation)
    review_request = CommittedReviewRequest(
        repo,
        review_inputs.revision["merge_base_sha"],
        config=RunConfig(
            backend="pi", output_mode="review",
            trajectory_hub_repo="fixture/requested-export" if challenge == "hub-export" else None,
            review_profile_path=profile_path,
        ),
        limits=ReviewLimits(90, 12, 10),
        hosted_policy=hosted_policy,
    )
    if challenge == "capacity-refusal":
        # The remaining finite bound (typed coverage scopes) refuses before any model work.
        monkeypatch.setattr(committed_review, "COVERAGE_MAX_SHARDS", 0)
    if challenge in {"capacity-refusal", "hub-export"}:
        expected = "retains at most 0 assignment parts" if challenge == "capacity-refusal" else "external export"
        with pytest.raises(ValueError, match=expected):
            await run_committed_review(review_request)
        assert dispatches == []
        assert not (repo / ".daydream/committed-review.json").exists()
        assert not (repo / ".review-output.md").exists()
        return
    if challenge == "cancelled":
        task = asyncio.create_task(run_committed_review(review_request))
        started = asyncio.create_task(challenge_started.wait())
        ready, _ = await asyncio.wait({task, started}, timeout=20, return_when=asyncio.FIRST_COMPLETED)
        if started not in ready:
            started.cancel()
            await asyncio.gather(started, return_exceptions=True)
            if task.done():
                pytest.fail(f"Challenge never started: {task.result().to_dict()['execution']}")
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            pytest.fail("Challenge did not start within the test deadline")
        task.cancel("caller interrupted challenge")
        with pytest.raises(asyncio.CancelledError, match="caller interrupted challenge"):
            await task
        # Read only after both the program and artifact session have exited.
        value = json.loads((repo / ".daydream/committed-review.json").read_text())
        validate_committed_result(value)
        assert value["analyzed_revision"] == review_inputs.revision
        assert value["terminal_result"]["pipeline_state"] == "cancelled"
        assert value["terminal_result"]["analysis_state"] == "incomplete"
        assert "interruption" in value["terminal_result"]["reason_codes"]
        assert value["claims"][0]["disposition"] == "unresolved" and value["findings"] == []
        interrupted = value["execution"][-1]
        assert interrupted == {
            "assignment": "challenge", "accepted": False, "settled": False,
            "cleanup_complete": True, "reason": "interruption",
        }
        assert all(not binding.authority_open and binding.cleanup_complete for binding in bindings)
        report = (repo / ".review-output.md").read_text()
        assert value["run_id"] in report and "Unresolved concerns" in report and "interruption" in report
        trajectory = root_trajectory(repo)
        assert trajectory["session_id"] == value["run_id"] and trajectory["extra"]["partial"]
        assert trajectory["final_metrics"]["total_prompt_tokens"] == 17
        assert trajectory["final_metrics"]["total_cost_usd"] == 0.002
        captured = captures[0]
        assert not captured.exists()
        assert f"worktree {captured}\n" not in git(repo, "worktree", "list", "--porcelain")
        return
    result = await run_committed_review(review_request)
    value = result.to_dict()
    assert len(value["assignments"][0]["source_refs"]) == 6
    assert all(
        len(decision["evidence_refs"]) == 1
        for scope in value["terminal_result"]["stack_outcomes"] for decision in scope["target_decisions"]
    )
    if output_failure:
        assert injected
    assert result.output_installed is (not output_failure)
    assert result.capture_state == "complete"
    if challenge == "exhausted":
        assert json.loads((repo / ".daydream/committed-review.json").read_text()) == value
        assert root_trajectory(repo)["session_id"] == value["run_id"]
        captured = captures[0]
        assert not captured.exists()
        assert f"worktree {captured}\n" not in git(repo, "worktree", "list", "--porcelain")
        assert value["settings"]["profile_name"] == "pipeline-only"
        assert value["settings"]["profile_source"] == "explicit"
        assert value["settings"]["limits"]["effective_investigation_s"] == pytest.approx(60)
        defaults = build_default_profile().strategies
        assert value["settings"]["strategies"]["discovery"] == defaults["discovery.generic_fallback"].content
        assert value["settings"]["strategies"]["challenge"] == defaults["arbitration"].content.replace(
            "{arbiter_input_path}", "the captured-source claim inventory below"
        )
    if output_failure:
        assert result.output_error is not None
        assert not (repo / ".daydream/committed-review.json").exists()
        assert not (repo / ".review-output.md").exists()
        if hosted_policy is not None:
            assert not (output_root / "committed-review.json").exists()
            assert not (output_root / "review-output.md").exists()
        assert value["terminal_result"]["run_id"] == value["run_id"]
        assert value["analyzed_revision"]["head_sha"] == review_inputs.revision["head_sha"]
    assert value["claims"] and [e["assignment"] for e in value["execution"]] == ["discovery", "challenge"]
    assert dispatches[0]["allocation"] == 5
    if challenge != "setup":
        assert dispatches[1]["allocation"] == 5
        assert dispatches[1]["deadline"] > dispatches[0]["deadline"]
    if challenge == "preparation":
        assert injected
        assert [d["deadline"] for d in dispatches] == [145, 190]
    complete = challenge in {"exhausted", "preparation", "beyond-512-parts"}
    assert result.complete is complete
    assert value["terminal_result"]["analysis_state"] == (
        "complete" if complete or output_failure else "incomplete"
    )
    confirmed = complete or challenge == "accepted-then-failed" or output_failure
    assert len(value["findings"]) == int(confirmed)
    assert value["claims"][0]["disposition"] == ("confirmed" if confirmed else "unresolved")
    assert value["execution"][0]["accepted"]
    if challenge == "last-call-quota":
        assert value["execution"][1]["reason"] == "host_tool_budget_exhaustion"
        assert not value["execution"][1]["accepted"] and not value["execution"][1]["settled"]
        assert "host_tool_budget_exhaustion" in value["terminal_result"]["reason_codes"]
        assert value["claims"][0]["challenge_evidence"] == []
        assert all(not binding.authority_open and binding.cleanup_complete for binding in bindings)
    if challenge == "accepted-then-failed":
        assert value["execution"][1]["accepted"] and not value["execution"][1]["settled"]
        # Persisted success flags cannot erase the actual retained execution failure.
        corrupted = deepcopy(value)
        corrupted["execution"][1]["settled"] = True
        assert corrupted["execution"][1]["reason"] == "backend_failure"
        _forge_completion(corrupted)
        with pytest.raises(ValueError, match="native execution"):
            validate_committed_result(corrupted)


@pytest.mark.parametrize('corruption', ['assigned-delete', 'support-bytes'])
@pytest.mark.parametrize('new_concern', [False, True], ids=['after-final-challenge', 'before-fresh-challenge'])
async def test_local_capture_corruption_retains_challenged_findings_and_unfinished_work(
    review_inputs: StageInputFactory, monkeypatch: pytest.MonkeyPatch,
    new_concern: bool, corruption: str,
) -> None:
    """Owner integrity tests miss corruption between admitted findings and a fresh challenge.

    Corrupt assigned or unchanged excluded support after a challenge admits a
    confirmed concern. Later submission/terminal checks must preserve that finding
    and unfinished work without consulting intact live bytes.
    """
    repo = review_inputs.work.source
    original = tracked_source_state(repo)
    calls: list[dict[str, Any]] = []

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, None]:
        assignment = _assignment(kwargs)
        calls.append(assignment)
        parts = _looked_up(kwargs['host_tools'])

        def selectors(file: str) -> list[dict[str, Any]]:
            part = next(part for part in parts if part['file'] == file)
            option = next(option for option in part['source_options'] if option['side'] == 'new')
            return [{'source_id': option['source_id'], 'basis': 'supplied_diff', 'source_range': None}]

        claim = _proposal(review_inputs)
        claim['payload']['evidence_refs'] = selectors('api.py')
        if assignment['kind'] == 'discovery':
            value = {'targets': [
                {'target_id': part['target_id'], 'status': 'reviewed', 'reason': 'Captured source inspected.',
                 'evidence_refs': selectors(part['file'])} for part in parts], 'claims': [claim]}
        else:
            claim['payload'].update(file='caller.py', grounds='A distinct changed caller needs fresh challenge.',
                                    evidence_refs=selectors('caller.py'))
            verdict = _verdict(review_inputs, assignment['claims'][0], evidence_refs=selectors('api.py'))
            value = {'verdicts': [verdict], 'claims': [claim] if new_concern else []}
        binding = kwargs['host_tools']
        binding.restrict(deadline=kwargs['deadline'], tool_call_budget=kwargs['tool_call_budget'])
        async with binding.session():
            binding.admit('terminal', 'structured_output', value, terminal=True)
            accepted = binding.submit('terminal', value)
            kwargs['investigation_budget'].observed_tool_starts += binding.calls
            binding.native_settled = True
        if assignment['kind'] == 'challenge':
            if corruption == 'assigned-delete':
                (cwd / 'api.py').unlink()
            else:
                (cwd / 'support/helper.py').write_text('SUPPORT = False\n')
        return accepted, None, None

    monkeypatch.setattr(committed_review, 'agent', SimpleNamespace(run_agent=execute, console=agent.console))
    result = await run_committed_review(CommittedReviewRequest(
        repo, review_inputs.revision['merge_base_sha'], local=True, limits=ReviewLimits(90, 12, 12),
        config=RunConfig(backend='pi', output_mode='review', ignore_paths=['support']),
    ))
    value = result.to_dict()
    assert result.output_installed and not result.complete
    assert [call['kind'] for call in calls] == ['discovery', 'challenge']
    assert value['schema_version'] == 6 and value['terminal_result']['schema_version'] == 2
    validate_committed_result(value)
    relabeled = deepcopy(value)
    relabeled['schema_version'] = 4
    with pytest.raises(ValueError, match='result version does not match captured source identity'):
        validate_committed_result(relabeled)
    # Pre-navigation results (single-scope coverage, 512-part inventories) are not converted.
    for superseded in (1, 2, 3):
        relabeled['schema_version'] = superseded
        with pytest.raises(ValueError, match='unsupported committed review result version; restart review'):
            validate_committed_result(relabeled)
    assert value['analyzed_revision']['head_sha'] == review_inputs.revision['head_sha']
    assert value['analyzed_revision']['source_id'] == calls[0]['analyzed_revision']['source_id']
    assert value['terminal_result']['analysis_state'] == 'incomplete'
    assert value['terminal_result']['reason_codes'] == (
        ['backend_failure', 'evidence_incomplete'] if new_concern else ['evidence_incomplete'])
    assert len(value['claims']) == (2 if new_concern else 1) and len(value['findings']) == 1
    assert value['findings'][0]['uid'] == calls[1]['claims'][0]['uid']
    assert all(entry['file'] != 'support/helper.py' for entry in calls[0]['catalog']['entries'])
    assert value['findings'][0]['payload']['file'] == 'api.py'
    assert value['findings'][0]['payload']['evidence_refs'] == value['claims'][0]['payload']['evidence_refs']
    if new_concern:
        assert value['claims'][1]['disposition'] == 'unresolved'
        assert value['execution'][-2]['assignment'] == 'challenge' and not value['execution'][-2]['accepted']
    assert value['execution'][-1]['assignment'] == 'source_integrity' and not value['execution'][-1]['accepted']
    assert tracked_source_state(repo) == original


@pytest.mark.parametrize("cleanup_change", ["none", "source-replacement", "git-routing"])
async def test_committed_capture_cleanup_keeps_ownership_after_session_close(
    review_inputs: StageInputFactory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    cleanup_change: str,
) -> None:
    """Independent capture cleanup ignores source replacement and preserves foreign storage.

    Runner cleanup tests do not exercise the committed API lifetime. Replacing its
    private Git directory must retain competing bytes without masking cancellation.
    """
    from tests.test_artifact_visibility import _Entry, _walk_entry

    repo = review_inputs.work.source
    captured_root: Path | None = None
    foreign_roots: list[Path] = []
    foreign_before: list[_Entry] = []
    primary = asyncio.CancelledError("caller cancelled committed review")

    @asynccontextmanager
    async def close_then_change(*args: Any, **kwargs: Any) -> AsyncIterator[ArtifactSession]:
        nonlocal captured_root
        try:
            async with open_artifact_session(*args, **kwargs) as session:
                captured_root = session.layout.repo
                yield session
        finally:
            assert captured_root is not None and not artifact_session_active()
            if cleanup_change == "source-replacement":
                removed = repo.with_name("removed-source")
                repo.rename(removed)
                seed_feature_branch(repo, base={"foreign.py": "FOREIGN = 1\n"},
                                    feature={"foreign.py": "FOREIGN = 2\n"})
                foreign_roots.append(repo)
            elif cleanup_change == "git-routing":
                foreign = repo.with_name("foreign")
                seed_feature_branch(foreign, base={"foreign.py": "FOREIGN = 1\n"},
                                    feature={"foreign.py": "FOREIGN = 2\n"})
                linked = repo.with_name("foreign-linked")
                git_ops.worktree_add(foreign, linked, "HEAD")
                foreign_git, _ = git_ops.git_dirs(linked)
                (captured_root / ".git").rename(captured_root / ".retained-git")
                (captured_root / ".git").write_text(f"gitdir: {foreign_git}\n", encoding="utf-8")
                foreign_roots.extend((foreign, linked))
            for root in foreign_roots:
                _walk_entry(repo.parent, root, foreign_before)

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, None]:
        binding = kwargs["host_tools"]
        binding.configure_schema(kwargs["output_schema"])
        async with binding.session():
            value = _discovery(review_inputs)
            value["claims"] = []
            binding.admit("terminal", "structured_output", value, terminal=True)
            accepted = binding.submit("terminal", value)
            if cleanup_change == "git-routing":
                raise primary
            binding.native_settled = True
        return accepted, None, None

    monkeypatch.setattr(committed_review, "open_artifact_session", close_then_change)
    monkeypatch.setattr(committed_review, "agent", SimpleNamespace(run_agent=execute, console=agent.console))
    review_request = CommittedReviewRequest(repo, review_inputs.revision["merge_base_sha"])
    if cleanup_change == "git-routing":
        with pytest.raises(asyncio.CancelledError) as caught:
            await run_committed_review(review_request)
        assert caught.value is primary
        payload = json.loads((repo / ".daydream/committed-review.json").read_text())
        validate_committed_result(payload)
        assert payload["terminal_result"]["pipeline_state"] == "cancelled"
    else:
        result = await run_committed_review(review_request)
        assert result.complete and result.output_installed
    assert captured_root is not None
    warning = capsys.readouterr().out
    if cleanup_change == "git-routing":
        assert captured_root.is_dir()
        assert "Captured source cleanup skipped" in warning and "retained" in warning
    else:
        assert not captured_root.exists()
        assert "Captured source cleanup" not in warning
    foreign_after: list[_Entry] = []
    for root in foreign_roots:
        _walk_entry(repo.parent, root, foreign_after)
    assert foreign_after == foreign_before


@pytest.mark.parametrize("fault", [
    "preparation", "backend", "empty-provider", "blank-provider", "invalid-provider",
    "empty-model", "blank-model", "invalid-model", "output-overlap", "missing-agent-dir", "file-agent-dir",
])
async def test_hosted_policy_refuses_unsupported_authority_before_source_preparation(
    review_inputs: StageInputFactory, operator_policy: HostedReviewPolicy, fault: str,
) -> None:
    """Hosted admission owns preparation authority; ordinary native tests cannot prove it.

    A post-checkout hook would execute before raw-byte validation if policy admission
    moved below worktree creation. No unsupported request may reach that effect.
    Missing/file configuration roots must produce the same actionable policy error,
    rather than escaping as raw filesystem errors during native context binding.
    Blank/non-string selections must fail here even though arbitrary nonblank
    identifiers are delegated to the isolated native configuration.
    """
    repo = review_inputs.work.source
    marker = repo.parent / "checkout-executed"
    hook = repo / ".git/hooks/post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o755)
    policy = operator_policy
    if fault == "missing-agent-dir":
        policy.agent_dir.rmdir()
    elif fault == "file-agent-dir":
        policy.agent_dir.rmdir()
        policy.agent_dir.write_text("not a configuration directory")
    changes: dict[str, dict[str, Any]] = {
        "preparation": {"trusted_prepared_source": False}, "backend": {"backend": "claude"},
        "empty-provider": {"provider": ""}, "blank-provider": {"provider": " \t\n"},
        "invalid-provider": {"provider": None},
        "empty-model": {"model": ""}, "blank-model": {"model": " \t\n"}, "invalid-model": {"model": 42},
        "output-overlap": {"output_dir": repo / "output"},
    }
    policy = replace(policy, **changes.get(fault, {}))
    before = git(repo, "worktree", "list", "--porcelain")
    message = ("agent_dir.*existing directory" if fault.endswith("agent-dir")
               else "nonblank.*provider and model strings" if fault.endswith(("provider", "model")) else None)
    with pytest.raises(UnsupportedHostedPolicy, match=message):
        await run_committed_review(CommittedReviewRequest(
            repo, review_inputs.revision["merge_base_sha"], hosted_policy=policy,
        ))
    assert not marker.exists()
    assert git(repo, "worktree", "list", "--porcelain") == before
    assert not policy.output_dir.exists()
    assert not policy.private_base.exists()


@pytest.mark.parametrize("corruption", ["wrong-source", "unchallenged-revision", "foreign-plan", "empty-execution"])
async def test_sealed_result_cannot_rebind_publication_or_planned_work(
    review_inputs: StageInputFactory,
    corruption: str,
) -> None:
    """Typed coverage alone cannot prove claim, source, plan or native-execution identity."""
    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    await _submit(review, "discovery", _discovery(review_inputs))
    concern = review.pending()[0]
    verdict = _verdict(review_inputs, concern)
    await _submit(review, "challenge", {"verdicts": [verdict], "claims": []})
    value = review.seal({})
    if corruption == "wrong-source":
        value["claims"][0]["payload"]["evidence_refs"][0]["revision"] = "different-head"
    elif corruption == "unchallenged-revision":
        value["claims"][0]["revision"] = 2
        value["claims"][0]["payload"]["trigger"] = "A new unchallenged trigger."
    elif corruption == "foreign-plan":
        value["assignments"][0]["target_id"] = "foreign"
        value["planned_work_id"] = planned_work_id(value["analyzed_revision"], value["assignments"])
    else:
        terminal = value["terminal_result"]
        assert terminal["analysis_state"] == "incomplete"
        assert terminal["reason_codes"] == ["evidence_incomplete"]
        assert value["findings"][0]["uid"] == concern["uid"]
        assert value["execution"] == []
        _forge_completion(value)
    value["findings"] = deepcopy(value["claims"])
    with pytest.raises(ValueError):
        validate_committed_result(value)


async def test_committed_review_materializes_raw_bytes_despite_checkout_transforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkout filters cannot change captured native bytes or the user's dirty source.

    The old refusal test covered transformed worktrees. Inert raw preparation now
    supports this input, so the real dispatch seam must observe the raw Git blob.
    """
    repo = tmp_path / "source"
    init_repo(repo)
    git(repo, "config", "filter.review.clean", "sed 's/NATIVE/CANONICAL/g'")
    git(repo, "config", "filter.review.smudge", "sed 's/CANONICAL/NATIVE/g'")
    write_and_stage(repo, ".gitattributes", "*.py filter=review\n")
    write_and_stage(repo, "api.py", "VALUE = 'CANONICAL base'\n")
    base = commit(repo, "base")
    write_and_stage(repo, "api.py", "VALUE = 'CANONICAL head'\n")
    head = commit(repo, "change")

    # Git's own clean-status check accepts the transformed worktree. That is
    # precisely why status cannot stand in for raw blob/native byte coherence.
    control = tmp_path / "transformed-control"
    git_ops.worktree_add(repo, control, head, detach=True)
    try:
        assert git(control, "status", "--porcelain") == ""
        assert (control / "api.py").read_bytes() == b"VALUE = 'NATIVE head'\n"
        assert git(control, "show", f"{head}:api.py") == "VALUE = 'CANONICAL head'"
    finally:
        git_ops.worktree_remove(repo, control, force=True)

    (repo / "api.py").write_bytes(b"VALUE = 'DIRTY_USER_BYTES'\n")
    original = tracked_source_state(repo)
    original_worktrees = git(repo, "worktree", "list", "--porcelain")

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, None]:
        assert (cwd / "api.py").read_bytes() == b"VALUE = 'CANONICAL head'\n"
        value = {"claims": [], "targets": [
            {"target_id": part["target_id"], "status": "reviewed", "reason": "Raw captured bytes inspected.",
             "evidence_refs": [{"source_id": part["source_options"][0]["source_id"],
                                "basis": "supplied_diff", "source_range": None}]}
            for part in _looked_up(kwargs["host_tools"])]}
        binding = kwargs["host_tools"]
        async with binding.session():
            binding.admit("terminal", "structured_output", value, terminal=True)
            accepted = binding.submit("terminal", value)
            binding.native_settled = True
        return accepted, None, None

    monkeypatch.setattr(committed_review, "agent", SimpleNamespace(run_agent=execute, console=agent.console))
    result = await run_committed_review(CommittedReviewRequest(repo, base))
    assert result.complete and result.to_dict()["analyzed_revision"]["head_sha"] == head
    assert tracked_source_state(repo) == original
    assert git(repo, "worktree", "list", "--porcelain") == original_worktrees


def test_committed_diff_uses_raw_commit_endpoints_despite_local_diff_drivers(tmp_path: Path) -> None:
    """Local diff drivers, colour and prefixes cannot erase raw text or metadata assignments."""
    repo = tmp_path / "source"
    init_repo(repo)
    write_and_stage(repo, ".gitattributes", "*.py diff=review\n")
    write_and_stage(repo, "api.py", "VALUE = 'base'\n")
    write_and_stage(repo, "data.bin", b"\0base\n")
    write_and_stage(repo, "mode.sh", "#!/bin/sh\n")
    base = commit(repo, "base")
    write_and_stage(repo, "api.py", "VALUE = 'head'\n")
    write_and_stage(repo, "data.bin", b"\0head\n")
    git(repo, "update-index", "--chmod=+x", "mode.sh")
    head = commit(repo, "change")
    (repo / "api.py").write_text("VALUE = 'dirty'\n")
    git(repo, "config", "diff.review.textconv", "echo DRIVER_OUTPUT")
    git(repo, "config", "diff.external", "false")
    git(repo, "config", "diff.noprefix", "true")
    git(repo, "config", "color.diff", "always")

    patch = git_ops.committed_diff(repo, base, head)

    inputs = _inputs(repo, tmp_path / "captured", ["api.py", "data.bin", "mode.sh"], patch)
    for assignment in (part for batch in inputs.assignment_batches for part in batch if part["file"] != "api.py"):
        [ref] = assignment["source_refs"]
        assert ref["side"] == "diff" and ref["fragment"]["bytes"] > 0
    assert "-VALUE = 'base'" in patch
    assert "+VALUE = 'head'" in patch
    assert "dirty" not in patch and "DRIVER_OUTPUT" not in patch
    assert "Binary files a/data.bin and b/data.bin differ" in patch
    assert "old mode 100644\nnew mode 100755" in patch


def immutable_input_for(repo: Path, *, requested_base: str | None = None) -> ImmutableReviewInput:
    """Capture independently computed original/local evidence for real Git journeys."""
    from daydream.git_ops.immutable import immutable_review_diff

    head = git(repo, "rev-parse", "HEAD")
    requested = requested_base or git(repo, "rev-parse", "main")
    comparison = git(repo, "merge-base", requested, head)
    return ImmutableReviewInput.from_dict({
        "schema_version": 1,
        "provenance": {"repository": "captured/repository", "pr_number": 1421, "head_sha": head,
            "requested_base_sha": requested, "comparison_base_sha": comparison,
            "title": "Historical captured title", "body": "Historical captured body", "context": "Captured intent."},
        "materialization": {"kind": "exact", "head_sha": head, "requested_base_sha": requested,
            "comparison_base_sha": comparison, "head_tree_sha": git(repo, "rev-parse", head + "^{tree}"),
            "comparison_tree_sha": git(repo, "rev-parse", comparison + "^{tree}"),
            "review_diff_key": hashlib.sha256(immutable_review_diff(repo, comparison, head).encode()).hexdigest(),
            "bundle_sha256": None},
    })


@pytest.mark.parametrize("route", ["copied", "bundle", "alternate"])
@pytest.mark.parametrize("fault", ["count", "delta-size"])
def test_immutable_object_population_bounds_precede_body_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, fault: str,
) -> None:
    """Compressed files cannot stand in for packed object count or reconstructed delta bytes."""
    from daydream.git_ops import immutable

    repo = tmp_path / "source"
    seed_feature_branch(repo, base={"api.py": "VALUE = 1\n"}, feature={"api.py": "VALUE = 0\n"})
    identity = immutable_input_for(repo)
    payload = tmp_path / "payload"
    oids = []
    for index in range(3 if fault == "count" else 2):
        payload.write_bytes((b"x" * (1280 * 1024) if fault == "delta-size" else b"tiny") + bytes([index]))
        oid = git(repo, "hash-object", "-w", str(payload))
        oids.append(oid)
        git(repo, "update-ref", f"refs/tags/stored-{index}", oid)
    git(repo, "-c", "pack.threads=1", "-c", "core.bigFileThreshold=1g", "repack", "-adf",
        "--window=20", "--depth=10")
    if fault == "delta-size":
        index_path, = (repo / ".git/objects/pack").glob("*.idx")
        rows = [row.split() for row in git(repo, "verify-pack", "-v", str(index_path)).splitlines()]
        assert any(row[0] in oids and len(row) == 7 and int(row[2]) < 1024 for row in rows)
        assert all(int(git(repo, "cat-file", "-s", oid)) == 1280 * 1024 + 1 for oid in oids)
    bundle = None
    if route == "bundle":
        bundle = tmp_path / "captured.bundle"
        git(repo, "bundle", "create", str(bundle), "--all")
        value = identity.to_dict()
        value["materialization"]["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
        identity = ImmutableReviewInput.from_dict(value)
    monkeypatch.setattr(immutable, "_MAX_OBJECT_BYTES", 2 * 1024 * 1024)
    if fault == "count":
        monkeypatch.setattr(immutable, "_MAX_FILES", 8)
    with pytest.raises(git_ops.GitError, match="reconstructed object population exceeds the supported size"):
        if route == "alternate":
            value = identity.to_dict()["materialization"]
            immutable.immutable_review_diff(repo, value["comparison_base_sha"], value["head_sha"])
        else:
            with immutable.prepare_immutable_source(repo, identity, tmp_path / "prepared", bundle=bundle):
                pytest.fail("oversized object population reached source admission")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("fault", ["allocation", "ref-output"])
def test_immutable_bundle_bounds_expansion_and_control_output_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """Tiny compressed bundles must fail at the resource guard, rather than later source checks."""
    from daydream.git_ops import immutable

    repo = tmp_path / "source"
    seed_feature_branch(repo, base={"api.py": "VALUE = 1\n"}, feature={"api.py": "VALUE = 0\n"})
    identity = immutable_input_for(repo)
    if fault == "allocation":
        payload = tmp_path / "compressed-payload"
        payload.write_bytes(b"x" * (3 * 1024 * 1024))
        oid = git(repo, "hash-object", "-w", str(payload))
        git(repo, "update-ref", "refs/tags/large", oid)
    else:
        head = git(repo, "rev-parse", "HEAD")
        for index in range(100):
            git(repo, "update-ref", f"refs/tags/extra-{index}", head)
    bundle = tmp_path / "captured.bundle"
    git(repo, "bundle", "create", str(bundle), "--all")
    assert bundle.stat().st_size < 16 * 1024
    value = identity.to_dict()
    value["materialization"]["bundle_sha256"] = hashlib.sha256(bundle.read_bytes()).hexdigest()
    identity = ImmutableReviewInput.from_dict(value)
    monkeypatch.setattr(immutable, "_MAX_OBJECT_BYTES", 2 * 1024 * 1024)
    if fault == "ref-output":
        monkeypatch.setattr(immutable, "_MAX_FILES", 8)
    expected = "allocate 3145729 over limit 2097153" if fault == "allocation" else "output exceeds the supported size"
    with pytest.raises(git_ops.GitError, match=expected), immutable.prepare_immutable_source(
        repo, identity, tmp_path / "prepared", bundle=bundle,
    ):
        pytest.fail("unbounded bundle reached source admission")
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_immutable_source_retains_the_documented_per_tree_blob_boundary(tmp_path: Path, offset: int) -> None:
    """Admit 8 MiB source blobs and larger historical blobs outside the compared trees."""
    from daydream.git_ops import immutable

    repo = tmp_path / "source"
    seed_feature_branch(repo, base={"api.py": "VALUE = 1\n"}, feature={"api.py": "VALUE = 0\n"})
    body = b"\0" * (8 * 1024 * 1024 + offset)
    if offset > 0:
        payload = tmp_path / "historical-blob"
        payload.write_bytes(body)
        oid = git(repo, "hash-object", "-w", str(payload))
        git(repo, "update-ref", "refs/tags/historical", oid)
    else:
        write_and_stage(repo, "boundary.bin", body)
        commit(repo, "capture a supported source blob")
    identity = immutable_input_for(repo)
    with immutable.prepare_immutable_source(repo, identity, tmp_path / "prepared") as prepared:
        if offset > 0:
            assert git(prepared, "cat-file", "-s", oid) == str(len(body))
            assert not (prepared / "boundary.bin").exists()
        else:
            assert (prepared / "boundary.bin").read_bytes() == body
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("outcome", ["complete", "attributes", "legal-ref", "packed", "empty", "no-diff", "partial",
                                    "native-failure", "install-failure", "copy-cancel"])
async def test_immutable_review_uses_only_captured_evidence_and_inert_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator_policy: HostedReviewPolicy, fake_gh: FakeGh, outcome: str,
) -> None:
    """Offline identity/export and inert preparation are absent from ordinary journey coverage.

    A live or credential lookup is recorded before raising, so swallowed cosmetic
    failures cannot pass. Hostile local/global Git configuration would execute a
    checkout filter or hook under the prior worktree materialization path.
    """
    from daydream import github_app
    from daydream.git_ops import process
    repo = tmp_path / "source"
    attributes = "api.py filter=hostile" + (" -diff" if outcome == "attributes" else "") + "\n"
    seed_feature_branch(
        repo, base={"api.py": "VALUE = 1\n", ".gitattributes": attributes},
        feature={"api.py": "VALUE = 0\n", ".gitattributes": attributes},
    )
    if outcome in {"legal-ref", "packed"}:
        branch = "fix@2+%!/東京\u2028\u00a0"
        git(repo, "check-ref-format", "refs/heads/" + branch)
        git(repo, "branch", "-m", branch)
        if outcome == "legal-ref":
            head = git(repo, "rev-parse", "--verify", "HEAD")
            (repo / ".git/HEAD").write_bytes(f"ref: refs/heads/{branch}\t \r\n".encode())
            assert git(repo, "rev-parse", "--verify", "HEAD") == head
    if outcome == "packed":
        git(repo, "repack", "-adb", "--write-bitmap-index")
        git(repo, "multi-pack-index", "write")
        pack = next((repo / ".git/objects/pack").glob("*.pack"))
        pack.with_suffix(".keep").write_text("Retain this ordinary maintained pack.\n")
        git(repo, "pack-refs", "--all")
        assert not (repo / ".git/refs/heads" / branch).exists()
    if outcome == "no-diff":
        git(repo, "update-ref", "refs/heads/main", git(repo, "rev-parse", "HEAD"))
    else:
        # The requested original base tip advances independently of the head.
        tip = git(repo, "commit-tree", git(repo, "rev-parse", "main^{tree}"), "-p",
                  git(repo, "rev-parse", "main"), "-m", "advanced original base")
        git(repo, "update-ref", "refs/heads/main", tip)
    if outcome == "attributes":
        (repo / ".git/info/attributes").write_text("api.py diff\n")
        git(repo, "config", "diff.context", "0")
        git(repo, "config", "diff.algorithm", "histogram")
        git(repo, "config", "diff.orderFile", str(tmp_path / "missing-order-file"))
        ambient = tmp_path / "ambient-gitconfig"
        ambient.write_text("[diff]\n\tcontext = 0\n\talgorithm = patience\n")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(ambient))
    identity = immutable_input_for(repo)
    envelope = identity.to_dict()
    marker = tmp_path / "forbidden-git-effect"
    hook = repo / ".git/hooks/post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o755)
    git(repo, "config", "filter.hostile.smudge", f"touch '{marker}'; cat")
    git(repo, "config", "core.autocrlf", "true")
    # Local and ambient includes are never opened by the immutable Git owner.
    git(repo, "config", "include.path", str(tmp_path / "malformed-global"))
    (tmp_path / "malformed-global").write_text("this is not valid Git configuration\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "malformed-global"))
    monkeypatch.setenv("GIT_EXEC_PATH", str(tmp_path / "missing-git-exec"))
    policy = replace(operator_policy, limits=ReviewLimits(60, 10, 10))
    attempts: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        attempts.append("GitHub invocation or credential resolution")
        raise AssertionError(attempts[-1])

    environment_get = type(os.environ).get

    def no_github_credentials(self: Any, key: Any, default: Any = None) -> Any:
        if self is os.environ and key in {
            "GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN",
            "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY",
        }:
            attempts.append(f"credential read: {key}")
            raise AssertionError(attempts[-1])
        return environment_get(self, key, default)

    async def readiness(self: PiBackend) -> None:
        return None

    monkeypatch.setattr(PiBackend, "validate_hosted_policy", readiness)
    seen: list[dict[str, Any]] = []

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, None]:
        assignment = _assignment(kwargs)
        seen.append(assignment)
        assert cwd != repo and (cwd / "api.py").read_bytes() == (repo / "api.py").read_bytes()
        assert not (cwd / ".git/hooks/post-checkout").exists()
        instructions = kwargs["review_system_instructions"]
        assert "captured/repository" not in prompt + instructions and '"pr_number"' not in prompt + instructions
        assert "Historical captured title" in instructions
        binding = kwargs["host_tools"]
        binding.configure_schema(kwargs["output_schema"])
        async with binding.session():
            part = _looked_up(binding)[0]
            assert part["source_options"][0]["source_id"] == "source:1"
            refs = [{"source_id": part["source_options"][0]["source_id"], "source_range": None,
                     "basis": "supplied_diff"}]
            if assignment["kind"] == "discovery":
                claim = {"uid": "", "revision": 0, "source_uids": [], "title": "Captured zero",
                    "payload": {"claim": "Captured zero reaches division.", "trigger": "A caller divides.",
                        "consequence": "The caller fails.", "grounds": "Captured source retains zero.",
                        "file": "api.py", "line": None, "evidence_refs": refs}}
                value = {"targets": [{"target_id": part["target_id"], "status": "reviewed", "reason": "Inspected.",
                    "evidence_refs": refs}], "claims": [] if outcome == "empty" else [claim]}
            else:
                retained = assignment["claims"][0]
                value = {"verdicts": [] if outcome == "partial" else [{"uid": retained["uid"],
                    "revision": retained["revision"], "verdict": "confirmed", "grounds": "The zero remains unguarded.",
                    "evidence_refs": refs}], "claims": []}
            binding.admit("submit", "structured_output", value, terminal=True)
            accepted = binding.submit("submit", value)
            if outcome == "native-failure" and assignment["kind"] == "challenge":
                raise RuntimeError("native failed after admission")
            binding.native_settled = True
        return accepted, None, None

    monkeypatch.setattr(committed_review, "agent", SimpleNamespace(run_agent=execute, console=agent.console))
    real_link = os.link
    injected = False

    def fail_install(source: Any, destination: Any, **kwargs: Any) -> None:
        nonlocal injected
        target = policy.output_dir / "findings.json"
        if outcome == "install-failure" and _matches_link_destination(
            destination, target, dir_fd=kwargs.get("dst_dir_fd")
        ):
            injected = True
            target.write_bytes(b"competing bytes")
            raise FileExistsError("competing output")
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", fail_install)
    review_request = CommittedReviewRequest(
        repo, envelope["materialization"]["requested_base_sha"], hosted_policy=policy, immutable_input=identity,
    )
    with monkeypatch.context() as analysis_boundaries:
        for owner, names in ((process, ("_run_gh", "_run_gh_async")),
                             (github_app, ("resolve_run_identity", "resolve_user_identity", "resolve_credentials"))):
            for name in names:
                analysis_boundaries.setattr(owner, name, forbidden)
        analysis_boundaries.setattr(type(os.environ), "get", no_github_credentials)
        if outcome == "copy-cancel":
            import asyncio

            cancellation = asyncio.CancelledError("cancel during immutable object copy")
            fdopen = os.fdopen
            preserved: list[Path] = []
            original_index = (repo / ".git/index").read_bytes()

            def cancel_copy(fd: int, *args: Any, **kwargs: Any) -> Any:
                stream = fdopen(fd, *args, **kwargs)
                write = stream.write

                def write_then_cancel(body: bytes) -> int:
                    written = int(write(body))
                    if not preserved:
                        prepared, = (policy.private_base / "immutable-inputs").iterdir()
                        leaf = next(p for p in (prepared / ".git/objects").glob("*/*")
                                    if p.stat().st_ino == os.fstat(fd).st_ino)
                        parent = leaf.parent.with_name("preserved-parent")
                        leaf.parent.rename(parent)
                        witness = parent / "witness"
                        witness.write_bytes(b"competing immutable bytes")
                        preserved.append(witness)
                        raise cancellation
                    return written

                if args and args[0] == "wb":
                    stream.write = write_then_cancel
                return stream

            analysis_boundaries.setattr(os, "fdopen", cancel_copy)
            with pytest.raises(asyncio.CancelledError) as caught:
                await run_committed_review(review_request)
            assert caught.value is cancellation
            assert len(preserved) == 1 and preserved[0].read_bytes() == b"competing immutable bytes"
            assert not seen and not attempts and not marker.exists()
            assert (repo / "api.py").read_bytes() == b"VALUE = 0\n"
            assert (repo / ".git/index").read_bytes() == original_index
            return
        if outcome == "install-failure":
            from daydream.artifact_visibility import ArtifactVisibilityError
            from daydream.findings import load_findings_artifact

            with pytest.raises(ArtifactVisibilityError, match="explicit artifact destination changed during recovery"):
                await run_committed_review(review_request)
            assert injected and attempts == [] and not marker.exists()
            assert (policy.output_dir / "findings.json").read_bytes() == b"competing bytes"
            frozen, = (policy.private_base / "runtime").glob("*/runs/*/frozen")
            retained, = frozen.glob(".explicit/*/committed-review.json")
            value = json.loads(retained.read_text())
            validate_committed_result(value)
            assert value["immutable_input"] == envelope
            assert value["terminal_result"]["analysis_state"] == "complete" and len(value["findings"]) == 1
            report, = frozen.glob(".explicit/*/review-output.md")
            assert "Captured zero" in report.read_text()
            findings, = frozen.glob(".explicit/*/findings.json")
            exported = load_findings_artifact(
                findings, expected_repo="captured/repository", expected_pr_number=1421,
                expected_head_sha=envelope["provenance"]["head_sha"],
                expected_base_sha=envelope["provenance"]["requested_base_sha"],
                expected_comparison_sha=envelope["provenance"]["comparison_base_sha"],
            )
            assert exported.terminal_result == value["terminal_result"]
            assert list((policy.private_base / "runtime").glob("*/unreconciled/publish-*"))
            assert not (repo / ".daydream").exists()
            return
        result = await run_committed_review(review_request)
    assert attempts == [] and fake_gh.process_calls() == [] and not marker.exists()
    value = result.to_dict()
    validate_committed_result(value)
    assert value["schema_version"] == 5 and value["immutable_input"] == envelope
    assert result.complete == (outcome in {"complete", "attributes", "legal-ref", "packed", "empty", "no-diff"})
    assert len(seen) == (0 if outcome == "no-diff" else 1 if outcome == "empty" else 2)
    if outcome == "attributes":
        assert all(ref["side"] == "diff" for part in value["assignments"] for ref in part["source_refs"])
    exported_value = json.loads((policy.output_dir / "findings.json").read_text())
    assert exported_value["head_sha"] == envelope["provenance"]["head_sha"]
    assert exported_value["terminal_result"] == value["terminal_result"]
    trajectory = json.loads((policy.output_dir / "trajectory.json").read_text())
    assert trajectory["extra"]["pr_repo"] == "captured/repository"
    assert trajectory["extra"]["pr_number"] == 1421
    assert trajectory["extra"]["immutable_input"] == envelope
    if outcome in {"partial", "native-failure"}:
        assert value["terminal_result"]["analysis_state"] == "incomplete"
        assert len(value["claims"]) == 1
    assert not (repo / ".daydream").exists()
    if outcome in {"complete", "empty"}:
        assert exported_value["schema_version"] == 3
        assert [finding["severity"] for finding in exported_value["findings"]] == (
            [None] if outcome == "complete" else []
        )
        assert post_findings_from_artifact(
            tmp_path, policy.output_dir / "findings.json", pr_number=1421, repo="captured/repository",
            head_sha=envelope["provenance"]["head_sha"], console=Console(),
            base_sha=envelope["provenance"]["requested_base_sha"],
            comparison_sha=envelope["provenance"]["comparison_base_sha"],
            bot_login="daydream", approve_on_clean=True,
        ) == 0
        posts = fake_gh.calls("POST", "/repos/captured/repository/pulls/1421/reviews")
        assert len(posts) == 1
        payload = posts[0].payload
        assert payload["event"] == ("COMMENT" if outcome == "complete" else "APPROVE")
        file_comments = fake_gh.calls("POST", "/repos/captured/repository/pulls/1421/comments")
        assert len(file_comments) == (1 if outcome == "complete" else 0)
        if outcome == "complete":
            comment = file_comments[0].payload
            assert (comment["path"], comment["subject_type"], comment["commit_id"]) == (
                "api.py", "file", envelope["provenance"]["head_sha"],
            )
            assert "Captured zero" in comment["body"] and "Captured zero reaches division." in comment["body"]
            assert parse_finding_markers(comment["body"]) == [exported_value["findings"][0]["fingerprint"]]


@pytest.mark.parametrize("fault", ["version", "float-version", "unknown", "original", "head", "base", "tree",
                                    "review-key", "missing-object", "object-fifo", "raw-bytes",
                                    "object-hash", "symlink-parent", "source-size", "base-source-size",
                                    "source-total-size", "authority", "ref-absolute", "ref-traversal",
                                    "ref-backslash", "ref-control", "ref-symlink", "ref-packed-suffix"])
async def test_immutable_invalid_evidence_fails_before_native_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator_policy: HostedReviewPolicy, fault: str,
) -> None:
    """Real object/checkout disagreement must refuse analysis, never become complete-empty.

    Ordinary sealed-result tests validate a produced plan; they cannot catch
    invalid immutable compilation evidence before that plan exists.
    """
    from daydream.immutable_review import ImmutableReviewInput

    repo = tmp_path / "source"
    seed_feature_branch(repo, base={"api.py": "VALUE = 1\n"}, feature={"api.py": "VALUE = 0\n"})
    if fault == "symlink-parent":
        (repo / "nested").mkdir()
        (repo / "nested/link").symlink_to("../api.py")
        git(repo, "add", "nested/link")
        commit(repo, "capture a symlink under a real parent")
    identity = immutable_input_for(repo)
    value = identity.to_dict()
    if fault in {"source-size", "base-source-size", "source-total-size"}:
        if fault == "base-source-size":
            git(repo, "checkout", "main")
        write_and_stage(repo, "too-large.txt", "x" * (8 * 1024 * 1024 - (fault == "source-total-size")) + "\n")
        if fault == "source-total-size":
            # 33 admitted blobs exceed the tree bound, sharing one inode and
            # one compressed Git object rather than 264 MiB of fixture storage.
            for index in range(32):
                name = f"copy-{index}.txt"
                os.link(repo / "too-large.txt", repo / name)
                git(repo, "add", name)
        commit(repo, "capture an unsupported compressible text blob")
        if fault == "base-source-size":
            git(repo, "checkout", "-b", "bounded-head")
            git(repo, "rm", "too-large.txt")
            commit(repo, "delete the oversized comparison blob")
        head, base = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "main")
        for group in ("provenance", "materialization"):
            value[group].update(head_sha=head, requested_base_sha=base, comparison_base_sha=base)
        value["materialization"].update(head_tree_sha=git(repo, "rev-parse", head + "^{tree}"),
                                         comparison_tree_sha=git(repo, "rev-parse", base + "^{tree}"))
        # Source bounds precede expansion, including validation of the stale
        # digest. Binary zeros never exercised text stdout allocation here.
        identity = ImmutableReviewInput.from_dict(value)
    if fault in {"version", "float-version", "unknown", "original"}:
        if fault == "unknown":
            value["unknown"] = True
        elif fault == "original":
            value["provenance"]["head_sha"] = "a" * 40
        else:
            value["schema_version"] = 9 if fault == "version" else 1.0
        with pytest.raises(ValueError, match="immutable"):
            ImmutableReviewInput.from_dict(value)
        return
    if fault in {"head", "base", "tree", "review-key"}:
        key = {"head": "head_sha", "base": "comparison_base_sha", "tree": "head_tree_sha",
               "review-key": "review_diff_key"}[fault]
        value["materialization"][key] = (
            value["materialization"]["head_sha"] if fault == "base"
            else "a" * (64 if fault == "review-key" else 40)
        )
        if fault in {"head", "base"}:
            value["provenance"][key] = value["materialization"][key]
        identity = ImmutableReviewInput.from_dict(value)
    elif fault in {"missing-object", "object-fifo", "object-hash"}:
        head = value["materialization"]["head_sha"]
        object_path = repo / ".git/objects" / head[:2] / head[2:]
        if fault == "object-hash":
            _, _, body = zlib.decompress(object_path.read_bytes()).partition(b"\0")
            body += b"\n"
            object_path.chmod(0o600)
            object_path.write_bytes(zlib.compress(f"commit {len(body)}".encode() + b"\0" + body))
        else:
            object_path.unlink()
            if fault == "object-fifo":
                os.mkfifo(object_path)
    elif fault == "raw-bytes":
        (repo / "api.py").write_text("checkout transform\n")
    elif fault == "symlink-parent":
        (repo / "nested").rename(tmp_path / "outside-nested")
        (repo / "nested").symlink_to(tmp_path / "outside-nested", target_is_directory=True)
    elif fault.startswith("ref-"):
        head = value["materialization"]["head_sha"]
        outside = tmp_path / "outside-ref"
        outside.write_text(head + "\n")
        ref = {"ref-absolute": str(outside), "ref-traversal": "refs/heads/../../../../outside-ref",
               "ref-backslash": "refs/heads/back\\slash", "ref-control": "refs/heads/fea\tture"}.get(
                   fault, "refs/heads/feature")
        if fault == "ref-backslash":
            (repo / ".git" / ref).write_text(head + "\n")
        elif fault == "ref-symlink":
            refs = repo / ".git/refs/heads"
            refs.rename(tmp_path / "outside-refs")
            refs.symlink_to(tmp_path / "outside-refs", target_is_directory=True)
        elif fault == "ref-packed-suffix":
            (repo / ".git" / ref).unlink()
            (repo / ".git/packed-refs").write_text(f"{head} invalid {ref}\n")
        (repo / ".git/HEAD").write_text("ref: " + ref + "\n")
    policy = replace(operator_policy, trusted_prepared_source=fault != "authority")
    attempts = []

    async def native_work(self: PiBackend) -> None:
        attempts.append("native")
        raise AssertionError("invalid evidence reached native work")

    monkeypatch.setattr(PiBackend, "validate_hosted_policy", native_work)
    expected = ("HEAD reference is unavailable" if fault == "ref-packed-suffix"
                else "HEAD reference is malformed") if fault.startswith("ref-") else {
        "head": "checkout HEAD differs", "base": "does not match the local merge base",
        "tree": "tree identity mismatch",
        "review-key": "review diff identity mismatch",
        "missing-object": "source validation failed", "object-hash": "source validation failed",
        "object-fifo": "bounded regular file", "raw-bytes": "checkout differs from captured raw",
        "symlink-parent": "checkout differs from captured raw", "source-size": "files exceed the supported size",
        "base-source-size": "files exceed the supported size",
        "source-total-size": "files exceed the supported size",
        "authority": "trusted_prepared_source=True",
    }[fault]
    bounded_source = fault in {"source-size", "base-source-size", "source-total-size"}
    if bounded_source:
        tracemalloc.start()
    try:
        with pytest.raises((ValueError, git_ops.GitError), match=expected):
            await run_committed_review(CommittedReviewRequest(
                repo, value["materialization"]["requested_base_sha"], hosted_policy=policy, immutable_input=identity))
    finally:
        if bounded_source:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    if bounded_source:
        assert peak < 16 * 1024 * 1024
    assert attempts == [] and not policy.output_dir.exists()


@pytest.mark.parametrize("fault", ["topology", "extra-history"])
async def test_immutable_synthetic_bundle_rejects_attested_extra_commit_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator_policy: HostedReviewPolicy, fault: str,
) -> None:
    """A correctly hashed two-ref bundle can still expose unintended history.

    Compiler bundle tests own compilation; this enters the runtime request with
    recomputed evidence so digest/ref guards cannot mask its topology obligation.
    """
    from daydream.benchmark.snapshot import build_bundle
    from daydream.git_ops import process
    from daydream.git_ops.immutable import immutable_review_diff
    original = tmp_path / "original"
    seed_feature_branch(original, base={"api.py": "VALUE = 1\n"}, feature={"api.py": "VALUE = 0\n"})
    original_input = immutable_input_for(original).to_dict()
    bundle = tmp_path / "captured.bundle"
    build_bundle(original, git(original, "rev-parse", "main"), git(original, "rev-parse", "HEAD"), bundle)
    source = tmp_path / "synthetic"
    git(tmp_path, "clone", "--no-local", str(bundle), str(source))
    base = git(source, "rev-parse", "origin/base")
    head = git(source, "rev-parse", "origin/head")
    git(source, "checkout", "--detach", head)
    git(source, "config", "user.name", "Fixture compiler")
    git(source, "config", "user.email", "compiler@example.invalid")
    extra = git(source, "commit-tree", git(source, "rev-parse", base + "^{tree}"),
                *(["-p", base] if fault == "topology" else []), "-m", "unexpected original history")
    if fault == "topology":
        head = git(source, "commit-tree", git(source, "rev-parse", head + "^{tree}"),
                   "-p", extra, "-m", "head with extra parent history")
        git(source, "checkout", "--detach", head)
        git(source, "update-ref", "refs/heads/base", base)
        git(source, "update-ref", "refs/heads/head", head)
        bundle.unlink()
        git(source, "bundle", "create", str(bundle), "refs/heads/base", "refs/heads/head")
    else:
        pack = process._run_git(
            source, ["pack-objects", "--stdout", "--revs"], input_text=f"{base}\n{head}\n{extra}\n",
            capture_bytes=True, error_context="fixture bundle packing failed",
        ).stdout
        header, _, _ = bundle.read_bytes().partition(b"\n\n")
        bundle.write_bytes(header + b"\n\n" + pack)
    local = {
        "kind": "synthetic", "head_sha": head, "requested_base_sha": base, "comparison_base_sha": base,
        "head_tree_sha": git(source, "rev-parse", head + "^{tree}"),
        "comparison_tree_sha": git(source, "rev-parse", base + "^{tree}"),
        "review_diff_key": hashlib.sha256(immutable_review_diff(source, base, head).encode()).hexdigest(),
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
    }
    identity = ImmutableReviewInput.from_dict({**original_input, "materialization": local})
    policy = operator_policy
    attempts = []

    async def native_work(self: PiBackend) -> None:
        attempts.append("native")
        raise AssertionError("invalid synthetic topology reached native work")

    monkeypatch.setattr(PiBackend, "validate_hosted_policy", native_work)
    message = "exactly two commits" if fault == "topology" else "contains extra commit history"
    with pytest.raises(git_ops.GitError, match=message):
        await run_committed_review(CommittedReviewRequest(
            source, base, hosted_policy=policy, immutable_input=identity, immutable_bundle=bundle))
    assert attempts == [] and not policy.output_dir.exists()

@pytest.mark.parametrize("corrupt", [False, True])
async def test_empty_selectors_still_validate_captured_source(
    review_inputs: StageInputFactory, tmp_path: Path, corrupt: bool,
) -> None:
    """A valid unfinished judgment cannot evade the final guard by submitting no source selectors."""
    capture = capture_source(review_inputs.work.repo, tmp_path / "captured",
        base_sha=review_inputs.revision['merge_base_sha'], head_sha=review_inputs.revision['head_sha'], local=False)
    review_inputs.captured_source = capture
    review = Review(review_inputs, "empty", deadline=clock.monotonic() + 60)
    value = _discovery(review_inputs)
    value['claims'] = []
    for target in value['targets']:
        target.update(status='not_reviewed', reason='Source work remains unfinished.', evidence_refs=[])
    if corrupt:
        (capture.root / 'caller.py').write_text('corrupt after preparation\n')
        with pytest.raises(HostToolFatal) as caught:
            await _submit(review, 'discovery', value)
        assert caught.value.code == 'input_unavailable'
    else:
        await _submit(review, 'discovery', value)
    sealed = review.seal({})
    assert sealed['terminal_result']['analysis_state'] == ('failed' if corrupt else 'incomplete')
    assert sealed['claims'] == []


async def test_sibling_submission_revalidates_after_prior_acceptance(
    review_inputs: StageInputFactory, tmp_path: Path,
) -> None:
    """A sibling cannot reuse successful source validation after intervening native corruption."""
    capture = capture_source(review_inputs.work.repo, tmp_path / "captured",
        base_sha=review_inputs.revision['merge_base_sha'], head_sha=review_inputs.revision['head_sha'], local=False)
    review_inputs.captured_source = capture
    first = Review(review_inputs, 'first', deadline=clock.monotonic() + 60)
    sibling = Review(review_inputs, 'sibling', deadline=clock.monotonic() + 60)
    await _submit(first, 'discovery', _discovery(review_inputs))
    prior = first.pending()
    (capture.root / 'caller.py').write_text('changed between callbacks\n')
    with pytest.raises(HostToolFatal) as caught:
        await _submit(sibling, 'discovery', _discovery(review_inputs))
    assert caught.value.code == 'input_unavailable'
    assert first.pending() == prior
    assert len(first.seal({})['claims']) == 1
    assert sibling.seal({})['claims'] == []


_LONG_LINE = "VALUE = '" + "é" * 30000 + "'\n"  # One UTF-8 line over 50 KB: unreadable by stock Pi read.


def _navigation_repo(repo: Path, shape: str) -> str:
    """About 20k changed lines either across more than 512 files or inside one large file."""
    if shape == "many-files":
        base = {f"pkg/module_{index:04d}.py": "BASE = 0\n" for index in range(600)}
        feature = {path: "".join(f"ITEM_{line} = {line}\n" for line in range(33)) for path in base}
        feature["zz_late/target.py"] = "LATE_MARKER = 'late'\n" + _LONG_LINE
    else:
        base = {"big.py": "BASE = 0\n"}
        feature = {"big.py": "".join(f"ITEM_{line} = {line}\n" for line in range(20000))
                   + "LATE_MARKER = 'late'\n" + _LONG_LINE}
    return seed_feature_branch(repo, base=base, feature=feature)


def _charged(binding: HostToolBinding, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Drive one host tool through its real charged admission and correlated call."""
    call_id = f"{name}-{binding.calls}"
    binding.admit(call_id, name, arguments)
    return binding.call(call_id, name, arguments)


def _read_target(binding: HostToolBinding, target_id: str) -> tuple[dict[str, Any], bytes]:
    """Reassemble every lookup page through the opaque cursor and verify each page digest."""
    pages: list[bytes] = []
    cursor = None
    while True:
        page = _charged(binding, "review_source", {"target_id": target_id, "path": None, "cursor": cursor})
        content = page["content"].encode()
        assert hashlib.sha256(content).hexdigest() == page["page"]["sha256"]
        assert page["page"]["offset"] == sum(map(len, pages)) and page["page"]["bytes"] == len(content)
        pages.append(content)
        if (cursor := page["next_cursor"]) is None:
            body = b"".join(pages)
            assert hashlib.sha256(body).hexdigest() == page["page"]["content_sha256"]
            assert len(body) == page["page"]["total_bytes"]
            return page["target"], body


@pytest.mark.parametrize("shape", ["many-files", "large-file"])
async def test_bounded_navigation_reaches_late_bytes_and_retains_partial_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shape: str,
) -> None:
    """Proof A: a ~20k-line real capture is navigable through bounded lookup, not inline inventory.

    The former 512-part refusal and whole inline inventory missed this regression:
    the population is computed before bounding, the system catalog is explicitly
    partial, and every frozen byte (including a multibyte line over 50 KB split into
    continuation fragments) is reachable by charged lookups. A late discovery failure
    keeps earlier admitted decisions and the challenged finding, types every
    undecided target as unfinished, and cannot report complete coverage.
    """
    repo = tmp_path / "source"
    head = _navigation_repo(repo, shape)
    base = git(repo, "rev-parse", "main")
    numstat = [line.split("\t") for line in git(repo, "diff", "--numstat", base, head).splitlines()]
    expected_files = {path for _, _, path in numstat}
    expected_added = sum(int(added) for added, _, _ in numstat)
    assert expected_added >= 19800
    observed: dict[str, Any] = {}

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, str | None]:
        instructions = kwargs["review_system_instructions"]
        assignment = _assignment(kwargs)
        binding: HostToolBinding = kwargs["host_tools"]
        binding.configure_schema(kwargs["output_schema"])
        binding.restrict(deadline=kwargs["deadline"], tool_call_budget=kwargs["tool_call_budget"])
        async with binding.session():
            if assignment["kind"] == "challenge":
                claim, = assignment["claims"]
                verdict = {"uid": claim["uid"], "revision": claim["revision"], "verdict": "confirmed",
                           "grounds": "The late captured bytes contain the marker after counterevidence search.",
                           "evidence_refs": observed["late_refs"]}
                value = {"verdicts": [verdict], "claims": []}
                binding.admit("challenge-final", "structured_output", value, terminal=True)
                accepted = binding.submit("challenge-final", value)
                kwargs["investigation_budget"].observed_tool_starts += binding.calls
                binding.native_settled = True
                return accepted, None, None
            catalog = assignment["catalog"]
            observed["system_catalog"] = catalog
            assert len(instructions.encode()) < 64 * 1024
            # Page the complete population through the host cursor.
            entries: list[dict[str, Any]] = []
            cursor = None
            while True:
                page = _charged(binding, "review_source", {"target_id": None, "path": None, "cursor": cursor})
                entries.extend(page["entries"])
                if (cursor := page["next_cursor"]) is None:
                    break
            observed["entries"] = entries
            contents: dict[str, bytes] = {}
            targets: dict[str, dict[str, Any]] = {}
            for entry in entries:
                targets[entry["target_id"]], contents[entry["target_id"]] = _read_target(binding, entry["target_id"])
            observed["contents"], observed["targets"] = contents, targets
            late = [entry["target_id"] for entry in entries if "LATE_MARKER" in contents[entry["target_id"]].decode()]
            late_target = targets[late[0]]
            observed["late_refs"] = [{"source_id": next(option["source_id"] for option in late_target["source_options"]
                                                        if option["side"] == "new"),
                                      "basis": "supplied_diff", "source_range": None}]
            reviewed = entries[:len(entries) // 2]
            for start in range(0, len(reviewed), 75):
                batch = {"targets": [{
                    "target_id": entry["target_id"], "status": "reviewed", "reason": "",
                    "evidence_refs": [{"source_id": targets[entry["target_id"]]["source_options"][0]["source_id"],
                                       "basis": "supplied_diff", "source_range": None}],
                } for entry in reviewed[start:start + 75]], "claims": [] if start else [{
                    "uid": "", "revision": 0, "source_uids": [], "title": "Late marker value",
                    "payload": {"claim": "The late marker changes captured behavior.", "trigger": "Import the file.",
                                "consequence": "The marker value is observed.", "grounds": "Late captured bytes.",
                                "file": late_target["file"], "line": None, "evidence_refs": observed["late_refs"]}}]}
                receipt = _charged(binding, "review_admit", batch)
                assert receipt["progress"]["decided"] == start + len(batch["targets"])
            kwargs["investigation_budget"].observed_tool_starts += binding.calls
        raise PiError("native discovery failed after progressive admissions")

    monkeypatch.setattr(committed_review, "agent", SimpleNamespace(run_agent=execute, console=agent.console))
    result = await run_committed_review(CommittedReviewRequest(repo, base, limits=ReviewLimits(600, 60, 8000)))
    value = result.to_dict()
    validate_committed_result(value)

    # Population precedes bounding: inventory equals the independent Git population.
    entries = observed["entries"]
    assert {part["file"] for part in value["assignments"]} == {entry["file"] for entry in entries} == expected_files
    assert [entry["target_id"] for entry in entries] == [part["target_id"] for part in value["assignments"]]
    assert len({entry["target_id"] for entry in entries}) == len(entries)
    if shape == "many-files":
        assert len(entries) > 512
    catalog = observed["system_catalog"]
    assert catalog["omitted_after"] == len(entries) - len(catalog["entries"])
    assert catalog["status"] == ("partial" if catalog["omitted_after"] else "complete")
    assert catalog["status"] == "partial" or shape == "large-file"
    assert catalog["total_targets"] == len(entries) and catalog["total_files"] == len(expected_files)
    # Every changed line is reachable: lookups expose at least the independently counted additions.
    added = sum(line.startswith("+") and not line.startswith("+++ ")
                for body in observed["contents"].values() for line in body.decode().split("\n"))
    assert all(not body.startswith(b"[Ordered continuation") for body in observed["contents"].values())
    fragments = [observed["contents"][entry["target_id"]] for entry in entries
                 if observed["targets"][entry["target_id"]]["kind"] == "continuation"
                 and observed["targets"][entry["target_id"]]["file"].endswith(("target.py", "big.py"))]
    assert fragments, "the long multibyte line must require continuation fragments"
    reassembled = b"".join(fragment.split(b"\n@@", 1)[1].split(b"\n", 1)[1] for fragment in fragments)
    assert ("+" + _LONG_LINE).encode() in reassembled
    assert added >= expected_added
    # Retained admissions survive the late discovery failure; unfinished work is typed per target.
    terminal = value["terminal_result"]
    assert terminal["analysis_state"] == "incomplete" and not result.complete
    decisions = {d["target_id"]: d for scope in terminal["stack_outcomes"] for d in scope["target_decisions"]}
    assert decisions.keys() == {entry["target_id"] for entry in entries}
    admitted = {e["target_id"] for e in entries[:len(entries) // 2]}
    assert {t for t, d in decisions.items() if d["status"] == "reviewed"} == admitted
    undecided = [d for t, d in decisions.items() if t not in admitted]
    assert undecided and all(d["status"] == "not_reviewed" and d["reason"].startswith("unsubmitted:")
                             and d["evidence_refs"] == [] for d in undecided)
    assert len(value["findings"]) == 1 and value["findings"][0]["payload"]["file"].endswith(("target.py", "big.py"))
    assert [e["assignment"] for e in value["execution"]] == ["discovery", "challenge"]
    assert value["execution"][0]["reason"] == "backend_failure" and not value["execution"][0]["accepted"]


async def test_lookup_is_read_only_cursor_bound_and_closes_with_authority(tmp_path: Path) -> None:
    """Navigation cannot mutate review state, alias another query's cursor, or outlive authority.

    Proof A drives lookups only along valid cursors; this catches a cursor reused across
    queries, a forged offset inside a multibyte character killing the invocation instead
    of a correctable rejection, unavailable selections reported as errors instead of
    typed data, and lookups served after the assignment closed.
    """
    repo = tmp_path / "subject"
    seed_feature_branch(repo, base={"api.py": "VALUE = 1\n", "uni.py": "TEXT = ''\n", "helper.py": "H = 1\n"},
                        feature={"api.py": "VALUE = 0\n", "uni.py": "TEXT = '\u00e9\u00e9\u00e9'\n",
                                 "helper.py": "H = 1\n"})
    review = Review(_inputs(repo, tmp_path / "inputs", ["api.py", "uni.py"]), "domain",
                    deadline=clock.monotonic() + 60)
    binding, _ = review.assignment("discovery", deadline=clock.monotonic() + 30)
    async with binding.session():
        path_page = _charged(binding, "review_source", {"target_id": None, "path": "api.py", "cursor": None})
        assert path_page["status"] == "complete" and {e["file"] for e in path_page["entries"]} == {"api.py"}
        assert path_page["capacity"]["basis"].startswith("current host ledger")
        assert path_page["capacity"]["remaining_tool_calls"] is None
        unavailable = _charged(binding, "review_source", {"target_id": None, "path": "helper.py", "cursor": None})
        assert unavailable["status"] == "unavailable" and "not proof of absence" in unavailable["reason"]
        unknown = _charged(binding, "review_source", {"target_id": "part:999999", "path": None, "cursor": None})
        assert unknown["status"] == "unavailable"
        uni = _charged(binding, "review_source", {"target_id": "uni.py", "path": None, "cursor": None})
        body = uni["content"].encode()
        inside = body.index("\u00e9".encode()) + 1
        # Forged cursors (built from the private format) model a model editing opaque cursors.
        uni_query = {"target_id": "uni.py", "path": None}
        for target, cursor, message in [
            (None, Review._cursor({"target_id": None, "path": "api.py"}, 0), "same target_id/path query"),
            ("uni.py", Review._cursor(uni_query, inside), "outside this target's content"),
            ("uni.py", Review._cursor(uni_query, len(body) + 1), "outside this target's content"),
            (None, Review._cursor({"target_id": None, "path": None}, 99), "outside this query's results"),
        ]:
            with pytest.raises(HostToolRejected, match=message):
                _charged(binding, "review_source", {"target_id": target, "path": None, "cursor": cursor})
        with pytest.raises(HostToolRejected, match="not both"):
            _charged(binding, "review_source", {"target_id": "api.py", "path": "api.py", "cursor": None})
        assert binding.authority_open and binding.fatal_error is None and binding.calls == 9
        catalog = _charged(binding, "review_source", {"target_id": None, "path": None, "cursor": None})
        assert catalog["decided_targets"] == 0 and all(entry["decided"] is None for entry in catalog["entries"])
    binding.close_authority()
    with pytest.raises(HostToolFatal, match="closed"):
        binding.admit("late", "review_source", {"target_id": None, "path": None, "cursor": None})
    value = review.seal({})
    assert value["claims"] == [] and all(
        d["status"] == "not_reviewed" for scope in value["terminal_result"]["stack_outcomes"]
        for d in scope["target_decisions"])


async def test_progressive_batches_are_atomic_and_earlier_admissions_survive(review_inputs: StageInputFactory) -> None:
    """Progressive admission replaces single-shot discovery without weakening atomicity.

    The single-submission path never exercised partial batches: a rejected batch must
    leave no decision, claim or coverage reservation, earlier decisions are final,
    oversized batches are refused, and sealing types undecided targets per target.
    """
    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    binding, _ = review.assignment("discovery", deadline=clock.monotonic() + 30)
    first, second = (part["target_id"] for part in review.assignments[:2])

    def decision(target: str, **fields: Any) -> dict[str, Any]:
        part = next(part for part in review.assignments if part["target_id"] == target)
        return {"target_id": target, "status": "reviewed", "reason": "", "evidence_refs": [{
            "source_id": part["source_options"][0]["source_id"], "basis": "supplied_diff", "source_range": None}],
            **fields}

    async with binding.session():
        receipt = _charged(binding, "review_admit",
                           {"targets": [decision(first)], "claims": [_proposal(review_inputs)]})
        assert receipt["progress"]["decided"] == 1 and receipt["admitted"]["claims"][0]["revision"] == 1
        bad = decision(second)
        bad["evidence_refs"][0]["source_id"] = "source:" + "0" * 64
        for batch, message in [
            ({"targets": [bad], "claims": [_proposal(review_inputs)]}, "outside this target"),
            ({"targets": [decision(first)], "claims": []}, "already admitted"),
            ({"targets": [decision(second), decision(second)], "claims": []}, "at most once per batch"),
            ({"targets": [decision(second, status="not_reviewed", reason="x" * 1024)], "claims": [
                {**_proposal(review_inputs), "title": "t" * 4096} for _ in range(33)]}, "exceeds 131072 UTF-8 bytes"),
            ({"targets": [decision(second)] * 129, "claims": []}, "at most 128 items"),
        ]:
            with pytest.raises(HostToolRejected, match=message):
                _charged(binding, "review_admit", batch)
        final = {"targets": [decision(second)], "claims": []}
        binding.admit("final", "structured_output", final, terminal=True)
        # Rejected batches installed nothing: one claim and exactly the two admitted decisions remain.
        progress = binding.submit("final", final)["progress"]
        assert (progress["decided"], progress["pending_challenge"]) == (2, 1)
    value = review.seal({})
    decisions = [d for scope in value["terminal_result"]["stack_outcomes"] for d in scope["target_decisions"]]
    assert {d["target_id"] for d in decisions if d["status"] == "reviewed"} == {first, second}
    assert all(d["reason"].startswith("unsubmitted:") for d in decisions if d["status"] == "not_reviewed")
    assert value["terminal_result"]["analysis_state"] == "incomplete" and len(value["claims"]) == 1


async def test_denied_over_limit_discovery_start_cannot_spend_protected_challenge_reserve(
    review_inputs: StageInputFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The binding charges a denied over-limit start; the program must not let it shrink challenge.

    Previously challenge received the shared ledger's remainder, so discovery's
    rejected extra start (charged before the limit check) consumed one protected call.
    """
    allocations: list[tuple[str, int]] = []

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, str | None]:
        kind = "discovery" if kwargs["phase"].value == "review" else "challenge"
        allocations.append((kind, kwargs["tool_call_budget"]))
        binding: HostToolBinding = kwargs["host_tools"]
        binding.configure_schema(kwargs["output_schema"])
        binding.restrict(deadline=kwargs["deadline"], tool_call_budget=kwargs["tool_call_budget"])
        async with binding.session():
            if kind == "discovery":
                parts = _looked_up(binding)
                value = {"targets": [{"target_id": part["target_id"], "status": "reviewed", "reason": "",
                                      "evidence_refs": [{"source_id": part["source_options"][0]["source_id"],
                                                         "basis": "supplied_diff", "source_range": None}]}
                                     for part in parts], "claims": [_proposal(review_inputs)]}
                binding.admit("terminal", "structured_output", value, terminal=True)
                accepted = binding.submit("terminal", value)
                for number in range(kwargs["tool_call_budget"]):  # The last start is over the limit.
                    with suppress(HostToolFatal):
                        binding.admit(f"extra-{number}", "read", {})
                assert binding.calls == kwargs["tool_call_budget"] + 1
            else:
                claim = _owner(binding).pending()[0]
                value = {"verdicts": [_verdict(review_inputs, claim)], "claims": []}
                binding.admit("terminal", "structured_output", value, terminal=True)
                accepted = binding.submit("terminal", value)
            kwargs["investigation_budget"].observed_tool_starts += binding.calls
            binding.native_settled = True
        return accepted, None, "tool_call_budget_exceeded" if kind == "discovery" else None

    monkeypatch.setattr(committed_review, "agent", SimpleNamespace(run_agent=execute, console=agent.console))
    result = await run_committed_review(CommittedReviewRequest(
        review_inputs.work.source, review_inputs.revision["merge_base_sha"], limits=ReviewLimits(90, 12, 10)))
    assert allocations == [("discovery", 5), ("challenge", 5)] and result.output_installed


async def test_claim_lines_inside_continuation_fragments_use_summarized_spans(tmp_path: Path) -> None:
    """A new 20k-line file is one hunk of null-range fragments; its claims previously had no valid line.

    The fragment identity stays canonical; only its summarized complete-line span admits
    the claim line, and a line outside that span is still rejected. The model learns that
    span from review_source, so the lookup itself must return it for a late fragment.
    """
    repo = tmp_path / "subject"
    seed_feature_branch(repo, base={"keep.py": "KEEP = 1\n"},
                        feature={"keep.py": "KEEP = 1\n", "big.py": "".join(f"ITEM_{n} = {n}\n" for n in range(20000))})
    inputs = _inputs(repo, tmp_path / "inputs", ["big.py"])
    review = Review(inputs, "domain", deadline=clock.monotonic() + 60)
    planned = review.assignments[-1]
    assert planned["kind"] == "continuation" and planned["source_refs"][0]["source_range"] is None
    binding, _ = review.assignment("discovery", deadline=clock.monotonic() + 30)
    lookup = binding.host_tools["review_source"].callback
    late = lookup({"target_id": planned["target_id"], "path": None, "cursor": None})["target"]
    assert late["fragment_line_offset"] == planned["fragment_line_offset"]
    span = late["source_summary"][0]["new"]
    selector = [{"source_id": late["source_options"][0]["source_id"], "basis": "supplied_diff", "source_range": None}]

    def claim(line: int) -> dict[str, Any]:
        return {"uid": "", "revision": 0, "source_uids": [], "title": "Late item", "payload": {
            "claim": "A late item changes.", "trigger": "Import big.py.", "consequence": "Value differs.",
            "grounds": "Captured late fragment.", "file": "big.py", "line": line, "evidence_refs": selector}}

    async with binding.session():
        with pytest.raises(HostToolRejected, match="claim.payload.line"):
            _charged(binding, "review_admit", {"targets": [], "claims": [claim(span["start"] - 1)]})
        receipt = _charged(binding, "review_admit", {"targets": [], "claims": [claim(span["end"])]})
    assert receipt["admitted"]["claims"][0]["revision"] == 1
    assert review.pending()[0]["payload"]["evidence_refs"][0]["fragment"] == planned["source_refs"][0]["fragment"]


async def test_challenge_verdict_must_name_the_claim_revision_current_at_admission(
    review_inputs: StageInputFactory,
) -> None:
    """A verdict on a revision superseded by an earlier batch must be rejected, not sealed.

    The assignment snapshots pending revisions once. Progressive batches can revise a
    claim first; a later verdict naming the dispatch-time revision previously passed,
    marked the newer revision confirmed without evidence, and made sealing raise so
    every admitted finding was lost.
    """
    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    await _submit(review, "discovery", _discovery(review_inputs))
    claim = review.pending()[0]
    binding, _ = review.assignment("challenge", deadline=clock.monotonic() + 30)
    revised = {key: deepcopy(claim[key]) for key in ("uid", "revision", "source_uids", "title", "payload")}
    revised["payload"].update(grounds="Refined grounds change the semantic payload.",
                              evidence_refs=_selectors(review_inputs))
    async with binding.session():
        receipt = _charged(binding, "review_admit", {"verdicts": [], "claims": [revised]})
        assert receipt["admitted"]["claims"][0]["revision"] == 2
        stale = {"verdicts": [_verdict(review_inputs, claim)], "claims": []}
        with pytest.raises(HostToolRejected, match="current revision at most once"):
            _charged(binding, "review_admit", stale)
        assert binding.authority_open and binding.fatal_error is None
    value = review.seal({})
    validate_committed_result(value)
    retained, = value["claims"]
    assert retained["revision"] == 2 and retained["disposition"] == "unresolved" and value["findings"] == []


async def test_unexpected_batch_installation_failure_is_indeterminate_and_keeps_prior_state(
    review_inputs: StageInputFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mutating host tool that fails inside installation cannot be replayed or half-installed.

    Endpoint tests only check the outcome label. Coverage capacity now installs before
    review state, so a failure there leaves earlier batches intact, the binding refuses
    further work (no replay), and the sealed result stays valid and incomplete.
    """
    from daydream.review_result import ReviewCoverage

    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    first, second = (part["target_id"] for part in review.assignments[:2])
    binding, _ = review.assignment("discovery", deadline=clock.monotonic() + 30)

    def batch(target: str) -> dict[str, Any]:
        part = review.assignments[review._index[target]]
        return {"targets": [{"target_id": target, "status": "reviewed", "reason": "", "evidence_refs": [{
            "source_id": part["source_options"][0]["source_id"], "basis": "supplied_diff", "source_range": None}]}],
            "claims": []}

    async with binding.session():
        assert _charged(binding, "review_admit", batch(first))["progress"]["decided"] == 1
        monkeypatch.setattr(ReviewCoverage, "admit_target_decisions_budgets",
                            lambda self, updates: (_ for _ in ()).throw(RuntimeError("storage fault")))
        with pytest.raises(HostToolFatal) as failure:
            _charged(binding, "review_admit", batch(second))
        assert failure.value.outcome == "indeterminate" and not failure.value.preacceptance
        assert binding.fatal_error is failure.value and not binding.authority_open
        with pytest.raises(HostToolFatal):
            _charged(binding, "review_admit", batch(second))
    monkeypatch.undo()
    value = review.seal({})
    validate_committed_result(value)
    decisions = {d["target_id"]: d for scope in value["terminal_result"]["stack_outcomes"]
                 for d in scope["target_decisions"]}
    assert decisions[first]["status"] == "reviewed"
    assert decisions[second]["reason"].startswith("unsubmitted:")
    assert value["terminal_result"]["analysis_state"] == "incomplete"
