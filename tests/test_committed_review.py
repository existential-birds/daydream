"""Review admission, revision and failure contracts; native journeys own transport proof."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from daydream import agent, clock, git_ops, json_utils
from daydream.backends import MetricsEvent, TextEvent
from daydream.backends.pi import PiBackend
from daydream.committed_review import CommittedReviewRequest, Review, run_committed_review
from daydream.committed_review_model import planned_work_id, validate_committed_result
from daydream.deep.detection import StackAssignment
from daydream.git_ops.source import FrozenSourceRejected
from daydream.host_tools import HostToolBinding, HostToolFatal, HostToolRejected
from daydream.hunk_index import write_hunk_index
from daydream.review_budget import ReviewLimits
from daydream.review_profile import build_default_profile
from daydream.review_result import validate_terminal_result
from daydream.review_source_refs import source_options
from daydream.review_stage_inputs import StageInputFactory
from daydream.run_config import RunConfig
from daydream.trajectory import get_current_recorder
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


@pytest.fixture
def review_inputs(tmp_path: Path) -> StageInputFactory:
    """Full three-hunk inventories exceed the bounded citations used by judgments."""
    repo = tmp_path / "subject"
    base_contents = {
        "api.py": "VALUE = 1\nOTHER = 1\n" + "# unchanged context\n" * 12 + "SECOND = 1\n"
        + "# unchanged context\n" * 12 + "THIRD = 1\n",
        "caller.py": "VALUE = 1\nOTHER = 1\n",
    }
    seed_feature_branch(repo, base=base_contents,
                        feature={name: contents.replace("= 1", "= 0") for name, contents in base_contents.items()})
    return _inputs(repo, tmp_path / "inputs", sorted(base_contents))


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


def _forge_completion(value: dict[str, Any]) -> None:
    terminal = value["terminal_result"]
    terminal["stack_outcomes"][0].update(status="complete", reason_codes=[], partial_evidence=False)
    terminal["phase_outcomes"][0].update(status="complete", reason_codes=[])
    terminal.update(analysis_state="complete", pipeline_state="completed", reason_codes=[],
                    completed_stacks=["committed"], uncovered_stacks=[])
    validate_terminal_result(terminal)


async def _submit(review: Review, kind: str, value: dict[str, Any]) -> dict[str, Any]:
    binding, schema = review.assignment(kind, deadline=clock.monotonic() + 30)
    binding.configure_schema(schema)
    async with binding.session():
        binding.admit("terminal", "structured_output", value, terminal=True)
        return binding.submit("terminal", value)


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


@pytest.mark.parametrize("fault", ["invalid-selector", "shared-capacity", "deadline", "closed"])
async def test_whole_admission_rejection_preserves_prior_work_and_obligations(
    review_inputs: StageInputFactory,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """Late checks cannot partially accept a new claim or consume semantic capacity.

    Existing native correction tests exercise a single stage; they do not protect
    a previously accepted claim/revision while a complete later update is rejected.
    """
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
                fd = real_open(path, *args, **kwargs)
                if Path(path) == review_inputs.canonical[0].path and not injected:
                    injected = True
                    if fault == "deadline":
                        binding.deadline = clock.monotonic() - 1
                    else:
                        binding.close_authority()
                return fd

            if fault in {"deadline", "closed"}:
                patching.setattr(os, "open", close_during_source_read)
            with pytest.raises((HostToolRejected, HostToolFatal)) as caught:
                binding.submit("terminal", update)
        if fault in {"deadline", "closed"}:
            assert isinstance(caught.value, HostToolFatal)
            assert injected and caught.value.code == ("deadline_expired" if fault == "deadline" else "authority_closed")
    assert review.pending() == prior
    value = review.seal({})
    assert value["terminal_result"]["analysis_state"] == "incomplete"
    assert value["findings"] == [] and len(value["claims"]) == 1


@pytest.mark.parametrize(
    "challenge", [
        "missing", "timeout", "accepted-then-failed", "exhausted", "setup", "output-install", "cancelled",
        "report-stage", "bundle-stage", "too-many-assignments", "hub-export", "hosted-output-install",
        "hosted-bundle-install",
    ]
)
async def test_program_retains_work_and_reserves_challenge_after_discovery(
    review_inputs: StageInputFactory,
    monkeypatch: pytest.MonkeyPatch,
    challenge: str,
    request: pytest.FixtureRequest,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Real host/Git/output/recorder composition preserves work across execution failures.

    Distinct rows cover cancellation usage, protected allowances, pre-dispatch
    refusals, pipeline-only strategy fallback, and real locked-worktree cleanup.
    Native journeys separately own transport/history/accounting.
    """
    output_failure = challenge in {
        "output-install", "report-stage", "bundle-stage", "hosted-output-install", "hosted-bundle-install",
    }
    injected = staging_output = False
    dispatches: list[dict[str, Any]] = []
    challenge_started = asyncio.Event()
    bindings: list[HostToolBinding] = []
    locked_captures: list[Path] = []

    def remove_locked_captures() -> None:
        for captured in locked_captures:
            if captured.exists():
                git(review_inputs.work.source, "worktree", "unlock", str(captured))
                git_ops.worktree_remove(review_inputs.work.source, captured, force=True)

    request.addfinalizer(remove_locked_captures)

    async def execute(backend: Any, cwd: Path, prompt: str, **kwargs: Any) -> tuple[Any, None, str | None]:
        assignment = json.loads(
            prompt.split("\nCommitted review assignment:\n")[1].split("\nEnd committed review assignment.\n")[0]
        )
        dispatches.append(
            {"kind": assignment["kind"], "allocation": kwargs["tool_call_budget"], "deadline": kwargs["deadline"]}
        )
        if assignment["kind"] == "discovery" and challenge in {"exhausted", "cancelled"}:
            git(review_inputs.work.source, "worktree", "lock", "--reason", "cleanup regression lock", str(cwd))
            locked_captures.append(cwd)
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
            if assignment["kind"] == "discovery":
                value = _discovery(review_inputs)
                for number in range(kwargs["tool_call_budget"] - 1):
                    binding.admit(f"read-{number}", "read", {})
            else:
                value = {"verdicts": [] if challenge == "missing" else [
                    _verdict(review_inputs, assignment["claims"][0])], "claims": []}
            binding.admit("terminal", "structured_output", value, terminal=True)
            accepted = binding.submit("terminal", value)
            kwargs["investigation_budget"].observed_tool_starts += binding.calls
            if assignment["kind"] == "challenge" and challenge == "accepted-then-failed":
                raise RuntimeError("native failed after host acceptance")
            binding.native_settled = True
        if assignment["kind"] == "discovery" and challenge == "setup":
            # The actual next input-capture filesystem operation must fail.
            for item in kwargs["sanctioned_inputs"].inputs:
                item.path.unlink()
        return accepted, None, None

    monkeypatch.setattr(agent, "run_agent", execute)
    repo = review_inputs.work.source
    profile_path: Path | None = None
    if challenge == "exhausted":
        profile_path = repo.parent / "pipeline-only.toml"
        profile_path.write_text('name = "pipeline-only"\n[pipeline]\nreviewer_investigation_budget_s = 60\n')
    if challenge == "too-many-assignments":
        paths = [f"extra-{number:03}.py" for number in range(511)]
        for path in paths:
            (repo / path).write_text("VALUE = 1\n")
        git(repo, "add", *paths)
        commit(repo, "add files beyond committed review capacity")
    output_root = repo
    hosted_policy = None
    if challenge.startswith("hosted-"):
        from daydream.hosted_review import HostedReviewPolicy
        from daydream.review_profile import ResolvedProfile

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
            if Path(destination) == output_root / output_name:
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
    if challenge in {"too-many-assignments", "hub-export"}:
        expected = "at most 512 assignment parts" if challenge == "too-many-assignments" else "external export"
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
        captured = locked_captures[0]
        assert captured.exists() and f"worktree {captured}\n" in git(repo, "worktree", "list", "--porcelain")
        warning = capsys.readouterr().out
        assert "Failed to remove captured worktree" in warning and "locked" in warning
        assert str(captured) in "".join(warning.replace("│", "").split())
        return
    result = await run_committed_review(review_request)
    value = result.to_dict()
    assert len(value["assignments"][0]["source_refs"]) == 6
    assert all(
        len(decision["evidence_refs"]) == 1
        for scope in value["terminal_result"]["stack_outcomes"] for decision in scope["target_decisions"]
    )
    assert result.output_installed is (not output_failure)
    assert result.capture_state == "complete"
    if challenge == "exhausted":
        assert json.loads((repo / ".daydream/committed-review.json").read_text()) == value
        assert root_trajectory(repo)["session_id"] == value["run_id"]
        captured = locked_captures[0]
        assert captured.exists() and f"worktree {captured}\n" in git(repo, "worktree", "list", "--porcelain")
        warning = capsys.readouterr().out
        assert "Failed to remove captured worktree" in warning and "locked" in warning
        assert str(captured) in "".join(warning.replace("│", "").split())
        assert value["settings"]["profile_name"] == "pipeline-only"
        assert value["settings"]["profile_source"] == "explicit"
        assert value["settings"]["limits"]["effective_investigation_s"] == 60
        defaults = build_default_profile().strategies
        assert value["settings"]["strategies"]["discovery"] == defaults["discovery.generic_fallback"].content
        assert value["settings"]["strategies"]["challenge"] == defaults["arbitration"].content.replace(
            "{arbiter_input_path}", "the captured-source claim inventory below"
        )
    if output_failure:
        assert injected and result.output_error is not None
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
    complete = challenge == "exhausted"
    assert result.complete is complete
    assert value["terminal_result"]["analysis_state"] == (
        "complete" if challenge == "exhausted" or output_failure else "incomplete"
    )
    confirmed = challenge in {"exhausted", "accepted-then-failed"} or output_failure
    assert len(value["findings"]) == int(confirmed)
    assert value["claims"][0]["disposition"] == ("confirmed" if confirmed else "unresolved")
    assert value["execution"][0]["accepted"]
    if challenge == "accepted-then-failed":
        assert value["execution"][1]["accepted"] and not value["execution"][1]["settled"]
        # Persisted success flags cannot erase the actual retained execution failure.
        corrupted = deepcopy(value)
        corrupted["execution"][1]["settled"] = True
        assert corrupted["execution"][1]["reason"] == "backend_failure"
        _forge_completion(corrupted)
        with pytest.raises(ValueError, match="native execution"):
            validate_committed_result(corrupted)


@pytest.mark.parametrize("fault", [
    "preparation", "backend", "empty-provider", "blank-provider", "invalid-provider",
    "empty-model", "blank-model", "invalid-model", "output-overlap", "missing-agent-dir", "file-agent-dir",
])
async def test_hosted_policy_refuses_unsupported_authority_before_source_preparation(
    review_inputs: StageInputFactory, fault: str,
) -> None:
    """Hosted admission owns preparation authority; ordinary native tests cannot prove it.

    A post-checkout hook would execute before raw-byte validation if policy admission
    moved below worktree creation. No unsupported request may reach that effect.
    Missing/file configuration roots must produce the same actionable policy error,
    rather than escaping as raw filesystem errors during native context binding.
    Blank/non-string selections must fail here even though arbitrary nonblank
    identifiers are delegated to the isolated native configuration.
    """
    from dataclasses import replace

    from daydream.hosted_review import HostedReviewPolicy, UnsupportedHostedPolicy
    from daydream.review_profile import ResolvedProfile

    repo = review_inputs.work.source
    marker = repo.parent / "checkout-executed"
    hook = repo / ".git/hooks/post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    hook.chmod(0o755)
    operator = repo.parent / "operator"
    operator.mkdir()
    if fault == "missing-agent-dir":
        operator.rmdir()
    elif fault == "file-agent-dir":
        operator.rmdir()
        operator.write_text("not a configuration directory")
    policy = HostedReviewPolicy(
        agent_dir=operator, environment={"PATH": os.environ["PATH"]},
        output_dir=repo.parent / "hosted-output", private_base=repo.parent / "hosted-private",
        profile=ResolvedProfile(build_default_profile(), "operator"), trusted_prepared_source=True,
    )
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


async def test_committed_review_refuses_clean_checkout_with_transformed_native_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clean Git status can hide changed native bytes; refuse before model dispatch."""
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

    async def unexpected_dispatch(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("incoherent source reached model dispatch")

    monkeypatch.setattr(agent, "run_agent", unexpected_dispatch)
    with pytest.raises(FrozenSourceRejected, match="api.py.*differs.*captured Git blob"):
        await run_committed_review(CommittedReviewRequest(repo, base))
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
