"""Shared Review obligations using real Git captures and the actual host callback.

Installed-Pi journeys own native history/configuration/transport proof. These
cases exercise semantic revisions and whole-admission failures once at Review.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from daydream import clock, git_ops
from daydream.backends import MetricsEvent, TextEvent
from daydream.backends.pi import PiBackend
from daydream.committed_review import CommittedReviewRequest, Review, run_committed_review
from daydream.committed_review_model import planned_work_id, validate_committed_result
from daydream.deep.detection import StackAssignment
from daydream.host_tools import HostToolBinding, HostToolFatal, HostToolRejected
from daydream.hunk_index import write_hunk_index
from daydream.review_budget import ReviewLimits
from daydream.review_profile import build_default_profile
from daydream.review_result import validate_terminal_result
from daydream.review_source_refs import source_options
from daydream.review_stage_inputs import StageInputFactory
from daydream.run_config import RunConfig
from daydream.trajectory import get_current_recorder
from tests.harness.git_helpers import commit, git, seed_feature_branch, work_context
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
    head = seed_feature_branch(
        repo,
        base=base_contents,
        feature={name: contents.replace("= 1", "= 0") for name, contents in base_contents.items()},
    )
    base = git(repo, "rev-parse", "main")
    diff = git_ops.diff(repo, base, head)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    diff_path = inputs / "diff.patch"
    diff_path.write_text(diff)
    index_path = write_hunk_index(inputs, diff)
    revision = {"head_sha": head, "merge_base_sha": base, "diff_key": hashlib.sha256(diff.encode()).hexdigest()}
    return StageInputFactory(
        PiBackend(model="unexecuted-domain-test"),
        work_context(repo, run_id="domain"),
        StackAssignment("committed", sorted(base_contents)),
        diff_path=diff_path,
        hunk_index_path=index_path,
        shared_paths={},
        revision=revision,
        artifact_session=None,
        allow_standalone=True,
        read_only=True,
    )


def _selectors(inputs: StageInputFactory, file: str = "api.py") -> list[dict[str, Any]]:
    part = next(p for batch in inputs.assignment_batches for p in batch if p["file"] == file)
    ref = next(option for option in source_options(part["source_refs"]) if option["side"] == "new")
    return [{"source_id": ref["source_id"], "basis": "supplied_diff", "source_range": None}]


def _proposal(inputs: StageInputFactory) -> dict[str, Any]:
    return {
        "uid": "",
        "revision": 0,
        "source_uids": [],
        "title": "Zero breaks the caller",
        "payload": {
            "claim": "The new zero reaches a divisor.",
            "trigger": "The caller divides by VALUE.",
            "consequence": "A request fails.",
            "grounds": "The changed assignment feeds the caller.",
            "file": "api.py",
            "line": 1,
            "evidence_refs": _selectors(inputs),
        },
    }


def _discovery(inputs: StageInputFactory) -> dict[str, Any]:
    return {
        "targets": [
            {
                "target_id": part["target_id"],
                "status": "reviewed",
                "reason": "Captured change inspected.",
                "evidence_refs": _selectors(inputs, part["file"]),
            }
            for batch in inputs.assignment_batches
            for part in batch
        ],
        "claims": [_proposal(inputs)],
    }


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
    """A verdict cannot clear changed meaning; first-pass/triage tests have no revisions.

    Each canonical semantic field is independently protected against accidental
    omission from applicability. Title-only formatting retains the same verdict.
    """
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
    verdict = {
        "uid": previous["uid"],
        "revision": 1,
        "verdict": "confirmed",
        "grounds": "The original claim survives counterevidence search.",
        "evidence_refs": _selectors(review_inputs),
    }
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
        rejection = {
            **verdict,
            "uid": pending[0]["uid"],
            "revision": pending[0]["revision"],
            "verdict": "rejected",
            "grounds": "The captured caller guards zero before division.",
        }
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
    verdict = {
        "uid": prior[0]["uid"],
        "revision": 1,
        "verdict": "confirmed",
        "grounds": "Counterevidence checked.",
        "evidence_refs": _selectors(review_inputs),
    }
    update = {"verdicts": [verdict], "claims": [_proposal(review_inputs)]}
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
        "too-many-assignments", "hub-export",
    ]
)
async def test_program_retains_work_and_reserves_challenge_after_discovery(
    review_inputs: StageInputFactory,
    monkeypatch: pytest.MonkeyPatch,
    challenge: str,
    request: pytest.FixtureRequest,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The actual program owns failure retention and fixed later-work reservations.

    This scripts only the external execution boundary and calls the real host
    admission, Git workspace, recorder and output owners. Installed native tests
    own transport accounting/history; old flow tests never enter this program.
    Cancellation is a distinct BaseException exit: accepted concerns, recorded
    usage and interrupted coverage must survive artifact-session teardown.
    Unsupported plan/export requests must refuse before native work; domain
    admission cases cannot catch a billable dispatch of an unrepresentable plan.
    The successful journey also uses a valid pipeline-only profile: default-profile
    journeys cannot detect missing per-stage strategy fallback.
    A real locked-worktree removal must warn without replacing a successful result
    or caller cancellation; ordinary cleanup never exercises that Git failure.
    """
    from daydream import agent

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
            selectors = {
                part["target_id"]: [
                    {
                        "source_id": next(ref["source_id"] for ref in part["source_options"] if ref["side"] == "new"),
                        "basis": "supplied_diff",
                        "source_range": None,
                    }
                ]
                for part in assignment["assignments"]
            }
            if assignment["kind"] == "discovery":
                proposal = _proposal(review_inputs)
                proposal["payload"]["evidence_refs"] = selectors["api.py"]
                value = {
                    "targets": [
                        {
                            "target_id": key,
                            "status": "reviewed",
                            "reason": "Original source inspected.",
                            "evidence_refs": refs,
                        }
                        for key, refs in selectors.items()
                    ],
                    "claims": [proposal],
                }
                for number in range(kwargs["tool_call_budget"] - 1):
                    binding.admit(f"read-{number}", "read", {})
            else:
                concern = assignment["claims"][0]
                value = {
                    "verdicts": []
                    if challenge == "missing"
                    else [
                        {
                            "uid": concern["uid"],
                            "revision": concern["revision"],
                            "verdict": "confirmed",
                            "grounds": "Original source corroborates the trigger after counterevidence search.",
                            "evidence_refs": selectors["api.py"],
                        }
                    ],
                    "claims": [],
                }
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
    if challenge == "output-install":
        real_link = os.link

        def fail_output_install(source: Any, destination: Any, **kwargs: Any) -> None:
            if Path(destination) == repo / ".review-output.md":
                raise OSError("new current report cannot install")
            real_link(source, destination, **kwargs)

        monkeypatch.setattr(os, "link", fail_output_install)
    review_request = CommittedReviewRequest(
        repo,
        review_inputs.revision["merge_base_sha"],
        config=RunConfig(
            backend="pi", output_mode="review",
            trajectory_hub_repo="fixture/requested-export" if challenge == "hub-export" else None,
            review_profile_path=profile_path,
        ),
        limits=ReviewLimits(90, 12, 10),
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
    assert result.output_installed is (challenge != "output-install")
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
    if challenge == "output-install":
        assert result.output_error is not None
        assert not (repo / ".daydream/committed-review.json").exists()
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
        "complete" if challenge in {"exhausted", "output-install"} else "incomplete"
    )
    confirmed = challenge in {"exhausted", "accepted-then-failed", "output-install"}
    assert len(value["findings"]) == int(confirmed)
    assert value["claims"][0]["disposition"] == ("confirmed" if confirmed else "unresolved")
    assert value["execution"][0]["accepted"]
    if challenge == "accepted-then-failed":
        assert value["execution"][1]["accepted"] and not value["execution"][1]["settled"]
        # Persisted success flags cannot erase the actual retained execution failure.
        corrupted = deepcopy(value)
        corrupted["execution"][1]["settled"] = True
        assert corrupted["execution"][1]["reason"] == "backend_failure"
        terminal = corrupted["terminal_result"]
        terminal["stack_outcomes"][0].update(status="complete", reason_codes=[], partial_evidence=False)
        terminal["phase_outcomes"][0].update(status="complete", reason_codes=[])
        terminal.update(
            analysis_state="complete", pipeline_state="completed", reason_codes=[],
            completed_stacks=["committed"], uncovered_stacks=[],
        )
        validate_terminal_result(terminal)
        with pytest.raises(ValueError, match="native execution"):
            validate_committed_result(corrupted)


@pytest.mark.parametrize("corruption", ["wrong-source", "unchallenged-revision", "foreign-plan", "empty-execution"])
async def test_sealed_result_cannot_rebind_publication_or_planned_work(
    review_inputs: StageInputFactory,
    corruption: str,
) -> None:
    """Persisted new-core output needs its own source/revision/plan validation.

    Existing terminal schema tests have no semantic claim/challenge records or
    complete assignment identity, so they cannot reject these forged envelopes.
    Valid typed coverage cannot replace missing native execution evidence.
    """
    review = Review(review_inputs, "domain", deadline=clock.monotonic() + 60)
    await _submit(review, "discovery", _discovery(review_inputs))
    concern = review.pending()[0]
    verdict = {
        "uid": concern["uid"],
        "revision": 1,
        "verdict": "confirmed",
        "grounds": "The original source corroborates this trigger.",
        "evidence_refs": _selectors(review_inputs),
    }
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
        terminal["stack_outcomes"][0].update(status="complete", reason_codes=[], partial_evidence=False)
        terminal["phase_outcomes"][0].update(status="complete", reason_codes=[])
        terminal.update(
            analysis_state="complete", pipeline_state="completed", reason_codes=[],
            completed_stacks=["committed"], uncovered_stacks=[],
        )
        assert value["execution"] == []
        validate_terminal_result(terminal)
    value["findings"] = deepcopy(value["claims"])
    with pytest.raises(ValueError):
        validate_committed_result(value)
