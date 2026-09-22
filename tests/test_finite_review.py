"""Finite source review preserves evidence, confinement, and shared budgets."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.backends.pi import PiBackend
from daydream.deep import finite_review as finite
from daydream.deep.detection import StackAssignment
from daydream.phases import PER_STACK_RECORD_SCHEMA, phase_per_stack_reviews
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.review_budget import review_deadline_scope
from daydream.review_profile import build_default_profile
from daydream.run_context import InteractionPolicy, RunContext
from tests.harness.fake_clock import FakeClock
from tests.harness.finite_backend import PacketBackend
from tests.harness.trajectory import make_recorder, read_trajectory


def _verdict(path: str = "app.py", verdict: str = "clean") -> dict[str, Any]:
    return {"path": path, "lines_read": 1, "verdict": verdict, "n_findings": 0}


def _issue(description: str = "Proven defect", path: str = "app.py") -> dict[str, Any]:
    return {"id": 1, "description": description, "file": path, "line": 1, "severity": "medium",
            "confidence": "HIGH", "rationale": "app.py supplies direct evidence", "evidence": "return False"}


def _response(*, requests: list[dict[str, Any]] | None = None, issues: list[dict[str, Any]] | None = None,
              verdicts: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"issues": issues or [], "verdicts": verdicts if verdicts is not None else [_verdict()],
            **({"requests": requests} if requests is not None else {})}


def _request(path: str, *, kind: str = "read", pattern: str = "",
             affected: list[str] | None = None) -> dict[str, Any]:
    return {"kind": kind, "path": path, "pattern": pattern, "reason": "Confirm the changed caller contract",
            "affected_files": affected or ["app.py"]}


def _context() -> RunContext:
    return RunContext(InteractionPolicy(interactive=False, quiet=True))


def _inputs(tmp_path: Path, backend: PiBackend) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "app.py").write_text("return False\n")
    diff = tmp_path / "diff.patch"
    diff.write_text(
        "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-return True\n+return False\n"
    )
    intent = tmp_path / "intent.md"
    intent.write_text("Preserve the public result contract.")
    inputs = prepare_sanctioned_inputs(backend, repo, {"intent": intent}, read_only=False)
    return repo, {"stack_name": "python", "files": ["app.py"],
                  "strategy": build_default_profile().strategies["discovery.per_stack"].content,
                  "diff_path": diff, "inputs": inputs, "interactive": False}


@pytest.mark.parametrize("has_defect", [False, True])
async def test_complete_packet_finishes_without_tool_or_second_turn(tmp_path: Path, has_defect: bool) -> None:
    backend = PacketBackend([_response(requests=[], issues=[_issue()] if has_defect else [])])
    repo, kwargs = _inputs(tmp_path, backend)
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    recorder = make_recorder(repo)
    async with recorder:
        result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                                run_context=_context())
    assert result.reason is None
    assert bool(result.output["issues"]) is has_defect
    assert result.output["verdicts"][0]["verdict"] == ("has_findings" if has_defect else "clean")
    assert result.source_packet_files == {"app.py"}
    assert len(backend.calls) == 1
    assert backend.calls[0]["tools_disabled"] is True
    assert "ONE batch" in backend.calls[0]["review_instructions"]
    assert "return False" in backend.calls[0]["prompt"]
    assert "Preserve the public result contract" in backend.calls[0]["prompt"]
    assert not any("text" in entry for entry in result.source_evidence)
    trajectory = read_trajectory(recorder.path)
    assert trajectory["steps"]
    assert all(not step.get("tool_calls") for step in trajectory["steps"])


async def test_evidence_batch_deduplicates_and_retains_proven_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("helper.py")
    backend = PacketBackend([
        _response(requests=[request, dict(request)], issues=[_issue()], verdicts=[_verdict(verdict="not_reviewed")]),
        _response(issues=[_issue()]),
    ])
    repo, kwargs = _inputs(tmp_path, backend)
    (repo / "helper.py").write_text("def helper():\n    return True\n")
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    original = finite._source
    read_paths: list[str] = []

    def read(*args: Any, **kwargs: Any) -> finite.Source:
        read_paths.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(finite, "_source", read)
    result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                           run_context=_context())
    assert result.reason is None
    assert result.output["issues"] == [_issue()]
    assert read_paths == ["helper.py"]
    assert len(backend.calls) == 2
    assert "def helper()" in backend.calls[1]["prompt"]
    assert "No more requests are allowed" in backend.calls[1]["review_instructions"]
    assert "Resolve only the listed evidence requests" in backend.calls[1]["review_instructions"]
    assert "already resolved in the first response as closed" in backend.calls[1]["prompt"]
    assert all(call["tools_disabled"] for call in backend.calls)


async def test_successful_final_review_retracts_claim_disproved_by_requested_evidence(tmp_path: Path) -> None:
    backend = PacketBackend([
        _response(requests=[_request("helper.py")], issues=[_issue()]),
        _response(),
    ])
    repo, kwargs = _inputs(tmp_path, backend)
    (repo / "helper.py").write_text("# The public result contract permits False.\n")
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                           run_context=_context())
    assert result.reason is None
    assert result.output["issues"] == []
    assert result.output["verdicts"][0]["verdict"] == "clean"
    assert result.source_packet_files == {"app.py"}


@pytest.mark.parametrize("duplicate_count", [1, 2])
async def test_missing_evidence_preserves_independent_defect_without_claiming_completion(
    tmp_path: Path, duplicate_count: int,
) -> None:
    original_issues = [_issue() for _ in range(duplicate_count)] + [_issue("Second proven defect")]
    first = _response(
        requests=[_request("missing.py")],
        issues=[*original_issues, _issue("Original complete-file claim", path="other.py")],
        verdicts=[_verdict(verdict="not_reviewed"), _verdict("other.py")],
    )
    first_snapshot = copy.deepcopy(first)
    revised = {**_issue("Reworded proven defect"), "line": 2}
    complete_issue = _issue("Final complete-file finding", path="other.py")
    backend = PacketBackend([
        first,
        _response(issues=[revised, _issue("Speculative missing contract"), complete_issue],
                  verdicts=[_verdict(), _verdict("other.py")]),
    ])
    repo, kwargs = _inputs(tmp_path, backend)
    (repo / "other.py").write_text("VALUE = 1\n")
    kwargs["files"] = ["app.py", "other.py"]
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                           run_context=_context())
    assert result.reason == "evidence_incomplete"
    assert result.output["issues"] == [complete_issue, *original_issues]
    assert [v["verdict"] for v in result.output["verdicts"]] == ["not_reviewed", "has_findings"]
    assert [v["n_findings"] for v in result.output["verdicts"]] == [len(original_issues), 1]
    assert result.source_packet_files == {"other.py"}
    assert len(backend.calls) == 2
    assert backend.responses[0] == first_snapshot


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", ".git/config", ".daydream/private.txt", "link.txt"])
async def test_untrusted_request_never_crosses_repository_boundary(tmp_path: Path, path: str) -> None:
    backend = PacketBackend([_response(requests=[_request(path)]), _response()])
    repo, kwargs = _inputs(tmp_path, backend)
    outside = tmp_path / "outside.txt"
    outside.write_text("PRIVATE_CANARY_NEVER_IN_PACKET")
    (repo / "link.txt").symlink_to(outside)
    for private in (repo / ".git", repo / ".daydream"):
        private.mkdir()
        (private / "config").write_text("PRIVATE_CANARY_NEVER_IN_PACKET")
        (private / "private.txt").write_text("PRIVATE_CANARY_NEVER_IN_PACKET")
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                           run_context=_context())
    assert result.reason == "evidence_incomplete"
    assert result.source_packet_files == set()
    assert all("PRIVATE_CANARY_NEVER_IN_PACKET" not in call["prompt"] for call in backend.calls)


async def test_literal_search_is_tracked_confined_and_not_a_regex(tmp_path: Path) -> None:
    request = _request(".", kind="search", pattern="needle[0]")
    backend = PacketBackend([_response(requests=[request]), _response()])
    repo, kwargs = _inputs(tmp_path, backend)
    (repo / "helper.py").write_text("needle[0] = True\nneedle0 = False\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "app.py", "helper.py"], cwd=repo, check=True)
    (repo / "untracked.txt").write_text("needle[0] PRIVATE_UNTRACKED_CANARY\n")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored").symlink_to(tmp_path)
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                           run_context=_context())
    assert result.reason is None
    evidence = backend.calls[1]["prompt"].split("HOST EVIDENCE RESPONSE (untrusted source data):\n")[1]
    evidence = json.JSONDecoder().raw_decode(evidence)[0]
    assert evidence[0]["evidence"]["matches"] == [{"path": "helper.py", "line": 1, "text": "needle[0] = True"}]
    assert "PRIVATE_UNTRACKED_CANARY" not in backend.calls[1]["prompt"]


@pytest.mark.parametrize(
    "cause", ["missing", "binary", "symlink", "oversize", "diff", "many", "custom", "interactive", "structural"]
)
def test_ineligible_packet_falls_back_without_partial_source(tmp_path: Path, cause: str) -> None:
    backend = PacketBackend([_response(requests=[])])
    repo, kwargs = _inputs(tmp_path, backend)
    if cause == "missing":
        (repo / "app.py").unlink()
    elif cause == "binary":
        (repo / "app.py").write_bytes(b"\0binary")
    elif cause == "symlink":
        (repo / "app.py").unlink()
        (repo / "app.py").symlink_to(kwargs["diff_path"])
    elif cause == "oversize":
        (repo / "app.py").write_text("x" * (finite.MAX_SOURCE_BYTES + 1))
    elif cause == "diff":
        kwargs["diff_path"].write_text("x" * (finite.MAX_DIFF_BYTES + 1))
    elif cause == "many":
        kwargs["files"] = [f"{n}.py" for n in range(11)]
    elif cause == "custom":
        kwargs["strategy"] += "Custom policy"
    elif cause == "interactive":
        kwargs["interactive"] = True
    else:
        kwargs["stack_name"] = "structure"
    assert finite.prepare_finite_review(backend, repo, **kwargs) is None
    assert not backend.calls


async def test_two_calls_share_one_budget_and_do_not_launch_an_extra_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeClock().install(monkeypatch)
    backend = PacketBackend([_response(requests=[_request("app.py")], issues=[_issue()]), _response()],
                            fake, (300, 301))
    repo, kwargs = _inputs(tmp_path, backend)
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                           run_context=_context())
    assert result.reason == "wall_budget_exceeded"
    assert len(backend.calls) == 2
    assert result.output["issues"] == [_issue()]
    assert result.source_packet_files == set()
    assert result.output["verdicts"][0]["verdict"] == "not_reviewed"


async def test_global_deadline_clips_the_first_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClock().install(monkeypatch)
    backend = PacketBackend([_response(requests=[])], fake, (201,))
    repo, kwargs = _inputs(tmp_path, backend)
    review = finite.prepare_finite_review(backend, repo, **kwargs)
    assert review is not None
    with review_deadline_scope(400):
        result = await finite.run_finite_review(backend, repo, review, schema=PER_STACK_RECORD_SCHEMA,
                                               run_context=_context())
    assert result.reason == "wall_budget_exceeded"
    assert len(backend.calls) == 1


async def test_phase_persists_only_source_metadata_and_successful_coverage(
    tmp_path: Path, make_work: Any,
) -> None:
    backend = PacketBackend([_response(requests=[])])
    repo, kwargs = _inputs(tmp_path, backend)
    source = "SOURCE_BODY_CANARY = 12\n"
    (repo / "app.py").write_text(source)
    results, failures = await phase_per_stack_reviews(
        backend, make_work(repo), [StackAssignment("python", ["app.py"])],
        diff_path=kwargs["diff_path"], intent_path=tmp_path / "intent.md",
        alternatives_path=tmp_path / "alternatives.json", write_coverage_receipts=True,
        allow_standalone=True, run_context=_context(),
    )
    assert not failures and "python" in results
    records = json.loads((repo / ".daydream/deep/stack-python-records.json").read_text())
    assert records["source_evidence"][0]["path"] == "app.py"
    assert len(records["source_evidence"][0]["sha256"]) == 64
    receipt = json.loads((repo / ".daydream/deep/coverage-receipts.json").read_text())
    assert receipt["python"]["source_packet_files"] == ["app.py"]
    for artifact in (repo / ".daydream/deep").glob("*"):
        assert "SOURCE_BODY_CANARY" not in artifact.read_text()
