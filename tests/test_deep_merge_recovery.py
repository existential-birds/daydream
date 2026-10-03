"""Recover malformed cross-stack merges into partial, resumable reports.

Coverage includes current envelope validation and bare-list rejection, structured error context, surviving
per-stack artifacts, typed synthesis failure outcomes, and fix/merge resume.
Salvage dedup uses host UIDs; malformed unbound pairs reject.
Phase tests mock only the backend; integration uses runner.run.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.deep.artifacts import (
    DeepArtifact,
    deep_dir,
)
from daydream.deep.merge_steps import _drop_cross_stack_duplicates, _step_cross_stack_merge, _step_load_items
from daydream.deep.reuse_store import ReuseCache
from daydream.extensions import get_registry
from daydream.flows.engine import FlowContext
from daydream.phases import CrossStackMergeError, phase_cross_stack_merge
from daydream.review_budget import review_warnings
from daydream.review_result import AnalyzedRevision, PlannedScope, ReasonCode, ReviewCoverage
from daydream.run_config import RunConfig
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.review_result import record_pool, review_coverage
from tests.harness.stub_backend import install_stub_backend, silence
from tests.harness.trajectory import make_recorder
from tests.test_deep_orchestrator import _merge_item, _run_deep

# Reconstruct Pi's prose-without-JSON response shape from run
# c48ca322-eb7d-4634-9fc3-fddbf349bacd. Original prose was not archived;
# the exact recorded error below is pinned from the saved typed coverage diagnostic.
ARCHIVED_MERGE_STR = ("I could not produce a JSON item list for the merged cross-stack findings. "
    "The per-stack reviews completed, but no consolidated item list was emitted."
)

# Both raised and persisted errors must match the archived message exactly.
CROSS_STACK_MERGE_ERR_MSG = "Cross-stack merge returned no item list (got StructuredOutputFailure)"


async def test_empty_merge_cold_reuse_and_resume_preserve_coverage_and_lifecycle(
    tmp_path: Path, make_work: Callable[..., WorkContext], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic merge remains a complete #733 unit and a resumable phase."""
    dd = deep_dir(tmp_path, allow_standalone=True)
    records = dd / "stack-python-records.json"
    records.write_text('{"issues": []}')
    alternatives = dd / "alternatives.json"
    alternatives.write_text("[]")
    intent = dd / "intent.md"
    intent.write_text("Review the new API contract")
    backend = ScriptedBackend(events=[AssertionError("empty synthesis must not dispatch")])
    cache = ReuseCache(dd.parent / "review-cache", run_id="empty-synthesis")
    coverage = ReviewCoverage("empty-synthesis", AnalyzedRevision("a" * 40, "b" * 40, "c" * 64),
        [PlannedScope("python", "python", ("api.py",)), PlannedScope("react", "react", ("App.tsx",))], ["merge"])
    coverage.record_scope("python", "incomplete", reasons=("host_tool_budget_exhaustion",),
                          partial_evidence=True, diagnostic="review budget exhausted")
    coverage.record_scope("react", "failed", reasons=(ReasonCode.HOST_WALL_BUDGET_EXHAUSTION,),
                          diagnostic="budget exhausted: wall deadline")
    ctx = FlowContext(
        RunConfig(target=str(tmp_path), review_cache_enabled=True), make_work(tmp_path), get_registry(),
        data={
            "dd": dd, "alts_path": alternatives, "intent_path": intent,
            "record_pool": record_pool(dd, paths=[records]),
            "exploration_dir": None, "reuse_cache": cache,
            "review_coverage": coverage,
        },
        allow_standalone_artifacts=True,
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    recorder = make_recorder(tmp_path)
    async with recorder:
        # Each successful synthesis must supersede its own stale salvage and
        # budget stop while retaining independent missing reviewer coverage.
        coverage.record_phase('merge', 'failed', reasons=('synthesis_failure',), diagnostic='old timeout')
        assert await _step_cross_stack_merge(ctx) is None
        assert json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text()) == {"items": []}
        assert json.loads(DeepArtifact.DEDUP_CANDIDATES.at(dd).read_text()) == {
            "record_alt_pairs": [], "record_duplicate_pairs": [],
        }
        assert coverage.unfinished_scopes == {"react": "budget exhausted: wall deadline",
                                          "python": "review budget exhausted"}
        assert review_warnings(dd) == ("python: review budget exhausted", "react: budget exhausted: wall deadline")
        manifests = list((cache.store_dir / "entries").glob("*/manifest.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        assert manifest["unit"] == "merge"
        assert {"merged-items.json", "dedup-candidates.json"} <= set(manifest["payload"])
        cold_items = DeepArtifact.MERGED_ITEMS.at(dd).read_bytes()

        # Same completed records restore through the ordinary reuse contract.
        DeepArtifact.MERGED_ITEMS.at(dd).write_text('{"items": [{"description": "stale result"}]}')
        assert await _step_cross_stack_merge(ctx) is None
        assert DeepArtifact.MERGED_ITEMS.at(dd).read_bytes() == cold_items
        provenance = cache.store_dir / "provenance" / "empty-synthesis.json"
        trace = json.loads(provenance.read_text())
        assert trace["units"]["merge"]["outcome"] == "hit"
        reused_trace = provenance.read_bytes()

        # An explicit merge resume recomputes the host output without looking
        # up or republishing cache units and still records a successful phase.
        ctx.config.start_at = "merge"
        DeepArtifact.MERGED_ITEMS.at(dd).write_text('{"items": [{"description": "stale result"}]}')
        assert await _step_cross_stack_merge(ctx) is None
        assert DeepArtifact.MERGED_ITEMS.at(dd).read_bytes() == cold_items
        assert provenance.read_bytes() == reused_trace
        assert await _step_load_items(ctx) is None
        report = ctx.data["merged_report"].read_text()
        assert "react" in report
        assert "python" in report
    assert backend.call_count == 0
    events = [event.to_dict() for event in recorder._phase_events]
    merge_ends = [event for event in events if event["phase"] == "merge" and event["event"] == "phase_end"]
    assert len(merge_ends) == 3
    assert all(event["status"] == "succeeded" for event in merge_ends)
    terminal = coverage.finalize("completed")
    assert terminal["analysis_state"] == "incomplete"
    assert terminal["failed_stacks"] == ["react"]
    assert terminal["phase_outcomes"] == [{"phase": "merge", "status": "complete", "reason_codes": [],
                                           "noop": True, "usable_evidence": False}]


async def test_empty_supervision_persists_complete_host_noop(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The required supervision stage records positive empty-input completion."""
    from daydream.runner import run
    from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config

    backend = EmptyReviewBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    assert await run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json")) == 0
    coverage = json.loads((multi_stack_target / ".daydream" / "deep" / "review-coverage.json").read_text())
    supervision = next(phase for phase in coverage["phase_outcomes"] if phase["phase"] == "supervision")
    assert supervision == {"phase": "supervision", "status": "complete", "reason_codes": [],
                            "noop": True, "usable_evidence": False}
    assert not any("supervisor adjudication" in call["prompt"].lower() for call in backend.calls)


@pytest.mark.parametrize("failed_phase", ["merge", "supervision"])
async def test_phase_failure_preserves_original_error_when_coverage_write_also_fails(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    make_config: Callable[..., RunConfig], failed_phase: str,
) -> None:
    """The production terminal boundary must retain the provider's first failure."""
    from daydream import json_utils
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import run
    from tests.harness.stub_backend import StubBackend
    from tests.test_deep_orchestrator import _pin_findings_pr

    original_error = RuntimeError(f"{failed_phase} provider failed first")
    failed = False

    class FailedPhaseBackend(StubBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> Any:
            nonlocal failed
            marker = "cross-stack merge agent" if failed_phase == "merge" else "supervisor adjudication"
            if marker in prompt.lower():
                failed = True
                raise original_error
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = FailedPhaseBackend(multi_stack_target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    stage = json_utils._stage_bytes

    def fail_after_provider(path: Path, content: bytes, **kwargs: Any) -> Path:
        if failed and path.name == "review-coverage.json":
            raise OSError("coverage stage failed after provider failure")
        return stage(path, content, **kwargs)

    monkeypatch.setattr(json_utils, "_stage_bytes", fail_after_provider)
    output = tmp_path / "absent-review.json"
    with pytest.raises(RuntimeError, match=f"{failed_phase} provider failed first") as caught:
        await run(make_config(multi_stack_target, findings_out=str(output), pr_number=pr.number,
                              file_config=DaydreamFileConfig(supervisor="llm")))
    assert caught.value is original_error
    assert any("Review coverage persistence failed: OSError" in note for note in caught.value.__notes__)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["synthesis", "projection"])
async def test_public_resume_retains_failure_until_same_revision_remerge(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    make_config: Callable[..., RunConfig], fault: str,
) -> None:
    """Fix-only export preserves a failed synthesis/projection; actual remerge repairs it."""
    from daydream.findings import load_findings_artifact
    from daydream.runner import run
    from tests.test_deep_orchestrator import _pin_findings_pr
    pr = _pin_findings_pr(monkeypatch, multi_stack_target)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_emit_str = "malformed synthesis response" if fault == "synthesis" else None
    outputs = [tmp_path / f"review-{stage}.json" for stage in ("cold", "fix", "merge")]
    for index, stage in enumerate(("review", "fix", "merge")):
        expected = 1 if index == 0 and fault == "synthesis" else 0
        assert await run(make_config(multi_stack_target, start_at=stage, findings_out=str(outputs[index]),
                                    pr_number=pr.number)) == expected
        if index == 0 and fault == "projection":
            DeepArtifact.MERGED_ITEMS.at(multi_stack_target / ".daydream" / "deep").write_text("{corrupt projection")
        if index == 1:
            assert not any("cross-stack merge agent" in call["prompt"].lower() for call in stub.calls)
        stub.merge_emit_str = None
        stub.calls.clear()
    loaded = [load_findings_artifact(path, expected_repo="o/r", expected_pr_number=pr.number,
                                    expected_head_sha=pr.head_sha) for path in outputs]
    assert all(artifact.terminal_result is not None for artifact in loaded)
    first, resumed, rerun = [cast(dict[str, Any], artifact.terminal_result) for artifact in loaded]
    assert len({result["run_id"] for result in (first, resumed, rerun)}) == 3
    assert first["analyzed_revision"] == resumed["analyzed_revision"] == rerun["analyzed_revision"]
    reason = "synthesis_failure" if fault == "synthesis" else "malformed_artifact"
    assert resumed["analysis_state"] == ("incomplete" if fault == "synthesis" else "failed")
    assert reason in resumed["reason_codes"]
    assert rerun["analysis_state"] == "complete" and reason not in rerun["reason_codes"] and loaded[2].findings
    assert loaded[0].findings
    if fault == "synthesis":
        assert first["analysis_state"] == "incomplete" and first["pipeline_state"] == "failed"
        assert resumed["pipeline_state"] == "completed" and reason in first["reason_codes"] and loaded[1].findings
    else:
        assert first["analysis_state"] == "complete" and loaded[1].findings == []


def _salvage_record() -> dict[str, object]:
    """Minimal per-stack record shape accepted by the merge phase."""
    return {"id": 1, "file": "api.py", "line": 1, "severity": "high", "confidence": "HIGH", "rationale": "r",
        "evidence": "api.py:1",
    }

def _write_merge_inputs(tmp_path: Path) -> dict[str, Path]:
    """Create the deep artifact directory and minimal files named by merge prompts."""
    deep = tmp_path / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    inputs = {
        "deep": deep, "intent": deep / "intent.md", "alts": deep / "alternatives.json", "dedup": deep / "dedup.json",
        "records": deep / "stack-python-records.json",
    }
    inputs["intent"].write_text("# Intent\n")
    inputs["alts"].write_text("{\"alternatives\": []}\n")
    inputs["dedup"].write_text('{"record_alt_pairs": [], "record_duplicate_pairs": []}\n')
    inputs["records"].write_text(json.dumps({"issues": [_salvage_record()]}))
    return inputs

@pytest.mark.parametrize('diagnostics', [None, [], {'scopes': {}, 'phases': {'extra': 'failure'}},
                                       {'scopes': {'python': 'x' * 1025}, 'phases': {}}])
def test_coverage_rejects_malformed_failure_diagnostics(diagnostics: Any) -> None:
    payload = review_coverage().to_dict()
    payload['diagnostics'] = diagnostics
    with pytest.raises(ValueError):
        ReviewCoverage.from_dict(payload)

async def test_merge_salvage_applies_dedup_prefilter(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None],
) -> None:
    """Salvage drops record_b duplicates before partial items reach the fix gate."""
    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    # python + react emit near-identical descriptions at distinct files, so the
    # D-27 pre-filter flags them as a single cross-stack duplicate pair.
    stub.parse_by_stack = {"python": {"severity": "high", "confidence": "HIGH",
            "file": "api.py", "line": 1, "description": "unbounded cache write",
        },
        "react": {"severity": "high", "confidence": "HIGH",
            "file": "App.tsx", "line": 1, "description": "unbounded cache write",
        },
    }
    stub.merge_emit_str = "no item list"
    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)
    dd = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text())
    # Recoverability lives in __merge__ failure evidence and the resumable stop.
    assert "partial" not in items
    per_stack = [i for i in items["items"] if i.get("lens") == "per-stack"]
    files = [i["file"] for i in per_stack]
    # The duplicate react side (record_b) is dropped; the python side survives.
    assert "api.py" in files
    assert "App.tsx" not in files, f"cross-stack duplicate leaked into partial items: {files}"


def _merge_args(tmp_path: Path) -> dict[str, Any]:
    """Common keyword args for a ``phase_cross_stack_merge`` call."""
    inputs = _write_merge_inputs(tmp_path)
    return {"record_pool": record_pool(tmp_path, json.loads(inputs["records"].read_text())["issues"],
            paths=[inputs["records"]]), "intent_path": inputs["intent"],
        "alternatives_path": inputs["alts"], "dedup_candidates_path": inputs["dedup"], "allow_standalone": True,
    }

def _merge_text_backend(text: str, structured: Any) -> ScriptedBackend:
    """Emit the Pi contract: prose TextEvents plus an optional structured ResultEvent."""
    def respond(cwd: Any, prompt: str, output_schema: Any = None, *args: Any) -> list[Any]:
        return [TextEvent(text=text),
            ResultEvent(structured_output=structured if output_schema else None, continuation=None),
        ]
    return ScriptedBackend(responder=respond, model="op-5")

@pytest.mark.parametrize("envelope", [True, False], ids=["current-envelope", "bare-array-rejected"])
async def test_merge_requires_current_result_envelope(
    tmp_path: Path, make_work: Callable[..., WorkContext], envelope: bool,
) -> None:
    args = _merge_args(tmp_path)
    records = [_merge_item(1, "api.py", "high", desc="unbounded cache write")]
    call = phase_cross_stack_merge(
        cast(Backend, _merge_text_backend("prose", {"items": records} if envelope else records)),
        make_work(tmp_path), **args,
    )
    if not envelope:
        with pytest.raises(CrossStackMergeError):
            await call
        return
    await call
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(deep_dir(tmp_path, allow_standalone=True)).read_text())
    assert [i["file"] for i in items["items"]] == ["api.py"]
    assert items.get("partial") is not True

@pytest.mark.parametrize("merge_text",
    ["The review is done; no JSON items list here.",
        # Reopen #361: the archived Pi str shape must raise a byte-identical CrossStackMergeError.
        ARCHIVED_MERGE_STR,
    ], ids=["generic", "archived"],
)
async def test_merge_raises_structured_error_on_str(
    tmp_path: Path, make_work: Callable[..., WorkContext], merge_text: Any,
) -> None:
    """R2/AC2: a genuinely-unparseable str raises CrossStackMergeError with shape + stacks."""
    args = _merge_args(tmp_path)
    with pytest.raises(CrossStackMergeError) as excinfo:
        await phase_cross_stack_merge(cast(Backend, _merge_text_backend(merge_text, None)), make_work(tmp_path), **args,
        )
    assert excinfo.value.response_shape == "StructuredOutputFailure"
    assert excinfo.value.stack_context == ["python"]
    assert str(excinfo.value) == CROSS_STACK_MERGE_ERR_MSG

async def test_merge_rejects_bare_list_and_salvages_bound_records(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_bare_list = [_merge_item(1, "store/cache.py", "high", desc="unbounded cache write"),
        _merge_item(2, "cli/main.py", "medium", desc="unused arg")]
    assert await _run_deep(multi_stack_target) == 1
    deep = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(deep).read_text())
    assert items["items"] and all(item["source_uids"] for item in items["items"])
    assert not {"store/cache.py", "cli/main.py"} & {item["file"] for item in items["items"]}
    assert items.get("partial") is not True
    coverage = ReviewCoverage.from_dict(json.loads(DeepArtifact.REVIEW_COVERAGE.at(deep).read_text()))
    assert coverage.phases["merge"]["status"] == "failed"

@pytest.mark.parametrize("merge_str,resume", [
    ("All stacks reviewed. No JSON item list to emit.", "inspect"), (ARCHIVED_MERGE_STR, "inspect"),
    ("prose with no item list", "fix"), (ARCHIVED_MERGE_STR, "fix"),
    ("prose with no item list", "merge"), ("prose with no item list", "fix-declined"),
])
async def test_merge_salvage_and_resume_preserve_findings_and_failure_context(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None],
    capsys: pytest.CaptureFixture[str], merge_str: str, resume: str,
) -> None:
    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity, stub.merge_emit_str = "high", merge_str
    assert await _run_deep(multi_stack_target) != 0
    dd = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text())
    assert "partial" not in items and items["items"]
    assert DeepArtifact.MERGED_REPORT.at(dd).is_file() and list(dd.glob("stack-*-records.json"))
    coverage = ReviewCoverage.from_dict(json.loads(DeepArtifact.REVIEW_COVERAGE.at(dd).read_text()))
    assert coverage.phases['merge']['status'] == 'failed'
    assert coverage.phases['merge']['reason_codes'] == ['synthesis_failure']
    assert coverage.diagnostics['phases']['merge'] == CROSS_STACK_MERGE_ERR_MSG
    if resume == "inspect":
        return
    stub.calls.clear()
    stub.merge_emit_str = None
    if resume == "merge":
        stub.merge_items = [_merge_item(1, "store/cache.py", "high", desc="unbounded cache write")]
        assert await _run_deep(multi_stack_target, start_at="merge") == 0
        prompts = [call["prompt"].lower() for call in stub.calls if "cross-stack merge agent" in call["prompt"].lower()]
        assert prompts and "__merge__" not in "\n".join(prompts)
    else:
        if resume == "fix":
            monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
            monkeypatch.delenv("CI", raising=False)
            monkeypatch.setattr("daydream.run_context._prompt_user",
                lambda _console, message, _default: "y" if "Apply fixes" in message else "n")
        assert await _run_deep(multi_stack_target, start_at="fix") == 0
        assert "Prior cross-stack synthesis failed; merged results are PARTIAL" in capsys.readouterr().out
        assert not any(
            "cross-stack merge agent" in call["prompt"].lower() or "per-stack review" in call["prompt"].lower()
            for call in stub.calls
        )
        if resume == "fix":
            salvaged = json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text())["items"]
            fixes = [
                call["prompt"] for call in stub.calls
                if call["prompt"].lower().startswith(("fix these", "fix this issue"))
            ]
            assert salvaged and any(item["description"] in prompt and item["file"] in prompt
                                    for item in salvaged for prompt in fixes)


async def test_merge_salvage_keeps_a_side_when_three_stacks_share_id_and_file(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None],
) -> None:
    """UID dedup retains one record when three stacks share ID, file, and description.

    Pairs (0,1), (0,2), and (1,2) must drop only records 1 and 2. An (id, file)
    key would delete all three; asserting the surviving generic:1 UID catches
    that loss even when structural findings keep the report nonempty.
    """
    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    # Identical IDs, files, and descriptions make all three stack pairs duplicates.
    collision = {"severity": "high", "confidence": "HIGH", "file": "api.py", "line": 1, "description": "Sample issue",}
    stub.parse_by_stack = {"generic": dict(collision), "python": dict(collision), "react": dict(collision),}
    stub.merge_emit_str = "no item list"
    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)
    dd = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text())["items"]
    per_stack = [i for i in items if i.get("lens") == "per-stack"]
    # The regression: on the old ``(id, file)`` key this list is EMPTY -- every
    # record matched the single dropped key ``("1", "api.py")``.
    assert len(per_stack) == 1, f"salvage lost the a-side of the duplicate group: {items}"
    # ... and it is the a-side, not an arbitrary survivor. ``generic`` sorts
    # first, so it is record 0 and the a-side of both pairs it appears in.
    assert per_stack[0]["uid"] == "generic:1"
    assert per_stack[0]["file"] == "api.py"
    # The structural item is worded differently and is never paired, so it must
    # be untouched -- it is also what masked this bug, by keeping items non-empty.
    assert [i["uid"] for i in items if i.get("lens") == "structural"] == ["structure:1"]
    # Pin the pairs so changed pre-filter behavior cannot pass vacuously.
    dedup = json.loads(DeepArtifact.DEDUP_CANDIDATES.at(dd).read_text())
    assert [(p["record_a_uid"], p["record_b_uid"]) for p in dedup["record_duplicate_pairs"]
    ] == [("generic:1", "python:1"), ("generic:1", "react:1"), ("python:1", "react:1")]
    assert {p["record_b_id"] for p in dedup["record_duplicate_pairs"]} == {"1"}
    assert {p["record_b_file"] for p in dedup["record_duplicate_pairs"]} == {"api.py"}

@pytest.mark.parametrize("payload", [
    {}, {"record_duplicate_pairs": None}, {"record_duplicate_pairs": [{}]},
    {"record_duplicate_pairs": [{"record_b_uid": ""}]},
])
def test_merge_salvage_rejects_dedup_without_current_identity(tmp_path: Path, payload: dict[str, Any]) -> None:
    DeepArtifact.DEDUP_CANDIDATES.at(tmp_path).write_text(json.dumps(payload))
    records = [{"id": 1, "file": "api.py", "description": "Sample issue", "uid": "generic:1"},
               {"id": 1, "file": "api.py", "description": "Sample issue", "uid": "python:1"}]
    with pytest.raises(ValueError):
        _drop_cross_stack_duplicates(tmp_path, records)
    assert [record["uid"] for record in records] == ["generic:1", "python:1"]

async def test_merge_salvage_partial_items_carry_source_uids(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full-flow salvage attributes each item to its surviving source record.

    Distinct per-stack files/descriptions avoid dedup so every record reaches
    the partial report. Host attribution must match the single-stack writer
    because no merge agent is available on this path.
    """
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_emit_str = "no item list"
    # Stay below arbiter severity and off the structural finding's api.py:1
    # location so no verdict rewrites the descriptions asserted below.
    stub.parse_by_stack = {"python": {"severity": "medium", "confidence": "MEDIUM", "file": "api.py", "line": 2,
            "description": "Unbounded cache write in the request handler",
        },
        "react": {"severity": "medium", "confidence": "MEDIUM", "file": "App.tsx", "line": 1,
            "description": "Missing key prop on the rendered list",
        },
        "generic": {"severity": "medium", "confidence": "MEDIUM", "file": "README.md", "line": 1,
            "description": "Setup instructions omit the migration step",
        },
    }

    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)

    dd = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(DeepArtifact.MERGED_ITEMS.at(dd).read_text())["items"]
    provenance = {str(i["description"]): i["source_uids"] for i in items}
    assert provenance == {"Setup instructions omit the migration step": ["generic:1"],
        "Unbounded cache write in the request handler": ["python:1"],
        "Missing key prop on the rendered list": ["react:1"], "Structural maintainability concern": ["structure:1"],
    }, items
    # Salvaged items retain birth UIDs consistent with their source_uids.
    for item in items:
        assert item["source_uids"] == [item["uid"]], item
