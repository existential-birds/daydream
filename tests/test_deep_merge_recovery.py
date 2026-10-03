"""Recover malformed cross-stack merges into partial, resumable reports.

Coverage includes bare-list normalization, structured error context, surviving
per-stack artifacts, reserved __merge__ failure records, and fix/merge resume.
Salvage dedup uses host UIDs; legacy pairs without UIDs retain both findings
with a warning. Phase tests mock only the backend; integration uses runner.run.
"""
from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.deep.artifacts import (
    _load_failures,
    dedup_candidates_path,
    deep_dir,
    merged_items_path,
    merged_report_path,
    per_stack_failures_path,
)
from daydream.deep.merge_steps import _drop_cross_stack_duplicates, _step_cross_stack_merge, _step_load_items
from daydream.deep.reuse_store import ReuseCache, review_cache_dir
from daydream.extensions import get_registry
from daydream.flows.engine import FlowContext
from daydream.phases import CrossStackMergeError, phase_cross_stack_merge
from daydream.review_budget import record_review_budget_stop, review_budget_path
from daydream.run_config import RunConfig
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence
from tests.harness.trajectory import make_recorder
from tests.test_deep_orchestrator import _merge_item, _run_deep

# Reconstruct Pi's prose-without-JSON response shape from run
# c48ca322-eb7d-4634-9fc3-fddbf349bacd. Original prose was not archived;
# the exact recorded error below is pinned from per-stack-failures.json.
ARCHIVED_MERGE_STR = ("I could not produce a JSON item list for the merged cross-stack findings. "
    "The per-stack reviews completed, but no consolidated item list was emitted."
)

# Both raised and persisted errors must match the archived message exactly.
CROSS_STACK_MERGE_ERR_MSG = "Cross-stack merge returned no item list (got str)"


async def test_empty_merge_cold_reuse_and_resume_preserve_coverage_and_lifecycle(
    tmp_path: Path, make_work: Callable[..., WorkContext],
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
    cache = ReuseCache(review_cache_dir(dd), run_id="empty-synthesis")
    ctx = FlowContext(
        RunConfig(target=str(tmp_path), review_cache_enabled=True), make_work(tmp_path), get_registry(),
        data={
            "dd": dd, "alts_path": alternatives, "intent_path": intent, "records": [], "record_sources": [],
            "records_paths": [records], "failed_stacks": {"react": "budget exhausted: wall deadline"},
            "structural_records_path": None, "exploration_dir": None, "reuse_cache": cache,
        },
        allow_standalone_artifacts=True, _backend_factory=lambda *args: backend,
    )
    recorder = make_recorder(tmp_path)
    async with recorder:
        # Each successful synthesis must supersede its own stale salvage and
        # budget stop while retaining independent missing reviewer coverage.
        per_stack_failures_path(dd).write_text(json.dumps({
            "react": "budget exhausted: wall deadline", "__merge__": {"message": "old"},
        }))
        record_review_budget_stop(dd, "Cross-stack merge", "old timeout")
        record_review_budget_stop(dd, "python", "review budget exhausted")
        assert await _step_cross_stack_merge(ctx) is None
        assert json.loads(merged_items_path(dd).read_text()) == {"items": []}
        assert json.loads(dedup_candidates_path(dd).read_text()) == {
            "record_alt_pairs": [], "record_duplicate_pairs": [],
        }
        assert _load_failures(per_stack_failures_path(dd)) == {"react": "budget exhausted: wall deadline"}
        assert json.loads(review_budget_path(dd).read_text()) == {"python": "review budget exhausted"}
        manifests = list((cache.store_dir / "entries").glob("*/manifest.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text())
        assert manifest["unit"] == "merge"
        assert {"merged-items.json", "dedup-candidates.json"} <= set(manifest["payload"])
        cold_items = merged_items_path(dd).read_bytes()

        # Same completed records restore through the ordinary reuse contract.
        merged_items_path(dd).write_text('{"items": [{"description": "stale result"}]}')
        assert await _step_cross_stack_merge(ctx) is None
        assert merged_items_path(dd).read_bytes() == cold_items
        provenance = cache.store_dir / "provenance" / "empty-synthesis.json"
        trace = json.loads(provenance.read_text())
        assert trace["units"]["merge"]["outcome"] == "hit"
        reused_trace = provenance.read_bytes()

        # An explicit merge resume recomputes the host output without looking
        # up or republishing cache units and still records a successful phase.
        ctx.config.start_at = "merge"
        merged_items_path(dd).write_text('{"items": [{"description": "stale result"}]}')
        assert await _step_cross_stack_merge(ctx) is None
        assert merged_items_path(dd).read_bytes() == cold_items
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
    inputs["records"].write_text(json.dumps([_salvage_record()]))
    return inputs

def test_load_failures_defaults_and_filters() -> None:
    with tempfile.TemporaryDirectory() as td:
        dd = Path(td) / ".daydream" / "deep"
        dd.mkdir(parents=True)
        p = per_stack_failures_path(dd)
        # Absent file -> {} (no prior failures default).
        assert _load_failures(p) == {}
        # Malformed JSON -> {}.
        p.write_text("{ not json")
        assert _load_failures(p) == {}
        # Non-dict root -> {}.
        p.write_text("[1, 2]")
        assert _load_failures(p) == {}
        # Verbatim content preserved (including the structured __merge__ entry).
        p.write_text(json.dumps({"s1": "boom", "__merge__": {"response_shape": "str", "message": "m"}}))
        assert _load_failures(p) == {"s1": "boom", "__merge__": {"response_shape": "str", "message": "m"},}

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
    items = json.loads(merged_items_path(dd).read_text())
    # Recoverability lives in __merge__ failure evidence and the resumable stop.
    assert "partial" not in items
    per_stack = [i for i in items["items"] if i.get("lens") == "per-stack"]
    files = [i["file"] for i in per_stack]
    # The duplicate react side (record_b) is dropped; the python side survives.
    assert "api.py" in files
    assert "App.tsx" not in files, f"cross-stack duplicate leaked into partial items: {files}"

async def _salvage_then_reset(
    monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None], target: Path, *,
    emit: str = "prose with no item list",
) -> StubBackend:
    """Run one salvageable merge, then reset the stub for the resume under test."""
    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, target)
    stub.parse_severity = "high"
    stub.merge_emit_str = emit
    assert await _run_deep(target) != 0  # merge salvaged -> Stop(1)
    stub.calls.clear()
    stub.merge_emit_str = None
    return stub


async def test_merge_failure_resume_surfaces_prior_failure(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fix resume warns about partial synthesis without treating __merge__ as a stack."""
    await _salvage_then_reset(monkeypatch, mute_side_effects, multi_stack_target)
    assert await _run_deep(multi_stack_target, start_at="fix") == 0
    out = capsys.readouterr().out
    assert "Prior cross-stack synthesis failed; merged results are PARTIAL" in out

def _merge_args(tmp_path: Path) -> dict[str, Any]:
    """Common keyword args for a ``phase_cross_stack_merge`` call."""
    inputs = _write_merge_inputs(tmp_path)
    return {"per_stack_records_paths": [inputs["records"]], "intent_path": inputs["intent"],
        "alternatives_path": inputs["alts"], "dedup_candidates_path": inputs["dedup"], "allow_standalone": True,
    }

def _merge_text_backend(text: str, structured: Any) -> ScriptedBackend:
    """Emit the Pi contract: prose TextEvents plus an optional structured ResultEvent."""
    def respond(cwd: Any, prompt: str, output_schema: Any = None, *args: Any) -> list[Any]:
        return [TextEvent(text=text),
            ResultEvent(structured_output=structured if output_schema else None, continuation=None),
        ]
    return ScriptedBackend(responder=respond, model="op-5")

async def test_merge_accepts_bare_list_result(tmp_path: Path, make_work: Callable[..., WorkContext]) -> None:
    args = _merge_args(tmp_path)
    await phase_cross_stack_merge(
        cast(Backend, _merge_text_backend("prose", [_salvage_record()],)), make_work(tmp_path), **args,
    )
    items = json.loads(merged_items_path(deep_dir(tmp_path, allow_standalone=True)).read_text())
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
    assert excinfo.value.response_shape == "str"
    assert excinfo.value.stack_context == ["python"]
    assert str(excinfo.value) == CROSS_STACK_MERGE_ERR_MSG

async def test_merge_accepts_bare_list_end_to_end(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_bare_list = [_merge_item(1, "store/cache.py", "high", desc="unbounded cache write"),
        _merge_item(2, "cli/main.py", "medium", desc="unused arg"),
    ]
    assert await _run_deep(multi_stack_target) == 0
    items = json.loads(merged_items_path(deep_dir(multi_stack_target, allow_standalone=True)).read_text())
    # Structural findings are host-appended; check the normalized bare-list items.
    per_stack = [i["file"] for i in items["items"] if i.get("lens") == "per-stack"]
    assert per_stack == ["store/cache.py", "cli/main.py"]
    assert items.get("partial") is not True

@pytest.mark.parametrize("merge_str",
    ["All stacks reviewed. No JSON item list to emit.",
        # Reopen #361: the archived Pi str shape salvages a byte-identical __merge__ message.
        ARCHIVED_MERGE_STR,
    ], ids=["generic", "archived"],
)
async def test_merge_str_response_is_salvaged_not_fatal(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, merge_str: Any,
) -> None:
    """R2/R3/R4/R5/S1/S2: a str merge writes partial items + failure record, stops resumably."""

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_str = merge_str
    assert await _run_deep(multi_stack_target) != 0  # controlled Stop(1), not a crash
    dd = deep_dir(multi_stack_target, allow_standalone=True)
    items = json.loads(merged_items_path(dd).read_text())
    # Salvage uses failure evidence and a resumable stop, without a partial flag.
    assert "partial" not in items
    assert len(items["items"]) > 0  # R3: consolidated from surviving records
    assert merged_report_path(dd).is_file()  # S1: partial review-output.md rendered
    failures = json.loads(per_stack_failures_path(dd).read_text())
    assert failures["__merge__"]["response_shape"] == "str"  # R4/AC2
    # Reopen #361: the byte-identical message persisted via "message": str(exc)
    # must match the record archived from run c48ca322.
    assert failures["__merge__"]["message"] == CROSS_STACK_MERGE_ERR_MSG
    # Assert exact language stacks; structure is partitioned out before merge.
    assert failures["__merge__"]["stack_context"] == ["generic", "python", "react"]
    assert len(list(dd.glob("stack-*-records.json"))) > 0  # R5: completed records survive

@pytest.mark.parametrize("merge_str",
    ["prose with no item list",
        # Reopen #361: after the archived Pi str shape salvages, the resume picks up partial items.
        ARCHIVED_MERGE_STR,
    ], ids=["generic", "archived"],
)
async def test_merge_failure_relaunch_picks_up_salvage(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None],
    capsys: pytest.CaptureFixture[str], merge_str: Any,
) -> None:
    """R6/AC4: --start-at fix after salvage picks up partial items; no re-review, no re-merge."""

    stub = await _salvage_then_reset(monkeypatch, mute_side_effects, multi_stack_target, emit=merge_str)
    # Accept the interactive fix gate so the resume consumes the salvaged
    # partial items; decline the later commit and posting gates.
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr("daydream.run_context._prompt_user",
        lambda _console, message, _default: "y" if "Apply fixes" in message else "n",
    )
    assert await _run_deep(multi_stack_target, start_at="fix") == 0
    # Reopen #361: the resume loader surfaces the prior merge failure.
    assert "Prior cross-stack synthesis failed; merged results are PARTIAL" in capsys.readouterr().out
    assert not any("cross-stack merge agent" in c["prompt"].lower() for c in stub.calls)
    assert not any("per-stack review" in c["prompt"].lower() for c in stub.calls)
    # The resumed fix must use a surviving artifact finding. Derive the expected
    # finding after dedup/evidence gating instead of assuming a stub item survives.
    salvaged = json.loads(merged_items_path(deep_dir(multi_stack_target, allow_standalone=True)).read_text())["items"]
    assert salvaged, "salvage wrote no items to consume"
    fix_calls = [c["prompt"]
        for c in stub.calls
        if "fix these" in c["prompt"].lower() or "fix this issue" in c["prompt"].lower()
    ]
    assert any(item["description"] in p and item["file"] in p
        for item in salvaged
        for p in fix_calls
    ), f"no fix prompt referenced a salvaged item: {salvaged}"

async def test_merge_failure_merge_resume_skips_merge_entry(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, mute_side_effects: Callable[..., None],
) -> None:
    """Merge resume excludes __merge__ from failed stacks and Uncovered stacks text."""
    stub = await _salvage_then_reset(monkeypatch, mute_side_effects, multi_stack_target)
    stub.merge_emit_bare_list = [_merge_item(1, "store/cache.py", "high", desc="unbounded cache write")]
    assert await _run_deep(multi_stack_target, start_at="merge") == 0
    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_calls, "expected a merge-agent relaunch"
    prompt = "\n".join(c["prompt"].lower() for c in merge_calls)
    assert "__merge__" not in prompt

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
    items = json.loads(merged_items_path(dd).read_text())["items"]
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
    dedup = json.loads(dedup_candidates_path(dd).read_text())
    assert [(p["record_a_uid"], p["record_b_uid"]) for p in dedup["record_duplicate_pairs"]
    ] == [("generic:1", "python:1"), ("generic:1", "react:1"), ("python:1", "react:1")]
    assert {p["record_b_id"] for p in dedup["record_duplicate_pairs"]} == {"1"}
    assert {p["record_b_file"] for p in dedup["record_duplicate_pairs"]} == {"api.py"}

def test_merge_salvage_keeps_both_sides_of_a_pre_uid_dedup_pair(tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Legacy dedup pairs without UIDs preserve both findings and warn.

    Current runs rewrite pairs with UIDs, so this enters the salvage helper
    with persisted legacy artifacts. Falling back to (id, file) could delete
    the finding the pair exists to preserve.
    """
    dd = tmp_path / ".daydream" / "deep"
    dd.mkdir(parents=True)
    # The pre-#1111 pair shape: a/b identified only by the non-unique (id, file).
    dedup_candidates_path(dd).write_text(json.dumps({"record_alt_pairs": [],
                "record_duplicate_pairs": [{
                        "record_a_id": "1", "record_a_file": "api.py", "record_a_description": "Sample issue",
                        "record_a_source": "stack-generic-records.json", "record_b_id": "1", "record_b_file": "api.py",
                        "record_b_description": "Sample issue", "record_b_source": "stack-python-records.json",
                        "similarity": 1.0,
                    }
                ],
            }
        )
    )
    records = [{"id": 1, "file": "api.py", "description": "Sample issue", "uid": "generic:1"},
        {"id": 1, "file": "api.py", "description": "Sample issue", "uid": "python:1"},
    ]

    kept = _drop_cross_stack_duplicates(dd, records)

    assert [r["uid"] for r in kept] == ["generic:1", "python:1"]
    out = " ".join(capsys.readouterr().out.split())
    assert "carries no record_b_uid" in out
    assert "keeping both records (issue #1111)" in out

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
    items = json.loads(merged_items_path(dd).read_text())["items"]
    provenance = {str(i["description"]): i["source_uids"] for i in items}
    assert provenance == {"Setup instructions omit the migration step": ["generic:1"],
        "Unbounded cache write in the request handler": ["python:1"],
        "Missing key prop on the rendered list": ["react:1"], "Structural maintainability concern": ["structure:1"],
    }, items
    # Salvaged items retain birth UIDs consistent with their source_uids.
    for item in items:
        assert item["source_uids"] == [item["uid"]], item
