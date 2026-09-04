"""Recoverable cross-stack merge failure (issue #361).

Regression coverage for turning a malformed cross-stack merge response into a
structured, salvageable failure instead of a fatal run abort:

  R1  -- a bare ``list`` merge result (surfaced by ``extract_json`` when the
         model emits a bare JSON array) is normalized + merged, not rejected.
  R2  -- a genuinely unparseable result raises a structured
         ``CrossStackMergeError`` carrying the response shape + stack context.
  R3  -- a merge failure writes a *partial* ``merged-items.json`` built from
         the surviving per-stack records.
  R4  -- the failure is recorded in ``per-stack-failures.json`` under the
         reserved ``__merge__`` key (shape + stack context).
  R5  -- completed per-stack records survive a merge failure.
  R6  -- a ``--start-at fix`` relaunch picks up the salvaged partial items
         without re-reviewing completed stacks or re-running the merge agent.
  R7  -- a ``str``-shaped merge response (mirroring the arbiter
         ``_SplitTextBackend`` "got str" pattern) triggers the salvage path,
         while a bare-``list`` response containing a parseable item list is
         merged normally.

Issue #1111 extends the salvage dedup coverage: the drop is keyed on the
host-minted ``uid`` rather than the non-unique ``(id, file)`` tuple, so a
duplicate group whose members are indistinguishable under that tuple keeps its
a-side instead of losing every member; and a pre-``uid``
``dedup-candidates.json`` keeps both sides with a warning rather than falling
back to the tuple.

The phase-level tests drive the real production path
(``phase_cross_stack_merge -> run_agent -> backend ResultEvent``) with only the
backend mocked; ``_MergeTextBackend`` reproduces the pi contract faithfully
(structured output delivered via ``ResultEvent``, prose via ``TextEvent``).
Integration tests run the full deep pipeline through ``runner.run``.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import AgentEvent, Backend, ResultEvent, TextEvent
from daydream.phases import phase_cross_stack_merge
from daydream.workspace import WorkContext
from tests.harness.stub_backend import install_stub_backend, silence
from tests.test_deep_orchestrator import _merge_item, _run_deep

# Faithful reconstruction of the Pi str-when-no-JSON shape from run
# c48ca322-eb7d-4634-9fc3-fddbf349bacd (backend Pi): pure prose/refusal with
# NO parseable JSON, so extract_json -> None and run_agent returns a str where
# the merge expects an item list. The exact archived prose is not preserved in
# the run archive; the byte-identical CONTRACT is the recorded message below,
# pinned from c48ca322's deep/per-stack-failures.json.
ARCHIVED_MERGE_STR = (
    "I could not produce a JSON item list for the merged cross-stack findings. "
    "The per-stack reviews completed, but no consolidated item list was emitted."
)

# The pinned error contract, named as a single source of truth: both the raised
# ``CrossStackMergeError`` str() and the persisted ``__merge__`` message must
# match the byte-identical record pinned from the archived run's
# per-stack-failures.json. Reused at every assertion site so a message drift
# breaks at one name instead of two inline literals.
CROSS_STACK_MERGE_ERR_MSG = "Cross-stack merge returned no item list (got str)"



def _write_merge_inputs(tmp_path: Path) -> dict[str, Path]:
    """Write the merged-findings inputs under *tmp_path*'s deep artifact dir.

    ``phase_cross_stack_merge`` anchors its artifact writes at
    ``target / .daydream / deep``, so that directory must exist before the phase
    runs. The merge prompt builder embeds the input *paths* (it does not parse
    their contents), so minimal placeholders are sufficient.
    """
    deep = tmp_path / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    inputs = {
        "deep": deep,
        "intent": deep / "intent.md",
        "alts": deep / "alternatives.json",
        "dedup": deep / "dedup.json",
        "records": deep / "stack-python-records.json",
    }
    inputs["intent"].write_text("# Intent\n")
    inputs["alts"].write_text("{\"alternatives\": []}\n")
    inputs["dedup"].write_text('{"record_alt_pairs": [], "record_duplicate_pairs": []}\n')
    inputs["records"].write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "file": "api.py",
                    "line": 1,
                    "severity": "high",
                    "confidence": "HIGH",
                    "rationale": "r",
                    "evidence": "api.py:1",
                }
            ]
        )
    )
    return inputs


def test_load_failures_defaults_and_filters() -> None:
    """F3: the shared ``_load_failures`` loader parses or degrades to {}."""
    import tempfile
    from pathlib import Path

    from daydream.deep.artifacts import per_stack_failures_path
    from daydream.deep.orchestrator import _load_failures

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
        p.write_text(
            json.dumps({"s1": "boom", "__merge__": {"response_shape": "str", "message": "m"}})
        )
        assert _load_failures(p) == {
            "s1": "boom",
            "__merge__": {"response_shape": "str", "message": "m"},
        }


async def test_merge_salvage_applies_dedup_prefilter(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Callable[..., None],
) -> None:
    """F2: the salvage path applies the D-27 dedup pre-filter to the partial items.

    With no merge agent to adjudicate cross-stack duplicates, a salvage drops
    the duplicate (record_b) side of record<->record pairs computed by the
    pre-filter; otherwise the partial merged-items.json carries duplicates into
    the resume verifier and fix gate.
    """
    from daydream.deep.artifacts import deep_dir, merged_items_path

    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    # python + react emit near-identical descriptions at distinct files, so the
    # D-27 pre-filter flags them as a single cross-stack duplicate pair.
    stub.parse_by_stack = {
        "python": {
            "severity": "high", "confidence": "HIGH",
            "file": "api.py", "line": 1, "description": "unbounded cache write",
        },
        "react": {
            "severity": "high", "confidence": "HIGH",
            "file": "App.tsx", "line": 1, "description": "unbounded cache write",
        },
    }
    stub.merge_emit_str = "no item list"
    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)
    dd = deep_dir(multi_stack_target)
    items = json.loads(merged_items_path(dd).read_text())
    # The dead ``partial: true`` root flag (unread by any consumer) is not
    # written; recoverability comes from the ``__merge__`` failure record +
    # resumable stop, not a flag in merged-items.json.
    assert "partial" not in items
    per_stack = [i for i in items["items"] if i.get("lens") == "per-stack"]
    files = [i["file"] for i in per_stack]
    # The duplicate react side (record_b) is dropped; the python side survives.
    assert "api.py" in files
    assert "App.tsx" not in files, f"cross-stack duplicate leaked into partial items: {files}"


async def test_merge_failure_resume_surfaces_prior_failure(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """F1/R6: a --start-at fix relaunch after salvage surfaces the merge failure.

    The structured ``__merge__`` entry is kept out of ``failed_stacks`` (so it
    cannot garble "Uncovered stacks") but resuming into a partial synthesis must
    not look clean -- the resume loader warns that the cross-stack synthesis
    failed and the merged results are partial.
    """
    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_str = "prose with no item list"
    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)
    stub.calls.clear()
    stub.merge_emit_str = None
    assert await _run_deep(multi_stack_target, start_at="fix") == 0
    out = capsys.readouterr().out
    assert "Prior cross-stack synthesis failed; merged results are PARTIAL" in out


def _merge_args(tmp_path: Path) -> dict[str, Any]:
    """Common keyword args for a ``phase_cross_stack_merge`` call."""
    inputs = _write_merge_inputs(tmp_path)
    return {
        "per_stack_records_paths": [inputs["records"]],
        "intent_path": inputs["intent"],
        "alternatives_path": inputs["alts"],
        "dedup_candidates_path": inputs["dedup"],
    }


class _MergeTextBackend:
    """Emits prose text and a structured merge result (the real pi contract).

    Mirrors the arbiter ``_SplitTextBackend`` in
    ``tests/test_arbiter_prose_extraction.py``: when a structured output schema
    is requested the ResultEvent carries the structured answer, otherwise the
    caller drives the unparseable-text path by passing ``structured=None``.
    """

    model = "op-5"
    fanout_concurrency = 4

    def __init__(self, text: str, structured: Any) -> None:
        self._text = text
        self._structured = structured

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        yield TextEvent(text=self._text)
        yield ResultEvent(
            structured_output=self._structured if output_schema else None, continuation=None
        )

    async def cancel(self) -> None:
        pass


async def test_merge_accepts_bare_list_result(tmp_path: Path, make_work: Callable[..., WorkContext]) -> None:
    """R1: a bare-list result is normalized + merged, not treated as a failure."""
    from daydream.deep.artifacts import deep_dir, merged_items_path

    args = _merge_args(tmp_path)
    await phase_cross_stack_merge(
        cast(Backend, _MergeTextBackend(
            "prose",
            [
                {
                    "id": 1,
                    "file": "api.py",
                    "line": 1,
                    "severity": "high",
                    "confidence": "HIGH",
                    "rationale": "r",
                    "evidence": "api.py:1",
                }
            ],
        )),
        make_work(tmp_path),
        **args,
    )
    items = json.loads(merged_items_path(deep_dir(tmp_path)).read_text())
    assert [i["file"] for i in items["items"]] == ["api.py"]
    assert items.get("partial") is not True


@pytest.mark.parametrize(
    "merge_text",
    [
        "The review is done; no JSON items list here.",
        # Reopen #361: the archived Pi str shape must raise a byte-identical CrossStackMergeError.
        ARCHIVED_MERGE_STR,
    ],
    ids=["generic", "archived"],
)
async def test_merge_raises_structured_error_on_str(
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
    merge_text: Any,
) -> None:
    """R2/AC2: a genuinely-unparseable str raises CrossStackMergeError with shape + stacks."""
    from daydream.phases import CrossStackMergeError

    args = _merge_args(tmp_path)
    with pytest.raises(CrossStackMergeError) as excinfo:
        await phase_cross_stack_merge(
            cast(Backend, _MergeTextBackend(merge_text, None)),
            make_work(tmp_path),
            **args,
        )
    assert excinfo.value.response_shape == "str"
    assert excinfo.value.stack_context == ["python"]
    assert str(excinfo.value) == CROSS_STACK_MERGE_ERR_MSG


async def test_merge_accepts_bare_list_end_to_end(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R7(i): a bare-list merge result is merged and the run succeeds."""
    from daydream.deep.artifacts import deep_dir, merged_items_path

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_bare_list = [
        _merge_item(1, "store/cache.py", "high", desc="unbounded cache write"),
        _merge_item(2, "cli/main.py", "medium", desc="unused arg"),
    ]
    assert await _run_deep(multi_stack_target) == 0
    items = json.loads(merged_items_path(deep_dir(multi_stack_target)).read_text())
    # Structural findings are appended after the merge independent of this branch,
    # so scope the assertion to the per-stack merge items (R7(i)): the bare-list
    # result is normalized + merged rather than rejected or silently lost.
    per_stack = [i["file"] for i in items["items"] if i.get("lens") == "per-stack"]
    assert per_stack == ["store/cache.py", "cli/main.py"]
    assert items.get("partial") is not True


@pytest.mark.parametrize(
    "merge_str",
    [
        "All stacks reviewed. No JSON item list to emit.",
        # Reopen #361: the archived Pi str shape salvages a byte-identical __merge__ message.
        ARCHIVED_MERGE_STR,
    ],
    ids=["generic", "archived"],
)
async def test_merge_str_response_is_salvaged_not_fatal(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    merge_str: Any,
) -> None:
    """R2/R3/R4/R5/S1/S2: a str merge writes partial items + failure record, stops resumably."""
    from daydream.deep.artifacts import (
        deep_dir,
        merged_items_path,
        merged_report_path,
        per_stack_failures_path,
    )

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_str = merge_str
    assert await _run_deep(multi_stack_target) != 0  # controlled Stop(1), not a crash
    dd = deep_dir(multi_stack_target)
    items = json.loads(merged_items_path(dd).read_text())
    # The ``partial: true`` root flag is dead metadata (no consumer reads it -
    # issue #361 follow-up); the salvage is recoverable via the ``__merge__``
    # failure record + resumable stop, not a merged-items flag.
    assert "partial" not in items
    assert len(items["items"]) > 0  # R3: consolidated from surviving records
    assert merged_report_path(dd).is_file()  # S1: partial review-output.md rendered
    failures = json.loads(per_stack_failures_path(dd).read_text())
    assert failures["__merge__"]["response_shape"] == "str"  # R4/AC2
    # Reopen #361: the byte-identical message persisted via "message": str(exc)
    # must match the record archived from run c48ca322.
    assert failures["__merge__"]["message"] == CROSS_STACK_MERGE_ERR_MSG
    # R4/AC2: pin the deterministic stack names instead of only asserting
    # non-empty. multi_stack_target routes api.py/App.tsx/README.md to the
    # python + react + generic stacks; the structural meta-stack is partitioned
    # out of the merge's per-stack records, so it is absent from the context.
    assert failures["__merge__"]["stack_context"] == ["generic", "python", "react"]
    assert len(list(dd.glob("stack-*-records.json"))) > 0  # R5: completed records survive


@pytest.mark.parametrize(
    "merge_str",
    [
        "prose with no item list",
        # Reopen #361: after the archived Pi str shape salvages, the resume picks up partial items.
        ARCHIVED_MERGE_STR,
    ],
    ids=["generic", "archived"],
)
async def test_merge_failure_relaunch_picks_up_salvage(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
    merge_str: Any,
) -> None:
    """R6/AC4: --start-at fix after salvage picks up partial items; no re-review, no re-merge."""
    from daydream.deep.artifacts import deep_dir, merged_items_path

    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_str = merge_str
    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)
    stub.calls.clear()
    stub.merge_emit_str = None
    # Force the fix gate to accept so the resume reaches the fix phase and
    # consumes the salvaged partial items; without this the gate declines in
    # non-interactive mode and the run stops (0) before any fix prompt exists.
    monkeypatch.setattr(
        "daydream.deep.orchestrator.resolve_or_prompt", lambda *_a, **_k: True
    )
    assert await _run_deep(multi_stack_target, start_at="fix") == 0
    # Reopen #361: the resume loader surfaces the prior merge failure.
    assert "Prior cross-stack synthesis failed; merged results are PARTIAL" in capsys.readouterr().out
    assert not any("cross-stack merge agent" in c["prompt"].lower() for c in stub.calls)
    assert not any("per-stack review" in c["prompt"].lower() for c in stub.calls)
    # R6 positive outcome: the fix phase consumed the salvaged partial items --
    # the fix prompt (built from merged-items.json) references a surviving
    # salvaged finding's description + file. A regression where --start-at fix
    # fails to load the partial merged-items.json would fail this assertion.
    # Read the expectation off the salvaged artifact rather than hard-coding one
    # stub description: which findings survive the salvage is decided by the
    # D-27 pre-filter and the evidence gate, not by this test.
    salvaged = json.loads(merged_items_path(deep_dir(multi_stack_target)).read_text())["items"]
    assert salvaged, "salvage wrote no items to consume"
    fix_calls = [
        c["prompt"]
        for c in stub.calls
        if "fix these" in c["prompt"].lower() or "fix this issue" in c["prompt"].lower()
    ]
    assert any(
        item["description"] in p and item["file"] in p
        for item in salvaged
        for p in fix_calls
    ), f"no fix prompt referenced a salvaged item: {salvaged}"


async def test_merge_failure_merge_resume_skips_merge_entry(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Callable[..., None],
) -> None:
    """R6/AC4: a --start-at merge resume does not surface __merge__ as a failed stack.

    Exercises the resume-loader skip directly: without it, the structured
    ``__merge__`` failure entry would be str-coerced into ``failed_stacks`` and
    re-surface as a garbled "Uncovered stacks" line in the relaunched merge
    prompt, corrupting the resume contract.
    """
    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_str = "prose with no item list"
    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)
    stub.calls.clear()
    stub.merge_emit_str = None
    stub.merge_emit_bare_list = [
        _merge_item(1, "store/cache.py", "high", desc="unbounded cache write")
    ]
    assert await _run_deep(multi_stack_target, start_at="merge") == 0
    merge_calls = [c for c in stub.calls if "cross-stack merge agent" in c["prompt"].lower()]
    assert merge_calls, "expected a merge-agent relaunch"
    prompt = "\n".join(c["prompt"].lower() for c in merge_calls)
    assert "__merge__" not in prompt



async def test_merge_salvage_keeps_a_side_when_three_stacks_share_id_and_file(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Callable[..., None],
) -> None:
    """F2/#1111: a salvage keeps the a-side when every duplicate shares ``(id, file)``.

    The reported live bug. Three stacks reviewing one diff each report ``id: 1``
    on ``api.py`` with the same description -- not an exotic shape but the norm:
    per-stack reviewer ids restart at 1 in every stack, and two stacks landing on
    one file is precisely what the cross-stack machinery exists to reconcile. The
    pre-filter pairs them (0,1), (0,2) and (1,2), so record 0 -- the a-side
    ``_drop_cross_stack_duplicates`` exists to KEEP -- is itself the a-side of two
    pairs whose b-sides are indistinguishable from it under an ``(id, file)`` key.
    The old set-membership filter therefore deleted all three and the partial
    report shipped with zero language findings; only the structural item (worded
    differently, so never paired) kept ``items`` non-empty and the pre-existing
    ``len(items) > 0`` salvage assertions green.

    Keying the drop on the host-minted ``uid`` names one record and only that
    record, so exactly the two b-sides go and ``generic:1`` survives -- asserted
    here on the uid the surviving item still carries, because the three findings
    are otherwise byte-identical and nothing else can tell them apart.
    """
    from daydream.deep.artifacts import dedup_candidates_path, deep_dir, merged_items_path

    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    # Byte-identical findings from all three language stacks of multi_stack_target.
    # Same reviewer id (the stub emits ``id: 1`` per stack), same file, same
    # description -> similarity 1.0 -> all three pairwise combinations are
    # cross-stack duplicate candidates.
    collision = {
        "severity": "high",
        "confidence": "HIGH",
        "file": "api.py",
        "line": 1,
        "description": "Sample issue",
    }
    stub.parse_by_stack = {
        "generic": dict(collision),
        "python": dict(collision),
        "react": dict(collision),
    }
    stub.merge_emit_str = "no item list"
    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)
    dd = deep_dir(multi_stack_target)
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
    # Pin the exact pairing that makes the collision reachable, so a future
    # pre-filter change that stops emitting these pairs cannot make the
    # assertions above pass vacuously.
    dedup = json.loads(dedup_candidates_path(dd).read_text())
    assert [
        (p["record_a_uid"], p["record_b_uid"]) for p in dedup["record_duplicate_pairs"]
    ] == [("generic:1", "python:1"), ("generic:1", "react:1"), ("python:1", "react:1")]
    assert {p["record_b_id"] for p in dedup["record_duplicate_pairs"]} == {"1"}
    assert {p["record_b_file"] for p in dedup["record_duplicate_pairs"]} == {"api.py"}


def test_merge_salvage_keeps_both_sides_of_a_pre_uid_dedup_pair(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#1111: a pre-uid ``dedup-candidates.json`` loses nothing, and says so.

    Every producer in the current pipeline stamps ``record_b_uid``, so this shape
    is only reachable through an artifact left by a run from before the field
    existed -- ``_step_cross_stack_merge`` rewrites ``dedup-candidates.json``
    before every merge, which is why this enters at the salvage helper with a
    real deep dir on disk rather than through ``runner.run``.

    The deliberate choice under test is that the b-side is NOT re-identified by
    falling back to ``(id, file)``: that fallback is the bug above. A duplicate
    surviving into a partial report is a far smaller error than deleting the
    finding the pair existed to preserve, so both sides are kept and the
    un-applied pair is named on the console instead of silently swallowed.
    """
    from daydream.deep.artifacts import dedup_candidates_path
    from daydream.deep.orchestrator import _drop_cross_stack_duplicates

    dd = tmp_path / ".daydream" / "deep"
    dd.mkdir(parents=True)
    # The pre-#1111 pair shape: a/b identified only by the non-unique (id, file).
    dedup_candidates_path(dd).write_text(
        json.dumps(
            {
                "record_alt_pairs": [],
                "record_duplicate_pairs": [
                    {
                        "record_a_id": "1",
                        "record_a_file": "api.py",
                        "record_a_description": "Sample issue",
                        "record_a_source": "stack-generic-records.json",
                        "record_b_id": "1",
                        "record_b_file": "api.py",
                        "record_b_description": "Sample issue",
                        "record_b_source": "stack-python-records.json",
                        "similarity": 1.0,
                    }
                ],
            }
        )
    )
    records = [
        {"id": 1, "file": "api.py", "description": "Sample issue", "uid": "generic:1"},
        {"id": 1, "file": "api.py", "description": "Sample issue", "uid": "python:1"},
    ]

    kept = _drop_cross_stack_duplicates(dd, records)

    assert [r["uid"] for r in kept] == ["generic:1", "python:1"]
    out = " ".join(capsys.readouterr().out.split())
    assert "carries no record_b_uid" in out
    assert "keeping both records (issue #1111)" in out


async def test_merge_salvage_partial_items_carry_source_uids(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#1111: a salvaged partial report still names the records behind each item.

    On this path there IS no merge agent -- that is what "salvage" means -- so
    the host is the only producer that can attribute anything, and the one that
    can do it with certainty, since every item here IS a record. Nothing extra
    writes that attribution: ``_salvage_merge_failure`` delegates to
    ``_write_single_stack_merged_items``, deliberately, so the salvage report
    cannot grow a second spelling of provenance that drifts from the bypass's.
    What this pins is that the delegation actually delivers it -- on precisely
    the path where a reader most needs it, since a partial report's provenance is
    what tells them which stacks made it into the file.

    Real path: the full deep flow with an unparseable (``str``) merge response.
    ``parse_by_stack`` gives each stack a finding on its own file, worded with
    no shared vocabulary, so the dedup pre-filter pairs nothing and every record
    reaches the partial write -- the salvage applies that pre-filter itself,
    having no merge agent to adjudicate the pairs.
    """
    from daydream.deep.artifacts import deep_dir, merged_items_path

    silence(monkeypatch)
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.merge_emit_str = "no item list"
    # Medium severity keeps every record below the arbiter's ``min_severity``,
    # and the python finding sits at api.py:2 rather than api.py:1 so it does not
    # collide with the structural stack's default finding either -- no
    # arbitration, so no verdict rewrites a description asserted on below.
    stub.parse_by_stack = {
        "python": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "api.py",
            "line": 2,
            "description": "Unbounded cache write in the request handler",
        },
        "react": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "App.tsx",
            "line": 1,
            "description": "Missing key prop on the rendered list",
        },
        "generic": {
            "severity": "medium",
            "confidence": "MEDIUM",
            "file": "README.md",
            "line": 1,
            "description": "Setup instructions omit the migration step",
        },
    }

    assert await _run_deep(multi_stack_target) != 0  # merge salvaged -> Stop(1)

    dd = deep_dir(multi_stack_target)
    items = json.loads(merged_items_path(dd).read_text())["items"]
    provenance = {str(i["description"]): i["source_uids"] for i in items}
    assert provenance == {
        "Setup instructions omit the migration step": ["generic:1"],
        "Unbounded cache write in the request handler": ["python:1"],
        "Missing key prop on the rendered list": ["react:1"],
        "Structural maintainability concern": ["structure:1"],
    }, items
    # These items never passed through the merge agent, so they still carry
    # their birth ``uid`` -- and the two answers agree, which is the invariant
    # ``item_source_uids`` exists to let a consumer rely on without re-deriving.
    for item in items:
        assert item["source_uids"] == [item["uid"]], item
