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

The phase-level tests drive the real production path
(``phase_cross_stack_merge -> run_agent -> backend ResultEvent``) with only the
backend mocked; ``_MergeTextBackend`` reproduces the pi contract faithfully
(structured output delivered via ``ResultEvent``, prose via ``TextEvent``).
Integration tests run the full deep pipeline through ``runner.run``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent, TextEvent
from daydream.phases import phase_cross_stack_merge
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
    multi_stack_target, monkeypatch, mute_side_effects
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
    multi_stack_target, monkeypatch, mute_side_effects, capsys
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
    ):
        yield TextEvent(text=self._text)
        yield ResultEvent(
            structured_output=self._structured if output_schema else None, continuation=None
        )

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


async def test_merge_accepts_bare_list_result(tmp_path: Path, make_work) -> None:
    """R1: a bare-list result is normalized + merged, not treated as a failure."""
    from daydream.deep.artifacts import deep_dir, merged_items_path

    args = _merge_args(tmp_path)
    await phase_cross_stack_merge(
        _MergeTextBackend(
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
        ),
        make_work(tmp_path),
        **args,
    )
    items = json.loads(merged_items_path(deep_dir(tmp_path)).read_text())
    assert [i["file"] for i in items["items"]] == ["api.py"]
    assert items.get("partial") is not True


async def test_merge_raises_structured_error_on_str(tmp_path: Path, make_work) -> None:
    """R2/AC2: a genuinely-unparseable str raises CrossStackMergeError with shape + stacks."""
    from daydream.phases import CrossStackMergeError

    args = _merge_args(tmp_path)
    with pytest.raises(CrossStackMergeError) as excinfo:
        await phase_cross_stack_merge(
            _MergeTextBackend("The review is done; no JSON items list here.", None),
            make_work(tmp_path),
            **args,
        )
    assert excinfo.value.response_shape == "str"
    assert excinfo.value.stack_context == ["python"]


async def test_archived_str_response_message_byte_identical(tmp_path, make_work) -> None:
    """Reopen #361: the archived Pi str shape raises CrossStackMergeError whose
    message is byte-identical to the record archived from run c48ca322
    (deep/per-stack-failures.json: 'Cross-stack merge returned no item list
    (got str)'), not merely a shape-typed error."""
    from daydream.phases import CrossStackMergeError

    args = _merge_args(tmp_path)
    with pytest.raises(CrossStackMergeError) as excinfo:
        await phase_cross_stack_merge(
            _MergeTextBackend(ARCHIVED_MERGE_STR, None),
            make_work(tmp_path),
            **args,
        )
    assert excinfo.value.response_shape == "str"
    assert excinfo.value.stack_context == ["python"]
    assert str(excinfo.value) == "Cross-stack merge returned no item list (got str)"


async def test_merge_accepts_bare_list_end_to_end(multi_stack_target, monkeypatch) -> None:
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


async def test_merge_str_response_is_salvaged_not_fatal(
    multi_stack_target, monkeypatch
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
    stub.merge_emit_str = "All stacks reviewed. No JSON item list to emit."
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
    # R4/AC2: pin the deterministic stack names instead of only asserting
    # non-empty. multi_stack_target routes api.py/App.tsx/README.md to the
    # python + react + generic stacks; the structural meta-stack is partitioned
    # out of the merge's per-stack records, so it is absent from the context.
    assert failures["__merge__"]["stack_context"] == ["generic", "python", "react"]
    assert len(list(dd.glob("stack-*-records.json"))) > 0  # R5: completed records survive


async def test_merge_failure_relaunch_picks_up_salvage(
    multi_stack_target, monkeypatch, mute_side_effects
) -> None:
    """R6/AC4: --start-at fix after salvage picks up partial items; no re-review, no re-merge."""
    silence(monkeypatch)
    mute_side_effects()
    stub = install_stub_backend(monkeypatch, multi_stack_target)
    stub.parse_severity = "high"
    stub.merge_emit_str = "prose with no item list"
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
    assert not any("cross-stack merge agent" in c["prompt"].lower() for c in stub.calls)
    assert not any("per-stack review" in c["prompt"].lower() for c in stub.calls)
    # R6 positive outcome: the fix phase consumed the salvaged partial items --
    # the fix prompt (built from merged-items.json) references the surviving
    # per-stack finding's description + file. A regression where --start-at fix
    # fails to load the partial merged-items.json would fail this assertion.
    fix_calls = [
        c["prompt"]
        for c in stub.calls
        if "fix these" in c["prompt"].lower() or "fix this issue" in c["prompt"].lower()
    ]
    assert any("Sample issue" in p and "api.py" in p for p in fix_calls)


async def test_merge_failure_merge_resume_skips_merge_entry(
    multi_stack_target, monkeypatch, mute_side_effects
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
