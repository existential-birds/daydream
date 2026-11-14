"""Record Identity And Retained Tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.deep.artifacts import deep_dir
from daydream.phases.findings import _write_single_stack_merged_items
from tests.deep_orchestrator.support import (
    _merged_items,
    _prime_source_uid_merge_resume,
    _provenance_item,
    _source_uids_by_description,
)
from tests.harness.console import collapse_panel_text as _panel_text
from tests.harness.review_result import saved_coverage
from tests.test_deep_orchestrator import (
    _TWIN_DESCRIPTION,
    _force_interactive,
    _install_stub_backend,
    _record,
    _run_deep,
    _silence,
)


async def test_hallucinated_merge_source_uid_is_dropped_and_reported(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """#1111 real-path: an invented uid is discarded; the finding is not."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _prime_source_uid_merge_resume(multi_stack_target)
    stub.merge_items = [
        # One real uid and one invention, on the same item: the real half must
        # survive the drop of the other rather than the whole list being voided.
        _provenance_item(1, "Partly attributed finding", source_uids=["python:1", "python:99"]),
        # Wholly invented, including a stack name this run has no records for.
        _provenance_item(2, "Wholly misattributed finding", file="App.tsx", source_uids=["ghost:7"]),
    ]

    assert await _run_deep(multi_stack_target, start_at="merge") == 0

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    by_description = _source_uids_by_description(deep)
    # The real uid survives; the invention beside it is gone.
    assert by_description["Partly attributed finding"] == ["python:1"]
    # Fail-open: the wholly misattributed finding SHIPS, with an honest empty
    # provenance. Losing the finding to protect a bookkeeping field would be the
    # bug, not the fix.
    assert by_description["Wholly misattributed finding"] == []
    raw = (deep / "merged-items.json").read_text()
    assert "python:99" not in raw, f"invented uid reached the canonical artifact: {raw}"
    assert "ghost:7" not in raw, f"invented uid reached the canonical artifact: {raw}"

    # ONE aggregate warning for the whole merge (a model that misreads the uid
    # convention misreads it for every item at once), naming both bad uids and
    # how much provenance the run lost.
    out = _panel_text(capsys)
    assert ("Merge agent cited 2 source_uid(s) that match no record in this run "
        "(ghost:7, python:99); dropped them from item provenance. "
        "1 of 2 merged item(s) now carry no record attribution." in out
    ), out

@pytest.mark.parametrize("provenance", ["null", "string"])
async def test_unattributable_merge_source_uids_require_current_schema(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    provenance: str,
) -> None:
    """Null attribution ships honestly as []; a wrong-typed key rejects synthesis but keeps valid siblings."""

    _silence(monkeypatch)
    _force_interactive(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    _prime_source_uid_merge_resume(multi_stack_target)
    stub.merge_items = [_provenance_item(1, "Unattributable finding",
        source_uids="python:1" if provenance == "string" else None)]
    stub.merge_emit_str = json.dumps({"items": [{"related_files": None, **item} for item in stub.merge_items]})

    assert await _run_deep(multi_stack_target, start_at="merge") == (0 if provenance == "null" else 1)

    deep = deep_dir(multi_stack_target, allow_standalone=True)
    by_description = _source_uids_by_description(deep)
    if provenance == "null":
        assert "Unattributable finding" in by_description, f"finding was lost: {sorted(by_description)}"
        assert by_description["Unattributable finding"] == []
    else:
        assert "Unattributable finding" not in by_description
        assert by_description, "valid sibling records must survive invalid merge output"
        assert saved_coverage(deep).phases["merge"]["status"] == "failed"

    # A null is a documented answer, not a bad claim: nothing was cited, so
    # nothing can have been invented, so the unknown-uid warning must stay quiet.
    out = _panel_text(capsys)
    assert "match no record in this run" not in out, out

def test_host_bypass_structural_fold_survivor_inherits_both_provenances(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unevidenced language twin folds into the structural survivor, keeping both identities and severity."""

    _silence(monkeypatch)
    deep = _prime_source_uid_merge_resume(multi_stack_target,
        structure=[_record(description=_TWIN_DESCRIPTION, line=5, evidence="api.py:5",
                           severity="high", uid="structure:1")],
    )
    language = [_record(
            id=1, description=_TWIN_DESCRIPTION, line=5,
            evidence="", uid="python:1",
        ), _record(id=2, description="Unrelated react concern", file="App.tsx", evidence="App.tsx:1", uid="react:1"),
    ]

    _write_single_stack_merged_items(
        multi_stack_target, deep, language, deep / "stack-structure-records.json", allow_standalone=True,
    )
    twins = [item for item in _merged_items(deep) if item["description"] == _TWIN_DESCRIPTION]
    assert len(twins) == 1
    survivor = "structural"
    source_uids = ["structure:1", "python:1"]
    assert twins[0]["lens"] == "structural"
    assert twins[0]["source_uids"] == source_uids
    assert twins[0]["severity"] == "high"
    assert _source_uids_by_description(deep)["Unrelated react concern"] == ["react:1"]

    folded = json.loads((deep / "folded-structural.json").read_text())
    assert folded["folded_count"] == 1
    assert folded["folded"][0]["survivor"] == survivor
    assert folded["folded"][0]["source_uids"] == source_uids



def test_persisted_record_artifacts_admit_only_current_host_identity() -> None:
    """Loaded records must carry this scope, revision, run lineage and canonical UIDs.

    The runner fault journeys corrupt bytes or shape; foreign identity metadata is
    otherwise only rejected implicitly through those slower whole-run paths.
    """
    from daydream.phases.review import valid_record_artifact

    revision = {'head_sha': 'h' * 40}
    valid = {'issues': [], 'scope_id': 'python', 'analyzed_revision': revision, 'originating_run_id': 'run-1'}
    assert valid_record_artifact(valid, scope_id='python', analyzed_revision=revision)
    faults = [{**valid, 'unexpected': True}, {**valid, 'scope_id': 'foreign'},
              {**valid, 'analyzed_revision': {'head_sha': 'f' * 40}}, {**valid, 'originating_run_id': ''},
              {**valid, 'incomplete': False}, {**valid, 'issues': {}}, {**valid, 'issues': [{'uid': 1}]},
              {**valid, 'issues': [{'uid': 'react:1'}]}, {**valid, 'issues': [{'uid': 'python:01'}]}]
    assert not any(valid_record_artifact(fault, scope_id='python', analyzed_revision=revision) for fault in faults)
