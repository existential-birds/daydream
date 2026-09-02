"""Task: projector emits schema-distinct process-trace and task-only records
behind the ``emit_process_traces`` flag (default off = byte-identical D8
behavior: non-decisive findings land in the adjudication report only)."""

import json
from pathlib import Path
from typing import Any

from tests.test_corpus_v2 import (
    _policy_file,
    _write_annotations_snapshot,
    _write_bundle,
)


def _cfg(out_dir: Path, bundle_dir: Path, snapshot: Path, **kw: Any) -> Any:
    from daydream.training.corpus_v2.projector import BuildCorpusV2Config

    return BuildCorpusV2Config(
        out_dir=out_dir,
        bundle_dir=bundle_dir,
        annotation_bundle_dir=snapshot.parent,
        license_policy_path=_policy_file(bundle_dir.parent),
        **kw,
    )


def _records(out_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (out_dir / "corpus-v2.jsonl").read_text().splitlines()
        if line.strip()
    ]


def _build(tmp_path: Path, **kw: Any) -> tuple[Path, Path, Path]:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(
        bundle_dir, dispositions=["accepted", "ambiguous"]
    )
    out_dir = tmp_path / "out"
    from daydream.training.corpus_v2.projector import run_build_corpus_v2

    out = run_build_corpus_v2(_cfg(out_dir, bundle_dir, snap, **kw))
    return out_dir, out, snap


def test_flag_off_emits_only_outcome_finding_records(tmp_path: Path) -> None:
    out_dir, _out, _snap = _build(tmp_path)
    records = _records(out_dir)
    assert records
    assert all(r["record_type"] == "outcome-finding" for r in records)
    summary = _out
    assert set(summary["records_by_type"]) == {"outcome-finding"}


def test_flag_off_is_the_default(tmp_path: Path) -> None:
    out_dir, _out, _snap = _build(tmp_path)
    off_dir, _out_off, _snap2 = _build(tmp_path / "b", emit_process_traces=False)
    assert (off_dir / "corpus-v2.jsonl").read_bytes() == (
        out_dir / "corpus-v2.jsonl"
    ).read_bytes()


def test_flag_on_emits_process_trace_and_task_only_records(tmp_path: Path) -> None:
    out_dir, out, _snap = _build(tmp_path, emit_process_traces=True)
    records = _records(out_dir)
    types = {r["record_type"] for r in records}
    assert "process-trace" in types
    assert "task-only" in types
    assert "outcome-finding" in types
    # The process-trace record is schema-distinct and NEVER gold: always
    # silver with no outcome label.
    traces = [r for r in records if r["record_type"] == "process-trace"]
    assert traces
    assert all(r["tier"] == "silver" for r in traces)
    assert all(r["outcome_label"] is None for r in traces)
    task_only = [r for r in records if r["record_type"] == "task-only"]
    assert task_only
    assert all(r["tier"] == "task-only" for r in task_only)
    assert all(r["outcome_label"] is None for r in task_only)
    # The summary counts the new types.
    assert out["records_by_type"].get("process-trace", 0) >= 1
    assert out["records_by_type"].get("task-only", 0) >= 1
    # Process-trace records count under the silver tier population.
    assert out["records_by_tier"].get("silver", 0) >= len(traces)


def test_flag_on_records_carry_identity_lineage_and_distinct_ids(
    tmp_path: Path,
) -> None:
    out_dir, _out, _snap = _build(tmp_path, emit_process_traces=True)
    records = _records(out_dir)
    derived = [
        r for r in records if r["record_type"] in ("process-trace", "task-only")
    ]
    assert derived
    ids = [r["record_id"] for r in records]
    assert len(ids) == len(set(ids)), "record_ids must be unique across types"
    for rec in derived:
        assert rec["schema_version"] == "2"
        assert rec["session_id"] == "sess-a"
        lineage = rec["lineage"]
        assert lineage["repo_slug"] == "owner/repo-a"
        assert lineage["license_decision"]["status"] == "admitted"
        assert lineage["split"] in ("train", "validation", "holdout")
        assert lineage["exclusion_reason"] is None
        # The adjudication report still carries the non-decisive finding (D8
        # report output is preserved), but the finding is no longer counted
        # as an exclusion — it was materialized as a record instead.
    lineage = json.loads((out_dir / "lineage.json").read_text())
    exclusions = lineage["exclusions_by_reason"]
    assert exclusions.get("non-decisive-adjudication", 0) == 0
    report = json.loads((out_dir / "adjudication-report.json").read_text())
    assert report


def test_flag_on_split_files_contain_derived_records(tmp_path: Path) -> None:
    out_dir, _out, _snap = _build(tmp_path, emit_process_traces=True)
    all_records = _records(out_dir)
    derived_types = {"process-trace", "task-only"}
    split_records: list[dict[str, Any]] = []
    for name in ("train.jsonl", "validation.jsonl", "holdout.jsonl"):
        split_records.extend(
            json.loads(line)
            for line in (out_dir / name).read_text().splitlines()
            if line.strip()
        )
    assert len(split_records) == len(all_records)
    assert any(r["record_type"] in derived_types for r in split_records)
