"""Current review-result and findings fixtures shared by unit and consumer tests."""
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from daydream.deep.detection import StackAssignment
from daydream.deep.records import RecordPool, record_uid, stack_name_from_uid, stamp_record_uids
from daydream.review_result import AnalyzedRevision, PlannedScope, ReviewCoverage


def review_coverage(
    *, run_id: str = "run-1", head_sha: str = "h" * 40,
    scope_ids: Iterable[str] = ("python", "structure"), phases: Iterable[str] = ("merge",),
    files: tuple[str, ...] = ("a.py",),
) -> ReviewCoverage:
    return ReviewCoverage(run_id, AnalyzedRevision(head_sha, "b" * 40, "diff-1"),
                          [PlannedScope(scope, scope, files) for scope in scope_ids], phases)


def terminal_result(state: str = "complete", *, head_sha: str = "h" * 40) -> dict[str, Any]:
    coverage = review_coverage(head_sha=head_sha, scope_ids=("python",))
    coverage.record_scope("python", state, reasons=() if state == "complete" else ("backend_failure",),
                          partial_evidence=state == "incomplete")
    coverage.record_phase("merge", "complete", noop=True)
    return coverage.finalize("completed")


def findings_artifact(
    findings: Iterable[dict[str, Any]] = (), *, kind: str = "review", **fields: Any,
) -> dict[str, Any]:
    artifact = {"schema_version": 2, "repo": "o/r", "pr_number": 7, "head_sha": "h" * 40,
                "findings": [{"location_distrust": False, "severity_before_demotion": None,
                              "severity_off_vocabulary": False, **finding} for finding in findings],
                "kind": kind, "run_info": None, "review_warnings": [], "diagrams": None, **fields}
    if kind == "review":
        artifact.setdefault("terminal_result", terminal_result(head_sha=artifact["head_sha"]))
    return artifact


async def review_scopes(backend: Any, work: Any, stacks: list[StackAssignment], **options: Any) -> Any:
    """Exercise the real provider phase with an explicit planned execution inventory."""
    from daydream.phases.review import phase_per_stack_reviews
    head, base = work.head_sha, work.base_sha
    if head is None or base is None:
        raise ValueError('review scopes require captured commit anchors')
    diff_path = options['diff_path']
    diff_key = hashlib.sha256(Path(diff_path).read_bytes()).hexdigest()
    coverage = ReviewCoverage("scope-test", AnalyzedRevision(head, base, diff_key),
                              [PlannedScope(s.stack_name, s.stack_name, tuple(s.files)) for s in stacks], ())
    return await phase_per_stack_reviews(backend, work, stacks, coverage=coverage, **options)


def merge_result(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Current producer envelope with explicit nullable relationships and attribution."""
    return {"items": [{"related_files": None, "source_uids": None, **item} for item in items]}


def records_artifact(coverage: ReviewCoverage, scope_id: str, issues: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """Bind validated producer records to their current scope and captured revision."""
    return {"issues": list(issues), "scope_id": scope_id, "analyzed_revision": coverage.revision.to_dict(),
            "originating_run_id": coverage.run_id}


def record_pool(root: Path, records: Iterable[dict[str, Any]] = (), structural: Iterable[dict[str, Any]] = (),
                *, paths: Iterable[Path] = (), structural_path: Path | None = None) -> RecordPool:
    """Build the current scoped state while sharing caller-owned record objects."""
    path_map = {path.name.removeprefix("stack-").removesuffix("-records.json"): path for path in paths}
    grouped: dict[str, list[dict[str, Any]]] = {scope: [] for scope in path_map}
    for scope_hint, rows in ((next(iter(path_map), "python"), list(records)), ("structure", list(structural))):
        stamp_record_uids(rows, scope_hint)
        for row in rows:
            scope = stack_name_from_uid(record_uid(row))
            grouped.setdefault(scope, []).append(row)
            path_map.setdefault(scope, root / f"stack-{scope}-records.json")
    if structural_path is not None:
        path_map["structure"] = structural_path
        grouped.setdefault("structure", [])
    coverage = review_coverage(scope_ids=grouped)
    return RecordPool({scope: records_artifact(coverage, scope, rows) for scope, rows in grouped.items()}, path_map)


def saved_coverage(root: Path) -> ReviewCoverage:
    """Read checked current evidence from a real run's deep artifact directory."""
    return ReviewCoverage.from_dict(json.loads((root / 'review-coverage.json').read_text()))
