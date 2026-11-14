"""Current review-result and findings fixtures shared by unit and consumer tests."""
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from daydream.deep.detection import StackAssignment
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




def merge_result(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Current producer envelope with explicit nullable relationships and attribution."""
    return {"items": [{"related_files": None, "source_uids": None, **item} for item in items]}


def records_artifact(coverage: ReviewCoverage, scope_id: str, issues: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    """Bind validated producer records to their current scope and captured revision."""
    return {"issues": list(issues), "scope_id": scope_id, "analyzed_revision": coverage.revision.to_dict(),
            "originating_run_id": coverage.run_id}


def frozen_target_decisions(target: Path, stack: StackAssignment, revision: AnalyzedRevision,
                            diff: str) -> list[dict[str, Any]]:
    """Seed a compatible prior review assertion from the actual frozen Git inputs."""
    from daydream.backends import Backend
    from daydream.hunk_index import write_hunk_index
    from daydream.review_stage_inputs import CanonicalChange, StageInputFactory
    from daydream.workspace import WorkContext
    from tests.harness.stub_backend import StubBackend

    dd = target / '.daydream'
    dd.mkdir(exist_ok=True)
    diff_path = dd / 'diff.patch'
    diff_path.write_text(diff, encoding='utf-8')
    index_path = write_hunk_index(dd, diff)
    work = WorkContext(repo=target, source=target, base_branch='main', base_sha=revision.merge_base_sha,
        head_branch=None, head_sha=revision.head_sha, is_ephemeral=False, run_id='primed-review-fixture')
    factory = StageInputFactory(cast(Backend, StubBackend(target)), work, stack,
        change=CanonicalChange.capture(diff_path, index_path), shared_paths={},
        revision=revision.to_dict(), artifact_session=None,
        allow_standalone=True, read_only=True)
    parts = [part for batch in factory.assignment_batches for part in batch]
    if stack.stack_name == 'structure':
        return [{'target_id': 'integration:structure', 'status': 'reviewed', 'reason': '',
                 'evidence_refs': parts[0]['source_refs'][:1]}]
    return [{'target_id': part['target_id'], 'status': 'reviewed', 'reason': '',
             'evidence_refs': part['source_refs'][:1]} for part in parts]




def saved_coverage(root: Path) -> ReviewCoverage:
    """Read checked current evidence from a real run's deep artifact directory."""
    return ReviewCoverage.from_dict(json.loads((root / 'review-coverage.json').read_text()))
