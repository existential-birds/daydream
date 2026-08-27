"""Code-review benchmark harness for daydream.

Hosts the private PR benchmark workspace (`daydream benchmark
init|status|validate|import-prs|curate`) — strict schemas, the mode-safe
storage/journal layer, the snapshot-freeze orchestration, and the UI-independent
curation service (``curate --apply-gold``, the issue-#5 seam).

The stable schema + workspace-service + snapshot-freeze types are exported for
consumers and future issues.
"""

from daydream.benchmark.curation import (  # noqa: F401 - stable service surface
    CurationError,
    StaleStateError,
    accept_candidate,
    add_edited_findings,
    add_finding,
    add_findings,
    apply_gold_fragment,
    attest_clean,
    exclude_case,
    exclude_evidence,
    exclude_evidence_batch,
    get_case,
    list_cases,
    mark_ready,
    reinclude_case,
    replace_findings,
    validate_case,
)
from daydream.benchmark.github_import import run_import_prs
from daydream.benchmark.schema import (
    BenchmarkManifest,
    Candidate,
    CaseDocument,
    CaseIndexEntry,
    EvidenceRecord,
    ImportDocument,
    PullRequestEntry,
    Snapshot,
    SnapshotImported,
    classify_validation,
    derive_workspace_state,
    normalize_hostname,
)
from daydream.benchmark.snapshot import freeze_one
from daydream.benchmark.workspace import (
    init_workspace,
    validate_workspace,
    workspace_status,
)

__all__ = [
    "BenchmarkManifest",
    "CurationError",
    "StaleStateError",
    "Candidate",
    "CaseDocument",
    "CaseIndexEntry",
    "EvidenceRecord",
    "ImportDocument",
    "PullRequestEntry",
    "Snapshot",
    "SnapshotImported",
    "classify_validation",
    "derive_workspace_state",
    "freeze_one",
    "init_workspace",
    "normalize_hostname",
    "run_import_prs",
    "accept_candidate",
    "add_edited_findings",
    "add_finding",
    "add_findings",
    "apply_gold_fragment",
    "attest_clean",
    "exclude_case",
    "exclude_evidence",
    "exclude_evidence_batch",
    "get_case",
    "list_cases",
    "mark_ready",
    "reinclude_case",
    "replace_findings",
    "validate_case",
    "validate_workspace",
    "workspace_status",
]
