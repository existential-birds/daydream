"""Code-review benchmark harness for daydream.

Scores daydream's deep-review findings against the withmartian
``code-review-benchmark`` offline set, and hosts the private PR benchmark
workspace (`daydream benchmark init|status|validate`) — strict schemas, the
mode-safe storage/journal layer, and the init/status/validate orchestration.

The pinned evaluable-PR registry and benchmark runner exports are kept; the
new stable schema + workspace-service types are exported for consumers and
future issues.
"""

from daydream.benchmark.config import BenchConfig
from daydream.benchmark.orchestrator import run_bench
from daydream.benchmark.prs import EVALUABLE_PRS, EvaluablePR, load_evaluable_prs
from daydream.benchmark.schema import (
    BenchmarkManifest,
    CaseDocument,
    CaseIndexEntry,
    PullRequestEntry,
    Snapshot,
    classify_validation,
    derive_workspace_state,
    normalize_hostname,
)
from daydream.benchmark.workspace import (
    init_workspace,
    validate_workspace,
    workspace_status,
)

__all__ = [
    "EVALUABLE_PRS",
    "BenchConfig",
    "BenchmarkManifest",
    "CaseDocument",
    "CaseIndexEntry",
    "EvaluablePR",
    "PullRequestEntry",
    "Snapshot",
    "classify_validation",
    "derive_workspace_state",
    "init_workspace",
    "load_evaluable_prs",
    "normalize_hostname",
    "run_bench",
    "validate_workspace",
    "workspace_status",
]
