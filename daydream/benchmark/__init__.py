"""Code-review benchmark harness for daydream.

Adapter that scores daydream's deep-review findings against the withmartian
``code-review-benchmark`` offline set. This package is built up across several
tasks; only the pinned evaluable-PR registry is exported so far.
"""

from daydream.benchmark.config import BenchConfig
from daydream.benchmark.orchestrator import run_bench
from daydream.benchmark.prs import EVALUABLE_PRS, EvaluablePR, load_evaluable_prs

__all__ = [
    "EVALUABLE_PRS",
    "BenchConfig",
    "EvaluablePR",
    "load_evaluable_prs",
    "run_bench",
]
