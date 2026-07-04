"""Distribution statistics over repeated benchmark trials.

Both the reviewer LLM and the LLM judge are stochastic, so a single trial's
precision/recall/F1 is one draw from a distribution with unknown variance. This
module aggregates N trials into per-metric summary statistics — mean, median,
sample standard deviation, min, max, and a seeded percentile bootstrap
confidence interval — so configs can be compared with their spread visible
rather than as bare point estimates.

Exports:
    MetricStats: Per-metric summary (mean/median/stddev/min/max/ci_low/ci_high).
    TrialDistribution: Precision/recall/F1 stats over N trials.
    compute_distribution: Aggregate a list of DaydreamScores into a TrialDistribution.
    bootstrap_ci: Seeded percentile bootstrap CI for a list of values.
    format_distribution_table: Render a TrialDistribution as a printable table.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass

from daydream.benchmark.score import DaydreamScores

#: Default bootstrap resample count. 10k keeps the CI endpoints stable to ~2
#: decimal places while staying fast for the small N (≤~30) trials produce.
_DEFAULT_N_BOOT = 10000

#: Fixed RNG seed so the bootstrap CI is reproducible across runs and tests.
_DEFAULT_SEED = 42


@dataclass(frozen=True)
class MetricStats:
    """Per-metric summary statistics across N trials."""

    mean: float
    median: float
    stddev: float
    min: float
    max: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class TrialDistribution:
    """Aggregated precision/recall/F1 statistics over N benchmark trials."""

    n: int
    precision: MetricStats
    recall: MetricStats
    f1: MetricStats


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = _DEFAULT_N_BOOT,
    ci: float = 0.95,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``values``.

    Resamples ``values`` with replacement ``n_boot`` times, takes each
    resample's mean, and returns the lower/upper percentile endpoints for the
    requested central mass. Seeded so the interval is deterministic (tests and
    reruns get identical endpoints).

    Args:
        values: The per-trial metric values (at least one).
        n_boot: Number of bootstrap resamples.
        ci: Central probability mass to bracket (e.g. 0.95 → 2.5%/97.5%).
        seed: RNG seed for reproducibility.

    Returns:
        ``(ci_low, ci_high)``. When ``values`` has a single element (or zero
        variance) both endpoints equal that value.

    Raises:
        ValueError: If ``values`` is empty, ``n_boot`` is not positive, or
            ``ci`` is not strictly inside ``(0, 1)``.
    """
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be positive, got {n_boot}")
    if not 0 < ci < 1:
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if len(values) == 1:
        return (values[0], values[0])

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_frac = (1.0 - ci) / 2.0
    hi_frac = 1.0 - lo_frac
    lo_idx = max(0, min(n_boot - 1, int(lo_frac * n_boot)))
    hi_idx = max(0, min(n_boot - 1, int(hi_frac * n_boot)))
    return (means[lo_idx], means[hi_idx])


def _metric_stats(values: list[float], *, seed: int = _DEFAULT_SEED) -> MetricStats:
    """Compute summary statistics (incl. bootstrap CI) for one metric's values."""
    ci_low, ci_high = bootstrap_ci(values, seed=seed)
    return MetricStats(
        mean=statistics.fmean(values),
        median=statistics.median(values),
        stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
        min=min(values),
        max=max(values),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def compute_distribution(scores: list[DaydreamScores], *, seed: int = _DEFAULT_SEED) -> TrialDistribution:
    """Aggregate N trials' scores into per-metric summary statistics.

    Args:
        scores: One :class:`DaydreamScores` per trial (at least one).
        seed: RNG seed forwarded to the bootstrap CI.

    Returns:
        A :class:`TrialDistribution` with precision/recall/F1 statistics.

    Raises:
        ValueError: If ``scores`` is empty.
    """
    if not scores:
        raise ValueError("compute_distribution requires at least one trial's scores")
    return TrialDistribution(
        n=len(scores),
        precision=_metric_stats([s.precision for s in scores], seed=seed),
        recall=_metric_stats([s.recall for s in scores], seed=seed),
        f1=_metric_stats([s.f1 for s in scores], seed=seed),
    )


def format_distribution_table(dist: TrialDistribution) -> str:
    """Render a :class:`TrialDistribution` as a fixed-width text table.

    Args:
        dist: The aggregated distribution.

    Returns:
        A multi-line string with a header row and one row per metric carrying
        mean, median, stddev, min, max, and the 95% bootstrap CI.
    """
    header = f"{'metric':<10} {'mean':>7} {'median':>7} {'stddev':>7} {'min':>7} {'max':>7} {'ci95':>17}"
    lines = [f"Distribution over {dist.n} trial(s):", header, "-" * len(header)]
    for name, stats in (("precision", dist.precision), ("recall", dist.recall), ("f1", dist.f1)):
        ci = f"[{stats.ci_low:.3f}, {stats.ci_high:.3f}]"
        lines.append(
            f"{name:<10} {stats.mean:>7.3f} {stats.median:>7.3f} {stats.stddev:>7.3f} "
            f"{stats.min:>7.3f} {stats.max:>7.3f} {ci:>17}"
        )
    return "\n".join(lines)


def distribution_to_dict(dist: TrialDistribution) -> dict:
    """Serialize a :class:`TrialDistribution` to a plain JSON-ready dict."""
    return {
        "n": dist.n,
        "precision": asdict(dist.precision),
        "recall": asdict(dist.recall),
        "f1": asdict(dist.f1),
    }
