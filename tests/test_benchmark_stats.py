"""Tests for the trial distribution statistics module."""

from __future__ import annotations

import statistics

import pytest

from daydream.benchmark.score import DaydreamScores
from daydream.benchmark.stats import (
    bootstrap_ci,
    compute_distribution,
    distribution_to_dict,
    format_distribution_table,
)


def _scores(precision: float, recall: float, f1: float) -> DaydreamScores:
    return DaydreamScores(precision=precision, recall=recall, f1=f1)


def test_compute_distribution_mean_median_stddev():
    trials = [
        _scores(0.4, 0.5, 0.44),
        _scores(0.6, 0.5, 0.55),
        _scores(0.5, 0.8, 0.62),
    ]
    dist = compute_distribution(trials)
    assert dist.n == 3
    assert dist.precision.mean == pytest.approx(statistics.fmean([0.4, 0.6, 0.5]))
    assert dist.precision.median == pytest.approx(0.5)
    assert dist.precision.stddev == pytest.approx(statistics.stdev([0.4, 0.6, 0.5]))
    assert dist.precision.min == 0.4 and dist.precision.max == 0.6
    assert dist.recall.mean == pytest.approx(0.6)


def test_compute_distribution_single_trial_zero_stddev():
    dist = compute_distribution([_scores(0.5, 0.5, 0.5)])
    assert dist.n == 1
    assert dist.precision.stddev == 0.0
    assert dist.precision.ci_low == 0.5 and dist.precision.ci_high == 0.5


def test_compute_distribution_empty_raises():
    with pytest.raises(ValueError):
        compute_distribution([])


def test_bootstrap_ci_is_seeded_and_deterministic():
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    a = bootstrap_ci(values, seed=42)
    b = bootstrap_ci(values, seed=42)
    assert a == b


def test_bootstrap_ci_brackets_the_mean():
    values = [0.4, 0.45, 0.5, 0.55, 0.6, 0.5, 0.48, 0.52]
    mean = statistics.fmean(values)
    lo, hi = bootstrap_ci(values, seed=42)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_ci_zero_variance_is_a_point():
    lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5], seed=42)
    assert lo == 0.5 and hi == 0.5


def test_bootstrap_ci_single_value():
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_ci_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_bootstrap_ci_nonpositive_n_boot_raises():
    with pytest.raises(ValueError, match="n_boot must be positive"):
        bootstrap_ci([0.4, 0.6], n_boot=0)


def test_bootstrap_ci_out_of_range_confidence_raises():
    for bad_ci in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"ci must be in \(0, 1\)"):
            bootstrap_ci([0.4, 0.6], ci=bad_ci)


def test_format_distribution_table_contains_metrics_and_stats():
    dist = compute_distribution([_scores(0.4, 0.5, 0.44), _scores(0.6, 0.7, 0.65)])
    out = format_distribution_table(dist)
    assert "precision" in out and "recall" in out and "f1" in out
    assert "mean" in out and "median" in out and "stddev" in out
    assert "ci95" in out
    assert "2 trial" in out


def test_distribution_to_dict_shape():
    dist = compute_distribution([_scores(0.4, 0.5, 0.44), _scores(0.6, 0.7, 0.65)])
    d = distribution_to_dict(dist)
    assert d["n"] == 2
    for metric in ("precision", "recall", "f1"):
        assert set(d[metric]) == {"mean", "median", "stddev", "min", "max", "ci_low", "ci_high"}
