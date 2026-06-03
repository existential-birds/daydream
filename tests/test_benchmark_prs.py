"""Tests for the pinned 26-PR evaluable benchmark registry."""

from collections import Counter

from daydream.benchmark.prs import load_evaluable_prs


def test_registry_is_the_26_evaluable_prs():
    prs = load_evaluable_prs()
    assert len(prs) == 26
    assert Counter(p.source_repo for p in prs) == {"sentry": 6, "grafana": 10, "cal.com": 10}
    assert all(p.golden_url.startswith("https://github.com/") for p in prs)
    assert all(len(p.base_sha) == 40 and len(p.head_sha) == 40 for p in prs)
    s = {p.pr_number: p for p in prs if p.source_repo == "sentry"}
    assert s[67876].base_sha.startswith("344aa10") and s[67876].head_sha.startswith("bb75657")
    assert len({p.golden_url for p in prs}) == 26  # golden_url == benchmark_data.json key
