"""Manual end-to-end acceptance test for the ``daydream bench`` pipeline.

This test is SKIPPED BY DEFAULT: a real run spends money (two daydream deep
reviews driven by local Claude credentials plus OpenRouter judge calls) and
takes many minutes over the network. It is committed as executable
documentation of the Task 11 acceptance procedure, so the real-path contract
is reproducible by hand:

    set -a; source daydream/.env; set +a
    export MARTIAN_BASE_URL=https://openrouter.ai/api/v1
    export MARTIAN_MODEL=anthropic/claude-opus-4.5
    DAYDREAM_BENCH_E2E_REPO=/path/to/code-review-benchmark/offline \\
        pytest tests/test_benchmark_e2e.py -s --no-skip   # (drop the skip mark)

See ``docs/benchmark.md`` and
``.beagle/concepts/code-review-benchmark-harness/research/smoke-run.md`` for the
measured per-PR cost/time anchor that gates the full ~26-PR sweep.

The assertions encode the OBSERVABLE acceptance contract, not bookkeeping:

  1. ``results/benchmark_data.json`` gains a ``daydream`` review for the 2 PRs.
  2. ``results/<sanitized-model>/evaluations.json`` gains ``daydream`` leaves
     carrying numeric ``tp``/``fp``/``fn`` (precision/recall computable).
  3. The command prints the scored-PR count and the aggregate precision/recall.
  4. A second identical run is incremental: it injects zero new reviews and
     still exits 0 (the regression-check Must-Have).
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

# Real run: spends money + network time. Kept skipped so ``make check`` stays
# green with zero warnings (no custom marker registered in pyproject, so a
# plain skip avoids an unknown-mark warning).
pytestmark = pytest.mark.skip(
    reason="manual: real run, spends money — see docs/benchmark.md and "
    "research/smoke-run.md. Set DAYDREAM_BENCH_E2E_REPO and remove this mark to run."
)

_MODEL = "anthropic/claude-opus-4.5"


def _grafana_daydream_reviews(data: dict) -> dict[str, list]:
    """Return {golden_url: daydream-tagged review list} for grafana entries.

    ``reviews`` is a list of review objects, each tagged with a tool/source
    name; a daydream review is one whose identifying field names daydream.
    """
    out: dict[str, list] = {}
    for url, entry in data.items():
        if "grafana" not in url.lower():
            continue
        reviews = entry.get("reviews", [])
        dd = [
            r
            for r in reviews
            if isinstance(r, dict)
            and "daydream" in json.dumps(r).lower()
        ]
        if dd:
            out[url] = dd
    return out


def test_bench_acceptance_2pr_subset_real_run():
    repo_env = os.environ.get("DAYDREAM_BENCH_E2E_REPO")
    assert repo_env, "set DAYDREAM_BENCH_E2E_REPO to the code-review-benchmark/offline checkout"
    repo = Path(repo_env)
    data_path = repo / "results" / "benchmark_data.json"
    evals_path = repo / "results" / _MODEL.replace("/", "_") / "evaluations.json"
    assert data_path.exists(), f"missing corpus: {data_path}"

    # MARTIAN_* must be exported by the caller (never hard-coded).
    assert os.environ.get("MARTIAN_API_KEY"), "export MARTIAN_API_KEY before running"

    cmd = [
        "daydream",
        "bench",
        "--benchmark-repo",
        str(repo),
        "--only",
        "grafana",
        "--limit",
        "2",
        "--score",
    ]

    # --- Step 1: first real run injects + scores 2 PRs. ---
    first = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    assert first.returncode == 0, f"first run failed: {first.stdout}\n{first.stderr}"

    # Step 2 contract (a): benchmark_data.json gained daydream reviews for 2 PRs.
    data = json.loads(data_path.read_text())
    injected = _grafana_daydream_reviews(data)
    assert len(injected) == 2, f"expected 2 daydream reviews, got {list(injected)}"

    # Step 2 contract (b): evaluations.json has numeric daydream leaves.
    evals = json.loads(evals_path.read_text())
    daydream_leaves = {
        url: tools["daydream"] for url, tools in evals.items() if "daydream" in tools
    }
    assert len(daydream_leaves) == 2, f"expected 2 daydream leaves, got {list(daydream_leaves)}"
    for url, leaf in daydream_leaves.items():
        for key in ("tp", "fp", "fn"):
            assert isinstance(leaf.get(key), int), f"{url} leaf.{key} not numeric: {leaf.get(key)}"
        assert isinstance(leaf.get("precision"), (int, float)), url
        assert isinstance(leaf.get("recall"), (int, float)), url

    # Step 2 contract (c): aggregate line printed with scored count.
    # Normalize whitespace: the Rich console wraps long lines at its detected
    # width (no TTY under capture → a default width), so assertions must not
    # depend on where a line happens to wrap.
    out = " ".join(first.stdout.split())
    assert "daydream aggregate over 2 PR(s)" in out, first.stdout
    assert "precision=" in out and "recall=" in out, first.stdout

    # --- Step 4: re-run is incremental — zero new reviews, exit 0. ---
    before = data_path.read_text()
    second = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ})
    assert second.returncode == 0, f"re-run failed: {second.stdout}\n{second.stderr}"
    second_out = " ".join(second.stdout.split())
    assert "already present" in second_out, f"expected skip messages: {second.stdout}"
    assert "Injected daydream review" not in second_out, f"re-run injected anew: {second.stdout}"
    after = data_path.read_text()
    assert json.loads(after) == json.loads(before), "re-run mutated the corpus (not incremental)"
    assert len(_grafana_daydream_reviews(json.loads(after))) == 2
