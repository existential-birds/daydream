"""A report-only review never enters pull request posting."""
from __future__ import annotations

from pathlib import Path

import pytest

from daydream.extensions import Registry
from daydream.workspace import WorkContext

# The merge agent returns a schema item list; the host renders the report.


async def test_report_only_review_mode_never_enters_pr_posting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review (report-only) mode stops before PR resolution or posting."""
    from daydream.deep.merge_steps import _step_post_review
    from daydream.flows.engine import FlowContext
    from daydream.run_config import RunConfig
    from daydream.run_context import InteractionPolicy, RunContext
    from tests.harness.review_result import review_coverage

    coverage = review_coverage()
    for scope in coverage.scopes:
        coverage.record_scope(scope, "complete")
    coverage.record_phase("merge", "complete", noop=True)
    coverage.finalize("completed")
    dd = tmp_path / "private-deep"
    ctx = FlowContext(
        config=RunConfig(), registry=Registry(), run_context=RunContext(InteractionPolicy()),
        work=WorkContext(repo=tmp_path, source=tmp_path, base_branch="main", base_sha="base", head_branch="feature",
                         head_sha="head", is_ephemeral=False, run_id="test-run"),
        data={"dd": dd, "items_file": tmp_path / "merged-items.json", "review_coverage": coverage, "mode": "review"},
    )

    async def _post_forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("report-only review mode must not resolve or post to a PR")

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _post_forbidden)
    assert await _step_post_review(ctx) is None
