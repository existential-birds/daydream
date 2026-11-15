"""Review coverage refuses malformed failure diagnostics."""
from __future__ import annotations

import pytest

from daydream.review_result import ReviewCoverage
from tests.harness.fake_gh import FakeGh
from tests.harness.review_result import review_coverage


@pytest.fixture(autouse=True)
def _no_open_pr(fake_gh: FakeGh) -> None:
    fake_gh.set_response("pr-list", value=[])


def test_coverage_rejects_malformed_failure_diagnostics() -> None:
    diagnostics = {'scopes': {}, 'phases': {'extra': 'failure'}}
    payload = review_coverage().to_dict()
    payload['diagnostics'] = diagnostics
    with pytest.raises(ValueError):
        ReviewCoverage.from_dict(payload)
