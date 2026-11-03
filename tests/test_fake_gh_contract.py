from pathlib import Path

import pytest

from daydream import git_ops
from tests.harness import github_schema
from tests.harness.fake_gh import install_fake_gh


def test_fake_gh_serves_recognizable_absence_and_validates_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valueless canned response must be a 404 production can classify.

    Callers key absence off gh's own ``(HTTP 404)`` stderr token, so a fake that renders the miss any other way
    turns the suite's "this resource is absent" idiom into an unclassifiable failure."""
    fake_gh = install_fake_gh(tmp_path, monkeypatch)
    fake_gh.set_response("GET", "repos/o/r/contents/a.py")

    with pytest.raises(git_ops.PathAbsentError) as excinfo:
        git_ops.gh_file_at_ref(tmp_path, "o/r", "0" * 40, "a.py")

    assert "HTTP 404" in str(excinfo.value)
    # The GraphQL schema subset the fake validates against must flag fields GitHub does not define.
    bad = """
    query X($o:String!,$n:String!,$p:Int!) {
      repository(owner:$o,name:$n) { pullRequest(number:$p) { reviewThreads(first:50) {
        nodes { id isResolved isResolvedBy side startSide
                comments(first:100) { nodes { author { login type isBot } } } }
      } } }
    }
    """
    assert {"side", "startSide", "isResolvedBy", "isBot", "type"} <= github_schema.unknown_query_fields(bad)
