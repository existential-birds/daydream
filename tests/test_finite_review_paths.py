"""Git-observed route names are data, not shell expressions."""

import subprocess
from pathlib import Path

import pytest

from daydream import clock
from daydream.deep.finite_review import _request_evidence, prepare_finite_review
from tests.harness.finite_backend import PacketBackend
from tests.test_finite_review import _inputs


def test_assigned_path_outside_verdict_schema_keeps_existing_reviewer(tmp_path: Path) -> None:
    backend = PacketBackend([])
    repo, kwargs = _inputs(tmp_path, backend)
    path = "$route.py"
    (repo / path).write_text("VALUE = 1\n")
    kwargs["files"] = [path]
    diff = kwargs["diff_path"]
    diff.write_text(diff.read_text().replace("app.py", path))

    assert prepare_finite_review(backend, repo, **kwargs) is None
    assert backend.calls == []


@pytest.mark.parametrize("scope", [".", "routes"])
def test_literal_search_accepts_git_observed_route_metacharacters(tmp_path: Path, scope: str) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = "routes/$shelfId.[tab].tsx"
    (tmp_path / "routes").mkdir()
    (tmp_path / path).write_text('const label = "Name your shelf";\n')
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", path], check=True)
    result = _request_evidence(
        tmp_path, {"kind": "search", "path": scope, "pattern": "Name your shelf"},
        4096, clock.monotonic() + 5,
    )
    assert result["complete"] is True
    assert result["matches"] == [{"path": path, "line": 1, "text": 'const label = "Name your shelf";'}]
