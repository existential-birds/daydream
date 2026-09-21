"""Git-observed route names are data, not shell expressions."""

import subprocess
from pathlib import Path

from daydream import clock
from daydream.deep.finite_review import _request_evidence


def test_literal_search_accepts_git_observed_route_metacharacters(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = "routes/$shelfId.[tab].tsx"
    (tmp_path / "routes").mkdir()
    (tmp_path / path).write_text('const label = "Name your shelf";\n')
    subprocess.run(["git", "-C", str(tmp_path), "add", "--", path], check=True)
    result = _request_evidence(
        tmp_path, {"kind": "search", "path": "routes", "pattern": "Name your shelf"},
        4096, clock.monotonic() + 5,
    )
    assert result["complete"] is True
    assert result["matches"] == [{"path": path, "line": 1, "text": 'const label = "Name your shelf";'}]
