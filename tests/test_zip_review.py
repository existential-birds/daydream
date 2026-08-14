"""Regression coverage for scripts/zip-review.sh archive-path handling.

The script derives its latest-run metadata from a positional directory and
historically interpolated that filesystem path into python3 -c program
strings. A path containing a single quote could break out of the program, and a
Python-shaped path could become executable source. These tests drive the real
executable script against archive roots whose single directory component is
Python-shaped, and assert that the path is treated as inert data (the embedded
pathlib call never runs) while branch-derived ZIP output still works.
"""

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from tests.harness.git_helpers import git, init_repo

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "zip-review.sh"

# A directory name that is Python-shaped: if it were interpolated into a
# python3 -c program string (the defect this regression guards against),
# executing it would write tmp_path/python-source-ran via the embedded
# pathlib call.
PYTHON_SHAPED_DIR = (
    "archive'+(__import__('pathlib').Path('python-source-ran')"
    ".write_text('executed') and '')+'"
)


@pytest.mark.parametrize(
    ("metadata_source", "expected_zip_name"),
    [
        pytest.param("manifest", "from-manifest.zip", id="manifest"),
        pytest.param("trajectory", "from-trajectory.zip", id="trajectory"),
    ],
)
def test_zip_review_treats_archive_paths_as_data(
    metadata_source: str,
    expected_zip_name: str,
    tmp_path: Path,
) -> None:
    """Archive paths reach python3 as data via sys.argv, never as source."""
    archive_root = tmp_path / PYTHON_SHAPED_DIR
    run_dir = archive_root / "runs" / "run-id"
    run_dir.mkdir(parents=True)
    (run_dir / "review-output.md").write_text("review\n")

    if metadata_source == "manifest":
        (run_dir / "trajectory.json").write_text("{}")
        (run_dir / "manifest.json").write_text(
            json.dumps({"git": {"branch": "team/from-manifest"}})
        )
    else:
        target_repo = tmp_path / "target-repo"
        init_repo(target_repo)
        git(target_repo, "checkout", "-b", "team/from-trajectory")
        (run_dir / "trajectory.json").write_text(
            json.dumps({"extra": {"target_dir": str(target_repo)}})
        )

    result = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        [str(SCRIPT), str(archive_root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"exit={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert f"Output: {expected_zip_name}" in result.stdout

    zip_path = tmp_path / expected_zip_name
    assert zipfile.is_zipfile(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        basenames = {Path(name).name for name in zf.namelist()}
    assert "trajectory.json" in basenames
    assert "review-output.md" in basenames

    # The Python-shaped path was inert data; the embedded pathlib call never ran.
    assert not (tmp_path / "python-source-ran").exists()
    # The branch-derived name was used, not the run-ID fallback.
    assert not (tmp_path / "run-id.zip").exists()
