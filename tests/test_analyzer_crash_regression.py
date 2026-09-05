"""Subprocess regression for the tree-sitter SIGSEGV bug (#1087).

A native SIGSEGV kills the whole pytest process, so the risky call runs in a
subprocess: a crash becomes an ordinary nonzero exit the test can assert on.
This test pins the GOOD-install contract (exit 0, populated quality output);
the bad-install behavior is pinned in-process by the version-guard tests.
"""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

_WORKER = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    from daydream.eval.analyzer import analyze_quality
    daydream_dir = Path(sys.argv[1])
    report = analyze_quality(daydream_dir)
    # "loc" lives in the per-file quality entry ("sloc"); the top-level
    # report carries only aggregates + scoped_files.
    loc = report["per_file"]["sample.py"]["sloc"]
    print(json.dumps({"scoped_files": report["scoped_files"], "loc": loc}))
    """
)


def test_analyze_quality_survives_in_subprocess(tmp_path: Path) -> None:
    # Production shape: analyze_quality(daydream_dir) scans daydream_dir.parent
    # (the workspace), so the analyzed dir is a .daydream inside the workspace.
    daydream_dir = tmp_path / ".daydream"
    daydream_dir.mkdir()
    (tmp_path / "sample.py").write_text(
        "def compute(x):\n"
        "    total = 0\n"
        "    for i in range(x):\n"
        "        total += x * i\n"
        "    return {k: v for k, v in [(\"t\", total)]}\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER, str(daydream_dir)],  # venv's own interpreter
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"crashed rc={proc.returncode}: {proc.stderr[-2000:]}"
    out = json.loads(proc.stdout)
    assert out["scoped_files"] == 1
    assert out["loc"] > 0
