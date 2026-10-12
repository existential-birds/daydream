"""Shared builders for test-execution fixtures."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any, Literal

from daydream.test_execution import TestExecutionIdentity


def test_execution_identity(**overrides: Any) -> TestExecutionIdentity:
    """Build a reusable host-passed execution identity, overriding named fields.

    The defaults describe a passed host ``uv run pytest`` run against the current working directory; pass
    ``cwd_relative``, ``interpreter``, ``config_digest``, ``absent_components``, ``kind``, ``outcome``, etc. to
    vary one dimension at a time."""
    fields: dict[str, Any] = {
        "session_id": "s", "argv": ("uv", "run", "pytest"), "cwd_relative": ".", "runner": "uv", "interpreter": None,
        "config_digest": "d" * 64, "absent_components": (), "input_tree_key": "t", "output_tree_key": "t",
        "head_sha": "a" * 40, "branch": "feature", "kind": "host", "outcome": "passed",
    }
    return TestExecutionIdentity(**{**fields, **overrides})


def host_test_command(directory: Path, *, mode: Literal["pass", "first", "all"] = "pass",
                      output: str | None = None) -> str:
    """Create a real suite with an external counter; first/all modes exit nonzero.

    Keep directory outside the target worktree so instrumentation cannot change
    the tree being validated. The counter is named suite-count.txt.
    """
    directory.mkdir(parents=True, exist_ok=True)
    counter = directory / "suite-count.txt"
    script = directory / "suite.py"
    script.write_text(
        "from pathlib import Path\nimport sys\n"
        f"counter = Path({str(counter)!r})\n"
        "count = int(counter.read_text()) + 1 if counter.exists() else 1\n"
        "counter.write_text(str(count))\n"
        f"failed = {mode == 'all'!r} or ({mode == 'first'!r} and count == 1)\n"
        f"print({output!r} if {output is not None!r} else ('1 failed' if failed else '2 passed'))\n"
        "sys.exit(1 if failed else 0)\n"
    )
    return shlex.join([sys.executable, str(script)])
