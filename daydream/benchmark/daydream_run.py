"""Non-interactive daydream review subprocess wrapper for the benchmark harness.

Issues the exact invocation
``daydream --non-interactive --base <sha> --trajectory <path> <checkout>``
(see ``research/daydream-invocation.md`` §0/§1/§3), then returns the path to
the canonical findings artifact ``<checkout>/.daydream/deep/merged-items.json``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: How many trailing characters of stderr to surface in the failure message.
_STDERR_TAIL = 4000


class DaydreamRunError(Exception):
    """Raised when the daydream subprocess exits non-zero."""


class DaydreamArtifactError(Exception):
    """Raised when daydream exits 0 but the expected findings artifact is absent."""


def run_daydream_review(checkout: Path, *, base_sha: str, trajectory_path: Path) -> Path:
    """Run a non-interactive deep daydream review against a checkout.

    Args:
        checkout: Path to the target repository checkout to review.
        base_sha: Raw commit-ish (SHA) to diff against, passed verbatim to ``--base``.
        trajectory_path: Destination for the ATIF v1.6 trajectory JSON.

    Returns:
        Path to the canonical ``merged-items.json`` findings artifact.

    Raises:
        DaydreamRunError: If the daydream process exits non-zero (includes a stderr tail).
        DaydreamArtifactError: If the run succeeds but the artifact is absent.
    """
    cmd = [
        "daydream",
        "--non-interactive",
        "--base",
        base_sha,
        "--trajectory",
        str(trajectory_path),
        str(checkout),
    ]
    result = subprocess.run(  # noqa: S603 - args are harness-controlled, not user input
        cmd,  # noqa: S607 - daydream is a trusted command
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-_STDERR_TAIL:]
        raise DaydreamRunError(
            f"daydream review failed (exit {result.returncode}) for {checkout}:\n{stderr_tail}"
        )

    artifact = checkout / ".daydream" / "deep" / "merged-items.json"
    if not artifact.exists():
        raise DaydreamArtifactError(
            f"daydream exited 0 but findings artifact is missing: {artifact}"
        )
    return artifact
