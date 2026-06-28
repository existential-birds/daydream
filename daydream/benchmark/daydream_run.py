"""Non-interactive daydream review subprocess wrapper for the benchmark harness.

Issues the invocation
``daydream --non-interactive --base <sha> --trajectory <path> [--backend <b>]
[--model <m>] <checkout>`` (see ``research/daydream-invocation.md`` §0/§1/§3),
then returns the path to the canonical findings artifact
``<checkout>/.daydream/deep/merged-items.json``. The reviewer ``provider`` is
forwarded via the ``PI_PROVIDER`` environment variable, never as an argv flag.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from daydream.github_app import APP_ID_ENV, APP_PRIVATE_KEY_ENV

#: How many trailing characters of stderr to surface in the failure message.
_STDERR_TAIL = 4000

#: Maximum wall-clock seconds to wait for the daydream subprocess.
#: A full deep review can take many minutes; 1 hour is a generous upper bound.
_DAYDREAM_TIMEOUT = 3600


class DaydreamRunError(Exception):
    """Raised when the daydream subprocess exits non-zero."""


class DaydreamArtifactError(Exception):
    """Raised when daydream exits 0 but the expected findings artifact is absent."""


def run_daydream_review(
    checkout: Path,
    *,
    base_sha: str,
    trajectory_path: Path,
    backend: str | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> Path:
    """Run a non-interactive deep daydream review against a checkout.

    Args:
        checkout: Path to the target repository checkout to review.
        base_sha: Raw commit-ish (SHA) to diff against, passed verbatim to ``--base``.
        trajectory_path: Destination for the ATIF v1.6 trajectory JSON.
        backend: Reviewer backend; appended as ``--backend <backend>`` when set.
        model: Reviewer model; appended as ``--model <model>`` when set.
        provider: Reviewer provider; forwarded via the ``PI_PROVIDER`` environment
            variable (never argv) when set.

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
    ]
    if backend:
        cmd += ["--backend", backend]
    if model:
        cmd += ["--model", model]
    cmd.append(str(checkout))

    env = os.environ.copy()
    # The bench reviews arbitrary local checkouts of upstream repos. GitHub App
    # credentials inherited from the operator's shell would make daydream attempt
    # an installation-token resolution for that upstream owner (e.g. grafana),
    # find no installation, and hard-abort (exit 1) before any review runs. App
    # auth is never needed to review a local checkout, so strip it from the env.
    for app_var in (APP_ID_ENV, APP_PRIVATE_KEY_ENV):
        env.pop(app_var, None)
    if provider:
        env["PI_PROVIDER"] = provider
    try:
        result = subprocess.run(  # noqa: S603 - args are harness-controlled, not user input
            cmd,  # noqa: S607 - daydream is a trusted command
            check=False,
            capture_output=True,
            text=True,
            timeout=_DAYDREAM_TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise DaydreamRunError(
            f"daydream review timed out after {_DAYDREAM_TIMEOUT}s for {checkout}"
        ) from exc
    if result.returncode != 0:
        # daydream prints its errors to stdout (Rich console), so a stderr-only
        # message is frequently empty; surface both streams' tails.
        tail = f"{result.stdout or ''}\n{result.stderr or ''}".strip()[-_STDERR_TAIL:]
        raise DaydreamRunError(
            f"daydream review failed (exit {result.returncode}) for {checkout}:\n{tail}"
        )

    artifact = checkout / ".daydream" / "deep" / "merged-items.json"
    if not artifact.exists():
        raise DaydreamArtifactError(
            f"daydream exited 0 but findings artifact is missing: {artifact}"
        )
    return artifact
