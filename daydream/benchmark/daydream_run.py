"""Non-interactive daydream review subprocess wrapper for the benchmark harness.

Issues the invocation
``daydream --non-interactive --base <sha> --trajectory <path> [--backend <b>]
[--model <m>] <checkout>`` (see ``research/daydream-invocation.md`` §0/§1/§3),
then returns the path to the canonical findings artifact
``<checkout>/.daydream/deep/merged-items.json``. The reviewer ``provider`` is
forwarded via the ``PI_PROVIDER`` environment variable, never as an argv flag.
"""

from __future__ import annotations

import collections
import os
import queue
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from daydream.github_app import APP_ID_ENV, APP_PRIVATE_KEY_ENV

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: How many trailing characters of stderr to surface in the failure message.
_STDERR_TAIL = 4000

#: Maximum wall-clock seconds to wait for the daydream subprocess.
#: A full deep review can take many minutes; 1 hour is a generous upper bound.
_DAYDREAM_TIMEOUT = 3600


class DaydreamRunError(Exception):
    """Raised when the daydream subprocess exits non-zero."""


class DaydreamArtifactError(Exception):
    """Raised when daydream exits 0 but the expected findings artifact is absent."""


def _run_captured(cmd: list[str], env: dict[str, str], checkout: Path) -> None:
    """Quiet path: run to completion, capturing output; raise on timeout/non-zero."""
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


def _run_streamed(
    cmd: list[str], env: dict[str, str], checkout: Path, on_line: Callable[[str], None]
) -> None:
    """Verbose path: stream merged stdout/stderr line-by-line to ``on_line``.

    A bounded tail of the most recent lines is retained so a non-zero exit can
    surface the same kind of error context as the captured path. A reader thread
    drains stdout onto a queue so the whole stream is bounded by a single
    ``_DAYDREAM_TIMEOUT`` wall-clock deadline: a child that holds stdout open
    without emitting output (or runs past the deadline mid-stream) is killed
    rather than blocking the read loop forever.
    """
    tail: collections.deque[str] = collections.deque(maxlen=40)
    lines: queue.Queue[str | None] = queue.Queue()
    with subprocess.Popen(  # noqa: S603 - args are harness-controlled, not user input
        cmd,  # noqa: S607 - daydream is a trusted command
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    ) as proc:
        assert proc.stdout is not None
        stdout = proc.stdout

        def _pump() -> None:
            for line in stdout:
                lines.put(line)
            lines.put(None)  # sentinel: stdout closed (child exiting)

        threading.Thread(target=_pump, daemon=True).start()
        deadline = time.monotonic() + _DAYDREAM_TIMEOUT
        while True:
            try:
                line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                proc.kill()
                raise DaydreamRunError(
                    f"daydream review timed out after {_DAYDREAM_TIMEOUT}s for {checkout}"
                ) from None
            if line is None:
                break
            on_line(line)
            tail.append(line)
        returncode = proc.wait()
    if returncode != 0:
        raise DaydreamRunError(
            f"daydream review failed (exit {returncode}) for {checkout}:\n{'\n'.join(tail)}"
        )


def run_daydream_review(
    checkout: Path,
    *,
    base_sha: str,
    trajectory_path: Path,
    backend: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    on_line: Callable[[str], None] | None = None,
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
        on_line: When set, the review runs via ``subprocess.Popen`` and each output
            line (stdout+stderr merged) is forwarded to this callback live instead of
            being captured silently. When ``None`` (default), the quiet
            ``subprocess.run`` capture path is used unchanged.

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
    # The provider argument is the single source of truth; an inherited
    # PI_PROVIDER must not leak past a run with no explicit override.
    if provider:
        env["PI_PROVIDER"] = provider
    else:
        env.pop("PI_PROVIDER", None)

    if on_line is None:
        _run_captured(cmd, env, checkout)
    else:
        _run_streamed(cmd, env, checkout, on_line)

    artifact = checkout / ".daydream" / "deep" / "merged-items.json"
    if not artifact.exists():
        raise DaydreamArtifactError(
            f"daydream exited 0 but findings artifact is missing: {artifact}"
        )
    return artifact
