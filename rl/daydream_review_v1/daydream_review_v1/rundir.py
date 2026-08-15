"""Pull daydream's archived run directory out of a live rollout sandbox.

Scoring runs while the runtime is still up (a ``@vf.reward`` with a required
``runtime`` parameter — verifiers 0.2.1 ``task.py:269-277``), so the reward reads
daydream's own artifacts straight off the sandbox filesystem and replays them
through the training pipeline's scorer on the host.

Only the handful of small files the scorer actually reads are copied. The full
archive bundle carries per-fork trajectories and diffs that can run to megabytes;
none of it feeds the reward.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import verifiers.v1 as vf

#: Fixed members of the archived run dir the reward path reads. Every one is
#: optional: a review-only rollout has no verdicts, a green run has no
#: fix-failures. ``deep/stack-*-records.json`` is collected separately by glob —
#: its names depend on which stacks the router detected.
#:
#: ``trajectory.json`` and any per-fork ``trajectories/*.json`` are deliberately
#: NOT listed here: they carry untrusted, model-directed operational text from
#: the committed golden run (test-only data) and must never be forwarded into
#: model context through the collector. Exclusion is by omission from this
#: allowlist and is pinned by tests/test_rundir.py::
#: test_fetch_run_dir_excludes_fixture_trajectories.
RUN_DIR_FILES: tuple[str, ...] = (
    "manifest.json",
    "review-output.md",
    "deep/review-output.md",
    "deep/recommendation-verdicts.json",
    "deep/merged-items.json",
    "deep/test-verdict.json",
    "deep/fix-failures.json",
)

DEFAULT_ARCHIVE_ROOT = "/rollout/archive"


async def _session_dir(runtime: vf.Runtime, archive_root: str) -> str | None:
    """Absolute path of the rollout's single archived run dir, or ``None``.

    One rollout is one daydream invocation, so exactly one session id is
    expected. Zero means the run crashed before archiving; more than one means
    the archive is not this rollout's alone and nothing here can be attributed.
    """
    root = shlex.quote(f"{archive_root}/runs")
    result = await runtime.run(["sh", "-c", f"ls -1 {root} 2>/dev/null"], {})
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 1:
        return None
    return f"{archive_root}/runs/{names[0]}"


async def _present_files(runtime: vf.Runtime, session_dir: str) -> list[str]:
    """Relative paths of the run-dir members that actually exist in the sandbox."""
    listing = " ".join(shlex.quote(name) for name in RUN_DIR_FILES)
    script = (
        f"cd {shlex.quote(session_dir)} || exit 0\n"
        f"for f in {listing}; do [ -f \"$f\" ] && printf '%s\\n' \"$f\"; done\n"
        "for f in deep/stack-*-records.json; do [ -f \"$f\" ] && printf '%s\\n' \"$f\"; done\n"
        "exit 0\n"
    )
    result = await runtime.run(["sh", "-c", script], {})
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


async def fetch_run_dir(
    runtime: vf.Runtime,
    dest: Path,
    archive_root: str = DEFAULT_ARCHIVE_ROOT,
) -> Path | None:
    """Copy the rollout's archived run dir into *dest* on the host.

    Args:
        runtime: The live rollout runtime.
        dest: Host directory to populate. Caller owns its lifetime — pass a
            ``tempfile.TemporaryDirectory()`` path so nothing is left behind
            across thousands of rollouts.
        archive_root: ``DAYDREAM_ARCHIVE_DIR`` as the harness set it.

    Returns:
        *dest* when a run dir was found and at least one member copied, else
        ``None``. ``None`` is the crash case and scores zero — it is never an
        exception, because a policy that crashes daydream must still receive a
        gradient rather than killing the rollout.
    """
    session_dir = await _session_dir(runtime, archive_root)
    if session_dir is None:
        return None

    copied = 0
    for rel in await _present_files(runtime, session_dir):
        data = await runtime.read(f"{session_dir}/{rel}")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        copied += 1
    return dest if copied else None


async def daydream_completed(
    runtime: vf.Runtime,
    archive_root: str = DEFAULT_ARCHIVE_ROOT,
) -> bool:
    """Whether daydream finished its pipeline, regardless of exit code.

    daydream writes the trajectory's ``final_metrics`` only at the very end, so
    its presence separates a legitimate non-zero outcome — tests still red after
    the fix pass, ``Stop(1)`` at ``deep/orchestrator.py:1389`` — from a crash.
    Mirror of ``daydream/benchmark/daydream_run.py:85-102`` ``_review_complete``.
    """
    session_dir = await _session_dir(runtime, archive_root)
    if session_dir is None:
        return False
    try:
        raw = await runtime.read(f"{session_dir}/trajectory.json")
        trajectory = json.loads(raw)
    except Exception:
        return False
    return isinstance(trajectory, dict) and bool(trajectory.get("final_metrics"))
