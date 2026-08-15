"""Pull daydream's archived run directory out of a live rollout sandbox.

Scoring runs while the runtime is still up (a ``@vf.reward`` with a required
``runtime`` parameter — verifiers 0.2.1 ``task.py:269-277``), so the reward reads
daydream's own artifacts straight off the sandbox filesystem and replays them
through the training pipeline's scorer on the host. The supervisor seals the
archived run dir (with the candidate diff) after the agent's write window, and
the reward verifies that seal against the staged copy before trusting any
value — a tampered archive must zero the reward, never record honest telemetry.

Only the handful of small files the scorer actually reads are copied. The full
archive bundle carries per-fork trajectories and diffs that can run to megabytes;
none of it feeds the reward.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import verifiers.v1 as vf

from daydream_review_v1.verifier import SealResult, seal_bytes, verify

#: Fixed members of the archived run dir the reward path reads. Every one is
#: optional: a review-only rollout has no verdicts, a green run has no
#: fix-failures. ``deep/stack-*-records.json`` is collected separately by glob —
#: its names depend on which stacks the router detected.
#:
#: ``seal.json`` is the supervisor-produced integrity seal over the other
#: members plus the candidate diff; it rides in the fetch so the reward can
#: verify the staged copy against it.
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
    "seal.json",
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


async def verify_seal(
    run_dir: Path,
    runtime: vf.Runtime,
    repo: str,
    head_sha: str,
    *,
    seal_expected: bool = False,
) -> bool | None:
    """Verify the staged run dir's supervisor-produced seal.

    Args:
        run_dir: The staged host copy of the archived run dir (``fetch_run_dir``
            output), including ``seal.json`` when the supervisor sealed the run.
        runtime: The live rollout runtime. The candidate diff is re-derived
            from the sandbox through it at scoring time, so the seal binds the
            diff the verifier checkout will actually apply.
        repo: The repository under review inside the sandbox.
        head_sha: The baked head SHA the rollout diffed against.
        seal_expected: Whether the harness claims to have sealed the run. A run
            the harness sealed whose ``seal.json`` is missing at scoring time
            is a vanished seal — a tamper, never a legacy unsealed run — so it
            must fail closed rather than score at full trust.

    Returns:
        ``True`` when the seal verifies against the staged members and the diff
        re-derived from the sandbox; ``False`` when a seal exists but is
        missing, malformed, or mismatched (a tamper must zero the reward, not
        crash scoring) — including a vanished seal on a run the harness claims
        to have sealed (*seal_expected*), and a diff that cannot be re-derived
        (a git failure must fail closed, never hash as the empty diff); ``None``
        when no seal was produced and none was expected (legacy/unsealed runs
        keep their pre-seal scoring — the harness seals every completed
        production run, so this is the test-only path). Never raises.
    """
    seal_path = run_dir / "seal.json"
    if not seal_path.is_file():
        # A missing seal is the legacy path only when none was expected. The
        # harness claimed to seal this run, so a vanished seal.json is an
        # internal contradiction that must read as a tamper, never as an
        # unsealed run at full trust.
        return False if seal_expected else None
    try:
        seal = SealResult.model_validate_json(seal_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # The seal covers the RUN_DIR_FILES members that exist plus the
    # deep/stack-*-records.json glob members — both are reward inputs (the
    # stack records feed the intrinsic scorer's format gate). seal.json itself
    # is the seal record and is never hashed into itself.
    present = [
        run_dir / rel for rel in RUN_DIR_FILES if rel != "seal.json" and (run_dir / rel).is_file()
    ]
    present += sorted(run_dir.glob("deep/stack-*-records.json"))
    # Re-derive the candidate diff from the sandbox exactly as the seal
    # producer did (and as the verifier checkout will apply it): the seal's
    # embedded copy is an audit record, never the verification input, so a
    # committed diff rewritten after sealing fails the digest check.
    try:
        diff_result = await runtime.run(["git", "-C", repo, "diff", head_sha, "HEAD"], {})
    except Exception:
        return False
    if diff_result.exit_code != 0:
        # The diff cannot be re-derived, so there is nothing to verify against.
        # Hashing b"" here would let a git failure at BOTH seal and scoring
        # time pass as a matching empty diff (seal_verified 1.0 on a run whose
        # diff was never re-derived); a failed re-derivation must fail closed
        # like any other unverifiable seal.
        return False
    return verify(seal, present, candidate_diff=diff_result.stdout.encode())


async def seal_archived_run(
    runtime: vf.Runtime,
    archive_root: str = DEFAULT_ARCHIVE_ROOT,
    *,
    repo: str,
    head_sha: str,
) -> bool:
    """Seal the archived run dir + the candidate diff; write ``seal.json`` into the sandbox.

    The supervisor (harness) runs this after the launch returns, so the seal is
    produced outside the agent's write window and the reward can verify the
    staged copy against it. The candidate diff is the rollout's own committed
    diff against the baked head (``b""`` when the runner cannot produce one).

    Returns:
        ``True`` when a seal was written (and, under docker, the run dir
        hardened root-owned read-only); ``False`` when there is no run dir to
        seal or sealing failed. Never raises: a missing seal must not crash the
        rollout. A sealing failure on a real run dir is fail-closed: the dir is
        marked with an unvalidatable ``seal.json`` so scoring reads a failed
        seal (``seal_verified`` 0.0, zero reward) rather than an unsealed run
        at full trust.
    """
    session_dir = await _session_dir(runtime, archive_root)
    if session_dir is None:
        return False
    try:
        artifacts: dict[str, bytes] = {}
        for rel in await _present_files(runtime, session_dir):
            # The seal covers every reward input: the RUN_DIR_FILES members the
            # reward reads AND the deep/stack-*-records.json glob members, which
            # feed the intrinsic scorer's format gate. seal.json is the seal
            # record itself.
            if rel == "seal.json":
                continue
            if rel not in RUN_DIR_FILES and not (
                rel.startswith("deep/stack-") and rel.endswith("-records.json")
            ):
                continue
            artifacts[rel] = await runtime.read(f"{session_dir}/{rel}")
        diff_result = await runtime.run(["git", "-C", repo, "diff", head_sha, "HEAD"], {})
        candidate_diff = diff_result.stdout.encode() if diff_result.exit_code == 0 else b""
        seal = seal_bytes(artifacts, candidate_diff)
        await runtime.write(f"{session_dir}/seal.json", seal.model_dump_json().encode())
        # base.Dockerfile documents that the supervisor re-chowns the sealed run
        # dir root-owned read-only at seal time — the mechanism that makes the
        # sealed artifacts agent-inaccessible once the agent's write window has
        # closed. The docker runtime execs as the container root, so the chown
        # lands there; the local subprocess path has no root boundary (it runs
        # as the host user, sharing the agent's uid), so there is nothing to
        # re-chown on that path. A failed hardening is a sealing failure: a
        # seal over still-agent-writable bytes would not be worth trusting.
        if runtime.type == "docker":
            hardened = await runtime.run(
                [
                    "sh",
                    "-c",
                    f"chown -R root:root {shlex.quote(session_dir)} "
                    f"&& chmod -R a-w {shlex.quote(session_dir)}",
                ],
                {},
            )
            if hardened.exit_code != 0:
                raise RuntimeError(
                    "could not re-chown the sealed run dir root-owned read-only: "
                    f"{hardened.stderr.strip() or 'chown/chmod failed'}"
                )
        return True
    except Exception:
        # Fail closed: a run whose seal could not be produced must score as a
        # failed seal (verify_seal -> False -> zero reward), never as an
        # unsealed full-trust run. Overwrite seal.json with an unvalidatable
        # marker; if even that write fails, the harness records the failure.
        try:
            await runtime.write(f"{session_dir}/seal.json", b'{"seal_failed": true}')
        except Exception:
            pass
        return False


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
