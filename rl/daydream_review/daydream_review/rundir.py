"""Fetch reward inputs from the live rollout runtime and replay them through the host scorer. The
supervisor seals the archive and candidate diff after the agent write window; verify the staged copy
before trusting it, with tampering yielding zero reward.

Copy only the small scorer inputs, excluding the megabyte-scale per-fork trajectories and diffs.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import verifiers.v1 as vf

from daydream_review.verifier import SealResult, seal_bytes, verify

#: Optional fixed reward inputs: review-only runs omit verdicts and green runs omit fix-failures.
#: Collect dynamic deep/stack-*-records.json separately. seal.json travels with the inputs for
#: verification.  Exclude trajectory.json and trajectories/*.json: golden-run trajectories contain
#: untrusted model-directed text that must never enter model context through the collector.
#: test_fetch_run_dir_excludes_fixture_trajectories pins this allowlist boundary.
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

#: Disable repository-controlled diff.external helpers (including trustExitCode) and .gitattributes
#: textconv drivers. Supervisors derive diffs as the trusted host/root identity; untrusted
#: repository configuration must not execute there. All candidate derivations share these hardening
#: flags.
GIT_DIFF_HARDENING_FLAGS: tuple[str, str] = ("--no-ext-diff", "--no-textconv")

#: Pass as a bare argv element, never shell-interpolate. Exclude tracked .daydream artifacts from
#: the product diff, matching both _fixes_applied probes so sealing and fix acceptance agree.
DAYDREAM_EXCLUDE = ":(exclude).daydream"


def candidate_diff_cmd(repo: str, head_sha: str) -> list[str]:
    """Derive the candidate identically for sealing, verification, and verifier-checkout construction.
    The one-revision git diff against head_sha includes committed, staged, and unstaged tracked
    changes, excluding untracked files; this matches _fixes_applied.
    """
    return [
        "git", "-C", repo, "diff",
        *GIT_DIFF_HARDENING_FLAGS,
        head_sha,
        "--",
        DAYDREAM_EXCLUDE,
    ]


def candidate_quiet_diff_cmd(
    repo: str,
    head_sha: str,
    pathspecs: list[str],
    *,
    include_head: bool = False,
) -> list[str]:
    """Quiet companion to candidate_diff_cmd for oracle probes. include_head selects head_sha HEAD for
    _fixes_applied; otherwise compare head_sha with the current tracked tree so
    _protected_test_paths_unchanged catches uncommitted tampering.
    """
    cmd = ["git", "-C", repo, "diff", *GIT_DIFF_HARDENING_FLAGS, "--quiet", head_sha]
    if include_head:
        cmd.append("HEAD")
    return [*cmd, "--", *pathspecs]


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
    """Copy archived reward inputs into caller-owned dest; archive_root is the harness's
    DAYDREAM_ARCHIVE_DIR. Use a TemporaryDirectory across scoring to avoid rollout leaks. Return
    dest if any member copied, else None: a pre-archive crash scores zero instead of aborting the
    rollout.
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
    """Verify fetch_run_dir's staged seal and artifacts against the candidate diff re-derived from the
    live sandbox at scoring time.

    Return True for a matching seal; False for missing expected seals, malformed/mismatched seals,
    or failed diff derivation. Never hash a failed git read as an empty diff. Return None only when
    no seal exists and none was expected (legacy tests; completed production runs are sealed). Never
    raise: unverifiable state must zero reward, not crash scoring.
    """
    seal_path = run_dir / "seal.json"
    if not seal_path.is_file():
        return False if seal_expected else None
    try:
        seal = SealResult.model_validate_json(seal_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # Hash present fixed inputs and stack records used by the intrinsic format gate; exclude
    # seal.json itself.
    present = [
        run_dir / rel for rel in RUN_DIR_FILES if rel != "seal.json" and (run_dir / rel).is_file()
    ]
    present += sorted(run_dir.glob("deep/stack-*-records.json"))
    # Re-derive the diff: the embedded seal copy is audit-only, and later tracked changes must fail
    # verification.
    try:
        diff_result = await runtime.run(candidate_diff_cmd(repo, head_sha), {})
    except Exception:
        return False
    if diff_result.exit_code != 0:
        # Fail closed: hashing b"" after git fails both here and during sealing could falsely verify
        # a diff that was never derived.
        return False
    return verify(seal, present, candidate_diff=diff_result.stdout.encode())


async def seal_archived_run(
    runtime: vf.Runtime,
    archive_root: str = DEFAULT_ARCHIVE_ROOT,
    *,
    repo: str,
    head_sha: str,
) -> bool:
    """Seal the archive and current tracked diff against the baked head after the agent write window.
    The candidate is b"" when the runner cannot derive it.

    Return True after writing seal.json and, under Docker, hardening the archive
    root-owned/read-only; return False for no archive or any failure. Never raise. If an existing
    archive cannot be sealed, write an invalid seal marker so scoring yields seal_verified=0 and
    zero reward instead of legacy full trust.
    """
    session_dir = await _session_dir(runtime, archive_root)
    if session_dir is None:
        return False
    try:
        artifacts: dict[str, bytes] = {}
        for rel in await _present_files(runtime, session_dir):
            if rel == "seal.json":
                continue
            artifacts[rel] = await runtime.read(f"{session_dir}/{rel}")
        diff_result = await runtime.run(candidate_diff_cmd(repo, head_sha), {})
        candidate_diff = diff_result.stdout.encode() if diff_result.exit_code == 0 else b""
        seal = seal_bytes(artifacts, candidate_diff)
        await runtime.write(f"{session_dir}/seal.json", seal.model_dump_json().encode())
        # Docker exec runs as root: harden the archive root-owned/read-only after the agent write
        # window. Local subprocess smoke runs share the host UID and have no root boundary. Failed
        # hardening is failed sealing; agent-writable sealed bytes cannot be trusted.
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
        # Mark failed sealing explicitly so verification returns False and zero reward. If even this
        # write fails, the harness records that failure.
        try:
            await runtime.write(f"{session_dir}/seal.json", b'{"seal_failed": true}')
        except Exception:
            pass
        return False


async def daydream_completed(
    runtime: vf.Runtime,
    archive_root: str = DEFAULT_ARCHIVE_ROOT,
) -> bool:
    """Detect pipeline completion from trajectory final_metrics, which is written only at the end. A
    nonzero outcome such as tests remaining red is complete; absence indicates a crash. Matches the
    benchmark _review_complete convention.
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
