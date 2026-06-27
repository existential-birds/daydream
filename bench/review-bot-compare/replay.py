#!/usr/bin/env python3
"""Replay daydream against the exact snapshot a review bot saw.

For each harvested PR record, reconstruct the bot's view of the diff:
  head = the bot review's commit_id
  base = merge-base(origin/<base_ref>, head)   # == GitHub's 3-dot compare base
checked out in a detached worktree, then run daydream in review-only mode and
collect its findings artifact + trajectory cost.

daydream is run in IN-PLACE mode (target = the detached worktree, no --branch,
no --worktree) so it diffs `--base ... HEAD` exactly over the bot's snapshot.

Usage:
  python replay.py --repo owner/repo --source /path/to/local/clone \
      --in ./out --backend codex [--shallow] [--limit 5] [--pr 123]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from common import read_json, repo_slug, run, write_json

from daydream.backends.pi import STREAM_DROP_SIGNATURES

# daydream resolves a GitHub App identity (and hard-aborts on failure) when these
# are set and the run is "posting" (--review counts). The benchmark reviews a
# detached local worktree with no posting, so run it under plain user identity.
_APP_ENV_VARS = ("DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY")


def _local_identity_env() -> dict[str, str]:
    env = dict(os.environ)
    for var in _APP_ENV_VARS:
        env.pop(var, None)
    return env


# Backend overload / rate-limit signatures worth a retry (vs. a real review
# failure). Matched case-insensitively against daydream's captured stdout.
_TRANSIENT_SIGNATURES = (
    "429",
    "service may be temporarily overloaded",
    "temporarily overloaded",
    "rate limit",
    "rate_limit",
    "overloaded_error",
    "try again later",
)

_ERROR_CONTEXT_MARKERS = (
    "pierror",
    "backend execution error",
    "fatal error",
)


def _is_transient(stdout: str) -> bool:
    low = (stdout or "").lower()
    if any(sig in low for sig in _TRANSIENT_SIGNATURES):
        return True
    # Stream-drop signatures only count when daydream actually errored out.
    if any(marker in low for marker in _ERROR_CONTEXT_MARKERS):
        return any(sig in low for sig in STREAM_DROP_SIGNATURES)
    return False


def _findings_complete(findings_path: Path, traj_path: Path) -> bool:
    """True when a prior attempt already produced a *complete* review on disk.

    daydream writes the findings artifact and the trajectory's `final_metrics`
    only at the very end of the review pipeline. A z.ai/GLM stream-drop at the
    closing stream (`PiError: terminated`) exits the subprocess non-zero but
    leaves both files fully written — the review is done, only the socket died.
    Re-running it wastes ~9 min and risks dropping again, so we treat this as a
    success and never retry it.
    """
    try:
        art = read_json(findings_path)
        if not isinstance(art.get("findings"), list):
            return False
        traj = read_json(traj_path)
    except (ValueError, OSError):
        return False
    return bool(traj.get("final_metrics"))


def _collect(findings_path: Path, traj_path: Path) -> dict:
    """Read finding count + cost/token metrics off the on-disk artifacts."""
    out: dict = {}
    if findings_path.exists():
        try:
            art = read_json(findings_path)
            out["n_findings"] = len(art.get("findings", []))
            out["findings_path"] = str(findings_path)
        except (ValueError, OSError):
            out["n_findings"] = None
    else:
        out["n_findings"] = 0
    if traj_path.exists():
        try:
            fm = read_json(traj_path).get("final_metrics") or {}
            out["cost_usd"] = fm.get("total_cost_usd")
            out["prompt_tokens"] = fm.get("total_prompt_tokens")
            out["completion_tokens"] = fm.get("total_completion_tokens")
            out["cached_tokens"] = fm.get("total_cached_tokens")
        except (ValueError, OSError):
            pass
    return out


def git(source: Path, args: list[str], *, check: bool = True, timeout: int | None = None):
    return run(["git", "-C", str(source), *args], check=check, timeout=timeout)


def ensure_commit(source: Path, pr_number: int, sha: str, base_ref: str) -> None:
    """Make `sha` and `origin/<base_ref>` reachable in the local clone."""
    # PR head ref makes every commit in the PR history (incl. older review SHAs)
    # reachable even when the source branch was deleted after merge.
    git(source, ["fetch", "origin", f"pull/{pr_number}/head"], check=False, timeout=300)
    git(source, ["fetch", "origin", base_ref], check=False, timeout=300)
    if git(source, ["cat-file", "-e", f"{sha}^{{commit}}"], check=False).returncode != 0:
        # Last resort: try fetching the bare SHA directly.
        git(source, ["fetch", "origin", sha], check=False, timeout=300)


def resolve_base(source: Path, base_ref: str, head: str) -> str | None:
    mb = git(source, ["merge-base", f"origin/{base_ref}", head], check=False)
    if mb.returncode == 0 and mb.stdout.strip():
        return mb.stdout.strip()
    # Fallback: first parent of head (two-dot diff). Less faithful but non-empty.
    parent = git(source, ["rev-parse", f"{head}^"], check=False)
    return parent.stdout.strip() if parent.returncode == 0 else None


def replay_one(
    record: dict, source: Path, out_dir: Path, backend: str, shallow: bool, timeout: int,
    retries: int = 2, backoff: int = 30,
) -> dict:
    n = record["pr_number"]
    head = record["review_commit_id"]
    base_ref = record["base_ref"] or "main"
    result: dict = {"pr_number": n, "head": head, "base_ref": base_ref}

    if not head:
        result["status"] = "skipped-no-commit"
        return result

    ensure_commit(source, n, head, base_ref)
    if git(source, ["cat-file", "-e", f"{head}^{{commit}}"], check=False).returncode != 0:
        result["status"] = "skipped-unreachable-head"
        return result

    base = resolve_base(source, base_ref, head)
    if not base:
        result["status"] = "skipped-no-base"
        return result
    result["base"] = base

    wt = (out_dir / "worktrees" / f"pr-{n}").resolve()
    findings_path = (out_dir / "findings" / f"pr-{n}.json").resolve()
    traj_path = (out_dir / "traj" / f"pr-{n}.json").resolve()
    log_path = (out_dir / "logs" / f"pr-{n}.log").resolve()
    for p in (findings_path, traj_path, log_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Resume: a prior attempt already produced a complete review (a tail-end
    # stream-drop leaves findings + trajectory fully written). Never re-run it.
    if _findings_complete(findings_path, traj_path):
        result.update(_collect(findings_path, traj_path))
        result["attempts"] = 0
        result["wall_seconds"] = 0.0
        result["reused_existing"] = True
        result["status"] = "ok"
        return result

    # Fresh worktree at the bot's snapshot.
    git(source, ["worktree", "remove", "--force", str(wt)], check=False)
    git(source, ["worktree", "add", "--detach", str(wt), head], timeout=300)

    try:
        cmd = [
            "daydream", "--review", "--non-interactive",
            "--backend", backend,
            "--pr-number", str(n),
            "--base", base,
            "--findings-out", str(findings_path),
            "--trajectory", str(traj_path),
        ]
        if shallow:
            cmd.append("--shallow")
        cmd.append(str(wt))

        # Retry transient backend overload/rate-limit (e.g. z.ai GLM "429 service
        # temporarily overloaded") with exponential backoff. A fresh worktree is
        # already in place; daydream re-runs cleanly over it.
        start = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            proc = run(cmd, cwd=wt, check=False, timeout=timeout, env=_local_identity_env())
            if proc.returncode == 0 or attempts > retries or not _is_transient(proc.stdout):
                break
            # The review may have completed and only the closing stream dropped —
            # findings + trajectory are on disk. That's a success; don't re-run it.
            if _findings_complete(findings_path, traj_path):
                break
            wait = backoff * (2 ** (attempts - 1))
            print(f"  transient backend error (attempt {attempts}/{retries+1}); "
                  f"retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
        elapsed = time.monotonic() - start
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\n=== STDOUT ===\n{proc.stdout}\n\n=== STDERR ===\n{proc.stderr}"
        )

        result["exit_code"] = proc.returncode
        result["attempts"] = attempts
        result["wall_seconds"] = round(elapsed, 1)
        if proc.returncode != 0 and _is_transient(proc.stdout):
            result["transient_error"] = True

        result.update(_collect(findings_path, traj_path))

        # A non-zero exit whose review nonetheless completed (stream dropped at the
        # tail) is a success: findings + metrics are intact on disk.
        if proc.returncode == 0:
            result["status"] = "ok"
        elif _findings_complete(findings_path, traj_path):
            result["status"] = "ok"
            result["stream_dropped_at_tail"] = True
        else:
            result["status"] = f"daydream-exit-{proc.returncode}"
    finally:
        # Always reclaim the detached worktree, even if daydream timed out or
        # raised — run(..., timeout=) raises TimeoutExpired through here, and
        # leaked worktrees would otherwise accumulate across a long batch.
        git(source, ["worktree", "remove", "--force", str(wt)], check=False)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="owner/repo")
    ap.add_argument("--source", required=True, type=Path, help="local clone with the GitHub remote")
    ap.add_argument("--in", dest="in_root", default="./out", type=Path, help="harvest output root")
    ap.add_argument("--backend", default="codex", choices=["claude", "codex", "pi"])
    ap.add_argument("--shallow", action="store_true", help="single-stack review (faster pilot)")
    ap.add_argument("--limit", type=int, default=0, help="max PRs to replay (0 = all)")
    ap.add_argument("--pr", type=int, action="append", help="replay only these PR numbers (repeatable)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-PR daydream timeout (seconds)")
    ap.add_argument("--retries", type=int, default=2, help="retries on transient backend 429/overload")
    ap.add_argument("--backoff", type=int, default=30, help="base backoff seconds (doubles per retry)")
    ap.add_argument("--retry-failed", action="store_true",
                    help="only re-run PRs whose existing replay record is not 'ok'")
    args = ap.parse_args()

    base_dir = args.in_root / repo_slug(args.repo)
    index = read_json(base_dir / "index.json")
    prs = index["prs"]

    if args.pr:
        wanted = set(args.pr)
        prs = [p for p in prs if p["pr_number"] in wanted]
    if args.retry_failed:
        failed = set()
        for f in sorted((base_dir / "replay").glob("pr-*.json")):
            rec = read_json(f)
            if rec.get("status") != "ok":
                failed.add(rec["pr_number"])
        prs = [p for p in prs if p["pr_number"] in failed]
        print(f"Retrying {len(prs)} previously-failed PRs: "
              f"{sorted(p['pr_number'] for p in prs)}", file=sys.stderr)
    # Only PRs with a usable snapshot + at least one inline comment.
    prs = [p for p in prs if p["review_commit_id"] and p["n_inline_comments"] > 0]
    if args.limit:
        prs = prs[: args.limit]

    if not args.source.exists():
        print(f"--source {args.source} does not exist", file=sys.stderr)
        return 2

    print(f"Replaying {len(prs)} PRs with backend={args.backend} shallow={args.shallow}",
          file=sys.stderr)

    results = []
    for p in prs:
        n = p["pr_number"]
        record = read_json(base_dir / f"pr-{n}.json")
        print(f"\n=== PR #{n} :: {record['title']!r} ===", file=sys.stderr)
        try:
            res = replay_one(record, args.source, base_dir, args.backend, args.shallow,
                             args.timeout, retries=args.retries, backoff=args.backoff)
        except Exception as e:  # noqa: BLE001 - isolate one PR's failure from the batch
            res = {"pr_number": n, "status": f"error: {e}"}
        results.append(res)
        write_json(base_dir / "replay" / f"pr-{n}.json", res)
        print(
            f"  -> {res.get('status')} | findings={res.get('n_findings')} "
            f"cost={res.get('cost_usd')} tok={res.get('prompt_tokens')}/"
            f"{res.get('completion_tokens')} {res.get('wall_seconds')}s "
            f"(attempts={res.get('attempts')})",
            file=sys.stderr,
        )

    # Rebuild summary from ALL per-PR records on disk so partial/retry runs
    # accumulate into one complete summary for compare.py.
    all_recs = [read_json(f) for f in sorted((base_dir / "replay").glob("pr-*.json"))]
    write_json(base_dir / "replay" / "summary.json", {"repo": args.repo, "results": all_recs})
    print(f"\nReplay complete -> {base_dir / 'replay'} "
          f"({len(all_recs)} PRs in summary)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
