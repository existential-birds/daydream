#!/usr/bin/env python3
"""Re-drive PR comment posting from existing deep review artifacts.

Usage:
    uv run python scripts/redrive_post.py /path/to/target/repo --pr 58

Reads .daydream/deep/{alternatives.json, stack-*-records.json, dedup-candidates.json}
and synthesizes .review-output.md, then calls the standard PR posting flow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _load_artifacts(deep_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load alternatives, per-stack records, and dedup candidates."""
    alts = json.loads((deep_dir / "alternatives.json").read_text())
    dedup = json.loads((deep_dir / "dedup-candidates.json").read_text())

    records: list[dict] = []
    for p in sorted(deep_dir.glob("stack-*-records.json")):
        records.extend(json.loads(p.read_text()))

    return alts, records, dedup


def _record_key(record: dict) -> tuple[str, int]:
    """Composite key for a per-stack record (file + id)."""
    return str(record["file"]), int(record["id"])


def _find_overlaps(
    alts: list[dict], records: list[dict]
) -> tuple[dict[int, list[dict]], set[tuple[str, int]]]:
    """Match alternatives to generic records by shared file paths."""
    by_file: dict[str, list[dict]] = {}
    for r in records:
        by_file.setdefault(r["file"], []).append(r)

    alt_matches: dict[int, list[dict]] = {}
    consumed: set[tuple[str, int]] = set()
    for alt in alts:
        matches: list[dict] = []
        for f in alt.get("files", []):
            for r in by_file.get(f, []):
                key = _record_key(r)
                if key not in consumed:
                    matches.append(r)
                    consumed.add(key)
        if matches:
            alt_matches[alt["id"]] = matches

    return alt_matches, consumed


def _fmt_body(
    *,
    severity: str = "",
    confidence: str = "",
    description: str = "",
    rationale: str = "",
    recommendation: str = "",
    extra: str = "",
) -> str:
    parts: list[str] = []
    if severity:
        parts.append(f"   **Severity:** {severity}")
    if confidence:
        parts.append(f"   **Confidence:** {confidence}")
    if description:
        parts.append(f"   {description}")
    if rationale:
        parts.append(f"   **Rationale:** {rationale}")
    if recommendation:
        parts.append(f"   **Recommendation:** {recommendation}")
    if extra:
        parts.append(f"   {extra}")
    return "\n\n".join(parts)


def build_report(
    alts: list[dict], records: list[dict], dedup: list[dict]
) -> str:
    """Merge alternatives + per-stack records into .review-output.md."""
    alt_matches, consumed_ids = _find_overlaps(alts, records)

    lines: list[str] = ["# Review\n"]
    lines.append("## Issues\n")

    num = 0

    for alt in alts:
        files = alt.get("files", [])
        title = alt.get("title", alt.get("description", ""))
        matches = alt_matches.get(alt["id"], [])

        file_lines: dict[str, int | None] = {f: None for f in files}
        for m in matches:
            if m["file"] in file_lines and m.get("line"):
                file_lines[m["file"]] = m["line"]

        extra = ""
        if matches:
            evidence = "; ".join(m["description"] for m in matches)
            extra = f"**Per-stack evidence:** {evidence}"

        for f in files:
            num += 1
            line = file_lines.get(f)
            loc = f"{f}:{line}" if line else f
            body = _fmt_body(
                severity=alt.get("severity", ""),
                confidence=alt.get("confidence", ""),
                description=alt.get("description", ""),
                rationale=alt.get("rationale", ""),
                recommendation=alt.get("recommendation", ""),
                extra=extra,
            )
            lines.append(f"{num}. [{loc}] {title}\n{body}\n")

    for r in records:
        if _record_key(r) in consumed_ids:
            continue
        num += 1
        loc = f"{r['file']}:{r['line']}" if r.get("line") else r["file"]
        body = _fmt_body(
            confidence=r.get("confidence", ""),
            description=r["description"],
            rationale=r.get("rationale", ""),
        )
        lines.append(f"{num}. [{loc}] {r['description']}\n{body}\n")

    return "\n".join(lines)


def _lookup_pr(target_dir: Path, pr_number: int) -> dict | None:
    """Fetch PR details via gh CLI."""
    try:
        r = subprocess.run(  # noqa: S603, S607 -- args are hardcoded
            [
                "gh", "pr", "view", str(pr_number),
                "--json", "number,headRefOid,baseRefOid,baseRefName,url",
            ],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def _repo_owner_name(target_dir: Path) -> tuple[str | None, str | None]:
    try:
        r = subprocess.run(  # noqa: S603, S607
            ["gh", "repo", "view", "--json", "owner,name"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
    except (subprocess.SubprocessError, OSError):
        return None, None
    if r.returncode != 0:
        return None, None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None, None
    return data.get("owner", {}).get("login"), data.get("name")


async def _run(target_dir: Path, pr_number: int, auto_yes: bool = False) -> None:
    from daydream.pr_review import (
        PRInfo,
        build_payload,
        classify,
        parse_report,
    )

    # --- Build report from artifacts ---
    deep_dir = target_dir / ".daydream" / "deep"
    if not deep_dir.is_dir():
        print(f"No .daydream/deep/ directory found at {target_dir}", file=sys.stderr)
        sys.exit(1)

    alts, records, dedup = _load_artifacts(deep_dir)
    report = build_report(alts, records, dedup)

    output_path = target_dir / ".review-output.md"
    output_path.write_text(report)
    print(f"Wrote merged report ({len(report)} bytes) to {output_path}")

    # --- Parse the report ---
    issues = parse_report(report)
    if not issues:
        print("No parseable issues in report", file=sys.stderr)
        sys.exit(1)
    print(f"Parsed {len(issues)} issues from report")

    # --- Resolve PR info ---
    pr_data = _lookup_pr(target_dir, pr_number)
    if pr_data is None:
        print(f"Could not find PR #{pr_number}", file=sys.stderr)
        sys.exit(1)

    owner, repo = _repo_owner_name(target_dir)
    if owner is None or repo is None:
        print("Could not determine repo owner/name", file=sys.stderr)
        sys.exit(1)

    pr = PRInfo(
        number=pr_data["number"],
        head_sha=pr_data["headRefOid"],
        base_sha=pr_data["baseRefOid"],
        base_ref=pr_data.get("baseRefName", ""),
        owner=owner,
        repo=repo,
        url=pr_data.get("url", ""),
    )
    print(f"PR #{pr.number} ({pr.url})")
    print(f"  head: {pr.head_sha[:12]}  base: {pr.base_sha[:12]}")

    # --- Classify issues (inline vs body-only) ---
    classified = classify(target_dir, pr, issues)
    inline_files = sorted({c["path"] for c in classified.inline})
    print(
        f"\n{len(classified.inline)} inline on "
        f"{', '.join(inline_files) if inline_files else '(none)'}, "
        f"{len(classified.body_only)} folded into body"
    )

    if not classified.inline and not classified.body_only:
        print("No postable issues after classification")
        return

    # --- Confirm and post ---
    if not auto_yes:
        answer = input("\nPost these as a PR review? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Skipped.")
            return

    payload = build_payload(pr, classified)
    with tempfile.NamedTemporaryFile(
        mode="w",
        prefix=f"pr-{pr.number}-review-",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tf:
        tf.write(json.dumps(payload, indent=2))
        payload_path = Path(tf.name)

    print(f"Payload written to {payload_path}")

    try:
        r = subprocess.run(  # noqa: S603, S607
            [
                "gh", "api",
                "--method", "POST",
                f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/reviews",
                "--input", str(payload_path),
            ],
            cwd=target_dir,
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"Failed to post: {exc}", file=sys.stderr)
        print(f"Payload kept at {payload_path}")
        sys.exit(1)

    if r.returncode != 0:
        print(f"gh api failed (rc={r.returncode}): {r.stderr}", file=sys.stderr)
        print(f"Payload kept at {payload_path}")
        sys.exit(1)

    try:
        data = json.loads(r.stdout)
        url = data.get("html_url", "(no URL returned)")
    except json.JSONDecodeError:
        url = "(response not JSON)"

    payload_path.unlink(missing_ok=True)
    print(f"\nPosted review: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-drive deep review PR posting")
    parser.add_argument("target_dir", type=Path, help="Path to the target repo")
    parser.add_argument("--pr", type=int, required=True, help="PR number to post to")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    target_dir = args.target_dir.resolve()
    if not target_dir.is_dir():
        print(f"Not a directory: {target_dir}", file=sys.stderr)
        sys.exit(1)

    import anyio

    anyio.run(_run, target_dir, args.pr, args.yes)


if __name__ == "__main__":
    main()
