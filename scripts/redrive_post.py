#!/usr/bin/env python3
"""Re-drive PR comment posting from the canonical deep merged-items file.

Usage:
    uv run python scripts/redrive_post.py /path/to/target/repo --pr N [--yes]

The sole input is `.daydream/deep/merged-items.json` — the canonical,
schema-validated finding list produced by the cross-stack merge. Conversion,
classification, confirmation, and submission are delegated to
`daydream.pr_review.post_review_to_pr_from_report`, so a redrive can never
reintroduce deduped findings or drop structural ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from daydream.deep.artifacts import merged_items_path
from daydream.pr_review import PostStatus, post_review_to_pr_from_report
from daydream.ui import create_console


async def _run(target_dir: Path, pr_number: int, auto_yes: bool = False) -> None:
    deep_dir = target_dir / ".daydream" / "deep"
    items_path = merged_items_path(deep_dir)

    if not items_path.is_file():
        print(f"No canonical merged-items.json found at {items_path}", file=sys.stderr)
        sys.exit(1)

    # A present-but-corrupt/partially-written merged-items.json is a real error,
    # not a "nothing to post": validate it here so it exits 1 (distinguishing
    # corruption from a genuinely-empty finding list) instead of silently
    # succeeding. The shared poster swallows JSONDecodeError into
    # NOTHING_TO_POST for the main deep flow, which would otherwise hide input
    # corruption behind a 0 exit code here (#400, round 2).
    try:
        json.loads(items_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read merged-items.json at {items_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        status = await post_review_to_pr_from_report(
            target_dir,
            items_path,
            console=create_console(),
            post=auto_yes,
            pr_number=pr_number,
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read merged-items.json at {items_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    if status in (PostStatus.NO_PR, PostStatus.FAILED):
        sys.exit(1)


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
