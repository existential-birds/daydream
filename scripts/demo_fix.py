#!/usr/bin/env python3
"""Demo script for the FIX phase only.

Applies fixes from parsed feedback with Haiku.
Requires a prior review run (demo_review.py).

Usage:
    python scripts/demo_fix.py [DIRECTORY] [--skip-setup]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from _demo_common import create_test_repo, DEFAULT_REPO_PATH, REVIEW_OUTPUT_FILE


def main():
    parser = argparse.ArgumentParser(description="Run fix phase demo with Haiku")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_REPO_PATH,
        help=f"Test repo location (default: {DEFAULT_REPO_PATH})",
    )
    parser.add_argument("--skip-setup", action="store_true", help="Skip repo creation, use existing review")
    args = parser.parse_args()

    repo_path = args.directory.resolve()

    if not args.skip_setup:
        target = create_test_repo(repo_path)
        if target is None:
            return 1
        print("\nNote: Running review first to generate .review-output.md...")
        subprocess.run(
            [
                sys.executable, "-m", "daydream",
                str(target),
                "--python",
                "--model", "haiku",
                "--review-only",
                "--no-cleanup",
            ],
            cwd=Path(__file__).parent.parent,
        )
    else:
        if not repo_path.exists():
            print(f"Error: {repo_path} does not exist")
            return 1
        review_file = repo_path / REVIEW_OUTPUT_FILE
        if not review_file.exists():
            print(f"Error: {review_file} does not exist")
            print("Run demo_review.py first or remove --skip-setup")
            return 1
        target = repo_path

    print(f"\nRunning FIX phase (parse + fix, skip test) on: {target}")
    print("-" * 60)

    # Start at fix phase - this will parse and apply fixes but skip to test phase
    # We use a custom approach: start at parse, but don't use --review-only
    result = subprocess.run(
        [
            sys.executable, "-m", "daydream",
            str(target),
            "--python",
            "--model", "haiku",
            "--start-at", "fix",
            "--no-cleanup",
        ],
        cwd=Path(__file__).parent.parent,
    )

    print(f"\nTest repo preserved at: {repo_path}")
    print("Run test phase with: python scripts/demo_test.py --skip-setup")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
