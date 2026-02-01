#!/usr/bin/env python3
"""Demo script for the REVIEW phase only.

Creates a buggy repo and runs only the review phase with Haiku.

Usage:
    python scripts/demo_review.py [DIRECTORY]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from _demo_common import DEFAULT_REPO_PATH, create_test_repo


def main():
    parser = argparse.ArgumentParser(description="Run review phase demo with Haiku")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_REPO_PATH,
        help=f"Where to create the test repo (default: {DEFAULT_REPO_PATH})",
    )
    parser.add_argument("--skip-setup", action="store_true", help="Skip repo creation, use existing")
    args = parser.parse_args()

    repo_path = args.directory.resolve()

    if not args.skip_setup:
        target = create_test_repo(repo_path)
        if target is None:
            return 1
    else:
        if not repo_path.exists():
            print(f"Error: {repo_path} does not exist")
            return 1
        target = repo_path

    print(f"\nRunning REVIEW phase on: {target}")
    print("-" * 60)

    result = subprocess.run(
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

    print(f"\nTest repo preserved at: {repo_path}")
    print("Run next phase with: python scripts/demo_parse.py --skip-setup")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
