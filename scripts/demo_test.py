#!/usr/bin/env python3
"""Demo script for the TEST phase only.

Runs tests and healing loop with Haiku.
Can be run on any repo with tests.

Usage:
    python scripts/demo_test.py [DIRECTORY] [--skip-setup]
"""

import argparse
import subprocess
import sys
from pathlib import Path

from _demo_common import create_test_repo, DEFAULT_REPO_PATH


def main():
    parser = argparse.ArgumentParser(description="Run test phase demo with Haiku")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_REPO_PATH,
        help=f"Test repo location (default: {DEFAULT_REPO_PATH})",
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

    print(f"\nRunning TEST phase on: {target}")
    print("-" * 60)

    result = subprocess.run(
        [
            sys.executable, "-m", "daydream",
            str(target),
            "--python",
            "--model", "haiku",
            "--start-at", "test",
            "--no-cleanup",
        ],
        cwd=Path(__file__).parent.parent,
    )

    print(f"\nTest repo preserved at: {repo_path}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
