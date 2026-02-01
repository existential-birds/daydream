#!/usr/bin/env python3
"""Demo script that creates a buggy Python/FastAPI repo and runs the full daydream loop.

Usage:
    python scripts/run_demo_python.py [DIRECTORY] [--cleanup]

Arguments:
    DIRECTORY    Where to create the test repo (default: ../test_buggy_demo)

Options:
    --cleanup    Remove the test repo after running
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from _demo_common import create_test_repo, DEFAULT_REPO_PATH


def run_daydream(target: Path) -> int:
    """Run daydream on the target repo."""
    print(f"\nRunning daydream on: {target}")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "daydream", str(target), "--python", "--model", "haiku", "--debug", "--no-cleanup"],
        cwd=Path(__file__).parent.parent,
    )

    return result.returncode


def cleanup_test_repo(repo_path: Path) -> None:
    """Remove the test repo."""
    if repo_path.exists():
        print(f"\nCleaning up: {repo_path}")
        shutil.rmtree(repo_path)


def main():
    parser = argparse.ArgumentParser(description="Run daydream demo on a buggy test repo")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=DEFAULT_REPO_PATH,
        help=f"Where to create the test repo (default: {DEFAULT_REPO_PATH})",
    )
    parser.add_argument("--cleanup", action="store_true", help="Remove test repo after running")
    args = parser.parse_args()

    repo_path = args.directory.resolve()

    try:
        # Create test repo
        target = create_test_repo(repo_path)
        if target is None:
            return 1

        # Run daydream
        exit_code = run_daydream(target)

        # Cleanup if requested
        if args.cleanup:
            cleanup_test_repo(repo_path)
        else:
            print(f"\nTest repo preserved at: {repo_path}")

        return exit_code

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
