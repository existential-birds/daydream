#!/usr/bin/env python3
"""Demo script that creates a buggy Python/FastAPI repo and runs the full daydream loop."""

import shutil
import sys

from _demo_common import create_argument_parser, run_daydream_command, validate_repo_path


def main():
    parser = create_argument_parser("Run full daydream demo on a buggy test repo")
    parser.add_argument("--cleanup", action="store_true", help="Remove test repo after running")
    args = parser.parse_args()

    try:
        target = validate_repo_path(args.repo_path, skip_setup=False)
        if target is None:
            return 1

        print(f"\nRunning daydream on: {target}\n" + "-" * 60)
        exit_code = run_daydream_command(target, "python", args.model, extra_args=["--debug"])

        if args.cleanup:
            print(f"\nCleaning up: {args.repo_path.resolve()}")
            shutil.rmtree(args.repo_path.resolve())
        else:
            print(f"\nTest repo preserved at: {args.repo_path.resolve()}")

        return exit_code
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
