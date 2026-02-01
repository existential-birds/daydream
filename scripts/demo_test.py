#!/usr/bin/env python3
"""Demo script for the TEST phase only. Can run on any repo with tests."""

import sys

from _demo_common import create_argument_parser, run_daydream_command, validate_repo_path


def main():
    parser = create_argument_parser("Run test phase demo")
    args = parser.parse_args()

    target = validate_repo_path(args.repo_path, args.skip_setup)
    if target is None:
        return 1

    print(f"\nRunning TEST phase on: {target}\n" + "-" * 60)
    return run_daydream_command(target, "python", args.model, start_at="test")


if __name__ == "__main__":
    sys.exit(main())
