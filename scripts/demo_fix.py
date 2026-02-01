#!/usr/bin/env python3
"""Demo script for the FIX phase only. Requires a prior review run."""

import sys

from _demo_common import create_argument_parser, run_daydream_command, validate_repo_path


def main():
    parser = create_argument_parser("Run fix phase demo")
    args = parser.parse_args()

    target = validate_repo_path(args.repo_path, args.skip_setup, require_review_file=True)
    if target is None:
        return 1

    print(f"\nRunning FIX phase on: {target}\n" + "-" * 60)
    return run_daydream_command(target, "python", args.model, start_at="fix")


if __name__ == "__main__":
    sys.exit(main())
