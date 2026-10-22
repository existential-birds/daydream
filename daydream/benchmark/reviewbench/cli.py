"""Public ReviewBench commands owned by the supplied Harbor workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys

from daydream import git_ops
from daydream.benchmark.reviewbench.candidate import candidate_identity
from daydream.review_profile import ProfileError


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the single public Harbor workflow."""
    parser = subparsers.add_parser('reviewbench', help='supplied ReviewBench through Harbor')
    commands = parser.add_subparsers(dest='reviewbench_command', required=True)
    from daydream.benchmark.reviewbench.harbor_cli import add_parsers

    add_parsers(commands)


def handle(args: argparse.Namespace) -> int:
    """Dispatch an explicit export or model launch with operator-facing errors."""
    from daydream.benchmark.reviewbench.harbor_cli import handle as harbor_handle

    try:
        return harbor_handle(args, candidate_identity())
    except (OSError, ValueError, KeyError, TypeError, ProfileError, git_ops.GitError,
            subprocess.CalledProcessError) as exc:
        print(f'ReviewBench: {exc}', file=sys.stderr)
        return 1
