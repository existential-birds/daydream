"""Shared cwd-grounding instruction for review/exploration prompts.

When daydream runs in a linked git worktree whose shared git dir lives in a
sibling main worktree, agents that derive the repo root from git topology
(`git worktree list`, `git rev-parse --git-common-dir`, the `.git` gitdir line)
resolve paths against the WRONG worktree. This instruction grounds every agent
to its actual working directory so file paths resolve correctly.
"""

from __future__ import annotations

CWD_GROUNDING_INSTRUCTION = (
    "Your working directory is {cwd}. This is a git worktree whose shared git "
    "dir may belong to a different worktree. Resolve every file path relative "
    "to this directory (or `git rev-parse --show-toplevel`). NEVER derive "
    "paths from `git worktree list`, `git rev-parse --git-common-dir`, or the "
    "`.git` file. Any path passed to a subagent must live under this working "
    "directory."
)

__all__ = ["CWD_GROUNDING_INSTRUCTION"]
