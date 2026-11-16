"""Throwaway #1890 probe: does a supporting new-side ref on a tracked-worktree-edits revision read reviewed bytes?

Run: uv run python .scratch/issue-1890/probe_supporting_facts.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from daydream import git_ops
from daydream.host_tools import HostToolRejected
from daydream.review_source_refs import _supporting_facts, canonicalize_source_selectors, source_id

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e", "GIT_COMMITTER_NAME": "t",
       "GIT_COMMITTER_EMAIL": "t@e", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, env=ENV, check=True, capture_output=True, text=True).stdout.strip()


with tempfile.TemporaryDirectory() as tmp:
    repo = Path(tmp) / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    (repo / "base.txt").write_text("base\n")
    git(repo, "add", "base.txt")
    git(repo, "commit", "-q", "-m", "initial")
    git(repo, "checkout", "-q", "-b", "feature")
    (repo / "api.py").write_text("VALUE = 2\n")
    git(repo, "add", "api.py")
    git(repo, "commit", "-q", "-m", "feature")
    reviewed = b"base\nowner dirty bytes\n"
    (repo / "base.txt").write_bytes(reviewed)  # tracked, uncommitted
    head, base = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "main")
    diff = git_ops.diff(repo, "main")  # the fixing path's live-checkout diff (orchestrator.py:312)
    print("has_worktree_edits:", git_ops.has_worktree_edits(repo))
    print("reviewed diff includes base.txt:", "+owner dirty bytes" in diff)
    revision = {"head_sha": head, "merge_base_sha": base, "diff_key": hashlib.sha256(diff.encode()).hexdigest(),
                "tracked_worktree_edits": True}
    facts = _supporting_facts(repo, revision, "new", "base.txt", None)
    print("_supporting_facts(new, base.txt):", facts)
    print("  sha256(HEAD bytes)    =", hashlib.sha256(b"base\n").hexdigest())
    print("  sha256(reviewed bytes)=", hashlib.sha256(reviewed).hexdigest())
    print("  RESOLVES AGAINST:", "HEAD (bug)" if facts.digest == hashlib.sha256(b"base\n").hexdigest()
          else "reviewed bytes" if facts.digest == hashlib.sha256(reviewed).hexdigest() else "other")
    # Selector into the uncommitted line 2, beside a minimal assigned ref.
    part = git(repo, "diff", "main", "--", "api.py") + "\n"
    assigned = {"artifact_sha256": revision["diff_key"], "part_sha256": hashlib.sha256(part.encode()).hexdigest(),
                "revision": head, "path": "api.py", "side": "new", "source_range": {"start": 1, "end": 1},
                "fragment": None, "basis": "supplied_diff", "role": "assigned"}
    selectors = [{"source_id": source_id(assigned), "basis": "supplied_diff", "source_range": None},
                 {"path": "base.txt", "side": "new", "source_range": {"start": 2, "end": 2},
                  "basis": "source_inspection"}]
    try:
        canonical = canonicalize_source_selectors(selectors, [assigned], revision=revision, source_root=repo)
        print("canonicalize (line 2): ADMITTED", canonical[1])
    except HostToolRejected as exc:
        print("canonicalize (line 2): REFUSED ->", exc)
