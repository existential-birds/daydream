"""Opportunistically materialize missing code_context.base_sha in legacy manifests.

Callers supply a discoverable local clone. Routine unresolved merge bases
return None without writing; read/write and unexpected Git failures propagate.
"""

from __future__ import annotations

import json
from pathlib import Path

import daydream.git_ops as git_ops
from daydream.json_utils import atomic_write_json


def materialize_base_sha(manifest_path: Path, *, repo_clone: Path) -> str | None:
    """Return an existing base_sha, or resolve and atomically persist it from repo_clone.

    The caller supplies an existing working tree. Routine no-merge-base returns
    None without mutation; malformed JSON, filesystem errors, and unexpected
    GitError (such as timeout/missing executable) propagate.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    code_ctx = manifest.get("code_context") or {}
    existing = code_ctx.get("base_sha")
    if isinstance(existing, str) and existing:
        return existing

    base_branch = code_ctx.get("base_branch") or (manifest.get("git") or {}).get("base_branch")
    head_sha = code_ctx.get("head_sha") or (manifest.get("git") or {}).get("head_sha")
    if not isinstance(base_branch, str) or not isinstance(head_sha, str):
        return None

    resolved = git_ops.merge_base(repo_clone, base_branch, head_sha)
    if resolved is None:
        return None

    manifest.setdefault("code_context", {})["base_sha"] = resolved
    atomic_write_json(manifest_path, manifest)
    return resolved
