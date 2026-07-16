"""Lazy ``base_sha`` materialization for older archive manifests.

Older daydream archives (pre-``code_context``) lack a ``base_sha`` in their
``manifest.json``. When training-record export needs that field, this module
shells out to ``git merge-base`` against a local clone of the target repo
and writes the resolved SHA back into the manifest atomically.

The materialization is opportunistic: callers invoke it only when
``base_sha`` is missing AND a repo clone is discoverable on disk. Failures
return ``None`` without mutating the manifest, so the caller can proceed
with ``base_sha=None``.

Spike rationale: a one-shot recoverability probe across the local archive
(`.beagle/concepts/training-ready-corpus/spike-base-sha.md`) returned a
100% resolve rate, which routed the design to lazy on-read materialization
rather than a one-shot offline backfill.
"""

from __future__ import annotations

import json
from pathlib import Path

import daydream.git_ops as git_ops
from daydream.json_utils import atomic_write_json


def materialize_base_sha(manifest_path: Path, *, repo_clone: Path) -> str | None:
    """Materialize ``code_context.base_sha`` into ``manifest_path`` if missing.

    Reads the manifest, returns any already-set ``base_sha`` without
    shelling out. Otherwise resolves it via
    :func:`daydream.git_ops.merge_base` against ``repo_clone`` and writes
    the manifest back atomically via
    :func:`daydream.json_utils.atomic_write_json`.

    Args:
        manifest_path: Path to the on-disk ``manifest.json``.
        repo_clone: Path to a live working tree of the manifest's
            ``repo_slug``. The caller is responsible for ensuring this
            clone exists; this function only invokes ``git`` against it.

    Returns:
        The resolved ``base_sha`` string on success (whether already set
        or freshly materialized), or ``None`` when ``git merge-base``
        could not resolve a SHA. Returning ``None`` does not mutate the
        manifest.

    Raises:
        OSError: Propagated from manifest read/write failures.
        json.JSONDecodeError: Raised when the manifest is not valid JSON.
        daydream.git_ops.GitError: Propagated from ``merge_base`` on
            unexpected subprocess failures (e.g. timeout, missing ``git``
            binary). Routine "no merge base" outcomes still return ``None``.
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
