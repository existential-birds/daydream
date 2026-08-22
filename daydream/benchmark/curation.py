"""UI-independent golden-review curation service for private benchmark workspaces.

This module is the issue-#5 browser/terminal seam: the fixed operation set a
curator (or the future interactive client) drives every gold-curation action
through. Every mutating operation:

1. loads the target case YAML strictly,
2. derives ``finding_id`` / ``provenance.kind`` / ``gold_status`` /
   ``gold_mode`` / ``state`` — never caller-supplied,
3. enforces state transitions via :meth:`schema.validate_case_transition`,
4. re-validates the whole resulting :class:`schema.CaseDocument` plus
   curation-service rules (location-vs-head from the shared bare mirror,
   >50 gold cap, duplicate canonical finding, historical byte-match,
   exclusion/re-inclusion contract),
5. stages the rewritten case through the existing
   :class:`storage.Transaction` journal, or raises :class:`CurationError`
   naming the violated invariant **before** opening the Transaction.

The service imports no Rich/input/editor/HTTP code; it depends only on the
fixed schema, the mode-safe storage/journal layer, the bare mirror, and
``git_ops``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream import git_ops
from daydream.benchmark import schema, snapshot, storage
from daydream.benchmark.storage import WorkspaceCorrupt, load_yaml_strict


class CurationError(Exception):
    """A curation operation violated an invariant and mutated nothing."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _head_file_line_count(root: Path, head_sha: str, path: str) -> int:
    """The line count of *path* in the frozen head tree (shared bare mirror).

    Runs ``git cat-file blob <head_sha>:<path>`` with cwd in the shared bare
    mirror at ``root/cache/repository.git`` — the same read source the snapshot
    module's ``rev_parse`` uses. Raises :class:`CurationError` when the mirror
    cannot serve the path (the case cannot be a verified `ready` snapshot
    without a mirror that carried its head). A present file returns
    ``len(content.splitlines())``; an empty file has line count 0.
    """
    mirror = snapshot.mirror(root)
    proc = git_ops._run_git(
        mirror, ["cat-file", "blob", f"{head_sha}:{path}"], retries=0
    )
    if proc.returncode != 0:
        raise CurationError(f"location path {path!r} not present in head {head_sha}")
    return len(proc.stdout.splitlines())


def _case_path(root: Path, case_id: str) -> Path:
    return Path(root) / "cases" / f"{case_id}.yaml"


def _load_case(root: Path, case_id: str) -> dict[str, Any]:
    """Load one case document strict, raising ``CurationError`` when absent."""
    path = _case_path(root, case_id)
    if not path.exists():
        raise CurationError(f"unknown case {case_id}")
    return load_yaml_strict(path)


def list_cases(root: Path) -> list[dict[str, Any]]:
    """Read-only index of the workspace's cases with derived curation state."""
    manifest = load_yaml_strict(Path(root) / "benchmark.yaml")
    out: list[dict[str, Any]] = []
    for case in manifest.get("cases", []):
        case_id = case.get("case_id")
        case_file = case.get("case_file")
        doc = load_yaml_strict(Path(root) / case_file) if case_file else {}
        curation = doc.get("curation") or {}
        findings = curation.get("findings") or []
        snapshot_doc = doc.get("snapshot") or {}
        snapshot_status = snapshot_doc.get("status", "imported")
        head_sha = snapshot_doc.get("original_head_sha") or ""
        out.append({
            "case_id": case_id,
            "pr_number": case.get("pr_number"),
            "state": curation.get("state"),
            "gold_mode": schema.derive_gold_mode(schema.Curation(**curation)),
            "gold_count": len(findings),
            "snapshot_status": snapshot_status,
            "head_prefix": head_sha[:12] if head_sha else "",
        })
    return out


def list_case(root: Path, case_id: str) -> dict[str, Any]:
    """Read-only view of one case document (its raw case dict)."""
    return _load_case(root, case_id)