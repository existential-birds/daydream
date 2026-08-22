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

import yaml

from daydream import git_ops
from daydream.benchmark import schema, snapshot, storage
from daydream.benchmark.storage import load_yaml_strict


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
            "gold_mode": schema.derive_gold_mode(_curation_model(curation)),
            "gold_count": len(findings),
            "snapshot_status": snapshot_status,
            "head_prefix": head_sha[:12] if head_sha else "",
        })
    return out


def list_case(root: Path, case_id: str) -> dict[str, Any]:
    """Read-only view of one case document (its raw case dict)."""
    return _load_case(root, case_id)


# ---------------------------------------------------------------------------
# derivation + validation
# ---------------------------------------------------------------------------

MAX_GOLD_FINDINGS = 50


def _curation_model(curation: dict[str, Any]) -> schema.Curation:
    """Build the fixed Curation model from a raw dict.

    The persisted ``gold_mode`` audit field is not schema field, so it is
    dropped here (it is recomputed by ``derive_gold_mode`` when read).
    """
    return schema.Curation(**{k: v for k, v in curation.items() if k != "gold_mode"})


def _schema_ready(raw: dict[str, Any]) -> dict[str, Any]:
    """A schema-valid copy of a raw case doc (persisted audit fields stripped)."""
    doc = dict(raw)
    curation = dict(raw.get("curation") or {})
    curation.pop("gold_mode", None)
    doc["curation"] = curation
    return doc


def _snapshot_head(raw: dict[str, Any]) -> str | None:
    """The 40-hex head SHA the case's snapshot was frozen at, or None."""
    snapshot_doc = raw.get("snapshot") or {}
    return snapshot_doc.get("original_head_sha")


def _projection_matches(candidate: dict[str, Any], finding: dict[str, Any]) -> bool:
    """True when a finding is byte-identical to one candidate projection.

    Compares the canonical content projection (title/body/location) — the
    severity is always ``None`` on a candidate projection.
    """
    return (
        candidate.get("title") == finding.get("title")
        and candidate.get("body") == finding.get("body")
        and candidate.get("location") == finding.get("location")
    )


def _derive_content(raw: dict[str, Any]) -> None:
    """Derive (and persist) ``gold_status`` + ``gold_mode`` from parsed findings.

    Never caller-supplied — derived always from the resulting findings, so an
    ``--apply-gold`` fragment (or a caller) cannot forge them.
    """
    curation = raw["curation"]
    model = _curation_model(curation)
    curation["gold_status"] = schema.derive_gold_status(model)
    curation["gold_mode"] = schema.derive_gold_mode(model)


def _validate_location(root: Path, raw: dict[str, Any], finding: dict[str, Any]) -> None:
    """Location-vs-head: path present in the head file, ordered positive lines."""
    location = finding.get("location")
    if location is None:
        return
    head_sha = _snapshot_head(raw)
    if not head_sha:
        raise CurationError("finding has a location but the snapshot carries no frozen head")
    path = location.get("path")
    start = location.get("start_line")
    end = location.get("end_line")
    line_count = _head_file_line_count(root, head_sha, path)
    if start < 1:
        raise CurationError(f"finding location start_line {start} must be >= 1")
    if end > line_count:
        raise CurationError(
            f"finding location {path!r} end_line {end} exceeds the head file's "
            f"line count {line_count}"
        )


def _validate_raw(root: Path, case_id: str, raw: dict[str, Any]) -> None:
    """Revalidate an in-memory case doc through curation rules + the full schema.

    Raises :class:`CurationError` naming the first violated curation rule; the
    fixed-schema :class:`ValidationError` (a contract failure) propagates
    unchanged. Never writes.
    """
    curation = raw.get("curation") or {}
    findings = curation.get("findings") or []

    if len(findings) > MAX_GOLD_FINDINGS:
        raise CurationError(f"case {case_id} exceeds 50 gold findings")
    ids = [f.get("finding_id") for f in findings]
    for fid in set(ids):
        if ids.count(fid) > 1:
            raise CurationError(f"case {case_id} has duplicate finding {fid}")

    candidates = raw.get("candidates") or []
    for finding in findings:
        _validate_location(root, raw, finding)
        provenance = finding.get("provenance") or {}
        if provenance.get("kind") == "historical":
            srcs = provenance.get("source_ids") or []
            if len(srcs) != 1:
                raise CurationError(
                    f"case {case_id} historical finding must reference exactly one source"
                )
            src = srcs[0]
            cand = next((c for c in candidates if c.get("source_id") == src), None)
            if cand is None:
                raise CurationError(f"historical finding references unknown candidate {src}")
            if not _projection_matches(cand, finding):
                raise CurationError(
                    f"historical finding source {src} does not byte-match its candidate projection"
                )

    schema.CaseDocument(**_schema_ready(raw))


def validate_case(root: Path, case_id: str) -> None:
    """Re-validate one case through the fixed schema plus curation-service rules.

    Returns ``None`` on success; raises :class:`CurationError` on the first
    curation-rule violation and lets the fixed-schema
    :class:`ValidationError` propagate as a contract failure. Never writes.
    """
    raw = _load_case(root, case_id)
    _validate_raw(root, case_id, raw)
    return None


def _stage_case(root: Path, case_id: str, raw: dict[str, Any], *, op: str) -> None:
    """Validate in-memory then atomically rewrite the single case YAML.

    The full validity check (curation rules + :class:`schema.CaseDocument`) runs
    **before** the :class:`storage.Transaction` opens, so a rejected mutation
    leaves the on-disk case byte-unchanged. No manifest is staged — curation
    state lives only in the case YAML.
    """
    _validate_raw(root, case_id, raw)
    with storage.Transaction(root, op_id=f"curate-{case_id}", kind=f"curation:{op}") as tx:
        tx.stage(
            f"cases/{case_id}.yaml",
            yaml.safe_dump(raw, sort_keys=False).encode("utf-8"),
        )
        tx.commit()


def accept_candidate(root: Path, case_id: str, source_id: str) -> None:
    """Accept one exact-exceptable candidate as a byte-identical ``historical`` finding.

    The finding's title/body/severity/location are taken straight from the
    candidate projection, ``provenance.kind`` is ``historical`` (the only path
    that produces it), and ``finding_id`` is derived from the content.
    """
    raw = _load_case(root, case_id)
    candidate = next(
        (c for c in (raw.get("candidates") or []) if c.get("source_id") == source_id),
        None,
    )
    if candidate is None:
        raise CurationError(f"no candidate {source_id} in case {case_id}")
    if not candidate.get("exact_acceptable"):
        raise CurationError(f"candidate {source_id} is not exact_acceptable")

    finding = {
        "title": candidate["title"],
        "body": candidate["body"],
        "severity": candidate.get("severity"),
        "location": candidate.get("location"),
        "provenance": {"kind": "historical", "source_ids": [source_id]},
    }
    finding["finding_id"] = schema.derive_finding_id(finding)
    raw.setdefault("curation", {}).setdefault("findings", []).append(finding)
    _derive_content(raw)
    _stage_case(root, case_id, raw, op="accept")
