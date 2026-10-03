"""Upgrade legacy cases without changing authored content or invalid input. V1 findings
gain case-scoped identities and schema version 2. Repair legacy curation pairing and
snapshot provenance only when local Git objects prove the requested tip, merge base, and
trees. Invalid cases remain byte-identical and appear in UpgradeReport.errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from daydream import git_ops
from daydream.benchmark import curation, schema, snapshot, storage


@dataclass
class CaseUpgrade:
    """One upgraded case's record."""

    case_id: str
    finding_ids_recomputed: int


@dataclass
class UpgradeReport:
    """The outcome of one :func:`migrate_workspace` run."""

    cases: list[CaseUpgrade] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _backfill_requested_base_sha(doc: dict[str, Any]) -> bool:
    """Preserve the sole base candidate on an imported snapshot that lacks it."""
    snapshot = doc.get("snapshot")
    if not isinstance(snapshot, dict):
        return False
    if snapshot.get("status") != "imported":
        return False
    original = snapshot.get("original_base_sha")
    if snapshot.get("requested_base_sha") is not None or not original:
        return False
    snapshot["requested_base_sha"] = original
    return True


def _repair_ready_base_provenance(root: Path, doc: dict[str, Any]) -> bool:
    """Verify and atomically repair one unmarked ready snapshot."""
    raw_snapshot = doc.get("snapshot")
    if not isinstance(raw_snapshot, dict) or raw_snapshot.get("status") != "ready":
        return False
    marker = raw_snapshot.get("base_resolution")
    if marker == "merge_base_v1":
        return False
    if marker is not None:
        raise ValueError("ready snapshot has an unsupported base-resolution marker")
    requested_tip = raw_snapshot.get("requested_base_sha") or raw_snapshot.get(
        "original_base_sha"
    )
    head_sha = raw_snapshot.get("original_head_sha")
    if not isinstance(requested_tip, str) or not isinstance(head_sha, str):
        raise ValueError("ready snapshot lacks source commit provenance")
    mirror = snapshot.mirror(root)
    if not mirror.exists():
        raise ValueError("ready snapshot provenance requires cache/repository.git")
    try:
        resolved_base = snapshot.resolve_original_base(mirror, requested_tip, head_sha)
        if resolved_base is None:
            raise ValueError("ready snapshot source commits have no merge base")
        trees = snapshot.resolve_trees(mirror, resolved_base, head_sha)
    except git_ops.GitError as exc:
        raise ValueError("ready snapshot provenance could not be verified") from exc
    if not isinstance(trees, tuple):
        raise ValueError("ready snapshot provenance references a missing Git object")
    base_tree, head_tree = trees
    if (
        base_tree != raw_snapshot.get("base_tree_sha")
        or head_tree != raw_snapshot.get("head_tree_sha")
    ):
        raise ValueError("ready snapshot source trees do not match recorded trees")
    raw_snapshot["requested_base_sha"] = requested_tip
    raw_snapshot["original_base_sha"] = resolved_base
    raw_snapshot["base_resolution"] = "merge_base_v1"
    return True


def _repair_legacy_unreplayable_curation(doc: dict[str, Any]) -> bool:
    """Repair the legacy producer's draft/unreplayable state pairing."""
    raw_snapshot = doc.get("snapshot")
    raw_curation = doc.get("curation")
    if (
        not isinstance(raw_snapshot, dict)
        or raw_snapshot.get("status") != "unreplayable"
        or not isinstance(raw_curation, dict)
        or raw_curation.get("state") != "draft"
    ):
        return False
    repaired = dict(raw_curation)
    repaired["state"] = "unreplayable"
    repaired["snapshot_attested"] = False
    repaired["clean_attested"] = False
    repaired["gold_status"] = "findings" if repaired.get("findings") else None
    curation._invalidate_task_spec_approval(repaired)
    doc["curation"] = repaired
    return True


def _upgrade_case(raw: dict[str, Any], case_id: str) -> tuple[dict[str, Any], int]:
    """Return a copy of *raw* with case-scoped finding ids and schema_version 2."""
    doc = dict(raw)
    _backfill_requested_base_sha(doc)
    findings = doc.get("curation", {}).get("findings") or []
    recomputed = 0
    new_findings = list(findings)
    for i, finding in enumerate(findings):
        expected = schema.derive_finding_id(finding, case_id=case_id)
        if finding.get("finding_id") != expected:
            new_findings[i] = {**finding, "finding_id": expected}
            recomputed += 1
    if recomputed:
        curation = dict(doc["curation"])
        curation["findings"] = new_findings
        doc["curation"] = curation
    doc["schema_version"] = 2
    return doc, recomputed


def migrate_workspace(root: Path, *, dry_run: bool = False) -> UpgradeReport:
    """Upgrade and validate cases, writing changes atomically under the workspace lock.
    Recompute V1 finding ids and legacy curation/provenance state; validate unchanged V2
    cases too. Invalid cases remain untouched and reported. Dry runs compute the report
    without writes or lock acquisition.
    """
    root = Path(root)
    if dry_run:
        return _migrate_workspace_unlocked(root, dry_run=True)
    with storage.WorkspaceLock(root):
        storage.recover_startup(root)
        return _migrate_workspace_unlocked(root, dry_run=False)


def _migrate_workspace_unlocked(root: Path, *, dry_run: bool) -> UpgradeReport:
    """Run the migration body (caller holds the workspace lock unless dry_run)."""
    manifest = storage.load_yaml_strict(root / "benchmark.yaml")
    report = UpgradeReport()
    writes: dict[str, bytes] = {}
    upgrades: list[CaseUpgrade] = []

    for _case in manifest.get("cases") or []:
        case_id = _case.get("case_id")
        case_file = _case.get("case_file")
        try:
            raw = storage.load_yaml_strict(root / case_file)
            current = raw.get("schema_version")
            changed = True
            if current == 2:
                # Repair legacy v2 base provenance/state pairings; valid v2 bytes stay untouched.
                new_raw = dict(raw)
                recomputed = 0
                changed = _backfill_requested_base_sha(new_raw)
            else:
                if current != 1:
                    raise ValueError(
                        f"case {case_id} has unsupported schema_version {current!r}"
                    )
                new_raw, recomputed = _upgrade_case(raw, case_id)
            # Both schema versions share the same repair suffix; a v1 upgrade is
            # already ``changed`` and never needs the repair return value.
            changed = _repair_ready_base_provenance(root, new_raw) or changed
            changed = _repair_legacy_unreplayable_curation(new_raw) or changed
            # strip the persisted audit field for validation (curation pattern),
            # but keep it in the written output — authored content is preserved.
            schema.CaseDocument.model_validate(schema._schema_ready(new_raw))
            if not changed:
                continue
            # Every staged case is written: the v1 schema_version bump is
            # unconditional, and a v2 repair is a real backfill.
            upgrades.append(CaseUpgrade(case_id=case_id, finding_ids_recomputed=recomputed))
            writes[case_file] = yaml.safe_dump(new_raw, sort_keys=False).encode("utf-8")
        except Exception as exc:  # never silently rewrite a case
            report.errors.append(f"{case_id}: {exc}")

    if not dry_run:
        for case_path, content in writes.items():
            with storage.Transaction(root, op_id=f"migrate-{case_path.replace('/', '_')}", kind="migrate") as tx:
                tx.stage(case_path, content)
                tx.commit()

    report.cases.extend(upgrades)
    return report
