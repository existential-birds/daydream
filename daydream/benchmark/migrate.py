"""Deterministic, non-destructive upgrade path for legacy authoring cases.

Issue #806 hardened the authoring schemas and made ``finding_id`` case-scoped
(``sha256(case_id, title, body, severity, path, start_line, end_line)``) via
:func:`daydream.benchmark.schema.derive_finding_id`, gated on
``CaseDocument.schema_version == 2`` so pre-change v1 workspaces stay loadable.
This module deterministically re-derives ``finding_id`` for every v1 case and
bumps its ``schema_version`` to 2, mutating only those two fields and never
touching authored content.

Invalid data is **never** silently rewritten: a case that fails to load or
validate is surfaced in ``UpgradeReport.errors`` and left byte-unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from daydream.benchmark import schema, storage


@dataclass
class CaseUpgrade:
    """One upgraded case's record."""

    case_id: str
    finding_ids_recomputed: int
    changed: bool


@dataclass
class UpgradeReport:
    """The outcome of one :func:`migrate_workspace` run."""

    cases: list[CaseUpgrade] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _upgrade_case(raw: dict, case_id: str) -> tuple[dict, int]:
    """Return a copy of *raw* with case-scoped finding ids and schema_version 2.

    Only ``finding_id`` values and ``schema_version`` are mutated; every
    authored field is preserved verbatim.
    """
    doc = dict(raw)
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
    """Deterministically upgrade every v1 case in the workspace to v2.

    Recomputes case-scoped ``finding_id`` and bumps ``schema_version`` to 2,
    writing changed cases atomically through ``storage.Transaction``. When
    *dry_run* is True the report is computed without writing. Invalid cases are
    recorded in ``report.errors`` and left byte-unchanged.

    The whole migration (read + write) runs under the workspace lock so it
    serializes against concurrent curators — otherwise a curator mutation and a
    migration could race on the same case file and lose an update.
    """
    root = Path(root)
    if dry_run:
        return _migrate_workspace_unlocked(root, dry_run=True)
    with storage.WorkspaceLock(root):
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
            if current == 2:
                continue
            if current != 1:
                raise ValueError(
                    f"case {case_id} has unsupported schema_version {current!r}"
                )
            new_raw, recomputed = _upgrade_case(raw, case_id)
            # strip the persisted audit field for validation (curation pattern),
            # but keep it in the written output — authored content is preserved.
            schema.CaseDocument.model_validate(schema._schema_ready(new_raw))
            # Every loadable v1 case is staged: the schema_version bump is
            # unconditional, so a write occurs even when no finding_id changed.
            changed = True
            upgrades.append(CaseUpgrade(case_id=case_id, finding_ids_recomputed=recomputed,
                                        changed=changed))
            writes[case_file] = yaml.safe_dump(new_raw, sort_keys=False).encode("utf-8")
        except Exception as exc:  # never silently rewrite a case
            report.errors.append(f"{case_id}: {exc}")

    if not dry_run:
        for case_path, content in writes.items():
            with storage.Transaction(root, op_id=f"migrate-{case_path}", kind="migrate") as tx:
                tx.stage(case_path, content)
                tx.commit()

    report.cases.extend(upgrades)
    return report
