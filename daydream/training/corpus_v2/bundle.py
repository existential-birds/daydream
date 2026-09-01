"""Fail-closed loader for hydrated curation bundles (corpus v2).

Mirrors ``daydream.archive.hydrate.verify_publication``'s SHA256SUMS
parse-and-verify loop, but reads read-only from a local checkout directory.
Every ingestion gate raises :class:`BundleError` naming the offending
path/field — ingestion errors are never discarded or softened.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from daydream.archive.hydrate_rules import (
    REASON_CODE_LICENSE_EVIDENCE_MISSING,
    REASON_CODE_REPO_IDENTITY_MISSING,
)

_SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "curation-manifest-v1.json"
# The only producer (daydream.archive.hydrate.finalize) writes the manifest as
# ``curation-manifest.json`` under ``curated/<curation-id>/``; the bundle root
# is that curated directory.
_MANIFEST_NAME = "curation-manifest.json"
_SUCCESS_NAME = "_SUCCESS"
_SUMS_NAME = "SHA256SUMS"


class BundleError(ValueError):
    """A curated bundle failed a fail-closed ingestion gate."""


class BundleBatch(BaseModel):
    """One batch row from the curation manifest.

    The optional ``repo_slug`` (the ``normalize_remote_url`` slug, never a raw
    remote URL) and ``license_evidence`` (``spdx_id`` + ``source``) fields are
    carried by newer manifests; the admission gate, not this parser, enforces
    their presence.
    """

    model_config = ConfigDict(frozen=True)

    session_id: str
    content_digest: str
    status: str
    reason_code: str | None
    artifact_relpath: str
    artifact_digest: str | None = None
    manifest_relpath: str | None = None
    repo_slug: str | None = None
    license_evidence: dict[str, Any] | None = None



@dataclass(frozen=True)
class CuratedBundle:
    curation_id: str
    source_hub_commit: str
    manifest: dict[str, Any]
    batches: tuple[BundleBatch, ...]

    @property
    def admitted(self) -> tuple[BundleBatch, ...]:
        return tuple(b for b in self.batches if b.status == "admitted")


def _gate(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _verify_sha256sums(root: Path, prefix: str) -> None:
    sums_path = root / _SUMS_NAME
    _gate(sums_path.is_file(), f"bundle {root}: missing {_SUMS_NAME}")
    # daydream.archive.hydrate writes SHA256SUMS relpaths relative to the
    # hub-checkout root under the manifest's canonical ``publication_prefix``
    # (``curated/<curation-id>/``); the bundle root is that curated directory,
    # so strip the prefix before resolving under the root.
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, sep, relpath = line.partition("  ")
        _gate(bool(sep), f"bundle {root}: malformed {_SUMS_NAME} line {line!r}")
        p = Path(relpath)
        _gate(
            not p.is_absolute() and ".." not in p.parts,
            f"bundle {root}: {_SUMS_NAME} relpath {relpath!r} is not relative to the bundle root",
        )
        resolved = relpath[len(prefix):] if relpath.startswith(prefix) else relpath
        target = root / resolved
        if not target.is_file():
            raise BundleError(f"bundle {root}: missing artifact {relpath!r} listed in {_SUMS_NAME}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        _gate(actual == digest, f"bundle {root}: digest mismatch for {relpath!r}")


def _validate_relpath(root: Path, relpath: str, what: str) -> Path:
    """Gate ``relpath`` under ``root`` and require the target to exist.

    Targets may be files or directories: the producer writes each batch as a
    directory (``batches/<session_id>/``) whose contents SHA256SUMS covers
    file-by-file. Existence (not ``is_file``) is the gate so both shapes
    resolve.
    """
    p = Path(relpath)
    _gate(not p.is_absolute(), f"bundle {root}: {what} {relpath!r} is not relative")
    _gate(".." not in p.parts, f"bundle {root}: {what} {relpath!r} contains '..' segment")
    resolved = root / p
    _gate(resolved.exists(), f"bundle {root}: missing artifact {relpath!r} ({what})")
    return resolved


def load_curated_bundle(root: Path) -> CuratedBundle:
    """Load a curated bundle from a local checkout, gating in order: the
    ``_SUCCESS`` marker, curation-manifest schema validation (which yields the
    canonical ``publication_prefix``), SHA256SUMS verification,
    relative-path existence for every batch's artifacts, and the admission
    gate (every ``admitted`` batch must carry non-blank ``repo_slug`` and
    ``license_evidence`` with a non-empty ``spdx_id``; quarantined/excluded
    rows are exempt). Raises :class:`BundleError` naming the offending
    path/field on any gate failure.
    """
    root = Path(root)
    _gate((root / _SUCCESS_NAME).is_file(), f"bundle {root}: missing {_SUCCESS_NAME} marker")

    manifest_path = root / _MANIFEST_NAME
    _gate(manifest_path.is_file(), f"bundle {root}: missing {_MANIFEST_NAME}")
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    from jsonschema import Draft202012Validator  # noqa: PLC0415  # lazy: ingestion-time only

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(doc), key=str)
    if errors:
        first = errors[0]
        raise BundleError(f"bundle {root}: curation manifest invalid at {first.json_path}: {first.message}")

    # The manifest is the untrusted input; its canonical ``publication_prefix``
    # (schema-pinned ``curated/<curation-id>/``) is what SHA256SUMS relpaths
    # are written relative to, so checksum resolution must see it.
    _verify_sha256sums(root, str(doc["publication_prefix"]))

    batches = tuple(BundleBatch(**b) for b in doc["batches"])
    for batch in batches:
        _validate_relpath(root, batch.artifact_relpath, "artifact_relpath")
        if batch.manifest_relpath is not None:
            _validate_relpath(root, batch.manifest_relpath, "manifest_relpath")

    # Admission gate: pure structural check on manifest rows — the admission
    # decision was made at hydration; this boundary refuses bundles that
    # predate the gate. Only admitted content is blocked; quarantined and
    # excluded rows are non-admitted by construction.
    for batch in batches:
        if batch.status != "admitted":
            continue
        if not batch.repo_slug or not batch.repo_slug.strip():
            raise BundleError(
                f"bundle {root}: admitted batch {batch.session_id!r}: "
                f"{REASON_CODE_REPO_IDENTITY_MISSING}"
            )
        evidence = batch.license_evidence
        if not evidence or not isinstance(evidence.get("spdx_id"), str) or not evidence["spdx_id"].strip():
            raise BundleError(
                f"bundle {root}: admitted batch {batch.session_id!r}: "
                f"{REASON_CODE_LICENSE_EVIDENCE_MISSING}"
            )

    return CuratedBundle(
        curation_id=doc["curation_id"],
        source_hub_commit=doc["source_hub_commit"],
        manifest=doc,
        batches=batches,
    )
