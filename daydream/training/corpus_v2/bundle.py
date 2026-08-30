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

_SCHEMA_PATH = Path(__file__).parent.parent / "schema" / "curation-manifest-v1.json"
_MANIFEST_NAME = "curation-manifest-v1.json"
_SUCCESS_NAME = "_SUCCESS"
_SUMS_NAME = "SHA256SUMS"


class BundleError(ValueError):
    """A curated bundle failed a fail-closed ingestion gate."""


class BundleBatch(BaseModel):
    """One batch row from the curation manifest (v1 shape)."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    content_digest: str
    status: str
    reason_code: str | None
    artifact_relpath: str
    artifact_digest: str | None = None
    manifest_relpath: str | None = None


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


def _verify_sha256sums(root: Path) -> None:
    sums_path = root / _SUMS_NAME
    _gate(sums_path.is_file(), f"bundle {root}: missing {_SUMS_NAME}")
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, sep, relpath = line.partition("  ")
        _gate(bool(sep), f"bundle {root}: malformed {_SUMS_NAME} line {line!r}")
        target = root / relpath
        if not target.is_file():
            raise BundleError(f"bundle {root}: missing artifact {relpath!r} listed in {_SUMS_NAME}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        _gate(actual == digest, f"bundle {root}: digest mismatch for {relpath!r}")


def _validate_relpath(root: Path, relpath: str, what: str) -> Path:
    p = Path(relpath)
    _gate(not p.is_absolute(), f"bundle {root}: {what} {relpath!r} is not relative")
    _gate(".." not in p.parts, f"bundle {root}: {what} {relpath!r} contains '..' segment")
    resolved = root / p
    _gate(resolved.is_file(), f"bundle {root}: missing artifact {relpath!r} ({what})")
    return resolved


def load_curated_bundle(root: Path) -> CuratedBundle:
    """Load a curated bundle from a local checkout, gating in order:
    ``_SUCCESS``, SHA256SUMS verification, manifest schema validation, and
    relative-path existence for every batch's artifacts. Raises
    :class:`BundleError` naming the offending path/field on any gate failure.
    """
    root = Path(root)
    _gate((root / _SUCCESS_NAME).is_file(), f"bundle {root}: missing {_SUCCESS_NAME} marker")
    _verify_sha256sums(root)

    manifest_path = root / _MANIFEST_NAME
    _gate(manifest_path.is_file(), f"bundle {root}: missing {_MANIFEST_NAME}")
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))

    from jsonschema import Draft202012Validator  # noqa: PLC0415  # lazy: ingestion-time only

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(doc), key=str)
    if errors:
        first = errors[0]
        raise BundleError(f"bundle {root}: curation manifest invalid at {first.json_path}: {first.message}")

    batches = tuple(BundleBatch(**b) for b in doc["batches"])
    for batch in batches:
        _validate_relpath(root, batch.artifact_relpath, "artifact_relpath")
        if batch.manifest_relpath is not None:
            _validate_relpath(root, batch.manifest_relpath, "manifest_relpath")

    return CuratedBundle(
        curation_id=doc["curation_id"],
        source_hub_commit=doc["source_hub_commit"],
        manifest=doc,
        batches=batches,
    )
