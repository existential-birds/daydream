"""Read back a run's local evidence and report its completeness with stable identities and digests."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from daydream.trajectory.journal import read_journal
from daydream.trajectory.layout import DOCUMENT_SUFFIX, run_document_path, siblings_directory


@dataclass(frozen=True)
class DocumentCheck:
    """One evidence document as read back from disk."""

    file: str
    trajectory_id: str | None
    records: int
    size: int
    sha256: str
    complete: bool
    problem: str | None


@dataclass(frozen=True)
class EvidenceCheck:
    """A run directory's documents; complete only when every seal verifies and reports complete capture."""

    documents: tuple[DocumentCheck, ...]
    capture_complete: bool | None

    @property
    def complete(self) -> bool:
        return bool(self.documents) and self.capture_complete is True and all(
            document.complete for document in self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {"complete": self.complete, "capture_complete": self.capture_complete,
                "documents": [asdict(document) for document in self.documents]}


def verify_run_evidence(run_dir: Path) -> EvidenceCheck:
    """Verify the root document and its siblings without trusting any reference or index."""
    root = run_document_path(run_dir)
    siblings = siblings_directory(run_dir)
    paths = [root, *(sorted(siblings.glob(f"*{DOCUMENT_SUFFIX}")) if siblings.is_dir() else [])]
    checks: list[DocumentCheck] = []
    capture_complete: bool | None = None
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            checks.append(DocumentCheck(path.name, None, 0, 0, "", False, f"unreadable ({type(exc).__name__})"))
            continue
        contents = read_journal(raw)
        opened = contents.records[0].body if contents.records else None
        trajectory_id = opened.get("trajectory_id") if isinstance(opened, dict) else None
        checks.append(DocumentCheck(path.name, trajectory_id, len(contents.records), len(raw),
                                    hashlib.sha256(raw).hexdigest(), contents.complete, contents.problem))
        if path == root and contents.seal is not None:
            capture_complete = contents.seal.body.get("capture_complete") is True
    return EvidenceCheck(tuple(checks), capture_complete)
