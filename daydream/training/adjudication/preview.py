"""Pre-harvest preview ledger over the hydrated index (issue #984, Task 9).

Builds the adjudication queue over a hydrated index and writes a
digest-pinned, canonical-JSON ledger so an operator can inspect exactly what a
subsequent canonical harvest would adjudicate *before* running it. A
hydrated staging archive with no ``sessions.jsonl`` is served through the
same read-only SQLite adapter materialize uses (never the read-write
``_get_connection``); preview never mutates the index. The ledger
pins per-finding evidence digests (delta on
``corpus_v2.bundle``'s bundle-digest SHA256SUMS verification and
the projector's two-bundle verification), and a re-preview against an
existing ledger reports any ``record_id`` whose evidence digest changed —
drift is surfaced, never silently merged.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import HubUnavailableError, RepoInfo, resolve_source_revision
from daydream.training.adjudication.queue import build_queue

__all__ = ["preview_ledger_digest", "run_preview"]

_SESSIONS_FILENAME = "sessions.jsonl"
_REVISION_FILENAME = "index-revision.txt"

_ITEM_KEYS = ("disposition", "evidence_digest", "fingerprint", "record_id", "status")


class _LocalIndexClient:
    """Minimal ``resolve_source_revision`` client over a local hydrated index.

    The index is already on disk, so a full 40-hex SHA is pinned by definition
    and "exists". Symbolic refs (moving branches/tags) resolve to nothing —
    the revision list is empty — so ``resolve_source_revision`` raises
    ``MovingBranchError`` with its own semantics; preview never re-implements
    pinned-revision policy.
    """

    def repo_info(self, revision: str | None = None) -> RepoInfo:
        return RepoInfo(sha=revision or "", private=True)

    def list_repo_files(self, revision: str | None = None) -> list[str]:
        return []

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
        raise HubUnavailableError(f"local index has no downloadable file {path_in_repo!r}")

    def upload_files(self, mapping: dict[str | Path, Path], commit_message: str) -> None:
        raise HubUnavailableError("local index client never uploads")

    @property
    def repo_private(self) -> bool:
        return True

    def list_revisions(self) -> list[str]:
        return []


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def preview_ledger_digest(ledger: dict[str, Any]) -> str:
    """SHA-256 over the canonical form of the ledger's pinned content."""
    pinned = {k: ledger[k] for k in ("index_revision", "items")}
    return hashlib.sha256(_canonical(pinned).encode("utf-8")).hexdigest()


def _load_sessions(index_root: Path) -> tuple[list[dict[str, Any]], str]:
    sessions_path = index_root / _SESSIONS_FILENAME
    if not sessions_path.is_file():
        raise HubUnavailableError(
            f"hydrated index sessions file not found: {sessions_path}"
        )
    sessions: list[dict[str, Any]] = []
    try:
        for line in sessions_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sessions.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise HubUnavailableError(f"unreadable hydrated index at {sessions_path}: {exc}") from exc
    index_revision = hashlib.sha256(sessions_path.read_bytes()).hexdigest()
    revision_file = index_root / _REVISION_FILENAME
    if revision_file.is_file():
        # Delegate pinned-revision resolution to hydrate.py's resolver: a
        # moving branch/tag raises MovingBranchError, a full SHA passes through.
        raw_revision = revision_file.read_text(encoding="utf-8").strip()
        if raw_revision:
            index_revision = resolve_source_revision(
                _LocalIndexClient(), raw_revision, exploratory=False
            )
    return sessions, index_revision


def run_preview(index_root: Path, ledger_path: Path) -> dict[str, Any]:
    """Preview the adjudication queue over the hydrated index at ``index_root``.

    Builds the queue (deterministic, ``record_id``-ordered), computes each
    item's evidence digest, and writes a canonical-JSON ledger
    ``{"index_revision", "items": [...], "ledger_digest"}`` to
    ``ledger_path`` (sorted keys, sorted items — identical index ⇒
    byte-identical ledger).

    Returns ``{"index_revision", "ledger_digest", "item_count",
    "drifted_record_ids"}`` where ``drifted_record_ids`` compares the fresh
    queue against any existing ledger at ``ledger_path``: a digest mismatch on
    a known ``record_id`` is reported, never silently merged. A first preview
    (no prior ledger) reports an empty drifted set.

    Failure policy: a missing/unreadable sessions file raises the
    ``HydrationError`` family (via ``HubUnavailableError``); a moving-branch
    source revision raises ``MovingBranchError`` through
    ``resolve_source_revision``; malformed evidence raises ``ValueError``
    from the queue builder naming the offending fingerprint.
    """
    if (index_root / _SESSIONS_FILENAME).is_file():
        sessions, index_revision = _load_sessions(index_root)
    else:
        # Hydrated staging archive: derive the sessions from the SQLite
        # index's label_observations via the shared materialize adapter
        # (read-only; lazy import — materialize imports this module's
        # ``_load_sessions``).
        from daydream.training.adjudication.materialize import (
            _sessions_from_hydrated_stage,
        )

        sessions, index_revision = _sessions_from_hydrated_stage(index_root)
    items = [
        {k: item[k] for k in _ITEM_KEYS} for item in build_queue(sessions)
    ]
    ledger: dict[str, Any] = {
        "index_revision": index_revision,
        "items": items,
    }
    ledger["ledger_digest"] = preview_ledger_digest(ledger)

    drifted: list[str] = []
    if ledger_path.is_file():
        try:
            prior = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HubUnavailableError(
                f"unreadable prior preview ledger at {ledger_path}: {exc}"
            ) from exc
        prior_digests = {
            str(item["record_id"]): str(item["evidence_digest"])
            for item in prior.get("items", [])
            if isinstance(item, dict) and item.get("record_id") and item.get("evidence_digest")
        }
        for item in items:
            record_id = str(item["record_id"])
            if record_id in prior_digests and prior_digests[record_id] != item["evidence_digest"]:
                drifted.append(record_id)

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(_canonical(ledger) + "\n", encoding="utf-8")
    return {
        "index_revision": index_revision,
        "ledger_digest": ledger["ledger_digest"],
        "item_count": len(items),
        "drifted_record_ids": drifted,
    }
