"""Build a deterministic evidence-digest ledger over the read-only hydrated index.

The same SQLite adapter serves preview and materialization. Re-preview reports
changed finding digests instead of silently merging drift.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import _FULL_SHA_RE, HubUnavailableError, MovingBranchError
from daydream.json_utils import canonical_json as _canonical
from daydream.training.adjudication.observations import load_observations, prior_adjudications
from daydream.training.adjudication.queue import build_queue

__all__ = ["preview_ledger_digest", "run_preview"]

_SESSIONS_OUT_FILENAME = "sessions.jsonl"
_REVISION_FILENAME = "index-revision.txt"

_ITEM_KEYS = ("disposition", "evidence_digest", "fingerprint", "record_id", "status")


def preview_ledger_digest(ledger: dict[str, Any]) -> str:
    """SHA-256 over the canonical form of the ledger's pinned content."""
    pinned = {k: ledger[k] for k in ("index_revision", "items")}
    return hashlib.sha256(_canonical(pinned).encode("utf-8")).hexdigest()


def _load_sessions(index_root: Path) -> tuple[list[dict[str, Any]], str]:
    sessions_path = index_root / _SESSIONS_OUT_FILENAME
    if not sessions_path.is_file():
        raise HubUnavailableError(
            f"hydrated index sessions file not found: {sessions_path}"
        )
    sessions: list[dict[str, Any]] = []
    try:
        source_bytes = sessions_path.read_bytes()
        for line in source_bytes.decode("utf-8").splitlines():
            if line.strip():
                sessions.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise HubUnavailableError(f"unreadable hydrated index at {sessions_path}: {exc}") from exc
    index_revision = hashlib.sha256(source_bytes).hexdigest()
    revision_file = index_root / _REVISION_FILENAME
    if revision_file.is_file():
        # A local revision pin has no remote authority to resolve symbolic refs.
        raw_revision = revision_file.read_text(encoding="utf-8").strip()
        if raw_revision:
            if _FULL_SHA_RE.fullmatch(raw_revision.lower()) is None:
                raise MovingBranchError(
                    f"ref {raw_revision!r} is a moving branch/tag, not a pinned commit; pass "
                    "exploratory=True to accept it (output is non-canonical), or pin an "
                    "exact 40-char commit SHA"
                )
            index_revision = raw_revision.lower()
    return sessions, index_revision


def run_preview(
    index_root: Path, ledger_path: Path, *, observations_path: Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic evidence-digest ledger over the read-only hydrated index.

    Return revision, ledger digest, item count and IDs whose evidence changed
    from a prior ledger. A first preview has no drift. Missing/unreadable input
    raises HubUnavailableError, moving refs raise MovingBranchError, and invalid
    evidence raises the queue builder's ValueError."""
    from daydream.training.adjudication.materialize import index_sessions

    sessions, index_revision, _runs = index_sessions(index_root)
    prior = prior_adjudications(load_observations(observations_path)) if observations_path else None
    items = [
        {k: item[k] for k in _ITEM_KEYS} for item in build_queue(sessions, prior_observations=prior)
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
