"""Crash-safe, workspace-local store for completed deep review results.

The store owns ``.daydream/review-cache/``. Each entry lives in
``<store>/entries/<key>/`` and is published in the same order as the merge
step's own completion contract (``deep/merge_steps.py``): the payload files
first, then ``manifest.json`` (payload digests **plus** the grounding digests
the result was produced under, MH16), then ``complete.marker`` last. A lookup is
a hit only when the manifest parses, names the directory it was found in, every
recorded payload digest matches the on-disk bytes, and the marker exists; a
crash before the marker therefore leaves a rerun, never a half-hit.

Reads are tolerant (an unreadable store is a named miss, never an exception);
writes are fail-loud (an ``OSError`` propagates so a caller can warn that it
did not cache what it claimed to).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.config import (
    DEFAULT_REVIEW_CACHE_ENABLED,
    DEFAULT_REVIEW_CACHE_MAX_AGE_DAYS,
    DEFAULT_REVIEW_CACHE_MAX_BYTES,
    DEFAULT_REVIEW_CACHE_MAX_ENTRIES,
)
from daydream.config_file import _coerce_non_negative_int
from daydream.deep.reuse_key import REUSE_KEY_FORMAT, PhaseIdentity
from daydream.deep.settings import _resolve_config_value
from daydream.json_utils import atomic_write_bytes

if TYPE_CHECKING:
    from daydream.runner import RunConfig

logger = logging.getLogger(__name__)

#: The ``.daydream`` child that owns the store.
REVIEW_CACHE_DIRNAME = "review-cache"
#: Entry subdirectory holding one completed result per content key.
ENTRIES_DIRNAME = "entries"
#: The manifest published before ``complete.marker``.
MANIFEST_NAME = "manifest.json"
#: The completion marker; a lookup is a hit only once it exists.
MARKER_NAME = "complete.marker"
#: Per-run provenance records written beside ``entries/``.
PROVENANCE_DIRNAME = "provenance"

_PRIVATE_DIR_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


@dataclass(frozen=True)
class ReuseBudget:
    """The three retention bounds, whichever binds first (MH12)."""

    max_entries: int
    max_bytes: int
    max_age_seconds: int


@dataclass(frozen=True)
class ReuseHit:
    """A complete, verified entry ready to be restored."""

    key: str
    payload_dir: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ReuseMiss:
    """A lookup that did not verify, with the failing element named."""

    key: str
    reason: str


# ---------------------------------------------------------------------------
# Config resolution (CLI tier -> file config -> built-in default)
# ---------------------------------------------------------------------------


def review_cache_enabled(config: RunConfig) -> bool:
    """Resolve the reuse-cache enable flag (MH13).

    Precedence mirrors :func:`daydream.deep.settings._resolve_config_value`: 1)
    ``RunConfig.review_cache_enabled`` (CLI ``--no-review-cache``), 2)
    ``DaydreamFileConfig.review_cache_enabled`` (file-config scalar), 3) the
    built-in default (reuse on). An explicit ``False`` at either tier wins.
    """
    value = _resolve_config_value(
        config, "review_cache_enabled", DEFAULT_REVIEW_CACHE_ENABLED
    )
    return bool(value)


def _budget_bound(config: RunConfig, attr: str, default: int) -> int:
    """Resolve one retention bound, coercing-and-degrading to the default."""
    coerced = _coerce_non_negative_int(_resolve_config_value(config, attr, default))
    return coerced if coerced is not None else default


def review_cache_budget(config: RunConfig) -> ReuseBudget:
    """Resolve the three retention bounds independently (MH12).

    Each bound rides the same three tiers as :func:`review_cache_enabled`; the
    enable flag never affects the budget and each bound resolves on its own.
    Config-file values are days since last use; the returned budget is seconds.
    """
    return ReuseBudget(
        max_entries=_budget_bound(
            config, "review_cache_max_entries", DEFAULT_REVIEW_CACHE_MAX_ENTRIES
        ),
        max_bytes=_budget_bound(
            config, "review_cache_max_bytes", DEFAULT_REVIEW_CACHE_MAX_BYTES
        ),
        max_age_seconds=86400
        * _budget_bound(
            config, "review_cache_max_age_days", DEFAULT_REVIEW_CACHE_MAX_AGE_DAYS
        ),
    )


# ---------------------------------------------------------------------------
# Path surface
# ---------------------------------------------------------------------------


def review_cache_dir(deep_dir: str | Path) -> Path:
    """The store directory, a sibling of the run's ``deep/`` output directory."""
    return Path(deep_dir).parent / REVIEW_CACHE_DIRNAME


def entries_dir(store_dir: str | Path) -> Path:
    """The directory holding one subdirectory per content key."""
    return Path(store_dir) / ENTRIES_DIRNAME


def entry_dir(store_dir: str | Path, key: str) -> Path:
    """The directory for one content key."""
    return entries_dir(store_dir) / key


def entry_manifest_path(store_dir: str | Path, key: str) -> Path:
    """The manifest path for one content key."""
    return entry_dir(store_dir, key) / MANIFEST_NAME


def entry_marker_path(store_dir: str | Path, key: str) -> Path:
    """The completion-marker path for one content key."""
    return entry_dir(store_dir, key) / MARKER_NAME


def provenance_path(store_dir: str | Path, run_id: str) -> Path:
    """The per-run provenance record path under ``<store>/provenance/``."""
    return Path(store_dir) / PROVENANCE_DIRNAME / f"{run_id}.json"


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIR_MODE)
    with suppress(OSError):
        os.chmod(path, _PRIVATE_DIR_MODE)


def _validate_payload_name(name: str) -> None:
    """Reject anything that is not a verbatim artifact basename."""
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"payload name must be a leaf basename: {name!r}")


def _payload_bytes(data: bytes | bytearray | str) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class ReuseCache:
    """A content-keyed store rooted at ``store_dir``.

    ``budget`` is optional here so the pure store can be exercised without
    retention; the configured default is supplied by the run's factory.
    """

    def __init__(
        self,
        store_dir: str | Path,
        *,
        budget: ReuseBudget | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.budget = budget
        self.run_id = run_id
        self.session_id = session_id

    def store(
        self,
        key: str,
        *,
        unit: str,
        payload: Mapping[str, bytes | bytearray | str],
        components: Mapping[str, Any],
        identity: PhaseIdentity,
        grounding: Mapping[str, str],
        grounding_status: Mapping[str, str],
        key_format: int = REUSE_KEY_FORMAT,
        run_id: str | None = None,
        session_id: str | None = None,
        now: float | None = None,
    ) -> None:
        """Write a completed entry: payload, then manifest, then marker.

        A failure before the marker propagates and leaves a directory that a
        later lookup reports as a miss, never a hit.
        """
        entry = entry_dir(self.store_dir, key)
        _ensure_private_dir(entry)

        payload_digests: dict[str, str] = {}
        for name, data in payload.items():
            _validate_payload_name(name)
            raw = _payload_bytes(data)
            atomic_write_bytes(entry / name, raw, mode=_PRIVATE_FILE_MODE)
            payload_digests[name] = hashlib.sha256(raw).hexdigest()

        recorded_at = time.time() if now is None else now
        manifest: dict[str, Any] = {
            "key": key,
            "unit": unit,
            "format": key_format,
            "components": dict(components),
            "payload": payload_digests,
            "grounding": dict(grounding),
            "grounding_status": dict(grounding_status),
            "profile_digest": identity.profile_digest,
            "backend": identity.backend,
            "model": identity.model,
            "effort": identity.effort,
            "origin": {
                "run_id": self.run_id if run_id is None else run_id,
                "session_id": self.session_id if session_id is None else session_id,
                "recorded_at": recorded_at,
            },
            "created_at": recorded_at,
            "last_used_at": recorded_at,
        }
        atomic_write_bytes(
            entry_manifest_path(self.store_dir, key),
            json.dumps(manifest, indent=2).encode("utf-8"),
            dir_fsync=True,
            mode=_PRIVATE_FILE_MODE,
        )
        atomic_write_bytes(
            entry_marker_path(self.store_dir, key),
            b"",
            dir_fsync=True,
            mode=_PRIVATE_FILE_MODE,
        )
        self.prune(keep_key=key)

    # -- retention ---------------------------------------------------------

    def _entry_last_used(self, entry: Path) -> float:
        """The manifest's ``last_used_at``; the directory mtime when unreadable."""
        try:
            manifest = json.loads((entry / MANIFEST_NAME).read_text(encoding="utf-8"))
            last_used = manifest.get("last_used_at")
            if isinstance(last_used, (int, float)):
                return float(last_used)
        except (OSError, ValueError):
            pass
        with suppress(OSError):
            return entry.stat().st_mtime
        return 0.0

    @staticmethod
    def _entry_bytes(entry: Path) -> int:
        total = 0
        for path in entry.rglob("*"):
            if path.is_file():
                with suppress(OSError):
                    total += path.stat().st_size
        return total

    def _remove_entry(self, entry: Path) -> None:
        """Remove one entry whole; a failure leaves a miss, never a partial hit."""
        try:
            shutil.rmtree(entry)
        except OSError:
            logger.debug("reuse cache: could not evict %s", entry, exc_info=True)

    def prune(self, *, keep_key: str) -> None:
        """Evict oldest-last-used entries while any retention bound is exceeded.

        Runs after a write, never on a timer. ``keep_key`` (the entry just
        written) is never a candidate, so the run's own result survives. A
        removal failure is logged and skipped: a cache that cannot prune
        slightly is still a correct cache.
        """
        if self.budget is None:
            return
        now = time.time()
        entries_path = entries_dir(self.store_dir)
        if not entries_path.is_dir():
            self._prune_provenance(now)
            return
        entries = [
            (entry, self._entry_last_used(entry), self._entry_bytes(entry))
            for entry in entries_path.iterdir()
            if entry.is_dir()
        ]
        entries.sort(key=lambda record: (record[1], record[0].name))
        count = len(entries)
        total_bytes = sum(record[2] for record in entries)
        for entry, last_used, size in entries:
            if entry.name == keep_key:
                continue
            too_old = (now - last_used) > self.budget.max_age_seconds
            over_count = count > self.budget.max_entries
            over_bytes = total_bytes > self.budget.max_bytes
            if too_old or over_count or over_bytes:
                self._remove_entry(entry)
                count -= 1
                total_bytes -= size
        self._prune_provenance(now)

    def _prune_provenance(self, now: float) -> None:
        """Age out per-run provenance records under the same age bound."""
        if self.budget is None:
            return
        provenance_dir = Path(self.store_dir) / PROVENANCE_DIRNAME
        if not provenance_dir.is_dir():
            return
        for record in provenance_dir.iterdir():
            if not record.is_file():
                continue
            try:
                age = now - record.stat().st_mtime
            except OSError:
                continue
            if age > self.budget.max_age_seconds:
                with suppress(OSError):
                    record.unlink()

    def lookup(self, key: str) -> ReuseHit | ReuseMiss:
        """Return a verified :class:`ReuseHit` or a named :class:`ReuseMiss`.

        Never raises: the store's own read failures become miss reasons.
        """
        entry = entry_dir(self.store_dir, key)
        manifest_path = entry_manifest_path(self.store_dir, key)
        if not manifest_path.is_file():
            return ReuseMiss(key, "manifest absent")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ReuseMiss(key, "manifest unreadable")
        if not isinstance(manifest, dict):
            return ReuseMiss(key, "manifest unreadable")
        if manifest.get("key") != key:
            return ReuseMiss(key, "manifest key mismatch")

        recorded = manifest.get("payload")
        if not isinstance(recorded, dict):
            return ReuseMiss(key, "manifest unreadable")
        for name, expected in recorded.items():
            try:
                actual = hashlib.sha256((entry / name).read_bytes()).hexdigest()
            except OSError:
                return ReuseMiss(key, f"payload missing: {name}")
            if actual != expected:
                return ReuseMiss(key, f"payload digest mismatch: {name}")

        if not entry_marker_path(self.store_dir, key).is_file():
            return ReuseMiss(key, "completion marker absent")

        # Refresh the eviction clock; bookkeeping never turns a hit into a miss
        # and never fabricates one.
        manifest["last_used_at"] = time.time()
        with suppress(OSError):
            atomic_write_bytes(
                manifest_path,
                json.dumps(manifest, indent=2).encode("utf-8"),
                dir_fsync=True,
                mode=_PRIVATE_FILE_MODE,
            )
        return ReuseHit(key, entry, manifest)


def grounding_delta(
    hit: ReuseHit, current: Mapping[str, str]
) -> dict[str, dict[str, str | bool]]:
    """Compare a hit's produced-under grounding with the current iteration's.

    Pure: reads only the manifest and the caller's digests, performs no I/O.
    A grounding input missing on either side is the literal ``"absent"``.
    """
    produced = hit.manifest.get("grounding")
    if not isinstance(produced, dict):
        produced = {}
    result: dict[str, dict[str, str | bool]] = {}
    for name in sorted(set(produced) | set(current)):
        was = produced.get(name, "absent")
        now = current.get(name, "absent")
        result[name] = {"produced": was, "current": now, "moved": was != now}
    return result
