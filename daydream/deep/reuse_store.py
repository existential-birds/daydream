"""Workspace-local completed review cache under .daydream/review-cache/.

Publish payload files, then their manifest and digests, then complete.marker.
Hits require a matching key, every payload digest, and the marker; incomplete
entries are misses. Read failures become named misses; write errors propagate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from daydream.config import (
    DEFAULT_REVIEW_CACHE_ENABLED,
    DEFAULT_REVIEW_CACHE_MAX_AGE_DAYS,
    DEFAULT_REVIEW_CACHE_MAX_BYTES,
    DEFAULT_REVIEW_CACHE_MAX_ENTRIES,
)
from daydream.deep.reuse_key import (
    REUSE_KEY_FORMAT,
    PhaseIdentity,
    absent_components,
    grounding_digests,
)
from daydream.deep.settings import _resolve_config_value, _resolve_non_negative_int
from daydream.json_utils import atomic_write_bytes, read_json_object

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext
    from daydream.run_config import RunConfig

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
    payload: Mapping[str, bytes]
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def restore(self, dest_dir: Path) -> str | None:
        """Restore the verified bytes; a partial write failure requires recomputation."""
        try:
            for name, raw in self.payload.items():
                (dest_dir / name).write_bytes(raw)
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}"
        return None


@dataclass(frozen=True)
class ReuseMiss:
    """A lookup that did not verify, with the failing element named."""

    key: str
    reason: str


# ---------------------------------------------------------------------------
# Per-unit reuse orchestration (shared by review, merge and the shard fan-out)
# ---------------------------------------------------------------------------

#: Provenance outcomes that mean the store served the unit, never a recompute.
REUSE_HIT_OUTCOMES = frozenset({"hit", "reused"})


def _reuse_outcome(entry: object) -> object:
    """The provenance ``outcome`` of one recorded unit, or ``None``."""
    return entry.get("outcome") if isinstance(entry, dict) else None


def record_absent_components(reuse: ReuseCache, unit: str, payload: Mapping[str, Any]) -> None:
    """Record the named miss for a payload with a ``None`` required component."""
    reuse.record(
        unit,
        outcome="miss",
        reason="absent components: " + ", ".join(absent_components(payload)),
    )


def reuse_grounding_statuses(reuse: ReuseCache, payload: Mapping[str, Any]) -> dict[str, str]:
    """Classify recorded grounding as reused only for served-unit provenance outcomes."""
    units = reuse.provenance().get("units")
    units = units if isinstance(units, dict) else {}
    return {
        unit: "reused" if _reuse_outcome(units.get(unit)) in REUSE_HIT_OUTCOMES else "regenerated"
        for unit in grounding_digests(payload)
    }


def record_reuse_hit(
    reuse: ReuseCache,
    unit: str,
    key: str,
    hit: ReuseHit,
    payload: Mapping[str, Any],
) -> None:
    """Record a complete reuse hit with its grounding delta and per-unit statuses."""
    reuse.record(
        unit,
        outcome="hit",
        reason="complete entry",
        key=key,
        origin_run_id=hit.manifest.get("origin", {}).get("run_id"),
        detail={
            "grounding": reuse.grounding_delta(hit, grounding_digests(payload)),
            "grounding_status": reuse_grounding_statuses(reuse, payload),
        },
    )


def lookup_reuse_entry(
    reuse: ReuseCache,
    unit: str,
    key: str,
    dest_dir: Path,
    *,
    on_restore_failure: Callable[[str], None] | None = None,
    expected_coverage: Mapping[str, Any] | None = None,
) -> ReuseHit | None:
    """Return a restored hit or record a miss; optionally report restore failures."""
    hit = reuse.lookup(key)
    if not isinstance(hit, ReuseHit):
        reuse.record(unit, outcome="miss", reason=hit.reason, key=key)
        return None
    if expected_coverage is not None and hit.manifest.get("coverage") != dict(expected_coverage):
        reuse.record(unit, outcome="miss", reason="complete coverage proof absent or mismatched", key=key)
        return None
    restore_reason = hit.restore(dest_dir)
    if restore_reason is None:
        return hit
    if on_restore_failure is not None:
        on_restore_failure(restore_reason)
    reuse.record(unit, outcome="miss", reason=f"restore failed: {restore_reason}", key=key)
    return None


# ---------------------------------------------------------------------------
# Config resolution (CLI tier -> file config -> built-in default)
# ---------------------------------------------------------------------------


def review_cache_enabled(config: RunConfig) -> bool:
    """Resolve CLI > file config > enabled default, preserving explicit False."""
    value = _resolve_config_value(
        config, "review_cache_enabled", DEFAULT_REVIEW_CACHE_ENABLED
    )
    return bool(value)


def review_cache_budget(config: RunConfig) -> ReuseBudget:
    """Resolve each bound independently; convert configured age in days to seconds."""
    return ReuseBudget(
        max_entries=_resolve_non_negative_int(
            config, "review_cache_max_entries", DEFAULT_REVIEW_CACHE_MAX_ENTRIES
        ),
        max_bytes=_resolve_non_negative_int(
            config, "review_cache_max_bytes", DEFAULT_REVIEW_CACHE_MAX_BYTES
        ),
        max_age_seconds=86400
        * _resolve_non_negative_int(
            config, "review_cache_max_age_days", DEFAULT_REVIEW_CACHE_MAX_AGE_DAYS
        ),
    )


def build_reuse_cache(ctx: FlowContext) -> ReuseCache:
    """Create the run's private cache root beside deep/, bound to its artifact session.

    entries/ and provenance/ remain lazy; the root exists for artifact publication.
    """
    deep_dir_path = ctx.data.get("dd")
    if not isinstance(deep_dir_path, Path):
        deep_dir_path = Path(str(deep_dir_path))
    session_id = None if ctx.artifacts is None else ctx.artifacts.layout.session_id
    store = ReuseCache(
        deep_dir_path.parent / REVIEW_CACHE_DIRNAME,
        budget=review_cache_budget(ctx.config),
        run_id=ctx.work.run_id,
        session_id=session_id,
        enabled=review_cache_enabled(ctx.config),
    )
    _ensure_private_dir(store.store_dir)
    return store


def reuse_cache_for(ctx: FlowContext) -> ReuseCache | None:
    """Return the published cache only when enabled; otherwise skip both reads and writes."""
    value = ctx.data.get("reuse_cache")
    if not isinstance(value, ReuseCache):
        return None
    if not review_cache_enabled(ctx.config):
        return None
    return value


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
    """Content-keyed completed results; an omitted budget disables retention limits."""

    def __init__(
        self,
        store_dir: str | Path,
        *,
        budget: ReuseBudget | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.store_dir = Path(store_dir)
        self.budget = budget
        self.run_id = run_id
        self.session_id = session_id
        self.enabled = enabled
        self._provenance_warned = False

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
        coverage: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> None:
        """Write a completed entry: payload, then manifest, then marker.

        A failure before the marker propagates and leaves a directory that a
        later lookup reports as a miss, never a hit.
        """
        entry = self._entry_dir(key)
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
            "format": REUSE_KEY_FORMAT,
            "components": dict(components),
            "payload": payload_digests,
            "grounding": dict(grounding),
            "grounding_status": dict(grounding_status),
            "profile_digest": identity.profile_digest,
            "backend": identity.backend,
            "model": identity.model,
            "effort": identity.effort,
            "origin": {
                "run_id": self.run_id,
                "session_id": self.session_id,
                "recorded_at": recorded_at,
            },
            "created_at": recorded_at,
            "last_used_at": recorded_at,
        }
        if coverage is not None:
            manifest["coverage"] = dict(coverage)
        atomic_write_bytes(
            entry / MANIFEST_NAME,
            json.dumps(manifest, indent=2).encode("utf-8"),
            dir_fsync=True,
            mode=_PRIVATE_FILE_MODE,
        )
        atomic_write_bytes(
            entry / MARKER_NAME,
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
        """Evict oldest entries after writes until retention bounds are met.

        Never evict keep_key. Log and skip removal failures, then age out provenance.
        """
        if self.budget is None:
            return
        now = time.time()
        entries_path = self.store_dir / ENTRIES_DIRNAME
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
        provenance_dir = self.store_dir / PROVENANCE_DIRNAME
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
        entry = self._entry_dir(key)
        manifest_path = entry / MANIFEST_NAME
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
        if manifest.get("format") != REUSE_KEY_FORMAT:
            return ReuseMiss(key, "manifest format mismatch")
        origin = manifest.get("origin")
        if (not isinstance(origin, dict)
                or any(origin.get(field) is not None and not isinstance(origin[field], str)
                       for field in ("run_id", "session_id"))):
            return ReuseMiss(key, "manifest origin unreadable")

        recorded = manifest.get("payload")
        if not isinstance(recorded, dict):
            return ReuseMiss(key, "manifest unreadable")
        payload: dict[str, bytes] = {}
        for name, expected in recorded.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                return ReuseMiss(key, "manifest payload unreadable")
            try:
                _validate_payload_name(name)
            except ValueError:
                return ReuseMiss(key, "manifest payload name invalid")
            try:
                raw = (entry / name).read_bytes()
                actual = hashlib.sha256(raw).hexdigest()
            except OSError:
                return ReuseMiss(key, f"payload missing: {name}")
            if actual != expected:
                return ReuseMiss(key, f"payload digest mismatch: {name}")
            payload[name] = raw

        if not (entry / MARKER_NAME).is_file():
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
        return ReuseHit(key, payload, manifest)

    # -- provenance --------------------------------------------------------

    def record(
        self,
        unit: str,
        *,
        outcome: str,
        reason: str,
        key: str | None = None,
        origin_run_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically merge one unit's outcome into per-run provenance.

        Other units survive. Read/write errors are best-effort: provenance is evidence,
        never an input to the work the cache stores.
        """
        path = self._provenance_path()
        entry: dict[str, Any] = {"outcome": outcome, "reason": reason}
        if key is not None:
            entry["key"] = key
        if origin_run_id is not None:
            entry["origin_run_id"] = origin_run_id
        if detail:
            entry.update(detail)
        try:
            record = read_json_object(path)
            units = record.get("units")
            if not isinstance(units, dict):
                units = {}
            units[unit] = entry
            record["units"] = units
            _ensure_private_dir(path.parent)
            atomic_write_bytes(
                path,
                json.dumps(record, indent=2).encode("utf-8"),
                dir_fsync=True,
                mode=_PRIVATE_FILE_MODE,
            )
        except OSError:
            if not self._provenance_warned:
                logger.warning(
                    "reuse cache: could not record reuse provenance at %s",
                    path,
                    exc_info=True,
                )
                self._provenance_warned = True

    def provenance(self) -> dict[str, Any]:
        """Read per-unit provenance and attach run identity and live store statistics.

        Records live in the cache so a fresh-run deep/ wipe preserves them.
        """
        record = read_json_object(self._provenance_path())
        record["run_id"] = self.run_id
        record["session_id"] = self.session_id
        record["enabled"] = self.enabled
        record["store"] = self._store_stats()
        return record

    def grounding_delta(
        self, hit: ReuseHit, current: Mapping[str, str]
    ) -> dict[str, dict[str, str | bool]]:
        """Compare current and produced-under digests without I/O; missing means "absent"."""
        produced = hit.manifest.get("grounding")
        if not isinstance(produced, dict):
            produced = {}
        result: dict[str, dict[str, str | bool]] = {}
        for name in sorted(set(produced) | set(current)):
            was = produced.get(name, "absent")
            now = current.get(name, "absent")
            result[name] = {"produced": was, "current": now, "moved": was != now}
        return result

    def _entry_dir(self, key: str) -> Path:
        return self.store_dir / ENTRIES_DIRNAME / key

    def _provenance_path(self) -> Path:
        run_id = self.run_id or self.session_id or "unknown"
        return self.store_dir / PROVENANCE_DIRNAME / f"{run_id}.json"

    def _store_stats(self) -> dict[str, Any]:
        """Walk ``entries/`` for the provenance summary's store row (SH2)."""
        entries_path = self.store_dir / ENTRIES_DIRNAME
        count = 0
        total_bytes = 0
        oldest_age: float | None = None
        now = time.time()
        if entries_path.is_dir():
            for entry in entries_path.iterdir():
                if not entry.is_dir():
                    continue
                count += 1
                total_bytes += self._entry_bytes(entry)
                age = now - self._entry_last_used(entry)
                if oldest_age is None or age > oldest_age:
                    oldest_age = age
        return {
            "entries": count,
            "bytes": total_bytes,
            "oldest_last_used_age_s": oldest_age,
        }
