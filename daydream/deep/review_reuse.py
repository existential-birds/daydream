"""Identity-bound reuse of completed review units and their shared grounding inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.deep.reuse_key import PhaseIdentity, digest_or_absent, exploration_digest, grounding_digests, unit_key
from daydream.deep.reuse_store import (
    ReuseCache,
    ReuseHit,
    lookup_reuse_entry,
    record_absent_components,
    record_reuse_hit,
    reuse_grounding_statuses,
)

if TYPE_CHECKING:
    from daydream.deep.state import DeepData
    from daydream.review_result import ReviewCoverage


def bind_reuse_coverage(payload: dict[str, Any], coverage: ReviewCoverage) -> None:
    """Bind versioned review caches to the captured snapshot and full scope inventory."""
    payload["components"]["review_coverage"] = {
        "schema_version": 1,
        "analyzed_revision": coverage.revision.to_dict(),
        "planned_scopes": [scope.to_dict() for scope in coverage.planned_scopes],
    }


def _reuse_expectation(coverage: ReviewCoverage, unit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "unit": unit,
        "analyzed_revision": coverage.revision.to_dict(),
        "planned_scopes": [scope.to_dict() for scope in coverage.planned_scopes],
        "status": "complete",
    }


@dataclass
class ReviewReuseUnit:
    """Keep a pre-dispatch key, identity and grounding together through restore/store."""

    cache: ReuseCache
    name: str
    identity: PhaseIdentity
    payload: dict[str, Any]
    coverage: ReviewCoverage
    key: str | None = field(init=False)

    def __post_init__(self) -> None:
        bind_reuse_coverage(self.payload, self.coverage)
        self.key = unit_key(self.payload)
        if self.key is None:
            record_absent_components(self.cache, self.name, self.payload)

    def lookup(
        self, destination: Path, *, on_restore_failure: Callable[[str], None] | None = None,
    ) -> ReuseHit | None:
        if self.key is None:
            return None
        return lookup_reuse_entry(
            self.cache, self.name, self.key, destination, on_restore_failure=on_restore_failure,
            expected_coverage=_reuse_expectation(self.coverage, self.name),
        )

    def record_hit(self, hit: ReuseHit) -> None:
        assert self.key is not None
        record_reuse_hit(self.cache, self.name, self.key, hit, self.payload)

    def restore(
        self, destination: Path, *, on_restore_failure: Callable[[str], None] | None = None,
    ) -> bool:
        hit = self.lookup(destination, on_restore_failure=on_restore_failure)
        if hit is None:
            return False
        self.record_hit(hit)
        return True

    def store(self, collect: Callable[[], dict[str, bytes] | None]) -> None:
        """Read and store outputs only for a usable key and a complete artifact set."""
        if self.key is None:
            return
        outcomes = self.coverage.scopes if self.name.startswith("shard:") else self.coverage.phases
        outcome = outcomes.get(self.name.removeprefix("shard:"))
        if outcome is None or outcome["status"] != "complete":
            return
        outputs = collect()
        if outputs is not None:
            self.cache.store(
                self.key,
                unit=self.name,
                payload=outputs,
                components=self.payload["components"],
                identity=self.identity,
                grounding=grounding_digests(self.payload),
                grounding_status=reuse_grounding_statuses(self.cache, self.payload),
                coverage=_reuse_expectation(self.coverage, self.name),
            )


def _records_bytes_by_basename(paths: list[Path]) -> dict[str, bytes | None]:
    """Map each records file by basename; an unreadable file maps to ``None``.

    A ``None`` entry makes the whole reuse unit a named miss rather than keying
    a partial set (see :func:`arbiter_key_payload` / :func:`merge_key_payload`).
    """
    records: dict[str, bytes | None] = {}
    for path in paths:
        try:
            records[path.name] = path.read_bytes()
        except OSError:
            records[path.name] = None
    return records


def _loop_grounding(deep_data: DeepData) -> dict[str, Any]:
    """The loop-re-derived inputs shared by the arbiter and merge units (MH2/MH16).

    Intent and alternatives are read back as each prompt sees them (the restored
    artifact on a hit), and the pre-scan is digested by directory content so
    its ``cache-key`` bookkeeping can never move anything.
    """
    alts_path = deep_data["alts_path"]
    try:
        alternatives_text = alts_path.read_text(encoding="utf-8") if alts_path.is_file() else None
    except OSError:
        alternatives_text = None
    return {
        "intent": digest_or_absent(deep_data.get("intent_summary")),
        "alternatives": digest_or_absent(alternatives_text),
        "exploration": {"digest": exploration_digest(deep_data["exploration_dir"])},
    }
