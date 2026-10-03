"""Fail-open, session-bound quality measurements around authorized fix rounds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from daydream.agent import console
from daydream.config import (
    DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE,
    DEFAULT_QUALITY_GATE_EROSION_DELTA,
    DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE,
    DEFAULT_QUALITY_GATE_VERBOSITY_DELTA,
)
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.settings import _resolve_non_negative_float
from daydream.trajectory import current_session_id
from daydream.ui import print_warning

if TYPE_CHECKING:
    from daydream.run_config import RunConfig


async def capture_quality(
    daydream_dir: Path, code_workspace: Path, candidate_paths: set[str] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Capture scoped metrics off the event loop; return (None, reason) on failure.

    None candidate_paths inspects the whole workspace. Capture never fails the
    run; callers persist the reason as unavailable evidence.
    """
    try:
        from daydream.eval.analyzer import analyze_quality

        return await anyio.to_thread.run_sync(
            partial(analyze_quality, daydream_dir, candidate_paths, code_workspace=code_workspace)
        ), None
    except Exception as exc:  # noqa: BLE001 -- fail-open: never fail the run
        return None, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class QualityGateThresholds:
    """The four resolved quality-gate thresholds threaded through one fix round."""

    erosion_delta: float
    verbosity_delta: float
    erosion_absolute: float
    verbosity_absolute: float

    @classmethod
    def from_config(cls, config: RunConfig) -> "QualityGateThresholds":
        return cls(
            erosion_delta=_resolve_non_negative_float(
                config, "quality_gate_erosion_delta", DEFAULT_QUALITY_GATE_EROSION_DELTA
            ),
            verbosity_delta=_resolve_non_negative_float(
                config, "quality_gate_verbosity_delta", DEFAULT_QUALITY_GATE_VERBOSITY_DELTA
            ),
            erosion_absolute=_resolve_non_negative_float(
                config, "quality_gate_erosion_absolute", DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE
            ),
            verbosity_absolute=_resolve_non_negative_float(
                config, "quality_gate_verbosity_absolute", DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE
            ),
        )

    def payload(self) -> dict[str, float]:
        """The threshold fields exactly as persisted in the gate artifact."""
        return {
            "erosion_delta_threshold": self.erosion_delta,
            "verbosity_delta_threshold": self.verbosity_delta,
            "erosion_absolute_threshold": self.erosion_absolute,
            "verbosity_absolute_threshold": self.verbosity_absolute,
        }


def _quality_delta(before: float | None, after: float | None) -> float | None:
    """Rounded per-file metric delta; ``None`` when either side is undefined."""
    if before is None or after is None:
        return None
    return round(after - before, 4)


def _quality_flagged(
    *,
    erosion_before: float | None,
    erosion_after: float | None,
    erosion_delta: float | None,
    verbosity_before: float | None,
    verbosity_after: float | None,
    verbosity_delta: float | None,
    thresholds: QualityGateThresholds,
) -> bool:
    """Flag metric deltas strictly above their thresholds.

    An undefined baseline uses the separate absolute-after threshold, including
    existing files that gained measurable content. Two undefined values do not flag.
    """
    if erosion_delta is not None and erosion_delta > thresholds.erosion_delta:
        return True
    if verbosity_delta is not None and verbosity_delta > thresholds.verbosity_delta:
        return True
    if erosion_before is None and erosion_after is not None and erosion_after > thresholds.erosion_absolute:
        return True
    if verbosity_before is None and verbosity_after is not None and verbosity_after > thresholds.verbosity_absolute:
        return True
    return False


def _load_quality_gate_rounds(gate_p: Path, session_id: str | None) -> list[dict[str, Any]]:
    """Load valid round objects only from the current session.

    Unreadable files start fresh. Warn on malformed payloads, missing round lists,
    or another session; malformed individual rounds are dropped with a warning.
    """
    try:
        raw = gate_p.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        existing = json.loads(raw)
    except (json.JSONDecodeError, AttributeError, TypeError):
        existing = None
    if not isinstance(existing, dict):
        print_warning(console, f"Quality gate artifact {gate_p} is malformed; starting rounds fresh")
        return []
    if existing.get("session_id") != session_id:
        print_warning(
            console,
            f"Quality gate artifact {gate_p} belongs to another run's session; "
            "its rounds were not carried forward",
        )
        return []
    existing_rounds = existing.get("rounds")
    if not isinstance(existing_rounds, list):
        print_warning(console, f"Quality gate artifact {gate_p} has no rounds list; starting rounds fresh")
        return []
    valid = [r for r in existing_rounds if isinstance(r, dict)]
    if len(valid) != len(existing_rounds):
        print_warning(console, f"Quality gate artifact {gate_p} has non-object round entries; dropping them")
    return valid


def _persist_quality_round(
    path: Path,
    rounds: list[dict[str, Any]],
    entry: dict[str, Any],
    session_id: str | None,
    thresholds: QualityGateThresholds,
) -> None:
    """Replace this round while retaining other rounds from the same session."""
    retained = [row for row in rounds if row.get("round") != entry["round"]]
    path.write_text(json.dumps({
        "enabled": True,
        **thresholds.payload(),
        "session_id": session_id,
        "rounds": [*retained, entry],
    }, indent=2))


def _persist_quality_gate_unavailable(
    *,
    gate_p: Path,
    rounds: list[dict[str, Any]],
    round_no: int,
    stage: str,
    reason: str,
    session_id: str | None,
    thresholds: QualityGateThresholds,
) -> None:
    """Replace this round with an unavailable entry, then warn; write errors propagate."""
    _persist_quality_round(
        gate_p, rounds, {"round": round_no, "unavailable": {"stage": stage, "reason": reason}},
        session_id, thresholds,
    )
    print_warning(
        console,
        f"Quality gate unavailable (round {round_no}, {stage}): {reason}",
    )


async def _evaluate_quality_gate(
    *,
    enabled: bool,
    thresholds: QualityGateThresholds,
    daydream_dir: Path,
    code_workspace: Path,
    dd: Path,
    candidates: set[str] | None,
    before: dict[str, Any] | None,
    before_unavailable_reason: str | None,
    iteration: int | None,
) -> None:
    """Persist session-bound quality evidence without blocking the fix run.

    Disabled gates record enabled:false. Otherwise replace this iteration's round
    (or append the next sequence number). Missing candidates, capture failures
    and persistence failures produce an unavailable entry when writing still works.
    Missing before/after file metrics are flagged, never treated as a clean pass.
    Warn on flagged files and any unavailable result, including failed persistence.
    """
    session_id = current_session_id()
    try:
        gate_p = DeepArtifact.FIX_QUALITY_GATE.at(dd)
        if not enabled:
            gate_p.write_text(
                json.dumps({"enabled": False, "session_id": session_id}, indent=2)
            )
            return
        rounds = _load_quality_gate_rounds(gate_p, session_id)
        round_no = iteration if iteration is not None else len(rounds) + 1

        def _unavailable(stage: str, reason: str) -> None:
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage=stage,
                reason=reason,
                session_id=session_id,
                thresholds=thresholds,
            )

        if candidates is None:
            _unavailable(
                "candidates",
                "could not enumerate files changed by the fix pass against the pre-fix snapshot",
            )
            return
        if before is None:
            _unavailable(
                "before",
                before_unavailable_reason or "pre-fix quality snapshot unavailable",
            )
            return
        after, unavailable = await capture_quality(daydream_dir, code_workspace, candidates)
        if after is None:
            _unavailable("after", unavailable or "post-fix quality snapshot unavailable")
            return
        before_per_file: dict[str, Any] = before.get("per_file") or {}
        after_per_file: dict[str, Any] = after.get("per_file") or {}

        per_file: dict[str, dict[str, Any]] = {}
        for rel in sorted(candidates):
            before_entry = before_per_file.get(rel)
            after_entry = after_per_file.get(rel)
            if before_entry is None and after_entry is None:
                continue
            entry: dict[str, Any] = {}
            for metric in ("erosion", "verbosity"):
                prior = before_entry.get(metric) if before_entry is not None else None
                after_value = after_entry.get(metric) if after_entry is not None else None
                entry[f"{metric}_before"] = prior
                entry[f"{metric}_after"] = after_value
                entry[f"{metric}_delta"] = _quality_delta(prior, after_value)
            # Issue #329 / Finding 5: a candidate that parsed pre-fix but is
            # MISSING from the post-fix analyzer output is unparseable after the
            # fix (analyze_quality omits malformed files). Recording null
            # after-metrics with ``flagged=false`` would read as a clean
            # verdict, so mark it explicitly unavailable and flagged -- a fix
            # that breaks a file is a regression, never a pass. Still fail-open:
            # never raises, never stops the run.
            if before_entry is not None and after_entry is None:
                entry["unparseable"] = True
                entry["flagged"] = True
                entry["reason"] = "file missing from post-fix analyzer output (unparseable?)"
            # A missing pre-fix baseline (e.g. a secondary edit outside the
            # reviewed diff) means the delta is unknowable: flag the file, never
            # read it as a clean pass. Fail-open (issue #329 / #457).
            elif before_entry is None and after_entry is not None:
                entry["flagged"] = True
                entry["reason"] = (
                    "missing pre-fix baseline: file edited by the fix pass but not "
                    "covered by the pre-fix quality snapshot"
                )
            else:
                entry["flagged"] = _quality_flagged(**entry, thresholds=thresholds)
            per_file[rel] = entry
        _persist_quality_round(
            gate_p, rounds, {"round": round_no, "per_file": per_file}, session_id, thresholds,
        )
        flagged = [rel for rel, entry in per_file.items() if entry["flagged"]]
        if flagged:
            lines = []
            for rel in flagged:
                entry = per_file[rel]
                # Unparseable and missing-baseline entries carry a ``reason``
                # naming the failure; surface it instead of before/after
                # numbers that would read like a normal regression.
                if entry.get("reason"):
                    lines.append(f"  - {rel}: {entry['reason']}")
                else:
                    lines.append(
                        f"  - {rel}: erosion {entry['erosion_before']} -> "
                        f"{entry['erosion_after']}, verbosity "
                        f"{entry['verbosity_before']} -> {entry['verbosity_after']}"
                    )
            print_warning(
                console,
                f"Quality gate flagged {len(flagged)} file(s) after fixes:\n" + "\n".join(lines),
            )
    except Exception as exc:  # noqa: BLE001 - fail-open: the gate must never fail the run
        # Stage "persist": the payload itself could not be written. Record the
        # failure as an unavailable round when the write path still works, and
        # ALWAYS warn -- a gate failure that surfaces nothing reads as a clean
        # pass, which is the exact hazard #329 describes.
        try:
            gate_p = DeepArtifact.FIX_QUALITY_GATE.at(dd)
            rounds = _load_quality_gate_rounds(gate_p, session_id)
            round_no = iteration if iteration is not None else len(rounds) + 1
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="persist",
                reason=f"{type(exc).__name__}: {exc}",
                session_id=session_id,
                thresholds=thresholds,
            )
        except Exception as inner:  # noqa: BLE001 - nothing left to persist; stay fail-open
            print_warning(
                console,
                f"Quality gate unavailable (round {iteration if iteration is not None else '?'}, "
                f"persist): {type(exc).__name__}: {exc}; could not persist unavailable verdict "
                f"({type(inner).__name__}: {inner})",
            )
