"""Authorized fix cycles, retained-tree evidence, test, commit, CI, and cleanup."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anyio
from rich.markup import escape as escape_markup

from daydream.agent import console
from daydream.artifact_visibility import artifact_dir_for, review_output_path_for
from daydream.config import (
    DEFAULT_GROUP_MAX_SERIAL_ITEMS,
    DEFAULT_GROUP_MAX_WALL_S,
    DEFAULT_QUALITY_GATE_ENABLED,
    DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE,
    DEFAULT_QUALITY_GATE_EROSION_DELTA,
    DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE,
    DEFAULT_QUALITY_GATE_VERBOSITY_DELTA,
    REVIEW_OUTPUT_FILE,
)
from daydream.config_file import _coerce_quality_threshold
from daydream.deep.artifacts import (
    fix_failures_path,
    fix_footprint_path,
    fix_leftover_untracked_path,
    fix_outcomes_path,
    fix_quality_gate_path,
    generated_file_violations_path,
    push_verdict_path,
    recommended_capture_path,
    remote_ci_handoff_path,
    remote_ci_verdict_path,
    stabilization_failed_path,
    test_verdict_path,
)
from daydream.deep.records import stamp_item_uids
from daydream.deep.scope_issues import _resolve_changed_files, enforce_authorized_fix_footprint
from daydream.deep.settings import _resolve_config_value
from daydream.deep.state import DeepState
from daydream.extensions.api import BreakLoop, Stop
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.flows.engine import FlowContext
from daydream.generated_files import is_generated_file, related_manifest_paths
from daydream.git_ops import GitError, GitPathState, IndexSnapshot, WorktreeRollbackSnapshot
from daydream.json_utils import atomic_write_json
from daydream.phases import (
    FIX_VERIFY_ACTIONABLE_VERDICTS,
    FIX_VERIFY_RETARGETABLE_VERDICTS,
    PushAttemptError,
    PushReceipt,
    TestAndHealResult,
    TestAttemptEvidence,
    phase_commit_push,
    phase_fix_parallel,
    phase_test_and_heal,
    phase_test_once,
    phase_verify_recommendations,
    require_empty_staged_index,
    severity_sorted,
)
from daydream.quote_scrub import scrub_smart_quotes_changed_files
from daydream.run_context import resolve_run_context
from daydream.trajectory import (
    DaydreamPhase,
    get_current_recorder,
    host_phase_scope,
    now_iso,
    phase_scope,
    redact_structured_text,
)
from daydream.ui import (
    format_verdict_join,
    print_error,
    print_fix_complete,
    print_info,
    print_success,
    print_verification_summary,
    print_warning,
)
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.remote_ci import RemoteCITarget, RemoteCIVerdict
    from daydream.runner import RunConfig


def _scope_issue_filing(config: RunConfig) -> bool:
    """Resolve the out-of-scope issue-filing opt-in (issue #1056).

    Precedence: 1) ``RunConfig.scope_issue_filing``
    (CLI tier), 2) ``DaydreamFileConfig.scope_issue_filing`` (file-config
    scalar), 3) built-in default ``False`` (no out-of-scope GitHub issues are
    filed unless a repo explicitly opts in).
    """
    if config.scope_issue_filing:
        return True
    file_config = config.file_config
    if file_config is not None and file_config.scope_issue_filing:
        return True
    return False


def _attach_verdicts(items: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach verifier verdicts to feedback items by matching `id` to `issue_id`.

    `phase_fix` reads the `verifier_verdict` / `evidence` / `unverified_assumptions`
    keys (advisory) and augments its prompt when present; items without a matching
    verdict are left untouched. Correctness rests on `normalize_items` having made
    item ids unique, so structural and per-stack findings can no longer collide on
    the same id.

    Args:
        items: Canonical feedback items, each with an integer `id`.
        payload: Verifier output; `payload["verdicts"]` is a list of entries each
            carrying `issue_id`, `verdict`, `evidence`, `unverified_assumptions`.

    Returns:
        The same `items` list (mutated in place) with verdict keys attached to any
        item whose `id` matched a verdict's `issue_id`.
    """
    payload = payload if isinstance(payload, dict) else {"verdicts": []}
    verdict_lookup: dict[int, dict[str, Any]] = {}
    for entry in payload.get("verdicts", []) or []:
        if not isinstance(entry, dict):
            continue
        issue_id = entry.get("issue_id")
        if not isinstance(issue_id, int):
            continue
        assumptions = entry.get("unverified_assumptions")
        verdict_lookup[issue_id] = {
            "verdict": entry.get("verdict", ""),
            "evidence": entry.get("evidence", ""),
            "unverified_assumptions": assumptions if isinstance(assumptions, list) else [],
        }
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        match = verdict_lookup.get(item_id)
        if match is not None:
            item["verifier_verdict"] = match["verdict"]
            item["evidence"] = match["evidence"]
            item["unverified_assumptions"] = match["unverified_assumptions"]
    return items


def _record_fix_preflight_rejection(dd: Path, items: list[dict[str, Any]]) -> None:
    """Record an admitted cycle's blocked items without reflecting unsafe paths."""
    failures = {
        item["item_uid"]: "fix_preflight_rejected: fix cycle did not start"
        for item in items
    }
    try:
        atomic_write_json(fix_failures_path(dd), failures, sort_keys=True)
    except OSError as exc:
        print_error(console, "Fix preflight failure audit failed", str(exc))


async def _step_fix_gate(ctx: FlowContext) -> Stop | None:
    """Fix-apply gate; on accept, load and severity-sort the canonical items."""
    deep_state = DeepState(ctx.data)
    # Fix-apply gate across the two interaction axes. ``--yes`` auto-applies;
    # an unattended run with no assumption declines (safe_default=False) so a
    # piped/CI run never mutates without intent; otherwise prompt.
    decision = resolve_run_context(ctx.run_context).confirm(
        safe_default=False,
        question="Apply fixes now? [y/N]",
        default="n",
        console=console,
    )
    if not decision:
        print_success(console, f"Report written to {deep_state.merged_report}. Exiting.")
        return Stop(0)

    from daydream import git_ops

    # A rejected index preflight must preserve prior artifacts as well as
    # source bytes. Only an admitted run may start a new evidence session.
    try:
        initial_index = require_empty_staged_index(ctx.work)
    except (OSError, git_ops.GitError) as exc:
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    # An accepted gate starts a new evidence session. No prior run's success,
    # patch, or policy audit may be inherited if this run later stops early.
    dd: Path = deep_state.dd
    stale_paths = (
        fix_footprint_path(dd),
        fix_outcomes_path(dd),
        test_verdict_path(dd),
        recommended_capture_path(dd),
        generated_file_violations_path(dd),
        fix_failures_path(dd),
        fix_leftover_untracked_path(dd),
        stabilization_failed_path(dd),
        artifact_dir_for(
            ctx.work.repo,
            session=ctx.artifacts,
            allow_standalone=ctx.allow_standalone_artifacts,
        ) / "recommended.patch",
    )
    try:
        for stale in stale_paths:
            stale.unlink(missing_ok=True)
    except OSError as exc:
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    # Read canonical merged items directly (validated above). Replaces an LLM
    # re-parse of the markdown, which silently dropped structural findings; here
    # they are ordinary tagged items that reach phase_fix like any other.
    items_file: Path = deep_state.items_file
    items: list[dict[str, Any]] = json.loads(items_file.read_text())["items"]
    # A leading ``./`` is a legal path spelling since the grammar relaxed
    # (#572/#573), but the reviewed-diff file set and the git tree are always
    # bare (never ``./``-prefixed). Normalize each finding's file once here so
    # a ``./x`` finding is not misfiled as out-of-scope by the fix gate (and
    # its own fix edit not reverted by the post-fix residual net), both of
    # which compare against bare git-derived paths.
    for _item in items:
        _file = _item.get("file")
        if isinstance(_file, str) and _file.startswith("./"):
            _item["file"] = _file[2:]
    stamp_item_uids(items)
    if not items:
        print_success(console, "No actionable items -- done.")
        return Stop(0)

    # The footprint deliberately admits canonical finding primary/related
    # paths even when they were not themselves in the reviewed diff. This is
    # the explicit authorization that lets a regression test or sibling source
    # file travel with a finding without turning every reviewed file into every
    # fixer's edit scope.
    changed_files = _resolve_changed_files(ctx)

    try:
        stable_head = git_ops.head_sha(ctx.work.repo)
        stable_ref = git_ops.stash_create(ctx.work.repo) or stable_head
        preexisting_untracked = git_ops.snapshot_untracked_paths(
            ctx.work.repo, include_runtime_artifacts=False
        )
        preexisting_gitlinks = git_ops.snapshot_worktree_gitlinks(ctx.work.repo)
        footprint = AuthorizedFixFootprint.build(
            ctx.work.repo, set(changed_files or []), items
        )
    except (OSError, ValueError, git_ops.GitError) as exc:
        _record_fix_preflight_rejection(dd, items)
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    session_id = _current_session_id() or ctx.work.run_id
    state = FixCycleState(
        session_id=session_id,
        stable_ref=stable_ref,
        stable_head=stable_head,
        initial_index=initial_index,
        preexisting_untracked=preexisting_untracked,
        preexisting_gitlinks=preexisting_gitlinks,
        footprint=footprint,
    )
    deep_state.fix_cycle_state = state
    try:
        initial_key = EvidenceKey(_capture_full_delta_key(ctx.work, state), footprint.policy_revision)
        _write_footprint_audit(ctx, state, initial_key)
    except (OSError, git_ops.GitError) as exc:
        _record_fix_preflight_rejection(dd, items)
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    # Severity-ordered (high before medium before low), stable within a
    # tier so equal-severity items keep their canonical merge order.
    deep_state.items = severity_sorted(items)
    return None


async def _step_verify(ctx: FlowContext) -> None:
    """Recommendation verification (#83) + verdict join rendering."""
    deep_state = DeepState(ctx.data)
    dd = deep_state.dd
    items: list[dict[str, Any]] = deep_state.items

    # Recommendation verification (#83). Runs ONLY after the apply-fixes
    # gate accepts, so a declined run (non-interactive / EOF / explicit "N")
    # skips both the verify pass and the recommendation-verdicts.json
    # artifact. A --start-at fix resume still produces verdicts whenever
    # fixes are applied (the gate still runs on resume; accept => verify runs).
    async with phase_scope(DaydreamPhase.VERIFY):
        verdicts_file, verdicts_payload = await phase_verify_recommendations(
            ctx.backend_for("verify"),
            ctx.work,
            merged_items_path=deep_state.items_file,
            deep_dir=dd,
            strategy=ctx.strategy("verification"),
            run_context=ctx.run_context,
        )
    print_verification_summary(console, verdicts_file)

    # Attach verifier verdicts to items by `id` (advisory; phase_fix reads them).
    items = _attach_verdicts(items, verdicts_payload)
    deep_state.items = items
    matched_ids = [i["id"] for i in items if i.get("verifier_verdict") is not None]
    unmatched_ids = [
        i["id"]
        for i in items
        if isinstance(i.get("id"), int)
        and i.get("verifier_verdict") is None
        and i.get("lens") != "structural"
    ]
    # Structural findings are verdict-exempt (in neither matched nor unmatched)
    # but still fixed; itemize them so the "X/Y matched" ratio isn't read as a
    # total that under-counts the items the fix loop iterates.
    structural_ids = [i.get("id") for i in items if i.get("lens") == "structural"]
    # Leftovers (no verdict, non-structural, missing/non-int id) so the
    # buckets always reconcile to len(items); surfaced only when present.
    other_ids = [
        i.get("id")
        for i in items
        if i.get("verifier_verdict") is None
        and i.get("lens") != "structural"
        and not isinstance(i.get("id"), int)
    ]
    console.print(
        format_verdict_join(
            matched=matched_ids,
            unmatched=unmatched_ids,
            structural=structural_ids,
            other=other_ids,
            total=len(items),
        )
    )


async def _capture_quality_before(
    daydream_dir: Path, code_workspace: Path, candidate_paths: set[str] | None
) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort pre-fix quality snapshot; ``(None, reason)`` when unavailable.

    ``analyze_quality`` is pure and deterministic (no backend, no network), but
    any failure here degrades to "gate unavailable" rather than failing the run
    -- the anti-degradation gate is fail-open by design (#315). The failure
    reason is returned so the gate can persist an auditable unavailable entry
    instead of leaving a silent blank (#329). The sync tree-walk runs off the
    event loop so parallel fix fan-out is never blocked by the analyzer
    (#329 / CodeRabbit Finding D). *candidate_paths* (issue #457) scopes the
    snapshot's parse/aggregate to the reviewed ``*.py`` set resolved by
    ``_step_fix`` before the capture; ``None`` keeps the whole-workspace
    snapshot (e.g. a resume that lost the diff context).
    """
    try:
        from daydream.eval.analyzer import analyze_quality

        return await anyio.to_thread.run_sync(
            partial(analyze_quality, daydream_dir, candidate_paths, code_workspace=code_workspace)
        ), None
    except Exception as exc:  # noqa: BLE001 -- fail-open: never fail the run
        return None, f"{type(exc).__name__}: {exc}"


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
    erosion_threshold: float,
    verbosity_threshold: float,
    erosion_absolute_threshold: float,
    verbosity_absolute_threshold: float,
) -> bool:
    """Whether a fixed file regressed past a threshold (#315).

    A file is flagged when its delta exceeds the threshold (both sides
    defined), or on its absolute AFTER value vs the absolute threshold when the
    BEFORE metric is undefined -- no functions / no non-blank lines pre-fix --
    but the AFTER metric is numeric. The absolute yardstick is a SEPARATE knob
    from the delta one (#329 / CodeRabbit Finding D): an undefined baseline has
    no delta, so a delta threshold is the wrong ruler for the absolute
    comparison. The absolute fallback fires on ANY file with an undefined
    baseline, not just new ones, so an EXISTING file that gains a CC>10
    function (erosion ``None`` pre-fix) is still caught. A metric with both
    sides undefined never flags.
    """
    if erosion_delta is not None and erosion_delta > erosion_threshold:
        return True
    if verbosity_delta is not None and verbosity_delta > verbosity_threshold:
        return True
    if erosion_before is None and erosion_after is not None and erosion_after > erosion_absolute_threshold:
        return True
    if verbosity_before is None and verbosity_after is not None and verbosity_after > verbosity_absolute_threshold:
        return True
    return False


def _quality_gate_threshold(config: RunConfig, attr: str, default: float) -> float:
    """Resolve a quality-gate threshold (delta or absolute), degrading invalid values to *default*.

    Mirrors ``_uncovered_sweep_max_files``: ``RunConfig`` field > file-config
    scalar > default, then the same finite non-negative guard the file-config
    parser applies (#329 / Finding 7). A negative threshold flags every
    unchanged file (a zero delta exceeds it); a NaN/infinite one disables the
    metric (every comparison against it is False) and writes a non-standard
    ``NaN`` to JSON. Both degrade to the named default so a directly-constructed
    ``RunConfig`` / ``DaydreamFileConfig`` cannot smuggle an invalid floor past
    the parser.
    """
    value = _resolve_config_value(config, attr, default)
    coerced = _coerce_quality_threshold(value)
    return coerced if coerced is not None else default


def _current_session_id() -> str | None:
    """Session id binding the quality-gate artifact to the current run."""
    recorder = get_current_recorder()
    return recorder.session_id if recorder is not None else None


def _load_quality_gate_rounds(gate_p: Path, session_id: str | None) -> list[dict[str, Any]]:
    """Load prior rounds for the CURRENT session, or start fresh when rebound.

    Rounds are carried forward only when the artifact's stored ``session_id``
    matches the current run's session (issue #329 / Finding 5): a ``--start-at
    fix`` resume of the SAME session appends rather than clobbers, while an
    artifact left by a DIFFERENT run is discarded with a warning so the first
    run's verdicts can never be archived as the new run's (corrupting the
    manifest / SQLite audit history). An artifact with no stored session is
    treated as belonging to another session.

    Never raises, and a malformed artifact is never treated as authoritative
    prior rounds (issue #329 / Finding 6): invalid JSON, a non-object payload
    (``[]``, ``42``, ...), a missing ``rounds`` list, or non-object round
    entries all degrade to an empty round list WITH a warning, so the current
    run repairs the artifact instead of silently losing the gate.
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


def _persist_quality_gate_unavailable(
    *,
    gate_p: Path,
    rounds: list[dict[str, Any]],
    round_no: int,
    stage: str,
    reason: str,
    session_id: str | None,
    erosion_delta_threshold: float,
    verbosity_delta_threshold: float,
    erosion_absolute_threshold: float,
    verbosity_absolute_threshold: float,
) -> None:
    """Persist an auditable ``unavailable`` round entry for THIS round.

    Any existing entry with the same round number is dropped first, so an
    unavailable verdict supersedes a stale successful one from the same round
    (e.g. a ``--start-at fix`` resume). Never raises.
    """
    rounds = [r for r in rounds if r.get("round") != round_no]
    rounds.append({"round": round_no, "unavailable": {"stage": stage, "reason": reason}})
    gate_p.write_text(
        json.dumps(
            {
                "enabled": True,
                "erosion_delta_threshold": erosion_delta_threshold,
                "verbosity_delta_threshold": verbosity_delta_threshold,
                "erosion_absolute_threshold": erosion_absolute_threshold,
                "verbosity_absolute_threshold": verbosity_absolute_threshold,
                "session_id": session_id,
                "rounds": rounds,
            },
            indent=2,
        )
    )
    print_warning(
        console,
        f"Quality gate unavailable (round {round_no}, {stage}): {reason}",
    )


async def _evaluate_quality_gate(
    *,
    enabled: bool,
    erosion_delta_threshold: float,
    verbosity_delta_threshold: float,
    erosion_absolute_threshold: float,
    verbosity_absolute_threshold: float,
    daydream_dir: Path,
    code_workspace: Path,
    dd: Path,
    candidates: set[str] | None,
    before: dict[str, Any] | None,
    before_unavailable_reason: str | None,
    iteration: int | None,
) -> None:
    """Compute and persist the fix-phase quality-gate verdict (issue #315).

    Fail-open by design: never raises, never stops the run. Every payload is
    bound to the current run's ``session_id`` so a later archive step cannot
    attribute another run's verdict to this one (#329). When disabled, writes
    ``{"enabled": false}``. When the gate would run but cannot be evaluated --
    a pre-fix capture failure, a post-fix capture failure, a changed-file
    enumeration failure (``candidates`` is ``None``, #329 / Finding 6), or a
    persist failure -- an explicit ``unavailable`` round entry is persisted for
    THIS round with the failed stage and reason, superseding any stale verdict
    for that round and keeping the failure auditable instead of reading as a
    clean gate. Each ``_step_fix`` invocation appends one ``rounds`` entry
    (keyed by the flow's loop iteration when present, else the next sequence
    number), so a resume or loop preserves the per-round trend. Flagged files
    are surfaced as warnings with their before/after numbers. A candidate the
    pre-fix snapshot did not cover -- the snapshot is scoped to the reviewed
    ``*.py`` set (#457), so an out-of-diff secondary edit the fix pass made has
    no baseline -- is recorded as flagged with a missing-baseline reason: an
    unverifiable edit must not read as a clean pass. The post-fix sync
    tree-walk runs off the event loop so parallel fix fan-out is never blocked
    by the analyzer (#329 / CodeRabbit Finding D).
    """
    session_id = _current_session_id()
    try:
        gate_p = fix_quality_gate_path(dd)
        if not enabled:
            gate_p.write_text(
                json.dumps({"enabled": False, "session_id": session_id}, indent=2)
            )
            return
        rounds = _load_quality_gate_rounds(gate_p, session_id)
        round_no = iteration if iteration is not None else len(rounds) + 1
        if candidates is None:
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="candidates",
                reason="could not enumerate files changed by the fix pass against the pre-fix snapshot",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
            return
        if before is None:
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="before",
                reason=before_unavailable_reason or "pre-fix quality snapshot unavailable",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
            return
        try:
            from daydream.eval.analyzer import analyze_quality

            # Issue #457: scope the post-fix capture to the candidate set
            # (reviewed *.py + files changed by the fix pass). ``candidates``
            # is always a set here -- the ``candidates is None`` branch above
            # returned already.
            after = await anyio.to_thread.run_sync(
                partial(analyze_quality, daydream_dir, candidates, code_workspace=code_workspace)
            )
        except Exception as exc:  # noqa: BLE001 -- fail-open: the gate must never fail the run
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="after",
                reason=f"{type(exc).__name__}: {exc}",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
            return
        before_per_file: dict[str, Any] = before.get("per_file") or {}
        after_per_file: dict[str, Any] = after.get("per_file") or {}

        per_file: dict[str, dict[str, Any]] = {}
        for rel in sorted(candidates):
            before_entry = before_per_file.get(rel)
            after_entry = after_per_file.get(rel)
            if before_entry is None and after_entry is None:
                continue
            # Issue #329 / Finding 5: a candidate that parsed pre-fix but is
            # MISSING from the post-fix analyzer output is unparseable after the
            # fix (analyze_quality omits malformed files). Recording null
            # after-metrics with ``flagged=false`` would read as a clean
            # verdict, so mark it explicitly unavailable and flagged -- a fix
            # that breaks a file is a regression, never a pass. Still fail-open:
            # never raises, never stops the run.
            if before_entry is not None and after_entry is None:
                per_file[rel] = {
                    "erosion_before": before_entry.get("erosion"),
                    "erosion_after": None,
                    "erosion_delta": None,
                    "verbosity_before": before_entry.get("verbosity"),
                    "verbosity_after": None,
                    "verbosity_delta": None,
                    "unparseable": True,
                    "flagged": True,
                    "reason": "file missing from post-fix analyzer output (unparseable?)",
                }
                continue
            # Issue #329 / #457: the pre-fix snapshot is scoped to the reviewed
            # diff's ``*.py`` set, so a candidate the fix pass edited that was
            # NOT in the reviewed diff -- a secondary edit that survived the
            # residual net, e.g. a newly-created untracked ``*.py`` -- has no
            # before baseline. A missing baseline must not read as a clean
            # pass: the delta is unknowable, so record the file explicitly
            # flagged with its reason instead of silently falling into the
            # absolute-only fallback, which can miss the exact delta regression
            # #329 added changed_after_fix for. Still fail-open: never raises,
            # never stops the run.
            if before_entry is None and after_entry is not None:
                per_file[rel] = {
                    "erosion_before": None,
                    "erosion_after": after_entry.get("erosion"),
                    "erosion_delta": None,
                    "verbosity_before": None,
                    "verbosity_after": after_entry.get("verbosity"),
                    "verbosity_delta": None,
                    "flagged": True,
                    "reason": (
                        "missing pre-fix baseline: file edited by the fix pass but not "
                        "covered by the pre-fix quality snapshot"
                    ),
                }
                continue
            erosion_before = before_entry.get("erosion") if before_entry is not None else None
            erosion_after = after_entry.get("erosion") if after_entry is not None else None
            verbosity_before = before_entry.get("verbosity") if before_entry is not None else None
            verbosity_after = after_entry.get("verbosity") if after_entry is not None else None
            erosion_delta = _quality_delta(erosion_before, erosion_after)
            verbosity_delta = _quality_delta(verbosity_before, verbosity_after)
            per_file[rel] = {
                "erosion_before": erosion_before,
                "erosion_after": erosion_after,
                "erosion_delta": erosion_delta,
                "verbosity_before": verbosity_before,
                "verbosity_after": verbosity_after,
                "verbosity_delta": verbosity_delta,
                "flagged": _quality_flagged(
                    erosion_before=erosion_before,
                    erosion_after=erosion_after,
                    erosion_delta=erosion_delta,
                    verbosity_before=verbosity_before,
                    verbosity_after=verbosity_after,
                    verbosity_delta=verbosity_delta,
                    erosion_threshold=erosion_delta_threshold,
                    verbosity_threshold=verbosity_delta_threshold,
                    erosion_absolute_threshold=erosion_absolute_threshold,
                    verbosity_absolute_threshold=verbosity_absolute_threshold,
                ),
            }
        rounds = [r for r in rounds if r.get("round") != round_no]
        rounds.append({"round": round_no, "per_file": per_file})
        payload = {
            "enabled": True,
            "erosion_delta_threshold": erosion_delta_threshold,
            "verbosity_delta_threshold": verbosity_delta_threshold,
            "erosion_absolute_threshold": erosion_absolute_threshold,
            "verbosity_absolute_threshold": verbosity_absolute_threshold,
            "session_id": session_id,
            "rounds": rounds,
        }
        gate_p.write_text(json.dumps(payload, indent=2))
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
            gate_p = fix_quality_gate_path(dd)
            rounds = _load_quality_gate_rounds(gate_p, session_id)
            round_no = iteration if iteration is not None else len(rounds) + 1
            _persist_quality_gate_unavailable(
                gate_p=gate_p,
                rounds=rounds,
                round_no=round_no,
                stage="persist",
                reason=f"{type(exc).__name__}: {exc}",
                session_id=session_id,
                erosion_delta_threshold=erosion_delta_threshold,
                verbosity_delta_threshold=verbosity_delta_threshold,
                erosion_absolute_threshold=erosion_absolute_threshold,
                verbosity_absolute_threshold=verbosity_absolute_threshold,
            )
        except Exception as inner:  # noqa: BLE001 - nothing left to persist; stay fail-open
            print_warning(
                console,
                f"Quality gate unavailable (round {iteration if iteration is not None else '?'}, "
                f"persist): {type(exc).__name__}: {exc}; could not persist unavailable verdict "
                f"({type(inner).__name__}: {inner})",
            )

MAX_POST_TEST_STABILIZATION_PASSES = 2
ACTIONABLE_VERDICTS = frozenset(FIX_VERIFY_ACTIONABLE_VERDICTS)
RETARGETABLE_VERDICTS = frozenset(FIX_VERIFY_RETARGETABLE_VERDICTS)


@dataclass(frozen=True)
class EvidenceKey:
    """Tree and policy identity required by verifier evidence."""

    tree_key: str
    policy_revision: int


@dataclass(frozen=True)
class RetainedTreeSnapshot:
    """Authorized retained patch plus full observed-tree identity."""

    paths: frozenset[str]
    states: tuple[GitPathState, ...]
    tree_key: str
    verifier_patch: str
    recommended_patch: bytes


@dataclass
class FixCycleState:
    """One accepted fix gate's stable policy, baseline, and evidence."""

    session_id: str
    stable_ref: str
    stable_head: str
    initial_index: IndexSnapshot
    preexisting_untracked: dict[str, GitPathState]
    preexisting_gitlinks: tuple[GitPathState, ...]
    footprint: AuthorizedFixFootprint
    latest_retained: RetainedTreeSnapshot | None = None
    verifier_key: EvidenceKey | None = None
    test_evidence: TestAttemptEvidence | None = None
    last_fix_target_by_uid: dict[str, str] = field(default_factory=dict)


def _fix_cycle_state(ctx: FlowContext) -> FixCycleState:
    return DeepState(ctx.data).fix_cycle_state


def _capture_full_delta_key(work: WorkContext, state: FixCycleState) -> str:
    from daydream import git_ops

    return git_ops.tree_key(
        git_ops.snapshot_worktree_delta(
            work.repo,
            state.stable_ref,
            preexisting_untracked=state.preexisting_untracked,
            preexisting_gitlinks=state.preexisting_gitlinks,
        )
    )


def capture_retained_tree(work: WorkContext, state: FixCycleState) -> RetainedTreeSnapshot:
    """Capture the authorized HEAD delta while keying the run-relative tree.

    ``stable_ref`` includes pre-gate tracked edits so it remains the authority
    for mutation attribution, rollback, and test identity.  Commit selection is
    deliberately relative to the original HEAD: authorized reviewed edits that
    predate the gate are part of the result even when no fixer touches them.
    Pre-existing untracked owner files remain protected even when a finding
    names them; authorization cannot silently enroll that private draft in a
    commit. New related files created after the gate are still retained.
    """
    from daydream import git_ops

    changed = set(git_ops.changed_paths_z(work.repo, state.stable_head))
    paths = frozenset(
        (changed & set(state.footprint.run_allowed_paths)) - set(state.preexisting_untracked)
    )
    states = git_ops.snapshot_worktree_paths(work.repo, paths)
    full_states = git_ops.snapshot_worktree_delta(
        work.repo,
        state.stable_ref,
        preexisting_untracked=state.preexisting_untracked,
        preexisting_gitlinks=state.preexisting_gitlinks,
    )
    recommended = git_ops.build_recommended_patch_strict(
        work.repo, state.stable_head, paths
    )
    return RetainedTreeSnapshot(
        paths=paths,
        states=states,
        tree_key=git_ops.tree_key(full_states),
        verifier_patch=recommended.decode("utf-8", errors="replace"),
        recommended_patch=recommended,
    )


def _evidence_payload(key: EvidenceKey) -> dict[str, Any]:
    return {"tree_key": key.tree_key, "policy_revision": key.policy_revision}


def _write_footprint_audit(ctx: FlowContext, state: FixCycleState, key: EvidenceKey) -> None:
    deep_state = DeepState(ctx.data)
    atomic_write_json(
        fix_footprint_path(deep_state.dd),
        state.footprint.audit_payload(state.session_id, evidence_key=_evidence_payload(key)),
    )


def _persist_stabilization_failure(ctx: FlowContext, state: FixCycleState, reason: str) -> None:
    deep_state = DeepState(ctx.data)
    atomic_write_json(
        stabilization_failed_path(deep_state.dd),
        {"session_id": state.session_id, "reason": reason},
    )


def _round_dispatch_items(ctx: FlowContext, canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive THIS round's fix dispatch set from prior fix-verify outcomes (#744).

    Round 1 (iteration unset or 1) dispatches the full canonical list. Round
    N+1 re-dispatches ONLY the actionable subset — findings whose prior-round
    verdict was ``unresolved`` / ``wrong_target`` / ``regressed`` — each
    carrying the verifier's reason forward into the fix prompt. A
    ``wrong_target(path)`` retarget applies only when *path* stays inside the
    allowed edit set (``changed_files ∪ {finding files}`` — the same set the
    residual net and allowed-files clause use); otherwise the item keeps its
    original file and the retarget is treated as not actionable (the #336 scope
    net is never widened). Never mutates the canonical items (copies only).
    """
    deep_state = DeepState(ctx.data)
    iteration = deep_state.iteration
    outcomes = deep_state.fix_outcomes or {}
    state = _fix_cycle_state(ctx)
    if iteration in (None, 1) or not outcomes:
        initial_dispatch = [dict(i) for i in canonical]
        for item in initial_dispatch:
            uid = item.get("item_uid")
            target = item.get("file")
            if isinstance(uid, str) and isinstance(target, str):
                state.last_fix_target_by_uid[uid] = target
        return initial_dispatch
    round_number = iteration if isinstance(iteration, int) else 1
    dispatched: list[dict[str, Any]] = []
    for item in canonical:
        uid = item.get("item_uid")
        outcome = outcomes.get(uid) if isinstance(uid, str) else None
        if not outcome or outcome.get("verdict") not in ACTIONABLE_VERDICTS:
            continue
        copy = dict(item)
        if outcome.get("verdict") in RETARGETABLE_VERDICTS:
            accepted = state.footprint.accept_retarget(
                ctx.work.repo,
                str(uid),
                outcome.get("path"),
                phase="fix",
                round_number=round_number,
            )
            if accepted is not None:
                copy["file"] = accepted
                copy["fix_verify_path"] = accepted
        copy["fix_verify_verdict"] = outcome.get("verdict")
        copy["fix_verify_reason"] = outcome.get("reason") or ""
        target = copy.get("file")
        if isinstance(uid, str) and isinstance(target, str):
            state.last_fix_target_by_uid[uid] = target
        dispatched.append(copy)
    return dispatched




def _round_rollback_snapshot(state: FixCycleState, work: WorkContext) -> WorktreeRollbackSnapshot:
    """Capture one exact rollback point for all authorized group paths."""
    from daydream import git_ops

    return WorktreeRollbackSnapshot(
        ref=state.stable_ref,
        index=git_ops.snapshot_index(work.repo),
        path_states=git_ops.snapshot_worktree_paths(
            work.repo, state.footprint.run_allowed_paths
        ),
        untracked=git_ops.snapshot_untracked_paths(
            work.repo, include_runtime_artifacts=False
        ),
    )


def _strict_scope_and_scrub(
    ctx: FlowContext,
    state: FixCycleState,
    *,
    phase: str,
    round_number: int | None,
) -> bool:
    """Apply the run-wide guard and quote scrub, returning whether bytes changed."""
    deep_state = DeepState(ctx.data)
    from daydream import git_ops

    before = _capture_full_delta_key(ctx.work, state)
    generated_restores: list[str] = []
    for path in git_ops.changed_paths_z(ctx.work.repo, state.stable_ref):
        if path.startswith(".daydream/") or path == REVIEW_OUTPUT_FILE:
            continue
        try:
            baseline = git_ops.show(ctx.work.repo, state.stable_ref, path)
        except git_ops.GitError:
            absolute = ctx.work.repo / path
            try:
                current = absolute.read_bytes()
            except OSError:
                current = None
            if current is not None and is_generated_file(path, current):
                state.footprint.authorize_new_generated(
                    ctx.work.repo,
                    path,
                    phase=phase,
                    round_number=round_number,
                    reason="new generated output approved by generated-file policy",
                )
            continue
        if is_generated_file(path, baseline):
            generated_restores.append(path)
            generated_restores.extend(related_manifest_paths(path))
    if generated_restores:
        restore_paths = sorted(set(generated_restores))
        git_ops.restore_worktree_paths_from_ref(
            ctx.work.repo, state.stable_ref, restore_paths
        )
        for path in restore_paths:
            state.footprint.record_git_event(
                action="restore",
                path=path,
                origin="guard",
                phase=phase,
                round_number=round_number,
                reason="restored an edit to existing generated output or its manifest",
            )
        atomic_write_json(
            generated_file_violations_path(deep_state.dd),
            {
                "session_id": state.session_id,
                "violations": restore_paths,
                "phase": phase,
                "round_number": round_number,
            },
            sort_keys=True,
        )
    enforced = enforce_authorized_fix_footprint(
        ctx.work,
        state.stable_ref,
        state.footprint,
        preexisting_untracked=state.preexisting_untracked,
        preexisting_gitlinks=state.preexisting_gitlinks,
        phase=phase,
        round_number=round_number,
        file_scope_issues=_scope_issue_filing(ctx.config),
        auth=ctx.github_execution.auth,
    )
    scrub_smart_quotes_changed_files(
        ctx.work.repo,
        sorted(enforced.retained_paths),
        pre_fix_ref=state.stable_ref,
    )
    after = _capture_full_delta_key(ctx.work, state)
    return bool(generated_restores) or enforced.mutated or before != after


def _enforce_terminal_confinement(
    ctx: FlowContext,
    state: FixCycleState,
    *,
    phase: str,
    round_number: int | None,
) -> str | None:
    """Restore all out-of-run/protected state and durably audit a failed exit."""
    try:
        enforce_authorized_fix_footprint(
            ctx.work,
            state.stable_ref,
            state.footprint,
            preexisting_untracked=state.preexisting_untracked,
            preexisting_gitlinks=state.preexisting_gitlinks,
            phase=phase,
            round_number=round_number,
            file_scope_issues=_scope_issue_filing(ctx.config),
            auth=ctx.github_execution.auth,
        )
        key = EvidenceKey(
            _capture_full_delta_key(ctx.work, state),
            state.footprint.policy_revision,
        )
        _write_footprint_audit(ctx, state, key)
    except Exception as exc:
        return str(exc)
    return None


def _stabilization_stop(
    ctx: FlowContext,
    state: FixCycleState,
    reason: str,
    *,
    round_number: int | None,
) -> Stop:
    """Fail closed after finalization while still restoring and auditing scope."""
    confinement_error = _enforce_terminal_confinement(
        ctx,
        state,
        phase="post_test_failure",
        round_number=round_number,
    )
    if confinement_error is not None:
        reason = f"{reason}; confinement failed: {confinement_error}"
    try:
        _persist_stabilization_failure(ctx, state, reason)
    except OSError as exc:
        print_error(console, "Stabilization failure audit failed", str(exc))
    return Stop(1)


async def _step_fix_authorized(ctx: FlowContext, state: FixCycleState) -> Stop | None:
    """Run one policy-bound fix round using its own complete rollback point."""
    deep_state = DeepState(ctx.data)
    items = _round_dispatch_items(ctx, deep_state.items)
    deep_state.fix_round_items = list(items)
    if not items:
        return None
    try:
        round_snapshot = _round_rollback_snapshot(state, ctx.work)
    except Exception as exc:
        print_error(console, "Fix snapshot failed", str(exc))
        return Stop(1)
    config = ctx.config
    quality_enabled = _resolve_config_value(
        config, "quality_gate_enabled", DEFAULT_QUALITY_GATE_ENABLED
    )
    reviewed_python = {
        path for path in (_resolve_changed_files(ctx) or []) if path.endswith(".py")
    }
    if quality_enabled:
        quality_before, quality_unavailable = await _capture_quality_before(
            artifact_dir_for(
                ctx.work.repo,
                session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            ),
            ctx.work.repo,
            reviewed_python,
        )
    else:
        quality_before, quality_unavailable = None, None
    exploration_dir = deep_state.exploration_dir_or_none
    exploration_dir = exploration_dir if isinstance(exploration_dir, Path) else None
    test_map_path = exploration_dir / "test-map.json" if exploration_dir else None
    intent_p: Path = deep_state.intent_path
    grounded = config.start_at not in ("per-stack", "merge", "fix")
    async with phase_scope(DaydreamPhase.FIX):
        try:
            failures = await phase_fix_parallel(
                ctx.backend_for("fix"),
                ctx.work,
                items,
                intent_path=intent_p if grounded and intent_p.exists() else None,
                group_max_wall_s=_resolve_config_value(
                    config, "group_max_wall_s", DEFAULT_GROUP_MAX_WALL_S
                ),
                group_max_serial_items=_resolve_config_value(
                    config, "group_max_serial_items", DEFAULT_GROUP_MAX_SERIAL_ITEMS
                ),
                exploration_dir=exploration_dir,
                test_map_path=test_map_path,
                footprint=state.footprint,
                round_snapshot=round_snapshot,
                run_context=ctx.run_context,
            )
        except Exception as exc:
            confinement_error = _enforce_terminal_confinement(
                ctx,
                state,
                phase="fix_failure",
                round_number=deep_state.iteration,
            )
            print_error(console, "Fix failed", str(exc))
            if confinement_error is not None:
                print_error(console, "Fix failure confinement failed", confinement_error)
            return Stop(1)
    budget_prefix = "file_group_budget_exceeded:"
    exception_failures = {
        path: reason
        for path, reason in failures.items()
        if not reason.startswith(budget_prefix)
    }
    failures_artifact = fix_failures_path(deep_state.dd)
    try:
        if failures:
            atomic_write_json(failures_artifact, failures, sort_keys=True)
        else:
            failures_artifact.unlink(missing_ok=True)
    except Exception as exc:
        confinement_error = _enforce_terminal_confinement(
            ctx,
            state,
            phase="fix_failure",
            round_number=deep_state.iteration,
        )
        print_error(console, "Fix failure audit failed", str(exc))
        if confinement_error is not None:
            print_error(console, "Fix failure confinement failed", confinement_error)
        return Stop(1)
    if exception_failures:
        from daydream import git_ops

        confinement_error = _enforce_terminal_confinement(
            ctx,
            state,
            phase="fix_failure",
            round_number=deep_state.iteration,
        )
        artifact_errors: list[str] = []
        try:
            leftover = sorted(
                set(
                    git_ops.snapshot_untracked_paths(
                        ctx.work.repo, include_runtime_artifacts=False
                    )
                )
                - set(state.preexisting_untracked)
            )
            if leftover:
                atomic_write_json(fix_leftover_untracked_path(deep_state.dd), leftover)
        except Exception as exc:
            artifact_errors.append(str(exc))
        print_warning(
            console,
            "Failed fix groups were restored; this run will not commit successful "
            "sibling-group edits: " + ", ".join(sorted(exception_failures)),
        )
        if confinement_error is not None:
            print_error(console, "Fix failure confinement failed", confinement_error)
        if artifact_errors:
            print_error(console, "Fix failure audit failed", "; ".join(artifact_errors))
        return Stop(1)
    try:
        _strict_scope_and_scrub(
            ctx,
            state,
            phase="fix",
            round_number=deep_state.iteration,
        )
        snapshot = capture_retained_tree(ctx.work, state)
    except Exception as exc:
        confinement_error = _enforce_terminal_confinement(
            ctx,
            state,
            phase="fix_failure",
            round_number=deep_state.iteration,
        )
        print_error(console, "Fix scope enforcement failed", str(exc))
        if confinement_error is not None:
            print_error(console, "Fix failure confinement failed", confinement_error)
        return Stop(1)
    deep_state.fix_round_snapshot = snapshot
    try:
        await _evaluate_quality_gate(
            enabled=quality_enabled,
            erosion_delta_threshold=_quality_gate_threshold(
                config, "quality_gate_erosion_delta", DEFAULT_QUALITY_GATE_EROSION_DELTA
            ),
            verbosity_delta_threshold=_quality_gate_threshold(
                config, "quality_gate_verbosity_delta", DEFAULT_QUALITY_GATE_VERBOSITY_DELTA
            ),
            erosion_absolute_threshold=_quality_gate_threshold(
                config, "quality_gate_erosion_absolute", DEFAULT_QUALITY_GATE_EROSION_ABSOLUTE
            ),
            verbosity_absolute_threshold=_quality_gate_threshold(
                config, "quality_gate_verbosity_absolute", DEFAULT_QUALITY_GATE_VERBOSITY_ABSOLUTE
            ),
            daydream_dir=artifact_dir_for(
                ctx.work.repo,
                session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            ),
            code_workspace=ctx.work.repo,
            dd=deep_state.dd,
            candidates={path for path in snapshot.paths if path.endswith(".py")}
            | {
                str(item["file"])
                for item in deep_state.items
                if isinstance(item.get("file"), str) and str(item["file"]).endswith(".py")
            },
            before=quality_before,
            before_unavailable_reason=quality_unavailable,
            iteration=deep_state.iteration,
        )
    except Exception as exc:
        confinement_error = _enforce_terminal_confinement(
            ctx,
            state,
            phase="fix_failure",
            round_number=deep_state.iteration,
        )
        print_error(console, "Fix quality evaluation failed", str(exc))
        if confinement_error is not None:
            print_error(console, "Fix failure confinement failed", confinement_error)
        return Stop(1)
    return None


async def _step_fix(ctx: FlowContext) -> Stop | None:
    """Run one policy-bound fix round against the stable fix-cycle baseline."""
    return await _step_fix_authorized(ctx, _fix_cycle_state(ctx))




async def verify_retained_tree(
    ctx: FlowContext,
    snapshot: RetainedTreeSnapshot,
    items: list[dict[str, Any]],
    *,
    pass_number: int,
) -> dict[str, dict[str, Any]]:
    """Verify all canonical findings and join numeric wire ids to durable uids."""
    from daydream.phases import phase_fix_verify

    async with phase_scope(DaydreamPhase.VERIFY):
        verdicts = await phase_fix_verify(
            ctx.backend_for("verify"),
            ctx.work,
            items,
            snapshot.verifier_patch,
            round_number=pass_number,
            run_context=ctx.run_context,
        )
    by_id = {
        item.get("id"): item
        for item in items
        if isinstance(item.get("id"), int) and isinstance(item.get("item_uid"), str)
    }
    state = _fix_cycle_state(ctx)
    outcomes: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        item = by_id.get(verdict.get("issue_id"))
        if item is None:
            continue
        stored = dict(verdict)
        stored["issue_id"] = item["id"]
        uid = item["item_uid"]
        if stored.get("verdict") == "resolved":
            target = state.last_fix_target_by_uid.get(uid)
            if target is not None:
                stored["path"] = target
        outcomes[uid] = stored
    return outcomes


def _persist_fix_outcomes_current(
    ctx: FlowContext,
    state: FixCycleState,
    key: EvidenceKey,
    outcomes: dict[str, dict[str, Any]],
) -> None:
    deep_state = DeepState(ctx.data)
    atomic_write_json(
        fix_outcomes_path(deep_state.dd),
        {
            "session_id": state.session_id,
            "evidence_key": _evidence_payload(key),
            "outcomes": outcomes,
        },
        sort_keys=True,
    )


async def _step_fix_verify_authorized(
    ctx: FlowContext, state: FixCycleState
) -> BreakLoop | Stop | None:
    deep_state = DeepState(ctx.data)
    snapshot = deep_state.fix_round_snapshot
    if snapshot is None:
        try:
            snapshot = capture_retained_tree(ctx.work, state)
        except Exception as exc:
            confinement_error = _enforce_terminal_confinement(
                ctx,
                state,
                phase="fix_verify_failure",
                round_number=deep_state.iteration,
            )
            print_error(console, "Fix verification capture failed", str(exc))
            if confinement_error is not None:
                print_error(console, "Fix failure confinement failed", confinement_error)
            return Stop(1)
    iteration = deep_state.iteration
    round_number = iteration if isinstance(iteration, int) else 1
    try:
        outcomes = await verify_retained_tree(
            ctx, snapshot, deep_state.items, pass_number=round_number
        )
        key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
        state.latest_retained = snapshot
        state.verifier_key = key
        deep_state.fix_outcomes = outcomes
        _persist_fix_outcomes_current(ctx, state, key, outcomes)
        _write_footprint_audit(ctx, state, key)
    except Exception as exc:
        confinement_error = _enforce_terminal_confinement(
            ctx,
            state,
            phase="fix_verify_failure",
            round_number=round_number,
        )
        print_error(console, "Fix verification failed", str(exc))
        if confinement_error is not None:
            print_error(console, "Fix failure confinement failed", confinement_error)
        return Stop(1)
    actionable = _actionable_verdicts(outcomes)
    if actionable and iteration not in (None, 3):
        return None
    _render_fix_outcome_summary(deep_state.dd, deep_state.items, outcomes)
    if actionable:
        return Stop(1)
    return BreakLoop()


async def _step_fix_verify(ctx: FlowContext) -> BreakLoop | Stop | None:
    """Verify every canonical finding against the complete retained tree."""
    return await _step_fix_verify_authorized(ctx, _fix_cycle_state(ctx))


def _actionable_verdicts(outcomes: dict[Any, dict[str, Any]]) -> list[str]:
    """Verdicts that schedule re-dispatch in a later round (issue #744)."""
    return [
        v["verdict"]
        for v in outcomes.values()
        if v.get("verdict") in ACTIONABLE_VERDICTS
    ]




def _render_fix_outcome_summary(
    dd: Path,
    items: list[dict[str, Any]],
    outcomes: dict[Any, dict[str, Any]],
) -> None:
    """Render the terminal per-finding fix-verdict lines (issue #744).

    Single production call site exercising ``print_fix_complete``'s verdict
    gating, so the honest ``resolved``/attempted-not-fixed branches are
    actually reached rather than dead next to a separate aggregate summary.
    ``items`` is the final round's dispatch set, so numbering reuses the same
    1-based counters printed during that round's fix turn: each terminal
    verdict line matches the neutral "Fix attempted" line for a finding
    dispatched that round. ``fix-outcomes.json`` (keyed by canonical id)
    remains the durable record, so a finding resolved in an earlier round but
    not re-dispatched may be absent from this terminal render.
    """
    if not outcomes:
        return
    numbering_by_id = {
        item["id"]: num
        for num, item in enumerate(items, start=1)
        if isinstance(item.get("id"), int)
    }
    numbering_by_uid = {
        item["item_uid"]: num
        for num, item in enumerate(items, start=1)
        if isinstance(item.get("item_uid"), str)
    }
    total = len(items)
    for outcome_key, verdict in outcomes.items():
        num = (
            numbering_by_uid.get(outcome_key)
            if isinstance(outcome_key, str)
            else numbering_by_id.get(outcome_key)
        )
        if num is None:
            num = numbering_by_id.get(verdict.get("issue_id"))
        if num is None:
            continue
        print_fix_complete(console, num, total, outcome=verdict.get("verdict"))


def _test_attempt_payload(attempt: TestAttemptEvidence) -> dict[str, Any]:
    return {
        "session_id": attempt.session_id,
        "kind": attempt.kind,
        "command": list(attempt.command) if attempt.command is not None else "agent-fallback",
        "passed": attempt.passed,
        "input_tree_key": attempt.input_tree_key,
        "output_tree_key": attempt.output_tree_key,
    }


def _persist_test_verdict(
    ctx: FlowContext,
    state: FixCycleState,
    *,
    passed: bool,
    ignored: bool,
    attempts: list[TestAttemptEvidence],
) -> None:
    deep_state = DeepState(ctx.data)
    from daydream.remote_ci import local_host_facts

    atomic_write_json(
        test_verdict_path(deep_state.dd),
        {
            "session_id": state.session_id,
            "passed": passed,
            "ignored": ignored,
            "retries": max(0, len(attempts) - 1),
            "attempts": [_test_attempt_payload(attempt) for attempt in attempts],
            "local_host": local_host_facts(),
        },
        sort_keys=True,
    )


def _authorize_final_red_override(ctx: FlowContext) -> bool:
    """Require a fresh interactive decision for a changed-tree red retest."""
    run_context = resolve_run_context(ctx.run_context)
    policy = run_context.policy
    if policy.assume is not None or not policy.interactive:
        return False
    return run_context.confirm(
        safe_default=False,
        question="Final no-heal validation is still red. Ignore and continue? [y/N]",
        default="n",
        console=console,
    )


async def finalize_retained_tree_after_test(
    ctx: FlowContext, result: TestAndHealResult
) -> Stop | None:
    """Strictly stabilize post-heal state in at most two guard passes."""
    deep_state = DeepState(ctx.data)
    state = _fix_cycle_state(ctx)
    attempts = list(result.attempts)
    if not attempts:
        return _stabilization_stop(
            ctx, state, "test produced no evidence", round_number=None
        )
    evidence = attempts[-1]
    state.test_evidence = evidence
    ignored = result.ignored

    for pass_number in range(1, MAX_POST_TEST_STABILIZATION_PASSES + 1):
        try:
            mutated = _strict_scope_and_scrub(
                ctx,
                state,
                phase="post_test",
                round_number=pass_number,
            )
            snapshot = capture_retained_tree(ctx.work, state)
            key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
            _write_footprint_audit(ctx, state, key)
        except Exception as exc:
            return _stabilization_stop(
                ctx,
                state,
                f"guard/capture/audit failed: {exc}",
                round_number=pass_number,
            )

        if state.verifier_key != key:
            try:
                outcomes = await verify_retained_tree(
                    ctx, snapshot, deep_state.items, pass_number=pass_number
                )
                _persist_fix_outcomes_current(ctx, state, key, outcomes)
            except Exception as exc:
                return _stabilization_stop(
                    ctx,
                    state,
                    f"final verifier failed: {exc}",
                    round_number=pass_number,
                )
            state.verifier_key = key
            deep_state.fix_outcomes = outcomes
            if _actionable_verdicts(outcomes):
                return _stabilization_stop(
                    ctx,
                    state,
                    "final verifier remains actionable",
                    round_number=pass_number,
                )

        matching_test = (
            evidence.session_id == state.session_id
            and evidence.input_tree_key == snapshot.tree_key
            and evidence.output_tree_key == snapshot.tree_key
        )
        ran_test = False
        if not matching_test and pass_number == 1:
            try:
                evidence, _, _ = await phase_test_once(
                    ctx.backend_for("test"),
                    ctx.work,
                    config=ctx.config,
                    session_id=state.session_id,
                    capture_tree_key=lambda: _capture_full_delta_key(ctx.work, state),
                    run_context=ctx.run_context,
                )
            except Exception as exc:
                return _stabilization_stop(
                    ctx,
                    state,
                    f"final test failed to run: {exc}",
                    round_number=pass_number,
                )
            attempts.append(evidence)
            state.test_evidence = evidence
            ignored = False if evidence.passed else _authorize_final_red_override(ctx)
            ran_test = True
            _persist_test_verdict(
                ctx,
                state,
                passed=evidence.passed,
                ignored=ignored,
                attempts=attempts,
            )

        if pass_number == 1 and (mutated or ran_test):
            continue
        stable_test = (
            evidence.input_tree_key == snapshot.tree_key
            and evidence.output_tree_key == snapshot.tree_key
            and (evidence.passed or ignored)
        )
        if mutated or state.verifier_key != key or not stable_test:
            return _stabilization_stop(
                ctx,
                state,
                "post-test tree did not stabilize",
                round_number=pass_number,
            )

        try:
            patch_path = artifact_dir_for(
                ctx.work.repo,
                session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            ) / "recommended.patch"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_bytes(snapshot.recommended_patch)
            atomic_write_json(
                recommended_capture_path(deep_state.dd),
                {
                    "session_id": state.session_id,
                    "capture_point": "post_test",
                    "tree_key": snapshot.tree_key,
                    "evidence_key": _evidence_payload(key),
                },
                sort_keys=True,
            )
        except OSError as exc:
            return _stabilization_stop(
                ctx,
                state,
                f"recommended capture failed: {exc}",
                round_number=pass_number,
            )
        state.latest_retained = snapshot
        return None

    return _stabilization_stop(
        ctx,
        state,
        "post-test pass bound exhausted",
        round_number=MAX_POST_TEST_STABILIZATION_PASSES,
    )



async def _step_test(ctx: FlowContext) -> Stop | None:
    """Run typed, identity-bound tests and strictly finalize the retained tree."""
    deep_state = DeepState(ctx.data)
    state = _fix_cycle_state(ctx)
    async with phase_scope(DaydreamPhase.TEST):
        try:
            result = await phase_test_and_heal(
                ctx.backend_for("test"),
                ctx.work,
                feedback_items=deep_state.items,
                config=ctx.config,
                session_id=state.session_id,
                capture_tree_key=lambda: _capture_full_delta_key(ctx.work, state),
                footprint=state.footprint,
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
            if not isinstance(result, TestAndHealResult):
                raise TypeError("phase_test_and_heal returned an invalid evidence result")
            _persist_test_verdict(
                ctx,
                state,
                passed=result.passed,
                ignored=result.ignored,
                attempts=list(result.attempts),
            )
        except Exception as exc:
            confinement_error = _enforce_terminal_confinement(
                ctx,
                state,
                phase="test_failure",
                round_number=None,
            )
            print_error(console, "Test evidence failed", str(exc))
            if confinement_error is not None:
                print_error(console, "Test failure confinement failed", confinement_error)
            return Stop(1)
    if not result.proceed:
        confinement_error = _enforce_terminal_confinement(
            ctx,
            state,
            phase="test_failure",
            round_number=None,
        )
        print_warning(console, "Tests failed after fix attempt.")
        if confinement_error is not None:
            print_error(console, "Test failure confinement failed", confinement_error)
        return Stop(1)
    return await finalize_retained_tree_after_test(ctx, result)


def _persist_push_verdict(
    ctx: FlowContext,
    receipt: PushReceipt,
    *,
    status: Literal["succeeded", "failed"],
    started_at: str,
    diagnostic: str | None = None,
) -> None:
    """Replace the current session's exact push-attempt outcome atomically."""
    deep_state = DeepState(ctx.data)
    state = _fix_cycle_state(ctx)
    payload: dict[str, object] = {
        "schema_version": 1,
        "session_id": state.session_id,
        "status": status,
        "remote": receipt.remote,
        "branch": receipt.branch,
        "pushed_sha": receipt.sha,
        "pushed_repository": receipt.pushed_repository,
        "started_at": started_at,
        "updated_at": now_iso(),
    }
    if diagnostic is not None:
        payload["diagnostic"] = redact_structured_text(diagnostic)[:2_000]
    atomic_write_json(
        push_verdict_path(deep_state.dd),
        payload,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )


async def _step_commit(ctx: FlowContext) -> Stop | None:
    """Stage the finalized retained paths once, then commit and push them."""
    deep_state = DeepState(ctx.data)
    state = _fix_cycle_state(ctx)
    snapshot = state.latest_retained
    if snapshot is None:
        print_error(console, "Commit/Push Failed", "retained tree was not finalized")
        return Stop(1)
    key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
    try:
        for path in sorted(snapshot.paths):
            state.footprint.record_git_event(
                action="stage",
                path=path,
                origin="staging",
                phase="commit",
                round_number=None,
                reason="final retained path selected for the validated index",
            )
        _write_footprint_audit(ctx, state, key)
    except Exception as exc:
        print_error(console, "Commit/Push Failed", str(exc))
        return Stop(1)
    started_at = now_iso()
    try:
        receipt = await phase_commit_push(
            ctx.backend_for("fix"),
            ctx.work,
            preexisting_untracked=set(state.preexisting_untracked),
            config=ctx.config,
            items=deep_state.items_or_empty or [],
            retained_paths=snapshot.paths,
            retained_states=snapshot.states,
            initial_index=state.initial_index,
            run_context=ctx.run_context,
        )
    except PushAttemptError as exc:
        try:
            _persist_push_verdict(
                ctx,
                exc.receipt,
                status="failed",
                started_at=started_at,
                diagnostic=str(exc),
            )
        except Exception as artifact_exc:
            print_error(console, "Push verdict persistence failed", str(artifact_exc))
        print_error(console, "Commit/Push Failed", str(exc))
        return Stop(1)
    except Exception as exc:
        print_error(console, "Commit/Push Failed", str(exc))
        return Stop(1)
    if receipt is not None:
        try:
            _persist_push_verdict(
                ctx,
                receipt,
                status="succeeded",
                started_at=started_at,
            )
        except Exception as exc:
            print_error(console, "Push verdict persistence failed", str(exc))
            return Stop(1)
        deep_state.push_receipt = receipt
    return None


def _resolve_remote_ci_target(ctx: FlowContext, receipt: PushReceipt) -> RemoteCITarget:
    """Bind the pushed repository/ref to one configured P04 pull request."""
    from daydream import git_ops, pr_review
    from daydream.remote_ci import RemoteCITarget

    configured_repo = ctx.config.pr_repo
    configured_pr = ctx.config.pr_number
    if git_ops.split_owner_repo(configured_repo or "") is None:
        raise GitError("remote CI requires a configured GitHub base repository")
    assert configured_repo is not None
    if not isinstance(configured_pr, int) or isinstance(configured_pr, bool) or configured_pr <= 0:
        raise GitError("remote CI requires a configured pull request number")
    if receipt.pushed_repository is None:
        raise GitError("the successful push remote has no GitHub repository identity")

    pr = pr_review.find_pr_by_number(
        ctx.work.repo, configured_pr, auth=ctx.github_execution.auth
    )
    if pr is None:
        raise GitError(f"configured pull request #{configured_pr} was not found")
    base_repository = f"{pr.owner}/{pr.repo}"
    expected = {
        "pull request": (configured_pr, pr.number),
        "configured base repository": (configured_repo.lower(), base_repository.lower()),
        "base ref": (ctx.work.base_branch, pr.base_ref),
        "head repository": (receipt.pushed_repository.lower(), (pr.head_repo or "").lower()),
        "head ref": (receipt.branch, pr.head_ref),
    }
    mismatches = [name for name, (wanted, actual) in expected.items() if wanted != actual]
    if mismatches:
        raise GitError(
            "remote CI identity does not match the pushed target: "
            + ", ".join(mismatches)
        )
    if pr.head_repo is None:
        raise GitError("pull request head repository is unavailable")
    return RemoteCITarget(
        target_dir=ctx.work.repo.resolve(),
        base_repository=base_repository,
        base_ref=pr.base_ref,
        head_repository=pr.head_repo,
        head_ref=pr.head_ref,
        pr_number=pr.number,
        pr_url=pr.url,
        remote=receipt.remote,
        pushed_sha=receipt.sha,
    )


def _print_remote_ci_result(verdict: RemoteCIVerdict) -> None:
    """Render only normalized GitHub evidence and its explicit limitations."""
    target = verdict.target
    if target is not None:
        print_info(
            console,
            escape_markup(
                f"Remote CI target: {target.base_repository} PR #{target.pr_number} "
                f"at {target.pushed_sha}"
            ),
        )
    print_info(
        console,
        escape_markup(f"Remote CI result: {verdict.status} — {verdict.reason}"),
    )
    if verdict.evidence_sha is not None:
        print_info(
            console,
            escape_markup(f"Remote CI evidence SHA: {verdict.evidence_sha}"),
        )
    if verdict.failing_contexts:
        print_warning(console, f"Failing CI: {', '.join(verdict.failing_contexts)}")
    if verdict.pending_contexts:
        print_warning(console, f"Pending CI: {', '.join(verdict.pending_contexts)}")
    if verdict.missing_contexts:
        print_warning(console, f"Missing CI: {', '.join(verdict.missing_contexts)}")
    advisory = [
        item.context
        for item in verdict.advisory_observations
        if item.state in {"fail", "pending"}
    ]
    if advisory:
        print_warning(console, f"Advisory CI not green: {', '.join(advisory)}")
    for url in verdict.urls:
        print_info(console, escape_markup(f"CI details: {url}"))


async def _step_remote_ci(ctx: FlowContext) -> Stop | None:
    """Bind and wait for exact pushed-SHA GitHub CI, failing closed."""
    deep_state = DeepState(ctx.data)
    from daydream.remote_ci import (
        DEFAULT_LIMITS,
        GitHubRemoteCIFetcher,
        pending_remote_ci_verdict,
        unavailable_remote_ci_verdict,
        wait_for_remote_ci,
        write_remote_ci_handoff,
        write_remote_ci_verdict,
    )

    receipt = deep_state.push_receipt
    if not isinstance(receipt, PushReceipt):
        return None
    state = _fix_cycle_state(ctx)
    limits = DEFAULT_LIMITS
    started_at = now_iso()
    monotonic_started = anyio.current_time()
    discovery_deadline = monotonic_started + limits.discovery_seconds
    completion_deadline = monotonic_started + limits.completion_seconds
    poll_count = 0
    verdict: RemoteCIVerdict | None = None

    def persist(snapshot: RemoteCIVerdict) -> None:
        deep_state = DeepState(ctx.data)
        nonlocal poll_count, verdict
        poll_count += 1
        verdict = snapshot
        write_remote_ci_verdict(
            remote_ci_verdict_path(deep_state.dd),
            snapshot,
            session_id=state.session_id,
            poll_count=poll_count,
            started_at=started_at,
            updated_at=now_iso(),
            discovery_deadline=discovery_deadline,
            completion_deadline=completion_deadline,
            limits=limits,
        )

    caught: BaseException | None = None
    cancelled_type = anyio.get_cancelled_exc_class()
    try:
        async with host_phase_scope(DaydreamPhase.REMOTE_CI) as phase:
            try:
                target = _resolve_remote_ci_target(ctx, receipt)
            except Exception as exc:
                verdict = unavailable_remote_ci_verdict(
                    reason="remote CI target identity is unavailable",
                    diagnostic=str(exc),
                )
                write_remote_ci_verdict(
                    remote_ci_verdict_path(deep_state.dd),
                    verdict,
                    session_id=state.session_id,
                    poll_count=0,
                    started_at=started_at,
                    updated_at=now_iso(),
                    discovery_deadline=discovery_deadline,
                    completion_deadline=completion_deadline,
                    limits=limits,
                )
            else:
                # One monotonic start owns both the durable deadline metadata
                # and the waiter's request budgets.  Resolution above is a
                # separate bounded P04 lookup and does not consume CI polling
                # time; persisting the initial state below does.
                started_at = now_iso()
                monotonic_started = anyio.current_time()
                discovery_deadline = monotonic_started + limits.discovery_seconds
                completion_deadline = monotonic_started + limits.completion_seconds
                write_remote_ci_verdict(
                    remote_ci_verdict_path(deep_state.dd),
                    pending_remote_ci_verdict(target),
                    session_id=state.session_id,
                    poll_count=0,
                    started_at=started_at,
                    updated_at=now_iso(),
                    discovery_deadline=discovery_deadline,
                    completion_deadline=completion_deadline,
                    limits=limits,
                )
                # The new target is durable before the previous attempt's
                # guidance is retired. Do this before any CI request so a
                # blocked or abruptly interrupted resume cannot expose it.
                remote_ci_handoff_path(deep_state.dd).unlink(missing_ok=True)
                print_info(
                    console,
                    f"Verifying remote CI for {target.base_repository} PR "
                    f"#{target.pr_number} at {target.pushed_sha}",
                )
                try:
                    verdict = await wait_for_remote_ci(
                        target,
                        fetcher=GitHubRemoteCIFetcher(
                            limits=limits, auth=ctx.github_execution.auth
                        ),
                        limits=limits,
                        monotonic_started_at=monotonic_started,
                        on_snapshot=persist,
                    )
                except (cancelled_type, KeyboardInterrupt) as exc:
                    phase.stop_reason = (
                        "cancelled" if isinstance(exc, cancelled_type) else "interrupted"
                    )
                    caught = exc
                else:
                    phase.stop_reason = verdict.status
            if caught is None and verdict is not None:
                phase.stop_reason = verdict.status
    except Exception as exc:
        print_error(console, "Remote CI verification failed", str(exc))
        return Stop(1)
    if caught is not None:
        if verdict is not None:
            try:
                with anyio.CancelScope(shield=True):
                    write_remote_ci_handoff(
                        remote_ci_handoff_path(deep_state.dd),
                        verdict,
                        session_id=state.session_id,
                    )
            except Exception as exc:
                print_error(console, "Remote CI handoff persistence failed", str(exc))
        raise caught
    if verdict is None:
        print_error(console, "Remote CI verification failed", "no verdict was produced")
        return Stop(1)

    _print_remote_ci_result(verdict)
    handoff = remote_ci_handoff_path(deep_state.dd)
    if verdict.status in {"passed", "no_ci"}:
        try:
            handoff.unlink(missing_ok=True)
        except OSError as exc:
            print_error(console, "Remote CI handoff cleanup failed", str(exc))
            return Stop(1)
        if verdict.status == "passed":
            print_success(console, "Exact pushed-SHA remote CI passed.")
        else:
            print_success(console, "Remote CI was observably not configured.")
        return None
    try:
        write_remote_ci_handoff(handoff, verdict, session_id=state.session_id)
    except Exception as exc:
        print_error(console, "Remote CI handoff persistence failed", str(exc))
    return Stop(1)


async def _perform_cleanup(ctx: FlowContext) -> None:
    """Terminal cleanup: remove the review output when enabled (#330).

    Restores the shallow ``commit-gate`` semantics the single-flow collapse
    dropped: ``--cleanup`` removes ``.review-output.md`` after a successful
    run, ``--no-cleanup`` keeps it, and an unspecified flag falls back to the
    old preamble gate (``--yes`` cleans up, unattended runs keep the artifact
    via ``safe_default=False``, interactive runs prompt). Not a flow step: it
    is invoked by ``_run_review_spine`` on any successful exit, so an early
    ``Stop(0)`` (e.g. a declined fix gate) still honors ``--cleanup`` while
    every failure path (a non-zero exit) skips it to keep evidence.
    """
    config = ctx.config
    target_dir = ctx.work.repo

    if config.cleanup is True:
        enabled = True
    elif config.cleanup is False:
        enabled = False
    else:
        enabled = resolve_run_context(ctx.run_context).confirm(
            safe_default=False,
            question="Cleanup review output after completion? [y/N]",
            default="n",
            console=console,
        )

    if not enabled:
        return
    review_output_path = review_output_path_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )
    if review_output_path.exists():
        review_output_path.unlink()
        print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")
