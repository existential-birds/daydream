"""Deep adjudication, merge, supervision, and review publication stages."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.agent import console
from daydream.artifact_visibility import review_output_path_for
from daydream.deep.arbiter import select_arbiter_targets, select_suppression_targets
from daydream.deep.artifacts import (
    MERGE_FAILURE_KEY,
    _load_failures,
    adjudication_complete_path,
    dedup_candidates_path,
    merged_items_path,
    merged_report_path,
    per_stack_failures_path,
    per_stack_records_path,
)
from daydream.deep.dedup import (
    CandidatePair,
    RecordDuplicatePair,
    build_dedup_candidates,
    build_record_dedup_candidates,
)
from daydream.deep.records import record_uid, stack_name_from_records_source, stack_name_from_uid
from daydream.deep.render import _PIPELINE_STAGE_NAMES, render_held_section, render_report
from daydream.deep.state import DeepState
from daydream.extensions.api import Stop
from daydream.flows.engine import FlowContext
from daydream.phases import (
    CrossStackMergeError,
    _write_single_stack_merged_items,
    phase_arbiter_review,
    phase_cross_stack_merge,
    phase_supervise_review,
    phase_suppression_review,
)
from daydream.supervision import RuleBasedSupervisor, apply_findings_verdicts, revise_finding_fields
from daydream.trajectory import DaydreamPhase, LifecycleReasonCode, LifecycleStatus, get_current_recorder, phase_scope
from daydream.ui import print_error, print_info, print_stage_progress, print_warning

if TYPE_CHECKING:
    from daydream.runner import RunConfig


def _precision_mode(config: RunConfig) -> bool:
    """Resolve the precision-mode opt-in (issue #232).

    Precedence (highest first), mirroring the composition root's fan-out threshold
    and ``_resolve_backend`` / ``_resolved_model`` at ``runner.py:295-326``:

      1. ``RunConfig.precision_mode`` (CLI tier / direct construction).
      2. ``DaydreamFileConfig.precision_mode`` (file-config scalar).
      3. Built-in default ``False`` (byte-identical behavior: the suppression
         predicate is never called and arbiter output is unchanged).

    Uses truthiness rather than ``is not None``: ``False`` is the meaningful
    "off" value, so a set-to-False file-config entry just falls through to the
    default rather than acting as a distinct sentinel.
    """
    if config.precision_mode:
        return True
    file_config = config.file_config
    if file_config is not None and file_config.precision_mode:
        return True
    return False


def _approve_on_clean(config: RunConfig) -> bool:
    """Resolve the approve-on-clean opt-in (issue #343).

    Precedence mirrors ``_precision_mode``: 1) ``RunConfig.approve_on_clean``
    (CLI tier), 2) ``DaydreamFileConfig.approve_on_clean`` (file-config
    scalar), 3) built-in default ``False`` (byte-identical behavior: the
    event stays COMMENT unless a repo explicitly opts in).
    """
    if config.approve_on_clean:
        return True
    file_config = config.file_config
    if file_config is not None and file_config.approve_on_clean:
        return True
    return False


def _supervisor_mode(config: RunConfig) -> str:
    """Resolve the file-config-only findings supervisor mode."""
    file_config = config.file_config
    mode = file_config.supervisor if file_config is not None else None
    return mode if mode in {"off", "rules", "llm"} else "off"


def _candidate_pair_to_json(pair: CandidatePair | RecordDuplicatePair) -> dict[str, Any]:
    """Serialize a CandidatePair dataclass into a JSON-compatible dict."""
    data = asdict(pair)
    # alt_files is a tuple -> convert to list for stable JSON.
    if isinstance(data.get("alt_files"), tuple):
        data["alt_files"] = list(data["alt_files"])
    return data


def _apply_adjudication_verdicts(
    records: list[dict[str, Any]],
    sources: list[str],
    targets: list[int],
    verdicts: dict[int, dict[str, Any]],
    *,
    pass_name: str,
    id_field: str,
    fail_closed: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fold arbiter / suppression verdicts back into the per-stack record set.

    The scoped arbiter (``#168``) and the precision-mode suppression pass
    (``#232``) share an identical positional-rebuild shape; they differ ONLY in
    the fail polarity of two branches -- a missing verdict and an ``id_field``
    mismatch -- which is why this is one parameterised helper rather than two
    ~80-line near-clones (that duplication is what let the stale-index bug at the
    call site hide). Each selected record (``targets[k]`` for 1-based
    ``id = k + 1``) is either revised in place (severity/confidence/description/
    rationale/evidence taken from the verdict) or dropped. ``file``/``line`` are
    never changed -- adjudication revises, it does not re-target findings.

    Host-side identity is the record's ``uid`` (issue #1111), not its position:
    the drop set holds uids, and the revise target is resolved through a uid map
    snapshotted before anything mutates. The positional ``arb_id`` / ``sup_id``
    is still the token the AGENT echoes, which is safe because the input
    artifact is written in the same call that reads the verdicts back. The
    hazard was only the host-side rebinding of indices *after* the compaction
    below -- which is why the caller used to have to re-derive its
    arbiter-exclusion set by ``id(record)`` object identity once this function
    returned. It no longer does: a set of uids survives this rebuild untouched,
    and survives a JSON round-trip or a dict copy that object identity would
    not.

    Fail polarity (the sole axis on which the two passes diverge):

    - ``fail_closed=False`` (arbiter): a record reaches arbitration because it is
      high-severity or contested, so a missing verdict or an ``id_field`` mismatch
      must NOT delete it. The original record is retained unchanged with a warning
      (fail OPEN); only an explicit ``keep:false`` drops it.
    - ``fail_closed=True`` (suppression): a record reaches suppression precisely
      because it is borderline (neither high-severity nor contested), so an
      unconfirmable verdict -- missing, mismatched, or ``keep:false`` -- drops it
      (fail CLOSED), the inverse polarity, safe because nothing important reaches
      this pass.

    Non-selected records always pass through untouched.

    Args:
        records: Per-stack records positionally aligned with ``sources``. Each
            carries a ``uid`` (guaranteed by ``_step_per_stack_parse``).
        sources: Per-record originating stack name.
        targets: Indices into ``records`` selected for this pass; ``targets[k]``
            carries 1-based ``id = k + 1`` echoed back in the verdict's
            ``id_field``. Read once, up front, to snapshot each target's uid.
        verdicts: ``id -> verdict`` mapping from the adjudication agent.
        pass_name: Human-readable pass name (``"arbiter"`` / ``"suppression"``)
            used only in warning text.
        id_field: Verdict key carrying the echoed positional id (``"arb_id"`` for
            the arbiter, ``"sup_id"`` for suppression).
        fail_closed: Fail polarity for the missing-verdict / id-mismatch branches
            (see above).

    Returns:
        New ``(records, sources)`` with the dropped uids' records removed and
        surviving selected records carrying the verdict's fields. Positional
        alignment between the two lists is preserved.
    """
    import warnings

    polarity_action = (
        "dropping the unconfirmed record"
        if fail_closed
        else "retaining the original record unchanged"
    )
    # Snapshot ``offset -> uid`` BEFORE anything mutates or drops a record
    # (issue #1111). This loop is where ``targets``' indices stop being
    # load-bearing: every host-side decision after it keys on the uid.
    target_uids: dict[int, str] = {
        offset: record_uid(records[record_index]) for offset, record_index in enumerate(targets)
    }
    # Revise targets resolve through this map rather than through
    # ``records[record_index]``: the uid is the record's identity, the index is
    # merely how it was selected. Built from the same list in the same call, so
    # a lookup for a snapshotted uid cannot miss; the duplicate-uid guard in
    # ``_step_per_stack_parse`` is what makes the mapping one-to-one.
    by_uid: dict[str, dict[str, Any]] = {}
    for record in records:
        record_key = record_uid(record)
        if record_key:
            by_uid[record_key] = record
    dropped: set[str] = set()
    # A targeted record with no uid is impossible after the birth stamp plus the
    # load-time backfill -- but "impossible" must not mean "silently
    # mis-dropped". Such a record cannot be named in ``dropped``, so it gets a
    # positional entry here, used ONLY to honour the caller's fail polarity for
    # a record we are unable to identify.
    dropped_positions: set[int] = set()

    def _warn_unconfirmable(message: str, *, record_index: int, uid: str) -> None:
        """Warn on an unconfirmable target and fail per ``fail_closed`` (#1111).

        The three branches below (no uid, no verdict, mismatched id_field) all
        warn, then either drop the record or retain it per the caller's fail
        polarity, then move on -- the one shape this function's docstring
        already unifies two ~80-line near-clones to avoid repeating. ``uid``
        empty means the record itself is unidentifiable, so the drop (when
        ``fail_closed``) can only be recorded positionally; otherwise it is
        recorded by uid like every uid-keyed drop in this function.
        """
        warnings.warn(message, stacklevel=3)
        if fail_closed:
            if uid:
                dropped.add(uid)
            else:
                dropped_positions.add(record_index)

    for offset, record_index in enumerate(targets):
        verdict_id = offset + 1
        uid = target_uids[offset]
        if not uid:
            # Broken invariant, reported loudly and failed per the caller's
            # polarity like every other unconfirmable branch here.
            _warn_unconfirmable(
                f"{pass_name.capitalize()} target {id_field}={verdict_id} "
                f"(record_index={record_index}) carries no uid, so its verdict cannot be bound "
                f"to a record identity; {polarity_action} (issue #1111).",
                record_index=record_index,
                uid="",
            )
            continue
        verdict = verdicts.get(verdict_id)
        if verdict is None:
            # No verdict returned for this id -- fail per the caller's polarity.
            _warn_unconfirmable(
                f"{pass_name.capitalize()} returned no verdict for {id_field}={verdict_id} "
                f"(record_index={record_index}, uid={uid}); {polarity_action}.",
                record_index=record_index,
                uid=uid,
            )
            continue
        if verdict.get(id_field) != verdict_id:
            # Secondary key guard: the id field in the verdict must match the key
            # we looked it up by. A mismatch would silently bind the verdict to the
            # wrong record -- fail per the caller's polarity rather than mis-apply.
            _warn_unconfirmable(
                f"{pass_name.capitalize()} verdict {id_field} mismatch: "
                f"expected {id_field}={verdict_id} "
                f"but verdict contains {id_field}={verdict.get(id_field)!r} "
                f"(record_index={record_index}, uid={uid}); {polarity_action}.",
                record_index=record_index,
                uid=uid,
            )
            continue
        if not verdict.get("keep", False):
            dropped.add(uid)
            continue
        # Revise IN PLACE rather than rebuilding the dict. A copied ``uid``
        # would survive a rebuild, so this is no longer the load-bearing
        # constraint it was when the suppression call site keyed its
        # arbiter-exclusion set by ``id(record)`` (#232) -- but in-place is still
        # the correct shape: the caller holds this same list and
        # ``_rewrite_stack_records`` persists these very dicts, so a fresh dict
        # would have to be threaded back into both.
        revise_finding_fields(by_uid[uid], verdict)

    new_records: list[dict[str, Any]] = []
    new_sources: list[str] = []
    for i, (record, source) in enumerate(zip(records, sources, strict=True)):
        # Dropped by uid; the positional set covers only the unidentifiable
        # records warned about above.
        if record_uid(record) in dropped or i in dropped_positions:
            continue
        new_records.append(record)
        new_sources.append(source)
    return new_records, new_sources


def _rewrite_stack_records(
    deep_dir_path: Path,
    stack_record_paths: list[Path],
    records: list[dict[str, Any]],
    sources: list[str],
) -> None:
    """Persist arbiter-revised records back to each per-stack records file (#168).

    The cross-stack merge reads per-stack records by path, so arbitration must
    be reflected on disk, not just in memory. Every language stack file is
    rewritten with its surviving records (an emptied stack becomes
    ``{"issues": [], "verdicts": [...]}`` rather than retaining stale
    pre-arbitration content).

    Routing is by the stack name encoded in each record's ``uid`` (issue
    #1111), falling back to the ``source`` string for a record carrying no uid
    at all, and a record that still routes outside ``stack_record_paths`` is
    reported loudly instead of vanishing -- see the comments in the loop for
    both halves of the defect this replaced.
    """
    by_stack: dict[Path, list[dict[str, Any]]] = {path: [] for path in stack_record_paths}
    for record, source in zip(records, sources, strict=True):
        # Route by the stack name in the record's own ``uid`` (issue #1111), not
        # by the ``source`` string. ``source`` has two spellings -- the records
        # filename on every path that loads records off disk, a bare stack name
        # for the uncovered sweep's in-memory append -- and the branch that used
        # to live here had to guess which one it held. The uid's stack half has
        # exactly one spelling. ``sources`` stays zipped in (``strict=True``)
        # both to assert the two lists are still aligned and to name the source
        # in the warning below.
        uid = record_uid(record)
        if uid:
            dest = per_stack_records_path(deep_dir_path, stack_name_from_uid(uid))
        else:
            # No uid at all: this is the case the uid-based routing above
            # cannot cover, so fall back to ``source`` -- the sole routing
            # signal before issue #1111 -- rather than letting the record fall
            # through to the "unroutable" branch below and be erased from disk.
            dest = per_stack_records_path(deep_dir_path, stack_name_from_records_source(source))
        if dest in by_stack:
            by_stack[dest].append(record)
        else:
            # There was no ``else`` here, and that was the second half of the
            # defect (issue #1111). This function rewrites each records file
            # WHOLESALE, so a record whose dest resolved outside
            # ``stack_record_paths`` was not merely skipped -- it was ERASED
            # from disk, silently. #1110 had to add the structural records path
            # to ``stack_record_paths`` precisely to keep records out of this
            # branch, and nothing would have failed loudly had that been missed.
            # "Every adjudicated record routes to a file being rewritten" is a
            # real invariant that was simply never enforced; this is where it is
            # enforced now. A warning rather than a raise: the adjudicated
            # verdicts for every OTHER record are already computed and belong on
            # disk, so aborting the rewrite would lose more than it protects.
            print_warning(
                console,
                f"Adjudicated record uid={record_uid(record) or '<none>'} (source {source}) "
                f"routes to {dest.name}, which is not among the records files being rewritten "
                f"({', '.join(sorted(path.name for path in stack_record_paths))}); "
                "its adjudication will not reach disk (issue #1111).",
            )
    for dest_path, stack_records in by_stack.items():
        # Issue #742: per-stack records files carry the dict shape
        # ``{"issues": [...], "verdicts": [...]}``. Preserve the verdicts from
        # the on-disk file (if a dict-shaped file is present) so arbitration
        # does not silently drop them; the dict shape is written back
        # regardless so every worker -- merge resume, the coverage evidence
        # path -- reads the same shape whether or not arbitration fired.
        verdicts: list[Any] = []
        if dest_path.is_file():
            try:
                existing = json.loads(dest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = None
            if isinstance(existing, dict):
                existing_verdicts = existing.get("verdicts", [])
                if isinstance(existing_verdicts, list):
                    verdicts = existing_verdicts
        dest_path.write_text(
            json.dumps({"issues": stack_records, "verdicts": verdicts}, indent=2)
        )


def _rejoin_structural_records(
    all_records: list[dict[str, Any]],
    record_sources: list[str],
    structural_records: list[dict[str, Any]],
    structural_sources: list[str],
    records_paths: list[Path],
    structural_path: Path | None,
) -> tuple[list[dict[str, Any]], list[str], set[str], list[Path], range]:
    """Rejoin structural records with language records for adjudication (#1103).

    The structural meta-stack is partitioned out of the dedup pool and the
    merge agent's record pool (`_step_per_stack_parse`) so its lens cannot be
    collapsed into a language bucket -- but that partition also made the
    contested-location branch unreachable for the one pair it exists to
    catch: a structural finding and a language finding reporting the same
    defect at the same place. This reverses the partition for arbitration;
    `_split_structural_records` restores it afterward so the dedup pre-filter
    and the merge prompt see exactly what they saw before.

    Returns the concatenated records/sources, the ``uid`` set of the structural
    records (used by :func:`_split_structural_records`), ``records_paths``
    extended with ``structural_path`` when present -- `_rewrite_stack_records`
    cannot persist a record that routes outside this path list, so the
    structural file must be included or an arbitrated structural verdict would
    never reach disk, and merge re-reads that very file to build the report's
    structural items -- and ``structural_range``: the positional index range
    every structural record occupies in the returned, freshly concatenated
    ``adjudicated`` list.

    ``structural_range`` (not the uid set) is what the caller must pass as
    :func:`~daydream.deep.arbiter.select_arbiter_targets`'s ``contested_only``:
    that exemption from the severity branch must cover EVERY structural
    record, including the no-uid edge case handled below, and a
    ``record_uid(rec) in structural_ids`` membership test silently drops that
    exemption for such a record (its uid is ``""``, which is never in
    ``structural_ids``) -- reopening exactly the widening this function's
    docstring says must never happen. Position cannot fail this way: it is
    read here, immediately after ``adjudicated`` is built and before anything
    reorders or compacts it.

    The uid set still holds uids, not ``id()`` object identities (issue
    #1111), because :func:`_split_structural_records` runs AFTER adjudication
    may have rebuilt the record list, where a positional range no longer
    identifies the same records. Object identity was the strictest possible
    key for that later use and also the most fragile one: it was invalidated
    by any stage that rebuilt a record as a fresh dict or round-tripped it
    through JSON, and such a record escaped the set silently, with the escape
    looking exactly like "not a structural record". A uid is carried inside
    the dict, so it survives both.
    """
    # A structural record with no uid cannot be named in this set and would come
    # back out of `_split_structural_records` as a language record -- i.e. into
    # the dedup pool the partition exists to keep it out of. Unreachable after
    # the duplicate/absence guarantees established in `_step_per_stack_parse`,
    # so this reports rather than stops: adjudication itself is still correct,
    # only the post-split routing of that one record is degraded.
    structural_ids = {uid for rec in structural_records if (uid := record_uid(rec))}
    unidentified = len(structural_records) - len(structural_ids)
    if unidentified:
        print_warning(
            console,
            f"{unidentified} structural record(s) carry no uid; they will be adjudicated but "
            "rejoin the language-stack pool after arbitration instead of the structural pool "
            "(issue #1111).",
        )
    adjudicated = all_records + structural_records
    adjudicated_sources = record_sources + structural_sources
    rewrite_paths = list(records_paths)
    if structural_path is not None:
        rewrite_paths.append(structural_path)
    structural_range = range(len(all_records), len(adjudicated))
    return adjudicated, adjudicated_sources, structural_ids, rewrite_paths, structural_range


def _split_structural_records(
    adjudicated: list[dict[str, Any]],
    adjudicated_sources: list[str],
    structural_ids: set[str],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
    """Split structural records back out after adjudication (#1103).

    Everything downstream of adjudication -- the dedup pre-filter, the merge
    prompt, the host-side structural append -- keeps the partitioned view it
    had before. Records the arbiter rejected are simply absent from both sides.

    The split keys on each record's ``uid`` against the set
    :func:`_rejoin_structural_records` built (issue #1111), so it does not care
    whether adjudication revised a record in place, replaced its dict, or
    round-tripped it through the records files -- the uid rides along inside the
    record either way. ``record_uid`` returns ``""`` for a record carrying none
    and ``""`` is never in the set, so an unidentifiable record lands on the
    language side, which is the outcome ``_rejoin_structural_records`` warns
    about when it drops one.
    """
    all_records: list[dict[str, Any]] = []
    record_sources: list[str] = []
    structural_records: list[dict[str, Any]] = []
    structural_sources: list[str] = []
    for rec, src in zip(adjudicated, adjudicated_sources, strict=True):
        if record_uid(rec) in structural_ids:
            structural_records.append(rec)
            structural_sources.append(src)
        else:
            all_records.append(rec)
            record_sources.append(src)
    return all_records, record_sources, structural_records, structural_sources


async def _step_arbiter(ctx: FlowContext) -> None:
    """Scoped arbiter over high-severity/contested findings (#168)."""
    deep_state = DeepState(ctx.data)
    config = ctx.config
    dd = deep_state.dd
    all_records: list[dict[str, Any]] = deep_state.records
    record_sources: list[str] = deep_state.record_sources
    # Issue #1103: adjudicate over language AND structural records together.
    # `_rejoin_structural_records` below reverses the partition applied in
    # `_step_per_stack_parse` (so structural findings can contest a language
    # finding restating them) and `_split_structural_records` restores it
    # afterward so the dedup pre-filter and the merge prompt see exactly what
    # they saw before.
    structural_records: list[dict[str, Any]] = deep_state.structural_records
    structural_sources: list[str] = deep_state.structural_record_sources

    # Scoped Opus arbiter (#168). Sonnet ran the per-stack reviews;
    # a single heavyweight arbiter now re-reviews ONLY the
    # high-severity / contested findings and writes its verdicts back
    # into the per-stack records before merge. A `--start-at merge`
    # resume re-runs arbitration from the on-disk records UNLESS the
    # completion marker proves a prior run already finalised them
    # (#175): a crash between the parse write and the rewrite would
    # otherwise let unarbitrated high-severity findings reach merge.
    #
    # The marker covers the WHOLE adjudication block (arbiter +, in
    # precision mode, suppression): it is written once after BOTH passes
    # have rewritten the per-stack records, so its presence proves the
    # records are fully adjudicated -- not just arbitrated. Renamed from
    # `arbiter_complete_path` so resume reasoning cannot under-read it as
    # arbiter-only (#232 review).
    adjudication_marker = adjudication_complete_path(dd)
    if (
        ctx.pipeline().arbitration.enabled
        and (config.start_at != "merge" or not adjudication_marker.is_file())
    ):
        structural_path: Path | None = deep_state.structural_records_path_or_none
        adjudicated, adjudicated_sources, structural_ids, rewrite_paths, structural_range = (
            _rejoin_structural_records(
                all_records, record_sources, structural_records, structural_sources,
                deep_state.records_paths, structural_path,
            )
        )

        arbiter_targets = select_arbiter_targets(
            adjudicated, adjudicated_sources,
            min_severity=ctx.pipeline().arbitration.min_severity,
            contested_location=ctx.pipeline().arbitration.contested_location,
            contested_only=structural_range,
        )
        # Capture the identities of records the arbiter will see, before
        # `_apply_adjudication_verdicts` compacts the list (#232). `arbiter_targets`
        # are indices into this pre-apply list; once records are dropped the
        # indices shift, so suppression exclusion must be keyed by per-record
        # identity -- not by the stale positional indices, and not by
        # `(file, line)`: two findings can share one location while only one is
        # arbitrated (a HIGH sibling arbitrated, a LOW sibling not), and a
        # `(file, line)` key would wrongly exclude BOTH, silently skipping the
        # LOW sibling from suppression.
        #
        # That identity is the record's `uid` (issue #1111). This used to be an
        # `id(record)` set, which only worked because
        # `_apply_adjudication_verdicts` happens to revise in place -- a stage
        # that rebuilt a kept record as a fresh dict would have escaped the set
        # silently and had it re-judged by the fail-CLOSED suppression pass. A
        # uid is inside the dict, so no rebuild or JSON round-trip can shake it
        # off. Records with no uid are left out of the set entirely rather than
        # collapsing onto a shared `""` key; the empty-uid case is handled at
        # the exclusion site below.
        arbitrated_ids = {uid for i in arbiter_targets if (uid := record_uid(adjudicated[i]))}
        if arbiter_targets:
            async with phase_scope(DaydreamPhase.DEEP, stage="arbiter"):
                arbiter_backend = ctx.backend_for("arbiter")
                verdicts, arbiter_continuation = await phase_arbiter_review(
                    arbiter_backend,
                    ctx.work,
                    selected_records=[adjudicated[i] for i in arbiter_targets],
                    diff_path=deep_state.diff_path,
                    intent_path=deep_state.intent_path,
                    alternatives_path=deep_state.alts_path,
                    exploration_dir=deep_state.exploration_dir,
                    intent_authoritative=deep_state.intent_authoritative,
                    strategy=ctx.strategy("arbitration"),
                    run_context=ctx.run_context,
                    artifact_session=ctx.artifacts,
                    allow_standalone=ctx.allow_standalone_artifacts,
                )
                # Identity gate: only resume when merge runs on the very same
                # backend instance. A per-phase override that resolves a
                # different backend gets the cold path.
                if arbiter_continuation is not None and arbiter_backend is ctx.backend_for("merge"):
                    deep_state.arbiter_continuation = arbiter_continuation
            adjudicated, adjudicated_sources = _apply_adjudication_verdicts(
                adjudicated, adjudicated_sources, arbiter_targets, verdicts,
                pass_name="arbiter",
                id_field="arb_id",
                fail_closed=False,
            )
            _rewrite_stack_records(
                dd, rewrite_paths, adjudicated, adjudicated_sources
            )

        # Precision-mode suppression pass (#232). OPT-IN: when precision_mode is
        # off (product default) this block never runs, `select_suppression_targets`
        # is never called, and arbiter output is byte-identical. When on, it gives
        # the borderline (LOW-confidence / low-severity uncontested) findings the
        # arbiter never sees a skeptical second opinion (for the profile's
        # ``Suppression.severity_classes`` severity classes), dropping any it cannot
        # confirm (fail-CLOSED, the inverse of the arbiter). The arbiter target set
        # is the exclusion set so nothing high-severity / contested is re-judged
        # here. One batched agent call, resolved via the cheaper `suppression`
        # phase key (Sonnet default) -- never per-finding Opus.
        if ctx.pipeline().suppression.enabled or _precision_mode(config):
            # Structural records join the exclusion set alongside the arbiter's
            # targets (issue #1103). Suppression is fail-CLOSED and selects on
            # low severity / LOW confidence; the structural lens is
            # high-conviction by construction and was never in this pass's pool
            # before, so letting the union widen it would drop structural
            # findings as a side effect of fixing the duplicate-post bug.
            # A record with no uid is excluded too (`not uid`): suppression is
            # fail-CLOSED, so a record we cannot match against either set would
            # otherwise be droppable on an identity we could not establish.
            # Unreachable after `_step_per_stack_parse`, and deliberately biased
            # toward keeping a finding rather than losing one.
            suppression_exclude = [
                i
                for i, r in enumerate(adjudicated)
                if not (uid := record_uid(r)) or uid in arbitrated_ids or uid in structural_ids
            ]
            suppression_targets = select_suppression_targets(
                adjudicated,
                adjudicated_sources,
                suppression_exclude,
                severity_classes=ctx.pipeline().suppression.severity_classes,
                confidence_classes=ctx.pipeline().suppression.confidence_classes,
            )
            if suppression_targets:
                async with phase_scope(DaydreamPhase.DEEP, stage="suppression"):
                    sup_verdicts = await phase_suppression_review(
                        ctx.backend_for("suppression"),
                        ctx.work,
                        selected_records=[adjudicated[i] for i in suppression_targets],
                        diff_path=deep_state.diff_path,
                        intent_path=deep_state.intent_path,
                        alternatives_path=deep_state.alts_path,
                        exploration_dir=deep_state.exploration_dir,
                        strategy=ctx.strategy("suppression"),
                        run_context=ctx.run_context,
                        artifact_session=ctx.artifacts,
                        allow_standalone=ctx.allow_standalone_artifacts,
                    )
                adjudicated, adjudicated_sources = _apply_adjudication_verdicts(
                    adjudicated, adjudicated_sources, suppression_targets, sup_verdicts,
                    pass_name="suppression",
                    id_field="sup_id",
                    fail_closed=True,
                )
                _rewrite_stack_records(
                    dd, rewrite_paths, adjudicated, adjudicated_sources
                )
        adjudication_marker.write_text("")
        all_records, record_sources, structural_records, structural_sources = (
            _split_structural_records(adjudicated, adjudicated_sources, structural_ids)
        )
    deep_state.records = all_records
    deep_state.record_sources = record_sources
    deep_state.structural_records = structural_records
    deep_state.structural_record_sources = structural_sources


def _clear_merge_failure(dd: Path) -> None:
    """Clear a stale ``__merge__`` salvage record after a successful re-merge.

    ``_salvage_merge_failure`` is the only writer of ``MERGE_FAILURE_KEY``; a
    later successful cross-stack merge must supersede it so a subsequent resume
    doesn't emit a misleading 'merged results are PARTIAL' warning for a merge
    that actually succeeded.
    """
    failures_p = per_stack_failures_path(dd)
    loaded = _load_failures(failures_p)
    if MERGE_FAILURE_KEY not in loaded:
        return
    loaded.pop(MERGE_FAILURE_KEY, None)
    if loaded:
        failures_p.write_text(json.dumps(loaded, indent=2, sort_keys=True))
    elif failures_p.exists():
        failures_p.unlink()


#: Cap on how many unidentifiable dedup pairs ``_drop_cross_stack_duplicates``
#: names individually in its aggregate warning below, mirroring
#: ``phases._MAX_REPORTED_UNKNOWN_UIDS``: an artifact written before
#: ``record_b_uid`` existed can carry many such pairs at once, and naming the
#: first few and counting the rest keeps the message actionable instead of
#: turning it into the per-pair flood the aggregation exists to avoid.
_MAX_REPORTED_UNIDENTIFIABLE_PAIRS = 10


def _drop_cross_stack_duplicates(dd: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the D-27 dedup pre-filter to a host-written partial merge (issue #361).

    In a full merge the merge agent adjudicates ``record_duplicate_pairs``
    (cross-stack records describing the same concern). A salvage writes the
    partial list with no merge agent, so it must apply the pre-filter's computed
    cross-stack duplicate pairs itself -- otherwise the partial
    ``merged-items.json`` carries duplicates into the resume verifier and fix
    gate. Keeps the ``record_a`` side of each pair (deterministic sort order)
    and drops the ``record_b`` side, matched on ``record_b_uid``.

    We used to match the b-side on ``(id, file)``, and that was our bug -- not a
    limitation of the artifact. That tuple is not unique (a reviewer-assigned
    ``id`` restarts at 1 in every stack), and the filter below is a
    set-membership test over the whole record list, so it deleted EVERY record
    matching a dropped key instead of the one b-side it meant to. Three stacks
    each reporting ``id: 1`` on ``api.py`` for the same defect yield pairs
    (0,1), (0,2) and (1,2), whose b-side keys are all ``("1", "api.py")`` --
    and record 0, the a-side this function exists to KEEP, matches that key too
    and died with them, leaving the partial report with zero language findings.
    It looked identical to the records being deleted, which is precisely why it
    had been paired with them. ``record_b_uid`` (issue #1111) names one record
    and only that record, so the same input now drops the two b-sides and keeps
    record 0.
    """
    dedup_p = dedup_candidates_path(dd)
    if not dedup_p.is_file():
        return records
    try:
        dedup = json.loads(dedup_p.read_text())
    except json.JSONDecodeError:
        return records
    dropped_uids: set[str] = set()
    unidentifiable_pairs: list[str] = []
    for pair in dedup.get("record_duplicate_pairs", []) or []:
        if not isinstance(pair, dict):
            continue
        b_uid = pair.get("record_b_uid")
        if isinstance(b_uid, str) and b_uid:
            dropped_uids.add(b_uid)
            continue
        # Within one run this is unreachable: ``dedup-candidates.json`` is
        # written unconditionally by ``_step_cross_stack_merge`` in the same call
        # that can go on to reach this salvage, from records
        # ``_step_per_stack_parse`` guaranteed carry uids. The guard is for the
        # artifact a resume reads back out of a deep dir written by an older run,
        # from before the field existed. Skip the pair rather than falling back
        # to the ``(id, file)`` key: that fallback is the bug documented above,
        # and leaving a duplicate in a partial report is a far smaller error
        # than deleting the finding the pair was supposed to preserve.
        unidentifiable_pairs.append(
            f"{pair.get('record_a_id')!r}/{pair.get('record_b_id')!r}"
        )
    if unidentifiable_pairs:
        # ONE warning for the whole salvage, not one per pair (mirrors
        # ``phases._validate_agent_source_uids``): an artifact written before
        # ``record_b_uid`` existed can carry many such pairs at once, and
        # per-pair reporting would bury the run's real output under identical
        # lines.
        shown = ", ".join(unidentifiable_pairs[:_MAX_REPORTED_UNIDENTIFIABLE_PAIRS])
        if len(unidentifiable_pairs) > _MAX_REPORTED_UNIDENTIFIABLE_PAIRS:
            shown += f", +{len(unidentifiable_pairs) - _MAX_REPORTED_UNIDENTIFIABLE_PAIRS} more"
        print_warning(
            console,
            "Cross-stack merge salvage: dedup pair(s) carries no record_b_uid, so "
            "neither side can be identified; keeping both records (issue #1111). "
            f"({len(unidentifiable_pairs)} pair(s): {shown})",
        )
    if not dropped_uids:
        return records
    # ``record_uid`` is ``""`` for an item with no pre-merge identity and ``""``
    # is never in ``dropped_uids``, so such an item is always kept.
    kept = [r for r in records if record_uid(r) not in dropped_uids]
    if len(kept) != len(records):
        print_info(
            console,
            f"Cross-stack merge salvage: dropped {len(records) - len(kept)} "
            "duplicate per-stack record(s) via the D-27 dedup pre-filter",
        )
    return kept


async def _step_cross_stack_merge(ctx: FlowContext) -> Stop | None:
    """Dedup pre-filter (D-27) + cross-stack merge (D-23..D-26).

    A genuinely unparseable merge response (issue #361) is salvaged rather than
    aborting the run: the completed stacks' verdicts are consolidated into a
    partial ``merged-items.json`` + failure record, and the run stops resumably
    (``Stop(1)``) so a relaunch picks up without re-reviewing completed stacks.
    """
    deep_state = DeepState(ctx.data)
    dd = deep_state.dd
    alts_p: Path = deep_state.alts_path
    all_records: list[dict[str, Any]] = deep_state.records
    failed_stacks: dict[str, str] = deep_state.failed_stacks

    async with phase_scope(
        DaydreamPhase.MERGE, stage="cross-stack-agent"
    ) as phase:
        # Dedup pre-filter (D-27).
        alt_issues_for_dedup: list[dict[str, Any]] = (
            json.loads(alts_p.read_text()) if alts_p.exists() else []
        )
        pairs = build_dedup_candidates(all_records, alt_issues_for_dedup)
        record_pairs = build_record_dedup_candidates(
            all_records, sources=deep_state.record_sources
        )
        dedup_p = dedup_candidates_path(dd)
        dedup_p.write_text(
            json.dumps(
                {
                    "record_alt_pairs": [_candidate_pair_to_json(p) for p in pairs],
                    "record_duplicate_pairs": [
                        _candidate_pair_to_json(p) for p in record_pairs
                    ],
                },
                indent=2,
            )
        )

        # Cross-stack merge (D-23..D-26).
        try:
            await phase_cross_stack_merge(
                ctx.backend_for("merge"),
                ctx.work,
                per_stack_records_paths=deep_state.records_paths,
                intent_path=deep_state.intent_path,
                alternatives_path=alts_p,
                dedup_candidates_path=dedup_p,
                exploration_dir=deep_state.exploration_dir,
                failed_stacks=failed_stacks or None,
                structural_records_path=deep_state.structural_records_path,
                intent_authoritative=deep_state.intent_authoritative,
                continuation=deep_state.arbiter_continuation,
                strategy=ctx.strategy("merge"),
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
        except CrossStackMergeError as exc:
            phase.finish(LifecycleStatus.FAILED, LifecycleReasonCode.DOMAIN_FAILURE)
            _salvage_merge_failure(ctx, exc)
            return Stop(1)
        # Issue #361: a successful re-merge supersedes any stale salvage record, so
        # the structured ``MERGE_FAILURE_KEY`` entry is cleared here -- otherwise a
        # later ``--start-at merge``/``fix`` resume still warns 'merged results are
        # PARTIAL' even though the cross-stack merge has since succeeded.
        _clear_merge_failure(dd)
    return None


def _salvage_merge_failure(ctx: FlowContext, exc: CrossStackMergeError) -> None:
    """Persist a salvageable cross-stack merge failure (issue #361).

    The merge agent returned a response containing no parseable item list (e.g.
    bare ``str`` prose/refusal/truncated JSON). Instead of aborting the run with
    the completed stacks' verdicts stranded on disk, consolidate the surviving
    per-stack records into a *partial* ``merged-items.json`` + ``review-output.md``
    and record the failure as a structured entry under the reserved
    ``MERGE_FAILURE_KEY`` in ``per-stack-failures.json``. The run then stops
    resumably so a relaunch picks up without re-review.

    Both fallible writes propagate through the project error type -- a genuinely
    unwritable salvage must surface, not silently degrade. The only tolerance is
    loading a missing/malformed existing ``per-stack-failures.json`` as ``{}``
    (the "no prior failures" default); existing per-stack entries are preserved.
    """
    deep_state = DeepState(ctx.data)
    dd = deep_state.dd
    print_error(
        console,
        "Cross-stack merge failed",
        f"{exc}; consolidating surviving per-stack records into a partial report. "
        "Relaunch with --start-at fix to resume.",
    )

    # Build the partial canonical merged-items.json + review-output.md from the
    # surviving per-stack records (reusing the single-stack write helper's shared
    # structural-tagging + render epilogue). Recoverability comes from the
    # structured ``__merge__`` failure record + resumable stop, not a root
    # ``partial`` flag in merged-items.json (no consumer reads it -- issue #361
    # follow-up). Apply the D-27 dedup pre-filter (issue #361): with no merge
    # agent to adjudicate, drop the duplicate side of cross-stack record pairs so
    # the partial list doesn't carry duplicates into the resume verifier/fix gate.
    # Issue #1111: these items are host-written -- no merge agent ran, by
    # definition of this path -- so their ``source_uids`` come from the records'
    # own uids. That attribution is not repeated here: it lives in
    # ``_write_single_stack_merged_items``, which is the single writer of this
    # path's items, and duplicating it would give the salvage report a second
    # spelling of provenance that could drift from the bypass's. The records
    # reaching here are uid-stamped (``_step_per_stack_parse``) and pass through
    # ``_drop_cross_stack_duplicates`` unmodified, so the uids survive.
    records = _drop_cross_stack_duplicates(dd, deep_state.records)
    _write_single_stack_merged_items(
        ctx.work.repo,
        dd,
        records,
        deep_state.structural_records_path,
        failed_stacks=deep_state.failed_stacks_or_none or None,
        artifact_session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )

    # Record the failure for resume. Never drop existing per-stack entries.
    failures_p = per_stack_failures_path(dd)
    failures = _load_failures(failures_p)
    failures[MERGE_FAILURE_KEY] = {
        "response_shape": exc.response_shape,
        "stack_context": exc.stack_context,
        "message": str(exc),
    }
    failures_p.write_text(json.dumps(failures, indent=2, sort_keys=True))
    print_info(console, f"Wrote partial merged items and merge-failure record to {dd}")


async def _step_single_stack_merge(ctx: FlowContext) -> None:
    """Tiny-diff single-stack bypass (#172): host-side merged-items write."""
    deep_state = DeepState(ctx.data)
    failed_stacks: dict[str, str] = deep_state.failed_stacks

    # Issue #172 — tiny-diff single-stack bypass. A ≤2-file diff
    # has nothing to cross-stack-merge and nothing contested to
    # arbitrate, so the host writes ``merged-items.json`` directly
    # via ``normalize_items`` + the exact structural-tagging logic
    # from ``phase_cross_stack_merge``. No arbiter, no dedup, no
    # merge agent. Downstream consumers (fix gate, verifier, PR
    # posting) read the canonical JSON unchanged (AC6).
    async with phase_scope(DaydreamPhase.MERGE, stage="single-stack-host"):
        _write_single_stack_merged_items(
            ctx.work.repo,
            deep_state.dd,
            deep_state.records,
            deep_state.structural_records_path,
            failed_stacks=failed_stacks or None,
            artifact_session=ctx.artifacts,
            allow_standalone=ctx.allow_standalone_artifacts,
        )


async def _step_load_items(ctx: FlowContext) -> Stop | None:
    """Host-side merged-items guard + render-only markdown recovery."""
    deep_state = DeepState(ctx.data)
    target_dir = ctx.work.repo
    dd = deep_state.dd

    print_stage_progress(console, 5, 5, _PIPELINE_STAGE_NAMES[4])
    merged_report = review_output_path_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )

    # merged-items.json is the canonical source of truth; review-output.md is
    # render-only. The missing-input guard keys on the JSON so a --start-at fix
    # resume with surviving JSON but absent markdown proceeds rather than bailing.
    items_file = merged_items_path(dd)
    if not items_file.is_file():
        print_error(
            console,
            "Missing Merged Items",
            f"Expected canonical merged items at {items_file}",
        )
        return Stop(1)

    # Best-effort recover the render-only markdown from the deep-dir copy for
    # the exit message when the canonical file is absent (e.g. a --start-at fix
    # resume where the copy to the canonical path never ran). Non-fatal.
    if not merged_report.exists():
        from daydream.deep.artifacts import merged_report_path as _deep_report_path

        deep_copy = _deep_report_path(dd)
        if deep_copy.exists():
            merged_report.write_text(deep_copy.read_text())

    # Issue #309: surface the uncovered-sweep coverage stats on the rendered
    # report. The sweep runs BEFORE the merge writes review-output.md, so the
    # section is appended here, once the report exists, to both the canonical
    # report and its deep-dir copy.
    _append_coverage_section(dd, merged_report, merged_report_path(dd))

    deep_state.merged_report = merged_report
    deep_state.items_file = items_file
    return None


def _emit_coverage_section(report: Path, deep_copy: Path, section: str) -> None:
    """Write ``section`` into the canonical report and its deep-dir copy.

    Shared by both branches of ``_append_coverage_section`` (the missing-index
    and normal coverage paths) so the two write loops cannot drift. Appends
    only when a target file does not already carry a ``## Coverage`` section;
    an absent file is a silent no-op.
    """
    for target in (report, deep_copy):
        if target.is_file():
            text = target.read_text(encoding="utf-8")
            if "## Coverage" not in text:
                target.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")


def _append_coverage_section(dd: Path, report: Path, deep_copy: Path) -> None:
    """Append a short ``## Coverage`` section when the sweep produced stats.

    Reads ``deep/coverage-stats.json`` and appends files_in_diff / files read /
    ratio / swept files to both the canonical report and its deep-dir copy. The
    ratio rendered is the POST-sweep value (recomputed after the sweep's forks
    landed); only files whose sweep review produced completed output are labeled
    covered. Failed sweep attempts are surfaced as failures, not claimed as
    coverage. A missing or malformed stats file is a silent no-op -- coverage
    surfacing is advisory, never a gate. ANY failure here (read error, invalid
    JSON, structurally-malformed root, a non-dict shape) warns and returns; it
    can never fail the merge step.
    """
    stats_p = dd / "coverage-stats.json"
    if not stats_p.is_file():
        return
    try:
        stats = json.loads(stats_p.read_text())
        if not isinstance(stats, dict):
            print_warning(
                console,
                "Ignoring malformed coverage stats (expected a JSON object): "
                f"{stats_p}",
            )
            return
        pre_sweep = stats.get("pre_sweep")
        if not isinstance(pre_sweep, dict):
            return
        # Issue #336: a missing hunk index leaves the changed-file set
        # unenumerated. Surface that gap instead of rendering an empty diff as
        # a full-coverage pass (the coverage ratio is ``None`` when the index
        # was absent, so no ratio line is emitted). ``load_hunk_index`` still
        # fails open -- this is reporting only.
        if pre_sweep.get("hunk_index_missing"):
            lines = [
                "## Coverage",
                "- Coverage not available: hunk index is missing.",
            ]
            section = "\n".join(lines) + "\n"
            _emit_coverage_section(report, deep_copy, section)
            return
        files_in_diff = pre_sweep.get("files_in_diff")
        if not isinstance(files_in_diff, int):
            return
        lines = [
            "## Coverage",
            f"- Files in diff: {files_in_diff}",
        ]
        # Prefer the POST-sweep numbers (the ratio the sweep actually achieved);
        # fall back to the pre-sweep snapshot when the sweep did not recompute.
        post_sweep = stats.get("post_sweep")
        read_source = post_sweep if isinstance(post_sweep, dict) else pre_sweep
        files_read = read_source.get("files_read_by_reviewers")
        if isinstance(files_read, int):
            lines.append(f"- Files read by reviewers: {files_read}")
        ratio = read_source.get("coverage_ratio")
        if isinstance(ratio, (int, float)):
            lines.append(f"- Coverage ratio: {ratio}")
        # Issue #309 finding 6: only files with a verified completed read are
        # labeled covered. A completed review output WITHOUT a read is a
        # completed attempt -- rendered as "reviewed (hunks only)" -- and never
        # appears on the covered line nor moves the ratio above.
        covered = stats.get("covered_files")
        if isinstance(covered, list) and covered:
            lines.append(f"- Second-pass sweep covered: {', '.join(str(f) for f in covered)}")
        completed = stats.get("completed_files")
        if isinstance(completed, list):
            hunks_only = [
                str(f) for f in completed if not (isinstance(covered, list) and f in covered)
            ]
            if hunks_only:
                lines.append(
                    f"- Second-pass sweep reviewed (hunks only): {', '.join(hunks_only)}"
                )
        failures = stats.get("sweep_failures")
        if isinstance(failures, dict) and failures:
            lines.append(f"- Best-effort sweep failures: {', '.join(sorted(str(f) for f in failures))}")
        skipped = stats.get("sweep_skipped_capacity")
        if isinstance(skipped, int) and skipped:
            lines.append(f"- Sweep capacity-skipped files: {skipped}")
        section = "\n".join(lines) + "\n"
        _emit_coverage_section(report, deep_copy, section)
    except Exception as exc:  # noqa: BLE001 -- advisory decoration: never fail the step
        print_warning(
            console,
            "Skipping coverage stats render (advisory; run continues): "
            f"{type(exc).__name__}: {exc}",
        )


async def _step_findings_out(ctx: FlowContext) -> Stop:
    """Two-phase findings artifact (Phase A): emit the strict-schema artifact and STOP."""
    deep_state = DeepState(ctx.data)
    from daydream.pr_review import resolve_review_renderers
    from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info
    from daydream.runner import _emit_findings_from_items

    items_file: Path = deep_state.items_file
    findings_items: list[dict[str, Any]] = json.loads(items_file.read_text())["items"]
    # Issue #1113: a review artifact carries the run's diagram payload when the
    # diagram step produced one, so Phase B can re-render the blocks into the
    # posted review from the validated specs.
    diagrams = (deep_state.diagrams or {}).get("payload")
    recorder = get_current_recorder()
    run_info = render_live_run_info(LiveRunInfoSource(recorder, ctx.artifacts))
    if run_info.diagnostic is not None:
        print_warning(console, run_info.diagnostic)
    return Stop(
        _emit_findings_from_items(
            ctx.work.repo,
            ctx.config,
            findings_items,
            run_info=run_info.markdown,
            renderers=resolve_review_renderers(ctx.registry),
            diagrams=diagrams,
            auth=ctx.github_execution.auth,
        )
    )


async def _step_supervise(ctx: FlowContext) -> None:
    """Apply the configured findings supervisor to canonical merged items."""
    deep_state = DeepState(ctx.data)
    mode = _supervisor_mode(ctx.config)
    file_config = ctx.config.file_config
    items_file: Path = deep_state.items_file
    items = json.loads(items_file.read_text())["items"]
    if mode == "rules":
        assert file_config is not None, "rules mode requires file_config (guaranteed by _supervisor_mode)"
        deny_globs = file_config.supervisor_deny_globs
        verdicts = RuleBasedSupervisor(deny_globs=deny_globs).review_findings(items)
    else:
        async with phase_scope(DaydreamPhase.DEEP, stage="supervise"):
            verdicts = await phase_supervise_review(
                ctx.backend_for("supervise"),
                ctx.work,
                items=items,
                diff_path=deep_state.diff_path,
                intent_path=deep_state.intent_path,
                alternatives_path=deep_state.alts_path,
                exploration_dir=deep_state.exploration_dir,
                strategy=ctx.strategy("supervision"),
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
    kept, held, events = apply_findings_verdicts(items, verdicts)
    items_file.write_text(json.dumps({"items": kept, "held": held}, indent=2))

    report = render_report(kept)
    held_section = render_held_section(held)
    if held_section:
        report = report.rstrip() + "\n\n" + held_section + "\n"
    deep_report = merged_report_path(deep_state.dd)
    deep_report.write_text(report)
    deep_state.merged_report.write_text(report)

    recorder = get_current_recorder()
    if recorder is not None:
        for finding_id, action, reason in events:
            recorder.emit_supervisor_verdict(finding_id, action, reason)
    return None


async def _step_post_review(ctx: FlowContext) -> Stop | None:
    """Offer to post in loop/shallow modes; ``--comment`` auto-posts.

    In comment mode posting is the run's deliverable, so a missing PR or a
    failed GitHub submission ends the run with exit code 1 instead of the
    warn-and-continue the default deep flow gets (#8). Report-only review mode
    never resolves a PR or enters the posting helper.
    """
    deep_state = DeepState(ctx.data)
    if deep_state.mode == "review":
        return None

    from daydream.pr_review import PostStatus, post_review_to_pr_from_report, resolve_review_renderers
    from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info

    recorder = get_current_recorder()
    run_info = render_live_run_info(LiveRunInfoSource(recorder, ctx.artifacts))
    if run_info.diagnostic is not None:
        print_warning(console, run_info.diagnostic)

    items_file: Path = deep_state.items_file
    pr_kwargs = (
        {"pr_number": ctx.config.pr_number}
        if ctx.config.pr_number is not None
        else {}
    )
    outcome = await post_review_to_pr_from_report(
        ctx.work.repo,
        items_file,
        run_info=run_info.markdown,
        renderers=resolve_review_renderers(ctx.registry),
        console=console,
        post=deep_state.mode == "comment",
        approve_on_clean=_approve_on_clean(ctx.config),
        diagram_blocks=(deep_state.diagrams or {}).get("blocks"),
        run_context=ctx.run_context,
        auth=ctx.github_execution.auth,
        **pr_kwargs,
    )
    if deep_state.mode == "comment" and outcome in (PostStatus.NO_PR, PostStatus.FAILED):
        return Stop(1)
    return None
