"""Deep exploration, intent, review fan-out, parsing, and coverage sweep stages."""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from daydream.agent import console, run_agent
from daydream.artifact_visibility import artifact_dir_for
from daydream.backends import effective_fanout_concurrency
from daydream.config import DEFAULT_TOOL_CALL_BUDGET, DEFAULT_WALL_BUDGET_S, STRUCTURE_STACK_NAME
from daydream.deep.artifacts import MERGE_FAILURE_KEY, _load_failures, per_stack_failures_path, per_stack_records_path
from daydream.deep.artifacts import alternatives_path as _alternatives_path
from daydream.deep.artifacts import intent_path as _intent_path
from daydream.deep.coverage import (
    _completed_read_paths,
    _finding_files_from_records,
    build_uncovered_sweep_prompt,
    compute_uncovered_files,
    coverage_receipt_path,
    diff_block_for_file,
    filter_sweepable_files,
    resolve_per_stack_verdicts,
)
from daydream.deep.diff import _read_full_diff, _ttt_diff_text
from daydream.deep.records import duplicate_record_uids, record_uid, stack_name_from_uid, stamp_record_uids
from daydream.deep.render import _PIPELINE_STAGE_NAMES
from daydream.deep.settings import fresh_ttt
from daydream.deep.state import DeepState
from daydream.eval.analyzer import _agent_label, _records_issues_or_empty, load_trajectories
from daydream.extensions.api import Stop
from daydream.flows.engine import FlowContext
from daydream.phases import (
    UNCOVERED_SWEEP_SCHEMA,
    phase_alternative_review,
    phase_per_stack_reviews,
    phase_understand_intent,
)
from daydream.prompt_budget import prepare_sanctioned_inputs
from daydream.trajectory import (
    DaydreamPhase,
    LifecycleReasonCode,
    LifecycleStatus,
    _safe_descriptor,
    dispatch_scope,
    get_current_recorder,
    maybe_fork,
    phase_scope,
)
from daydream.ui import (
    phase_subtitle,
    print_dim,
    print_error,
    print_phase_hero,
    print_stage_progress,
    print_warning,
    render_exploration_summary,
)

if TYPE_CHECKING:
    from daydream.trajectory import PhaseScopeHandle, TrajectoryRecorder

try:
    from daydream.exploration import ExplorationContext, safe_explore
    from daydream.exploration_runner import (
        count_changed_files as count_changed_files,
    )
    from daydream.exploration_runner import (
        pre_scan,
    )
    from daydream.exploration_runner import (
        select_tier as select_tier,
    )

    EXPLORATION_AVAILABLE = True
except ImportError:  # pragma: no cover -- optional exploration dependency
    EXPLORATION_AVAILABLE = False


def _uncovered_sweep_max_files(ctx: FlowContext) -> int:
    """Resolve the per-run uncovered-file sweep capacity cap (issue #309).

    Reads the profile pipeline's ``uncovered_sweep_max_files`` bound, which
    ``review_profile._parse_pipeline`` has already host-clamped (HOST_CAPS
    floor 1, ceiling 10) before the digest is computed, so no per-run
    coercion is needed here.
    """
    return ctx.pipeline().uncovered_sweep_max_files


def _uncovered_sweep_min_hunk_lines(ctx: FlowContext) -> int:
    """Resolve the minimum hunk size for a file to be swept (issue #309).

    Reads the profile pipeline's ``uncovered_sweep_min_hunk_lines`` bound,
    already host-clamped (HOST_CAPS floor 5) by ``review_profile._parse_pipeline``.
    """
    return ctx.pipeline().uncovered_sweep_min_hunk_lines


async def _step_exploration(ctx: FlowContext) -> None:
    """Exploration pre-scan (D-43), reused on an exact key match."""
    deep_state = DeepState(ctx.data)
    from daydream.exploration import cache_key_path, exploration_cache_key, read_cache_key
    from daydream.runner import _compute_diff_ref

    config = ctx.config
    target_dir = ctx.work.repo
    daydream_dir = artifact_dir_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )
    # Issue #644 — the pre-scan must be grounded in the FULL diff, never the
    # bounded in-memory ``ctx.data["diff"]``: ``detect_affected_files`` seeds a
    # specialist per affected file, so a dropped-block file would otherwise get
    # zero exploration context, and the exact-match cache key would cover only
    # the retained blocks (a change confined to a dropped-block region could
    # then hit the cache). ``diff_path`` is always written full at gather; a
    # read failure degrades to the bounded text with a warning rather than
    # failing the run — exploration is fail-open by design.
    diff = deep_state.diff
    try:
        diff = _read_full_diff(ctx)
    except OSError as exc:
        print_warning(
            console,
            f"Could not read the full diff for the exploration pre-scan "
            f"({exc}); falling back to the bounded in-memory diff",
        )
    tier = deep_state.tier
    exploration_path = daydream_dir / "exploration"

    exploration_dir: Path | None = None
    if not EXPLORATION_AVAILABLE:
        print_warning(
            console,
            "Exploration infrastructure not installed; running deep pipeline "
            "without pre-scan grounding",
        )
    elif config.exploration_context is None:
        # The in-process context short-circuits first; the disk cache is only
        # consulted when there is no in-memory context to reuse.
        cache_key = exploration_cache_key(
            ctx.work.head_sha or "", diff, tier
        )
        if (
            exploration_path.is_dir()
            and read_cache_key(exploration_path) == cache_key
        ):
            # Early return BEFORE the pre_scan/write_to_dir block below: routing
            # a hit through it with an empty in-memory context would overwrite
            # the cached files with "No data collected" stubs.
            print_dim(console, f"Reusing exploration pre-scan from {exploration_path}")
            deep_state.exploration_dir = exploration_path
            return

        # Miss: drop any stale directory so a partial previous result cannot be
        # read as this run's grounding.
        if exploration_path.is_dir():
            shutil.rmtree(exploration_path, ignore_errors=True)

        if tier == "skip":
            print_dim(console, "Skipping exploration -- trivial diff")
            config.exploration_context = ExplorationContext()
        else:
            print_phase_hero(console, "EXPLORE", phase_subtitle("EXPLORE"))
            explore_backend = ctx.backend_for("exploration")
            async with phase_scope(DaydreamPhase.EXPLORATION):
                config.exploration_context = await safe_explore(
                    pre_scan,
                    explore_backend,
                    target_dir,
                    diff,
                    diff_ref=_compute_diff_ref(target_dir),
                    strategies={
                        "exploration.pattern_scan": ctx.strategy("exploration.pattern_scan"),
                        "exploration.dependency_trace": ctx.strategy("exploration.dependency_trace"),
                        "exploration.test_mapping": ctx.strategy("exploration.test_mapping"),
                        "exploration.repository_survey": ctx.strategy("exploration.repository_survey"),
                    },
                    run_context=ctx.run_context,
                )
            # Osprey resolves an omitted model from its own config and reports
            # the authoritative value in session_start, during safe_explore.
            # Log after that boundary so the backend name is never presented as
            # the model id.
            print_dim(console, f"Exploration model: {explore_backend.model}")
            console.print(render_exploration_summary(config.exploration_context))
        if config.exploration_context is not None:
            exploration_dir = config.exploration_context.write_to_dir(exploration_path)
            if config.exploration_context.completed:
                cache_key_path(exploration_path).write_text(cache_key, encoding="utf-8")
            deep_state.exploration_dir = exploration_dir
            return
    if EXPLORATION_AVAILABLE and config.exploration_context is not None:
        exploration_dir = config.exploration_context.write_to_dir(exploration_path)
    deep_state.exploration_dir = exploration_dir


async def _step_intent(ctx: FlowContext) -> None:
    """TTT intent analysis, grounded by the PR description when it is fresh.

    On exit, ``ctx.data["intent_authoritative"]`` is set to True when a fresh,
    head-matched PR description with non-whitespace content grounded the intent
    phase (issue #279). Downstream reviewers read this key to determine whether
    to include the authoritative-intent precedence rule in their prompts.
    """
    deep_state = DeepState(ctx.data)
    from daydream import git_ops

    config = ctx.config
    work = ctx.work
    target_dir = work.repo

    print_stage_progress(console, 1, 5, _PIPELINE_STAGE_NAMES[0])
    pr_description: str | None = None
    if config.pr_number is not None:
        try:
            pr_view = git_ops.gh_pr_view(
                target_dir, config.pr_number, auth=ctx.github_execution.auth
            )
        except git_ops.GitError as exc:
            print_warning(
                console,
                f"Could not load PR #{config.pr_number} description ({exc}); "
                "continuing without PR description context",
            )
            pr_view = None
        if pr_view is not None:
            pr_state = pr_view.get("state", "")
            pr_head_oid = pr_view.get("headRefOid", "")
            local_head = work.head_sha
            if pr_state and pr_state.upper() != "OPEN":
                print_warning(
                    console,
                    f"PR #{config.pr_number} state is {pr_state!r} (not OPEN); "
                    "skipping PR description to avoid trusting a stale body",
                )
            elif pr_head_oid and local_head and pr_head_oid != local_head:
                print_warning(
                    console,
                    f"PR #{config.pr_number} head SHA ({pr_head_oid[:12]}) "
                    f"does not match local HEAD ({local_head[:12]}); "
                    "skipping PR description to avoid trusting a mismatched body",
                )
            else:
                pr_description = pr_view.get("body") or None
    # Issue #279: publish whether a fresh, head-matched PR description grounded
    # the intent phase, so downstream reviewers can include the precedence rule.
    # Match build_intent_prompt: whitespace-only bodies are ignored after strip.
    deep_state.intent_authoritative = bool(pr_description and pr_description.strip())
    async with phase_scope(DaydreamPhase.INTENT):
        deep_state.intent_summary = await phase_understand_intent(
            ctx.backend_for("intent"),
            work,
            deep_state.diff_path,
            deep_state.log,
            deep_state.branch,
            exploration_dir=deep_state.exploration_dir,
            pr_description=pr_description,
            diff_text=_ttt_diff_text(ctx),
            strategy=ctx.strategy("intent"),
            run_context=ctx.run_context,
        )
    # Each TTT step persists its own half, so a later step's failure cannot
    # discard an artifact this one already produced.
    intent_p = _intent_path(deep_state.dd)
    intent_p.write_text(deep_state.intent_summary)
    deep_state.intent_path = intent_p


async def _wonder(ctx: FlowContext) -> None:
    """TTT alternative-review (tier-gated) + its artifact write."""
    deep_state = DeepState(ctx.data)
    intent_summary = deep_state.intent_summary

    print_stage_progress(console, 2, 5, _PIPELINE_STAGE_NAMES[1])
    if deep_state.tier == "skip":
        alt_issues: list[dict[str, Any]] = []
        print_dim(console, "Skipping alternatives -- trivial diff")
    else:
        async with phase_scope(DaydreamPhase.ALTERNATIVES):
            alt_issues = await phase_alternative_review(
                ctx.backend_for("wonder"),
                ctx.work,
                deep_state.diff_path,
                intent_summary,
                exploration_dir=deep_state.exploration_dir,
                diff_text=_ttt_diff_text(ctx),
                strategy=ctx.strategy("alternatives"),
                run_context=ctx.run_context,
            )

    alts_p = _alternatives_path(deep_state.dd)
    alts_p.write_text(json.dumps(alt_issues, indent=2))
    deep_state.alts_path = alts_p


async def _step_wonder_and_per_stack(ctx: FlowContext) -> None:
    """Wonder (TTT alternative-review) alongside the per-stack review fan-out.

    On a fresh multi-stack run the two are siblings in one task group: wonder
    only feeds the merge agent and the dedup pre-filter, so the reviewers do not
    need to wait for it. Their prompts drop the ``alternatives.json`` pointer,
    since the file does not exist yet.

    Single-stack mode and every ``--start-at`` resume keep today's serial order
    and the pointer — in single-stack mode there is no merge agent, so the
    reviewer pointer is the ONLY path wonder findings take into the report.
    """
    deep_state = DeepState(ctx.data)
    # A resume (--start-at per-stack/merge/fix) skips wonder entirely — its
    # artifact is already on disk, which is also why the pointer stays on.
    run_wonder = fresh_ttt(ctx.config)
    concurrent = run_wonder and not deep_state.single_stack_mode
    holder: dict[str, BaseException | None] = {"exc": None}

    async def _wonder_guarded() -> None:
        # Held, not degraded: a wonder failure must fail the run, but only after
        # the fan-out's outputs are on disk for a later resume.
        try:
            await _wonder(ctx)
        except Exception as exc:  # noqa: BLE001 -- re-raised after the join
            holder["exc"] = exc

    if run_wonder and not concurrent:
        await _wonder(ctx)

    async with anyio.create_task_group() as tg:
        if concurrent:
            tg.start_soon(_wonder_guarded)
        await _per_stack_body(ctx, include_alternatives=not concurrent)

    if holder["exc"] is not None:
        raise holder["exc"]


async def _per_stack_body(ctx: FlowContext, *, include_alternatives: bool) -> None:
    """Per-stack review fan-out, with failure persistence and resume reconstruction."""
    deep_state = DeepState(ctx.data)
    config = ctx.config
    dd = deep_state.dd
    stacks = deep_state.stacks

    failed_stacks: dict[str, str] = deep_state.failed_stacks
    if config.start_at not in ("merge", "fix"):
        print_stage_progress(console, 3, 5, _PIPELINE_STAGE_NAMES[2])
        async with phase_scope(DaydreamPhase.DEEP, stage="review"):
            per_stack_outputs, failed_stacks = await phase_per_stack_reviews(
                ctx.backend_for("per_stack_review"),
                ctx.work,
                stacks,
                diff_path=deep_state.diff_path,
                intent_path=deep_state.intent_path,
                alternatives_path=deep_state.alts_path,
                exploration_dir=deep_state.exploration_dir,
                diff_text=deep_state.diff,
                intent_authoritative=deep_state.intent_authoritative,
                include_alternatives=include_alternatives,
                strategies={
                    "discovery.per_stack": ctx.strategy("discovery.per_stack"),
                    "discovery.structural": ctx.strategy("discovery.structural"),
                    "discovery.generic_fallback": ctx.strategy("discovery.generic_fallback"),
                },
                # Issue #731: always write deterministic coverage receipts so
                # the sweep credits reviewed files on every run (decoupled from
                # sharding; #740 updates the evidence gate and bounds).
                write_coverage_receipts=True,
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
        # Persist so a later `--start-at merge` resume can still surface
        # uncovered stacks (the in-memory failure map otherwise dies here).
        failures_p = per_stack_failures_path(dd)
        if failed_stacks:
            failures_p.write_text(json.dumps(failed_stacks, indent=2, sort_keys=True))
        elif failures_p.exists():
            # Fresh successful run supersedes any stale failures record.
            failures_p.unlink()
    else:
        # Resume: resurrect any prior failure summary before reconstructing
        # outputs, so failed stacks never re-enter the parse pipeline.
        from daydream.deep.artifacts import per_stack_review_path

        failures_p = per_stack_failures_path(dd)
        loaded = _load_failures(failures_p)
        # Surface a prior cross-stack synthesis failure (issue #361): the
        # structured ``MERGE_FAILURE_KEY`` entry is deliberately excluded from
        # ``failed_stacks`` (below) so it can't be misread as a failed stack /
        # garbled "Uncovered stacks" line, but resuming into a *partial* review
        # must not look clean -- say so explicitly so a ``--start-at fix``
        # relaunch doesn't fix + commit partial findings as if the cross-stack
        # merge had succeeded.
        _warn_prior_merge_failure(loaded)
        # Legacy entries are ``{stack_name: reason}`` str->str. Skip the
        # structured merge-failure entry (``MERGE_FAILURE_KEY``, a dict)
        # so it is never misread as a failed stack that would surface as
        # a garbled "Uncovered stacks" line on a resume (issue #361).
        failed_stacks = {
            str(k): str(v) for k, v in loaded.items() if isinstance(v, str)
        }
        per_stack_outputs = {
            stack.stack_name: per_stack_review_path(dd, stack.stack_name)
            for stack in stacks
            if stack.stack_name not in failed_stacks
        }
    deep_state.per_stack_outputs = per_stack_outputs
    deep_state.failed_stacks = failed_stacks


async def _step_per_stack_parse(ctx: FlowContext) -> Stop | None:
    """Load per-stack records (written by the reviewers) + structural partition.

    Issue #745 (AC4): there is no separate ``parse-<stack>`` stage -- the
    per-stack reviewers emit PER_STACK_RECORD_SCHEMA records directly. This
    step loads them off disk (same path fresh runs and ``--start-at merge``
    resumes take) and partitions the structural meta-stack records out before
    dedup.
    """
    deep_state = DeepState(ctx.data)
    dd = deep_state.dd
    stacks = deep_state.stacks
    failed_stacks: dict[str, str] = deep_state.failed_stacks

    print_stage_progress(console, 4, 5, _PIPELINE_STAGE_NAMES[3])
    # Issue #745 (AC4): the per-stack reviewers emit PER_STACK_RECORD_SCHEMA
    # records directly (no separate `parse-<stack>` stage), so BOTH a fresh run
    # and a `--start-at merge` resume load the on-disk per-stack records. A
    # fresh run separates records by the structural meta-stack below exactly as
    # the resume path does.
    per_stack_records_paths: list[Path] = []
    all_records: list[dict[str, Any]] = []
    record_sources: list[str] = []
    recorder = get_current_recorder()
    # Issue #742/#745: reconcile each stack's per-file verdicts against its own
    # completed-read evidence NOW that EVERY review fork is finalized on disk
    # (record-writing moved into the fan-out; reconciliation needs all forks
    # present, so it happens here after the task group, not inside it -- the
    # fork cache would otherwise read an incomplete set). Fail-open: a missing
    # fork degrades to ``[]`` (unread files stay swept, never recorded clean).
    stack_files: dict[str, list[str]] = {s.stack_name: list(s.files) for s in stacks}
    # Require a records file per detected stack (except ones in
    # `failed_stacks`). A bare glob would silently drop a stack whose records
    # file is absent, yielding a merged report missing a bucket. The same
    # per-stack records_path also drives verdict reconciliation below.
    expected_paths: list[Path] = []
    missing_stacks: list[str] = []
    for stack in stacks:
        if stack.stack_name in failed_stacks:
            continue
        records_path = per_stack_records_path(dd, stack.stack_name)
        if not records_path.is_file():
            missing_stacks.append(stack.stack_name)
            continue
        loaded = json.loads(records_path.read_text())
        issues = _records_issues_or_empty(loaded)
        declared = loaded.get("verdicts") if isinstance(loaded, dict) else []
        declared = declared if isinstance(declared, list) else []
        # Issue #745/#774: a `--start-at merge`/`fix` resume replays this step
        # under a NEW session id, so the current session carries none of the
        # prior run's `deep-<stack>` review forks and `_stack_review_reads`
        # resolves to ``[]`` here. Re-running the reconciliation would downgrade
        # the prior run's finalized clean verdicts back to `not_reviewed` and
        # rewrite them to disk. The on-disk verdicts are already finalized;
        # reconcile (and rewrite) only when this session actually ran the
        # per-stack review fan-out above.
        if ctx.config.start_at not in ("merge", "fix"):
            verdicts = _reconcile_stack_verdicts(
                dd.parent,
                recorder,
                stack.stack_name,
                assigned_files=stack_files[stack.stack_name],
                declared_verdicts=declared,
                parsed_records=issues,
            )
            records_path.write_text(json.dumps({"issues": issues, "verdicts": verdicts}, indent=2))
        expected_paths.append(records_path)
    if missing_stacks:
        print_error(
            console,
            "Missing Per-Stack Records",
            "Missing parsed records for: " + ", ".join(sorted(missing_stacks)),
        )
        return Stop(1)
    # Issue #309: a prior run's uncovered-file sweep records are
    # per-stack-style findings already finalized on disk (the sweep itself
    # is a no-op on merge resume). Load them so a merge resume keeps the
    # sweep's findings instead of silently dropping them.
    sweep_path = per_stack_records_path(dd, "uncovered")
    if sweep_path.is_file():
        expected_paths.append(sweep_path)
    for records_path in sorted(expected_paths):
        loaded = json.loads(records_path.read_text())
        # Issue #742: per-stack records files carry the dict shape
        # ``{"issues": [...], "verdicts": [...]}``. Merge consumes a bare
        # issues list, so normalize the dict shape here; legacy bare-list
        # records pass through unchanged.
        records = _records_issues_or_empty(loaded)
        # Issue #1111: this loop is the single choke point that populates
        # ``ctx.data["records"]`` -- on a fresh run and on a ``--start-at merge``
        # resume alike -- so it is where the ``uid`` invariant is guaranteed.
        # ``phase_per_stack_reviews`` stamps at record birth, but two shapes
        # still arrive here unstamped: records written by a run from before this
        # field existed simply lack the key, and so would any future producer
        # that missed the birth stamp. Re-deriving the uid from
        # ``(stack_name, position)`` reproduces exactly the value the producing
        # run would have minted -- which is precisely why the format is
        # deterministic rather than a uuid4 -- so the backfill is
        # indistinguishable from a birth stamp. ``stamp_record_uids`` normalizes
        # the records filename to a bare stack name itself and PRESERVES any uid
        # already on disk, so this is idempotent and never re-mints a uid the
        # producing run already handed out (which matters on resume, where the
        # on-disk list may be shorter than the list those uids were minted from).
        stamp_record_uids(records, records_path.name)
        per_stack_records_paths.append(records_path)
        source_name = records_path.name
        all_records.extend(records)
        record_sources.extend(source_name for _ in records)

    # Issue #1111: every uid-keyed stage downstream of here -- the dedup
    # pre-filter's b-side drop, adjudication's drop set, the structural
    # partition/rejoin, and ``_rewrite_stack_records``' file routing -- resolves
    # a record by its uid. Two records sharing one uid make each of those act on
    # the wrong record, which is exactly the over-delete this field exists to
    # prevent (see ``_drop_cross_stack_duplicates``). Checked here, over the
    # whole loaded pool, BEFORE the structural partition below splits it: one
    # check then covers both sides and also catches a collision that spans them.
    #
    # FATAL rather than a warning. A uid is host-minted and deterministic with
    # no content-derived input, so a collision is never the near-miss judgement
    # call a fingerprint match is -- it means two records files claim the same
    # stack name, or an artifact was written with a partially stamped list.
    # Continuing would emit a report quietly missing findings, and CLAUDE.md's
    # rule is that loss is never silently absorbed. A false positive costs a
    # bounded, visible, actionable stop (the message names the colliding uids
    # and the remedy); a false negative costs an invisible wrong answer.
    duplicate_uids = duplicate_record_uids(all_records)
    if duplicate_uids:
        loaded_names = ", ".join(sorted(path.name for path in per_stack_records_paths))
        print_error(
            console,
            "Duplicate Record Identities",
            f"Per-stack records carry duplicate uid(s): {', '.join(duplicate_uids)}\n\n"
            f"Records were loaded from: {loaded_names}\n\n"
            "Every stage after this one (dedup, arbitration, suppression, the structural fold, "
            "the per-stack records rewrite) resolves a record by its uid, so continuing would "
            "drop or revise the wrong findings.\n"
            "Re-run without --start-at to regenerate the per-stack records.",
        )
        return Stop(1)

    # Partition structural meta-stack records out before dedup: its lens
    # (file-size budgets, layering, canonical-helper gaps) differs from the
    # language stacks and collapsing it into their dedup pool would demote
    # those findings. The partition keys on the stack name encoded in each
    # record's own uid (issue #1111) and rebuilds ``all_records`` /
    # ``record_sources`` as pairs, so the positional index invariant between
    # those two lists survives the split exactly as before.
    #
    # This test used to compare ``src`` against both ``STRUCTURE_STACK_NAME``
    # and the records filename, on the stated belief that ``source`` is a bare
    # stack name on a fresh run and a filename on resume. That belief was
    # wrong: ``source_name`` is ``records_path.name`` above on every path, fresh
    # run and resume alike, so the bare-stack-name half of the test was dead
    # code (the only bare-name source in this module is the uncovered sweep's
    # ``"uncovered"``). A uid's stack half has one spelling and cannot rot that
    # way.
    #
    # The partition is scoped to the dedup pre-filter and the merge agent's
    # record pool -- the two places that could collapse a structural finding
    # into a language bucket. It is NOT a partition out of adjudication: the
    # records are kept here under their own keys so ``_step_arbiter`` can put
    # them back in front of the contested-location branch, which is the one
    # mechanism designed to catch a structural/language twin (issue #1103).
    structural_path_candidate = per_stack_records_path(dd, STRUCTURE_STACK_NAME)
    structural_records: list[dict[str, Any]] = []
    structural_record_sources: list[str] = []
    if structural_path_candidate in per_stack_records_paths:
        structural_records_path: Path | None = structural_path_candidate
        per_stack_records_paths = [
            p for p in per_stack_records_paths if p != structural_path_candidate
        ]
        kept_pairs = []
        for rec, src in zip(all_records, record_sources, strict=True):
            if stack_name_from_uid(record_uid(rec)) == STRUCTURE_STACK_NAME:
                structural_records.append(rec)
                structural_record_sources.append(src)
            else:
                kept_pairs.append((rec, src))
        all_records = [rec for rec, _ in kept_pairs]
        record_sources = [src for _, src in kept_pairs]
    else:
        structural_records_path = None

    deep_state.records_paths = per_stack_records_paths
    deep_state.records = all_records
    deep_state.record_sources = record_sources
    deep_state.structural_records_path = structural_records_path
    deep_state.structural_records = structural_records
    deep_state.structural_record_sources = structural_record_sources
    return None


def _clear_sweep_artifacts(dd: Path) -> None:
    """Delete the uncovered-file sweep's owned artifacts from ``dd``.

    ``coverage-stats.json``, ``stack-uncovered-records.json``, and every
    ``uncovered-*-review.md`` are the sweep's outputs. A ``--start-at per-stack``
    resume re-runs the sweep, so any artifact left from the prior run must be
    removed BEFORE new per-stack work: otherwise a rerun whose sweep is
    disabled / finds nothing / produces no output would leave stale records
    behind, and a later merge resume would reload them as this run's coverage.
    Merge/fix resumes keep the artifacts (the sweep is a no-op there and the
    records must survive).

    Fail-CLOSED: this cleanup is resume-safety-critical, not best-effort
    diagnostics. When a targeted artifact cannot be removed (a ``OSError`` from
    ``unlink()``, or an artifact that survives the loop), the function raises an
    ``OSError`` with an actionable message and the per-stack resume stops --
    it must never continue with a stale ``stack-uncovered-records.json`` in
    place that a later merge resume would reload as current findings.
    """
    patterns = (
        "coverage-stats.json",
        "stack-uncovered-records.json",
        "uncovered-*-review.md",
    )
    for pattern in patterns:
        for path in dd.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
    remaining = [p for pattern in patterns for p in dd.glob(pattern)]
    if remaining:
        names = ", ".join(sorted(p.name for p in remaining))
        raise OSError(
            f"stale sweep artifact(s) could not be removed: {names}; "
            "refusing to resume per-stack"
        )


async def _step_uncovered_sweep(ctx: FlowContext) -> None:
    """Issue #309: fail-open second-pass sweep over diff files no reviewer read.

    After per-stack reviews + parse, computes which diff files no ``deep-``
    reviewer read (via ``analyze_coverage``), budget-filters them (hunk size +
    capacity cap), dispatches one cheap reviewer per surviving file, parses the
    findings into ordinary ``PER_STACK_RECORD_SCHEMA`` records, and appends
    them to ``ctx.data`` so the arbiter/merge consume them exactly like any
    per-stack stack's records. Coverage stats land in ``deep/coverage-stats.json``.

    Fail-open: any exception here is caught, logged as a warning, and the step
    returns normally -- the sweep must NEVER fail the run.
    """
    if ctx.config.start_at in ("merge", "fix"):
        # Resume: records are already finalized on disk; a sweep would
        # re-review stale coverage against a diff that already ran.
        return None
    async with phase_scope(DaydreamPhase.DEEP, stage="uncovered") as phase:
        try:
            await _run_uncovered_sweep(ctx, phase=phase)
        except Exception as exc:  # noqa: BLE001 -- fail-open: never fail the run
            phase.finish(LifecycleStatus.FAILED, LifecycleReasonCode.DOMAIN_FAILURE)
            print_warning(
                console,
                f"Uncovered-file sweep failed (fail-open): {type(exc).__name__}: {exc}",
            )


def _load_coverage_receipts(ctx: FlowContext) -> dict[str, Any] | None:
    """Load this run's coverage receipts for the sweep (issue #731).

    Receipts are written on every deep run (decoupled from sharding; the sweep
    always attempts the load). Fail-open: a missing or malformed receipts file
    degrades to ``None`` (never raises, never skips the sweep) and
    ``compute_uncovered_files`` takes its forensic Reads-only path --
    byte-identical to today's behavior without receipts.
    """
    deep_state = DeepState(ctx.data)
    try:
        loaded = json.loads(coverage_receipt_path(deep_state.dd).read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else None
    except (OSError, ValueError):
        return None


async def _run_uncovered_sweep(
    ctx: FlowContext, *, phase: "PhaseScopeHandle | None" = None
) -> None:
    """Run the uncovered-file sweep body (issue #309)."""
    deep_state = DeepState(ctx.data)
    from daydream.hunk_index import load_hunk_index

    config = ctx.config
    dd = deep_state.dd
    recorder = get_current_recorder()
    session_id = recorder.session_id if recorder is not None else None

    uncovered_files, coverage_stats = compute_uncovered_files(
        dd.parent, session_id, receipts=_load_coverage_receipts(ctx)
    )

    # Issue #644 — the sweep's block extraction must source the FULL on-disk
    # diff (``ctx.data["diff_path"]``, always written full at gather) because
    # the coverage file set above derives from the same full ``diff.patch``:
    # a bounded in-memory ``ctx.data["diff"]`` would silently route a
    # truncated-away file into ``skipped_small`` (block lookup -> None) and it
    # would never be swept. A read error propagates to the step's
    # fail-open wrapper (the sweep must NEVER fail the run); it is not
    # swallowed with a silent empty-diff fallback. A ctx built without
    # ``diff_path`` (defensive legacy fallback only, never the default)
    # degrades to the in-memory diff rather than crashing the sweep.
    full_diff = _read_full_diff(ctx)

    swept_files, skipped_small_files, skipped_capacity_files = filter_sweepable_files(
        uncovered_files,
        load_hunk_index(dd.parent),
        min_hunk_lines=_uncovered_sweep_min_hunk_lines(ctx),
        max_files=_uncovered_sweep_max_files(ctx),
    )

    stats: dict[str, Any] = {
        "pre_sweep": {
            "files_in_diff": coverage_stats["files_in_diff"],
            "files_read_by_reviewers": coverage_stats["files_read_by_reviewers"],
            "coverage_ratio": coverage_stats["coverage_ratio"],
            # Issue #336: propagate the missing-index gap into the persisted
            # stats so the report renders it instead of a false full-coverage
            # pass (``load_hunk_index`` still fails open -- reporting only).
            "hunk_index_missing": coverage_stats.get("hunk_index_missing", False),
            "uncovered_files": uncovered_files,
            # Issue #731: per-evidence-type coverage counts (source_read /
            # inline_hunk_reviewed / dependency_frontier_read); receipts are
            # written and loaded on every deep run (decoupled from sharding,
            # #740), so the counts surface whenever the run produced them.
            "coverage_by_evidence": coverage_stats.get("coverage_by_evidence", {}),
        },
        "attempted_files": swept_files,
        "completed_files": [],
        # Issue #309 finding 6: ``covered_files`` is filled from the POST-sweep
        # recompute (verified completed reads of the swept files) and may be a
        # strict subset of ``completed_files`` -- a review written without a
        # Read of the file is an attempt, never coverage. Until the recompute
        # runs it starts empty (fail-open: unverifiable means not claimed).
        "covered_files": [],
        # POST-sweep ratio is recomputed below after the sweep forks land; this
        # pre-sweep snapshot is the fallback when the sweep produces no reads.
        "post_sweep": {
            "files_read_by_reviewers": coverage_stats["files_read_by_reviewers"],
            "coverage_ratio": coverage_stats["coverage_ratio"],
        },
        "sweep_finding_count": 0,
        # The integer skip counts are derived from the filename lists so the
        # two views cannot diverge (issue #309 finding 10).
        "sweep_skipped_capacity": len(skipped_capacity_files),
        "sweep_skipped_small_hunks": len(skipped_small_files),
        "sweep_skipped_capacity_files": skipped_capacity_files,
        "sweep_skipped_small_hunks_files": skipped_small_files,
    }
    stats_p = dd / "coverage-stats.json"

    if not swept_files:
        # A re-run that finds nothing to sweep must still refresh the records
        # artifact to a current empty list so a stale prior file (from the run
        # being resumed) cannot linger and be reloaded by a later merge resume.
        if config.start_at == "per-stack":
            per_stack_records_path(dd, "uncovered").write_text(json.dumps([]))
        stats_p.write_text(json.dumps(stats, indent=2))
        if phase is not None:
            phase.finish(
                LifecycleStatus.SKIPPED, LifecycleReasonCode.NO_ELIGIBLE_WORK
            )
        return

    # Cheap-tier dispatch (parse tier), parallel, in diff order. Each sweep
    # fork is `deep-uncovered-<n>` so post-run analyze_coverage counts its
    # reads (the coverage-ratio-improves acceptance criterion). Issue #745
    # (AC4): the sweep reviewer emits UNCOVERED_SWEEP_SCHEMA structured output
    # directly -- there is no `parse-uncovered-<n>` fork.
    parse_backend = ctx.backend_for("parse")
    limiter = anyio.CapacityLimiter(effective_fanout_concurrency(10, parse_backend))
    completed_reviews: set[str] = set()
    sweep_failures: dict[str, str] = {}
    sweep_records_by_file: dict[str, list[dict[str, Any]]] = {}

    descriptors = tuple(
        f"deep-uncovered-{n}" for n, _file in enumerate(swept_files)
    )
    # Loop-invariant for the whole fan-out: every sweep fork sanctions the same
    # intent and pre-scan artifacts, so they are captured once.
    sweep_inputs = {"intent": deep_state.intent_path}
    sweep_exploration = deep_state.exploration_dir_or_none
    if isinstance(sweep_exploration, Path):
        sweep_inputs["exploration-summary"] = sweep_exploration / "summary.md"
        sweep_inputs["exploration-affected-files"] = sweep_exploration / "affected_files.md"
    sanctioned_inputs = (
        prepare_sanctioned_inputs(parse_backend, ctx.work.repo, sweep_inputs, read_only=False)
        if ctx.artifacts is not None
        else None
    )
    async with dispatch_scope(
        recorder, phase=DaydreamPhase.DEEP, descriptors=descriptors
    ) as dispatch:
        async with anyio.create_task_group() as tg:
            for n, file in enumerate(swept_files):
                output_path = dd / f"uncovered-{n}-review.md"
                prompt = build_uncovered_sweep_prompt(
                    strategy=ctx.strategy("uncovered_review"),
                    file=file,
                    hunks=diff_block_for_file(full_diff, file) or "",
                    intent_path=deep_state.intent_path,
                    cwd=ctx.work.repo,
                    output_path=output_path,
                    exploration_dir=deep_state.exploration_dir,
                )

                async def _sweep_one(
                    file: str = file,
                    task_prompt: str = prompt,
                    n: int = n,
                ) -> None:
                    async with limiter:
                        try:
                            async with maybe_fork(
                                recorder,
                                f"deep-uncovered-{n}",
                                dispatch=dispatch,
                            ):
                                structured, _, budget_reason = await run_agent(
                                    parse_backend,
                                    ctx.work.repo,
                                    task_prompt,
                                    phase=DaydreamPhase.DEEP,
                                    output_schema=UNCOVERED_SWEEP_SCHEMA,
                                    tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
                                    wall_budget_s=DEFAULT_WALL_BUDGET_S,
                                    sanctioned_inputs=sanctioned_inputs,
                                    run_context=ctx.run_context,
                                )
                            if budget_reason:
                                sweep_failures[file] = (
                                    f"budget exhausted: {budget_reason}"
                                )
                            elif not isinstance(structured, dict):
                                sweep_failures[file] = "no structured output produced"
                            else:
                                issues = structured.get("issues")
                                # Schema validation guarantees ``issues`` is a list
                                # but not that each entry is a dict (nested item
                                # validity is deliberately the consumers' salvage
                                # domain -- see ``agent.py``); a non-dict entry
                                # would otherwise crash ``stamp_record_uids`` below
                                # and discard this whole fail-open sweep.
                                issues = (
                                    [
                                        item
                                        for item in issues
                                        if isinstance(item, dict)
                                    ]
                                    if isinstance(issues, list)
                                    else []
                                )
                                sweep_records_by_file[file] = issues
                                # Structured records are the authoritative sweep
                                # output. Markdown review files are optional backend
                                # byproducts and cannot gate persistence. Coverage is
                                # still computed independently from verified Reads.
                                completed_reviews.add(file)
                        except Exception as exc:  # noqa: BLE001 -- parallel isolation; fail-open
                            sweep_failures[file] = f"{type(exc).__name__}: {exc}"

                tg.start_soon(_sweep_one)
        if sweep_failures:
            status = (
                LifecycleStatus.PARTIAL
                if completed_reviews
                else LifecycleStatus.FAILED
            )
            reason = (
                LifecycleReasonCode.SOME_CHILDREN_FAILED
                if completed_reviews
                else LifecycleReasonCode.ALL_CHILDREN_FAILED
            )
            if dispatch is not None:
                dispatch.finish(status, reason)

    # Recompute coverage AFTER the sweep so the report shows the ratio the
    # sweep actually achieved (the ``deep-uncovered-*`` forks' completed reads
    # now count), never the pre-sweep snapshot. The recompute is fail-open: a
    # failure here falls back to the pre-sweep numbers already stored. The same
    # recompute drives ``covered_files`` (issue #309 finding 6): a swept file is
    # covered only when the post-sweep uncovered list no longer contains it --
    # i.e. a verified completed read of the file happened. A successful review
    # output WITHOUT a read leaves the file in the uncovered list, so it is
    # never claimed as covered.
    post_uncovered: list[str] | None = None
    try:
        post_uncovered, post_coverage = compute_uncovered_files(dd.parent, session_id)
        stats["post_sweep"] = {
            "files_read_by_reviewers": post_coverage["files_read_by_reviewers"],
            "coverage_ratio": post_coverage["coverage_ratio"],
        }
        stats["covered_files"] = sorted(f for f in completed_reviews if f not in post_uncovered)
    except Exception:  # noqa: BLE001 -- fail-open: keep the pre-sweep fallback
        pass

    # Merge the sweep records into the per-stack record set exactly like the
    # per-stack parse loop does, so arbiter/merge consume them as ordinary
    # per-stack records (no separate score path). The records file is written
    # whenever at least one sweep review produced output -- an emptied sweep
    # stack is ``[]``, mirroring ``_rewrite_stack_records`` semantics. When NO
    # review produced output, a current empty records artifact is still written
    # so a stale file from a prior run cannot linger.
    sweep_records: list[dict[str, Any]] = []
    for file in sorted(sweep_records_by_file):
        sweep_records.extend(sweep_records_by_file[file])
    stats["sweep_failures"] = {**sweep_failures}
    stats["completed_files"] = sorted(completed_reviews)
    # Issue #309 finding 6: per-file attempt status. A completed review output
    # is a completed ATTEMPT; only files with a verified post-sweep completed
    # read are "read". Anything else is "reviewed (hunks only)" and must not
    # move files_read_by_reviewers / coverage_ratio.
    covered_set = set(stats.get("covered_files") or [])
    stats["sweep_attempt_status"] = {
        file: ("read" if file in covered_set else "reviewed (hunks only)")
        for file in sorted(completed_reviews)
    }
    if completed_reviews:
        records_path = per_stack_records_path(dd, "uncovered")
        # Issue #1111: the sweep is the pipeline's SECOND record-birth site, and
        # the load-time backfill in ``_step_per_stack_parse`` cannot reach it --
        # ``STEPS`` orders ``per-stack-parse`` BEFORE ``uncovered-sweep``, and on
        # a fresh run ``stack-uncovered-records.json`` does not exist yet when
        # that loop runs. So stamp here, BEFORE the write, and both the on-disk
        # artifact and the in-memory pool extended below carry uids. A
        # ``--start-at merge`` resume reloads this same file through that loop,
        # where ``stamp_record_uids``' preserve-if-present behaviour keeps these
        # exact uids instead of re-minting them.
        #
        # These uids need no duplicate check of their own: they are freshly
        # minted over an unstamped list under a stack name no other producer
        # uses, and the pool they join cannot already hold an ``uncovered:N``.
        # The sweep is disabled outright on a merge/fix resume
        # (``_uncovered_sweep_enabled``), a per-stack resume deletes this file
        # before any new per-stack work (``_clear_sweep_artifacts``), and a fresh
        # run wipes the whole deep dir -- so on every path that reaches this
        # line, the parse loop found no uncovered records file to load.
        stamp_record_uids(sweep_records, "uncovered")
        records_path.write_text(json.dumps(sweep_records, indent=2))
        deep_state.records_paths.append(records_path)
        deep_state.records.extend(sweep_records)
        deep_state.record_sources.extend("uncovered" for _ in sweep_records)
        stats["sweep_finding_count"] = len(sweep_records)
    else:
        # No structured review produced output: on a re-run, write a current
        # empty records artifact so a stale file cannot linger.
        if config.start_at == "per-stack":
            per_stack_records_path(dd, "uncovered").write_text(json.dumps([]))
    stats_p.write_text(json.dumps(stats, indent=2))
    if sweep_failures and phase is not None:
        phase.finish(
            LifecycleStatus.PARTIAL
            if completed_reviews
            else LifecycleStatus.FAILED,
            LifecycleReasonCode.SOME_CHILDREN_FAILED
            if completed_reviews
            else LifecycleReasonCode.ALL_CHILDREN_FAILED,
        )


def _warn_prior_merge_failure(loaded: dict[str, Any]) -> None:
    """Warn on resume that a prior cross-stack synthesis failed (issue #361).

    The structured ``MERGE_FAILURE_KEY`` entry is deliberately excluded from
    ``failed_stacks`` (see caller) so it can't be misread as a failed stack /
    garbled "Uncovered stacks" line, but resuming into a *partial* review must
    not look clean -- say so explicitly so a ``--start-at fix`` relaunch doesn't
    fix + commit partial findings as if the cross-stack merge had succeeded.
    """
    merge_entry = loaded.get(MERGE_FAILURE_KEY)
    if merge_entry is not None:
        merge_message = (
            merge_entry.get("message")
            if isinstance(merge_entry, dict)
            else str(merge_entry)
        )
        print_warning(
            console,
            "Prior cross-stack synthesis failed; merged results are PARTIAL. "
            f"{merge_message} (issue #361) -- this resume fixes/verifies the "
            "partial per-stack findings as-is.",
        )


@lru_cache(maxsize=None)
def _loaded_review_forks(daydream_dir: Path, session_id: str) -> tuple[dict[str, Any], ...]:
    """Trajectory forks for a session, parsed once (issue #742 finding 4).

    ``load_trajectories`` re-parses every JSON file under
    ``runs/<session>/trajectories/`` on every call; the per-stack parse task
    group fans ``_stack_review_reads`` out once per stack concurrently, so cache
    the parsed fork set per ``(daydream_dir, session_id)`` and reap a single
    parse. The gate only consumes the ``forked`` list; the parsed dicts are
    read-only for the run, so caching is safe. ``Path`` and ``str`` are
    hashable, satisfying ``functools.lru_cache``'s key contract.
    """
    return tuple(load_trajectories(daydream_dir, session_id).get("forked", []))


def _stack_review_reads(
    daydream_dir: Path, recorder: TrajectoryRecorder | None, stack_name: str
) -> set[str]:
    """Completed reads from the ``deep-<stack>`` review fork (issue #742).

    The clean-verdict gate consumes evidence, never reviewer self-report: the
    reads that count are the ones the per-stack review agent actually completed
    in its own fork trajectory (``deep-<stack_name>.json``, written when the
    review fork exits — always before parse starts). The fork set is loaded once
    and cached (``_loaded_review_forks``) so N concurrent per-stack parse tasks
    do not re-parse the session's trajectories on every call (finding 4). A
    ``None`` recorder (no trajectory recording this run) or an absent fork
    yields the empty set (fail-open: every assigned file without a finding
    resolves to ``not_reviewed`` and is swept, never recorded clean).
    """
    if recorder is None:
        return set()
    # A sharded stack is named ``deep-<stack>#<n>`` but its fork is
    # filesystem-slugified (``_safe_descriptor``) to ``deep-<stack>-<n>`` on
    # disk, so match the slug -- not the raw descriptor (#742 finding 1).
    lookup = _safe_descriptor(f"deep-{stack_name}")
    for fork in _loaded_review_forks(daydream_dir, recorder.session_id):
        label = _agent_label(fork["_source_file"])
        if label == lookup or label.startswith(f"{lookup}--"):
            return _completed_read_paths(fork)
    return set()


def _reconcile_stack_verdicts(
    daydream_dir: Path,
    recorder: TrajectoryRecorder | None,
    stack_name: str,
    *,
    assigned_files: list[str],
    declared_verdicts: list[dict[str, Any]],
    parsed_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fail-open per-file verdict reconciliation for one stack (issue #742).

    Collects the evidence-gathering glue the per-stack parse needs: the stack's
    own ``deep-<stack>`` review-fork completed reads plus the parse's
    normalized finding files, then resolves the evidence-gated verdict for each
    assigned file. Fail-open (never fails the run, never records an unread file
    clean): any failure (e.g. a missing fork) yields ``[]`` so the stack's
    records stay writable and its unread files resolve to ``not_reviewed``
    instead of a pass.
    """
    try:
        stack_reads = _stack_review_reads(daydream_dir, recorder, stack_name)
        finding_files = _finding_files_from_records(parsed_records)
        return resolve_per_stack_verdicts(
            assigned_files=assigned_files,
            declared_verdicts=declared_verdicts,
            completed_read_paths=stack_reads,
            finding_files=finding_files,
        )
    except Exception:  # noqa: BLE001 -- fail-open: never fail the run on a missing fork
        return []
