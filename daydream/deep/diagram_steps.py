"""Grounded diagram authoring, repair, report application, and publication."""

from __future__ import annotations

import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from daydream.agent import console, run_agent
from daydream.artifact_visibility import artifact_dir_for
from daydream.backends import effective_fanout_concurrency
from daydream.config import (
    DEFAULT_DIAGRAM_MIN_BRANCH_POINTS,
    DEFAULT_DIAGRAM_MIN_CODE_FILES,
    DEFAULT_DIAGRAM_MIN_MODULES,
    DEFAULT_TOOL_CALL_BUDGET,
    DEFAULT_WALL_BUDGET_S,
    DIAGRAM_KINDS,
    DIAGRAM_MODES,
)
from daydream.deep.artifacts import diagram_markdown_path, diagram_path, merged_report_path
from daydream.deep.coverage import _completed_read_paths
from daydream.deep.detection import detect_stacks
from daydream.deep.diagram_grounding import RepoSymbols, ground_flowchart, ground_sequence
from daydream.deep.diagram_render import render_diagram_blocks, render_flowchart_mermaid, render_sequence_mermaid
from daydream.deep.diagram_schema import (
    FLOWCHART_SPEC_SCHEMA,
    SEQUENCE_SPEC_SCHEMA,
    coerce_flowchart_spec,
    coerce_sequence_spec,
)
from daydream.deep.diagram_trigger import Eligibility, decide_eligibility
from daydream.deep.diagram_types import DiagramResult, DiagramThresholds
from daydream.deep.diff import _ttt_diff_text
from daydream.deep.prompts import build_diagram_repair_prompt
from daydream.deep.render import insert_diagrams_section
from daydream.deep.settings import _resolve_config_value
from daydream.deep.state import DeepState
from daydream.extensions import get_registry
from daydream.extensions.api import Stop
from daydream.flows.engine import FlowContext
from daydream.json_utils import atomic_write_json
from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    SanctionedInputTransport,
    fits_inline_diff_budget,
    prepare_sanctioned_inputs,
)
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.trajectory import (
    DaydreamPhase,
    LifecycleReasonCode,
    LifecycleStatus,
    dispatch_scope,
    get_current_recorder,
    maybe_fork,
    phase_scope,
)
from daydream.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from daydream.runner import RunConfig
    from daydream.trajectory import DispatchHandle, PhaseScopeHandle, TrajectoryRecorder


@dataclass(frozen=True)
class DiagramSettings:
    """One run's resolved grounded-diagram configuration (issue #1113).

    Attributes:
        mode: The resolved diagram mode -- one of
            :data:`~daydream.config.DIAGRAM_MODES`.
        thresholds: The eligibility thresholds the decision is taken against.
        service_roots: Declared service-root globs for participant grouping;
            empty means "fall back to the improve list, then to inference".
    """

    mode: str
    thresholds: DiagramThresholds
    service_roots: list[str]


def _diagram_mode_for(config: RunConfig, mode: str) -> str:
    """Resolve the diagram mode for a run: CLI > file config > ``"auto"``.

    Split from :func:`_resolved_diagram_mode` so the spine can consult it
    before a ``FlowContext`` exists (it decides whether to build the import
    graph the cross-module rule needs).

    In ``--diagram-only`` mode ``config.diagram`` carries the requested kind
    and a repository file's ``mode = "off"`` is deliberately ignored: the user
    asked for this run by name, and silently doing nothing would be the worst
    possible answer. Every other mode honors the file's off switch.
    """
    if config.diagram in DIAGRAM_MODES:
        return str(config.diagram)
    if mode == "diagram":
        return "auto"
    file_config = config.file_config
    file_mode = file_config.diagram_mode if file_config is not None else None
    return file_mode if file_mode in DIAGRAM_MODES else "auto"


def _resolved_diagram_mode(ctx: FlowContext) -> str:
    """The active diagram mode for this flow context (issue #1113)."""
    deep_state = DeepState(ctx.data)
    return _diagram_mode_for(ctx.config, deep_state.mode)


def _diagram_settings(ctx: FlowContext) -> DiagramSettings:
    """Resolve mode + thresholds + service roots for the diagram step.

    Thresholds resolve through :func:`_resolve_config_value` (``RunConfig``
    attr, then file config, then the ``config.py`` default); there are no
    per-threshold CLI flags, so in practice the file config is the only
    override source. Resolved once per step and passed by value, which is what
    keeps ``decide_eligibility`` a pure function of its arguments and its
    verdict reproducible from ``diagram.json``.
    """
    file_config = ctx.config.file_config
    return DiagramSettings(
        mode=_resolved_diagram_mode(ctx),
        thresholds=DiagramThresholds(
            min_code_files=_resolve_config_value(
                ctx.config, "diagram_min_code_files", DEFAULT_DIAGRAM_MIN_CODE_FILES
            ),
            min_modules=_resolve_config_value(
                ctx.config, "diagram_min_modules", DEFAULT_DIAGRAM_MIN_MODULES
            ),
            min_branch_points=_resolve_config_value(
                ctx.config, "diagram_min_branch_points", DEFAULT_DIAGRAM_MIN_BRANCH_POINTS
            ),
        ),
        service_roots=list(file_config.diagram_service_roots) if file_config is not None else [],
    )


# --- Grounded diagrams (issue #1113) ----------------------------------------
#
# Two agent turns at most per kind, and no mermaid from either of them: the
# model proposes a JSON spec whose every element carries file:line evidence,
# ``ground_*`` verifies each element against the head tree and the turn's own
# read receipts, one repair turn fixes or removes what failed, survivors are
# pruned/capped, and a pure renderer emits the diagram. What the checker could
# not confirm is never drawn.


def _diagram_result(status: str, reason: str | None) -> DiagramResult:
    """A no-spec result for a kind that never produced one.

    ``skipped`` (not eligible) and ``failed`` (agent or budget error) share
    this shape: no spec, no grounding, no mermaid, and a reason the omission
    notice and ``diagram.json`` can both render.
    """
    return {
        "status": status,
        "reason": reason,
        "spec_proposed": None,
        "spec_final": None,
        "grounding": None,
        "omit_reasons": [],
        "mermaid": None,
    }


def _diagram_read_paths(fork_path: Path | None) -> set[str]:
    """Completed diagram-phase read paths recorded in one fork's trajectory.

    Fail-CLOSED: a missing, unreadable, or malformed fork file yields the empty
    set, which makes every citation fail ``FILE_NOT_READ_BY_MODEL``. The
    alternative -- treating "no receipts" as "all reads happened" -- would turn
    a recording failure into an unverified diagram.

    The fork file is written by ``_ForkCM.__aexit__`` even when the body
    raised, but ``_write`` short-circuits on a fork with no steps, so absence
    is a real and expected case.

    Note that the receipts are the UNION across ``run_agent``'s retry attempts:
    a failed retryable attempt's invocation is still flushed into the fork, so
    a file read during an attempt that later errored still counts. That is
    fail-open in the model's favour and is deliberate -- the read did happen,
    and the file content it returned is what grounding cares about.
    """
    if fork_path is None:
        return set()
    try:
        trajectory = json.loads(fork_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(trajectory, dict):
        return set()
    return _completed_read_paths(trajectory, phases={DaydreamPhase.DIAGRAM.value})


def _files_by_module(eligibility: Eligibility) -> dict[str, list[str]]:
    """Group the changed code files by module for the sequence prompt."""
    grouped: dict[str, list[str]] = {}
    for path, module in sorted(eligibility.modules.items()):
        grouped.setdefault(module, []).append(path)
    return grouped


def _inline_exploration_text(exploration_dir: Path | None) -> tuple[str | None, str | None]:
    """Best-effort host-side reads of the exploration summary and dependencies.

    Issue #1123: on read-only disposable-clone backends the host-only
    ``.daydream/`` artifacts are absent from the clone, so the prompt must
    carry their content inline. Both files are read best-effort (an
    ``OSError`` yields ``None`` — the block is omitted, never faked) and
    share one ``INLINE_DIFF_BUDGET_BYTES`` budget: the summary takes the
    first slice, the dependency edges fill the remainder. Each over-budget
    piece is truncated with an explicit marker rather than dropped. The
    summary is scrubbed first (``_scrub_exploration_summary``) so the
    standalone-artifact scaffolding the pre-scan writer emits cannot dangle
    in the inline rendering.
    """
    if exploration_dir is None:
        return None, None
    budget = INLINE_DIFF_BUDGET_BYTES
    try:
        summary: str | None = (exploration_dir / "summary.md").read_text(encoding="utf-8")
    except OSError:
        summary = None
    if summary is not None:
        summary = _scrub_exploration_summary(summary)
        encoded = summary.encode("utf-8")
        if len(encoded) > budget:
            truncated = encoded[:budget].decode("utf-8", errors="ignore")
            summary = (
                f"{truncated}\n[exploration summary truncated]\n" if truncated else None
            )
            encoded = summary.encode("utf-8") if summary is not None else b""
        if summary is not None:
            budget = max(budget - len(encoded), 0)
    try:
        dependencies: str | None = (exploration_dir / "dependencies.md").read_text(
            encoding="utf-8"
        )
    except OSError:
        dependencies = None
    if dependencies is not None:
        if budget <= 0:
            dependencies = None
        else:
            encoded = dependencies.encode("utf-8")
            if len(encoded) > budget:
                truncated = encoded[:budget].decode("utf-8", errors="ignore")
                dependencies = (
                    f"{truncated}\n[exploration summary truncated]\n"
                    if truncated
                    else None
                )
    return summary, dependencies


def _scrub_exploration_summary(summary: str) -> str:
    """Drop the standalone-artifact scaffolding ``summary.md`` carries.

    The pre-scan writer emits the summary as a standalone artifact: an
    embedded untrusted-content blockquote plus a table whose rows name the
    sibling artifacts (``affected_files.md``, ``conventions.md``,
    ``dependencies.md``). On a disposable clone only the summary body and the
    dependency edges travel inline — the siblings are absent, so rows that
    name them would dangle, and the clone-mode block builder re-emits the
    boundary as its opening block. Strip that scaffolding and keep the prose.
    """
    kept: list[str] = []
    in_artifact_table = False
    for line in summary.splitlines():
        if line == f"> {UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}":
            continue  # re-emitted by the clone-mode block builder
        if line == "| File | Contents |":
            in_artifact_table = True
            continue
        if in_artifact_table and line.startswith("|"):
            continue  # separator and every data row (each names a sibling)
        in_artifact_table = False
        kept.append(line)
    return "\n".join(kept)


def _prompt_builder_accepts_inline_kwargs(builder: Any) -> bool:
    """Whether ``builder`` accepts the clone-mode inline kwargs.

    Fork overrides written against the documented extension contract predate
    ``clone_mode``/``inline_exploration``/``inline_dependencies``; splatting
    them into such a builder would raise ``TypeError`` on every
    disposable-clone run, degrading the kind to failed. The builtin builders
    accept all three; a legacy override keeps the documented kwarg set, with
    ``exploration_dir`` arriving as ``None`` on clone runs (the host path
    would dangle in the disposable clone).
    """
    try:
        params = inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return {"clone_mode", "inline_exploration", "inline_dependencies"} <= params.keys()


def _diagram_author_prompt(
    ctx: FlowContext, kind: str, eligibility: Eligibility, backend: Any, *, inline_artifacts: bool = False
) -> str:
    """Build one kind's first-turn author prompt through the registry.

    On backends whose read-only profile executes in a disposable clone (the
    protocol-level ``read_only_disposable_clone`` capability) the host-only
    ``.daydream/`` artifact paths the pointer blocks name would dangle, so
    the prompt is made self-sufficient: exploration summary and dependency
    edges are inlined under the untrusted boundary (best-effort reads, shared
    prompt budget) and the diff is handed to the builder un-truncated — the
    builder owns clone-mode truncation. Other backends keep the budget-gated
    pointer path byte-for-byte.

    The clone-mode kwargs are only passed when the registered builder accepts
    them: fork overrides written against the documented extension contract
    predate the inline kwargs, and splatting them in would raise ``TypeError``
    on every disposable-clone run, degrading the kind to failed. A legacy
    override keeps the documented kwarg set; on a clone run its
    ``exploration_dir`` arrives as ``None`` rather than the dangling host path.
    """
    deep_state = DeepState(ctx.data)
    diff_path: Path = deep_state.diff_path
    inline_diff = _ttt_diff_text(ctx)
    exploration_dir: Path | None = deep_state.exploration_dir_or_none
    clone_mode = bool(getattr(backend, "read_only_disposable_clone", False))
    builder = get_registry().prompt(
        "diagram_sequence" if kind == "sequence" else "diagram_flowchart"
    )
    inline_kwargs: dict[str, Any]
    if clone_mode and _prompt_builder_accepts_inline_kwargs(builder):
        # A live artifact session routes the pre-scan through sanctioned inputs
        # instead of inlining it here.
        legacy = _inline_exploration_text(exploration_dir) if ctx.artifacts is None else (None, None)
        inline_kwargs = {
            "exploration_dir": None,
            "clone_mode": True,
            "inline_exploration": legacy[0],
            "inline_dependencies": legacy[1],
        }
    else:
        # A legacy override keeps its documented kwarg set, but a clone run
        # must not name the host-only exploration_dir: the path dangles in
        # the disposable clone, so it arrives as ``None`` there and untouched
        # otherwise.
        inline_kwargs = {"exploration_dir": None if clone_mode or inline_artifacts else exploration_dir}
    if kind == "sequence":
        return str(
            builder(
                diff_path=diff_path,
                inline_diff=inline_diff,
                files_by_module=_files_by_module(eligibility),
                cwd=ctx.work.repo,
                schema=SEQUENCE_SPEC_SCHEMA,
                **inline_kwargs,
            )
        )
    return str(
        builder(
            diff_path=diff_path,
            inline_diff=inline_diff,
            candidate_roots=[asdict(root) for root in eligibility.candidate_roots],
            forced=eligibility.flowchart.rule == "forced",
            cwd=ctx.work.repo,
            schema=FLOWCHART_SPEC_SCHEMA,
            **inline_kwargs,
        )
    )


async def _run_diagram_kind(
    ctx: FlowContext,
    *,
    kind: str,
    eligibility: Eligibility,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    symbols: RepoSymbols,
    recorder: "TrajectoryRecorder | None",
    backend: Any,
    dispatch: "DispatchHandle | None" = None,
) -> DiagramResult:
    """Author, ground, repair once, prune and render one diagram kind.

    Each turn runs in its own fork (``diagram-<kind>`` then
    ``diagram-<kind>-repair``) and the forks are strictly sequential: the first
    must EXIT before grounding runs, because the read receipts that decide
    ``FILE_NOT_READ_BY_MODEL`` only reach disk on exit, and the repair decision
    depends on that grounding. Nested forks would also be illegal -- the
    recorder ContextVar is reset LIFO.

    Returns:
        The kind's result dict (see
        :data:`~daydream.deep.diagram_types.DiagramResult`).
    """
    deep_state = DeepState(ctx.data)
    schema = SEQUENCE_SPEC_SCHEMA if kind == "sequence" else FLOWCHART_SPEC_SCHEMA

    def _ground(spec: dict[str, Any], read_paths: set[str]) -> Any:
        if kind == "sequence":
            return ground_sequence(
                spec,
                repo_root=ctx.work.repo,
                hunk_ranges=hunk_ranges,
                read_paths=read_paths,
                symbols=symbols,
            )
        return ground_flowchart(
            spec,
            repo_root=ctx.work.repo,
            hunk_ranges=hunk_ranges,
            read_paths=read_paths,
            candidate_roots=eligibility.candidate_roots,
            symbols=symbols,
        )

    coerce = coerce_sequence_spec if kind == "sequence" else coerce_flowchart_spec
    read_paths: set[str] = set()

    diff_path: Path = deep_state.diff_path
    exploration_dir = deep_state.exploration_dir_or_none
    diagram_diff = _ttt_diff_text(ctx)
    candidates: dict[str, Path] = {}
    # A disposable clone can read neither host artifact; the prompt inlines the
    # diff itself when it fits the budget.
    if not getattr(backend, "read_only_disposable_clone", False):
        candidates["hunk-index"] = diff_path.parent / "hunk-index.json"
        if not diagram_diff or not fits_inline_diff_budget(diagram_diff):
            candidates["diff"] = diff_path
    if isinstance(exploration_dir, Path):
        candidates |= {
            "exploration-summary": exploration_dir / "summary.md",
            "exploration-affected-files": exploration_dir / "affected_files.md",
            "exploration-dependencies": exploration_dir / "dependencies.md",
        }
    sanctioned_inputs = (
        prepare_sanctioned_inputs(
            backend,
            ctx.work.repo,
            {label: path for label, path in candidates.items() if path.is_file()},
            read_only=True,
        )
        if ctx.artifacts is not None
        else None
    )
    inline_artifacts = (
        sanctioned_inputs is not None and sanctioned_inputs.transport is SanctionedInputTransport.INLINE
    )

    async with maybe_fork(
        recorder, f"diagram-{kind}", dispatch=dispatch
    ) as fork:
        structured, continuation, budget_reason = await run_agent(
            backend,
            ctx.work.repo,
            _diagram_author_prompt(ctx, kind, eligibility, backend, inline_artifacts=inline_artifacts),
            phase=DaydreamPhase.DIAGRAM,
            output_schema=schema,
            read_only=True,
            wall_budget_s=DEFAULT_WALL_BUDGET_S,
            tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
            sanctioned_inputs=sanctioned_inputs,
            run_context=ctx.run_context,
        )
    read_paths |= _diagram_read_paths(getattr(fork, "path", None))
    if budget_reason:
        # A truncated author turn did not really answer: recording it as an
        # omission would claim the model looked and found nothing to draw.
        return _diagram_result("failed", f"budget exhausted: {budget_reason}")
    if not isinstance(structured, dict):
        return _diagram_result("failed", "no structured output produced")

    spec = coerce(structured)
    report = _ground(spec, read_paths)
    grounded_first_pass = int(report.summary["grounded"])
    repaired = 0

    # Exactly one repair turn, and only when the session can be resumed: a
    # fresh session would have to re-derive the whole spec from scratch, which
    # is a new proposal, not a repair.
    if report.ungrounded() and continuation is not None:
        repair_prompt = build_diagram_repair_prompt(
            kind=kind,
            failures=[check.to_dict() for check in report.ungrounded()],
            candidate_roots=(
                [asdict(root) for root in eligibility.candidate_roots]
                if kind == "flowchart"
                else None
            ),
            schema=schema,
        )
        async with maybe_fork(
            recorder, f"diagram-{kind}-repair", dispatch=dispatch
        ) as repair_fork:
            repaired_output, _, repair_budget = await run_agent(
                backend,
                ctx.work.repo,
                repair_prompt,
                phase=DaydreamPhase.DIAGRAM,
                output_schema=schema,
                continuation=continuation,
                read_only=True,
                wall_budget_s=DEFAULT_WALL_BUDGET_S,
                tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
                sanctioned_inputs=sanctioned_inputs,
                run_context=ctx.run_context,
            )
        read_paths |= _diagram_read_paths(getattr(repair_fork, "path", None))
        if not repair_budget and isinstance(repaired_output, dict):
            spec = coerce(repaired_output)
            report = _ground(spec, read_paths)
            repaired = max(int(report.summary["grounded"]) - grounded_first_pass, 0)

    omit_reasons = list(report.omit_reasons)
    mermaid: str | None = None
    if omit_reasons:
        status = "omitted"
    else:
        status = "rendered"
        mermaid = (
            render_sequence_mermaid(report.spec_final)
            if kind == "sequence"
            else render_flowchart_mermaid(report.spec_final)
        )
    return {
        "status": status,
        "reason": report.rejected,
        "spec_proposed": spec,
        "spec_final": report.spec_final,
        "grounding": {
            "elements": [check.to_dict() for check in report.elements],
            "summary": {
                "proposed": int(report.summary["proposed"]),
                "grounded_first_pass": grounded_first_pass,
                "repaired": repaired,
                "pruned": int(report.summary["pruned"]),
            },
            "capped": dict(report.capped),
            "root_range": list(report.root_range) if report.root_range is not None else None,
        },
        "omit_reasons": omit_reasons,
        "mermaid": mermaid,
    }


def _diagram_payload_without_mermaid(payload: dict[str, Any]) -> dict[str, Any]:
    """The ``diagram.json`` payload with every rendered ``mermaid`` string dropped.

    What travels in the Phase A findings artifact. The privileged poster
    re-renders from ``spec_final``, so shipping the mermaid would only offer it
    a model-adjacent string to trust by mistake.
    """
    results = payload.get("results")
    stripped: dict[str, Any] = {}
    if isinstance(results, dict):
        for kind, result in results.items():
            stripped[kind] = (
                {key: value for key, value in result.items() if key != "mermaid"}
                if isinstance(result, dict)
                else result
            )
    return {"eligibility": payload.get("eligibility"), "results": stripped}


def _apply_diagrams_to_report(ctx: FlowContext, blocks: str) -> None:
    """Insert the ``## Diagrams`` section into both copies of the rendered report.

    Textual insertion rather than a re-render: by the time this step runs,
    ``review-output.md`` has been written by the merge write and possibly
    rewritten by ``supervise``, and ``load-items`` has appended a ``##
    Coverage`` section that a re-render would erase.
    ``insert_diagrams_section`` is idempotent, so a repeated application is a
    no-op rather than a duplicate section.
    """
    deep_state = DeepState(ctx.data)
    targets = [merged_report_path(deep_state.dd)]
    canonical = deep_state.merged_report_or_none
    if canonical is not None:
        targets.append(Path(canonical))
    for target in targets:
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8")
        target.write_text(insert_diagrams_section(text, blocks), encoding="utf-8")


async def _step_diagram(ctx: FlowContext) -> Stop | None:
    """Decide, author, ground and render this run's grounded diagrams (#1113).

    Always writes ``diagram.json`` when the step is enabled, even when nothing
    is eligible: the recorded eligibility signals are the audit trail for why a
    PR did or did not get a diagram, and producing them costs zero agent calls.

    Fail-open in every review mode -- one kind's failure warns, records
    ``status="failed"`` and leaves the rest of the review untouched. In
    ``--diagram-only`` mode the diagram IS the deliverable, so a failure exits
    1 (after the artifact is written, so the evidence survives).
    """
    async with phase_scope(DaydreamPhase.DIAGRAM, stage="diagram") as phase:
        return await _run_diagram_step(ctx, phase=phase)


async def _run_diagram_step(
    ctx: FlowContext, *, phase: "PhaseScopeHandle"
) -> Stop | None:
    """Run the durable diagram step inside its identified phase scope."""
    deep_state = DeepState(ctx.data)
    settings = _diagram_settings(ctx)
    mode = deep_state.mode
    target_dir = ctx.work.repo
    dd: Path = deep_state.dd

    from daydream.hunk_index import head_side_ranges_by_file, load_hunk_index
    from daydream.runner import _file_config_or_empty
    from daydream.services import enumerate_services

    changed_files = sorted(str(path) for path in deep_state.changed_files)
    hunk_ranges = head_side_ranges_by_file(
        load_hunk_index(
            artifact_dir_for(
                target_dir,
                session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
        )
    )
    file_config = _file_config_or_empty(ctx.config)
    eligibility = decide_eligibility(
        repo_root=target_dir,
        changed_files=changed_files,
        hunk_ranges=hunk_ranges,
        # Detection is re-run here rather than read off ``ctx.data["stacks"]``:
        # that list is published AFTER the tiny-diff collapse and the sharder,
        # so on a small two-language diff every file would sit in one
        # ``generic`` assignment and the whole diff would read as non-code.
        stacks=detect_stacks(changed_files),
        services=enumerate_services(
            target_dir, file_config, service_roots=settings.service_roots or None
        ),
        import_graph=deep_state.import_graph or {},
        thresholds=settings.thresholds,
        force=settings.mode,
    )

    kinds = eligibility.eligible_kinds()
    results: dict[str, DiagramResult | None] = {}
    for kind in DIAGRAM_KINDS:
        if kind in kinds:
            continue
        decision = eligibility.sequence if kind == "sequence" else eligibility.flowchart
        results[kind] = _diagram_result("skipped", decision.reason)

    failures: dict[str, str] = {}
    if kinds:
        print_info(console, f"Grounded diagrams: authoring {', '.join(kinds)}")
        backend = ctx.backend_for("diagram")
        recorder = get_current_recorder()
        # One shared definition index: both kinds cite the same handful of
        # files, and every method on it is synchronous, so the two sibling
        # tasks cannot interleave inside one lookup.
        symbols = RepoSymbols(target_dir)
        limiter = anyio.CapacityLimiter(effective_fanout_concurrency(2, backend))
        descriptors = tuple(f"diagram-{kind}" for kind in kinds)
        async with dispatch_scope(
            recorder, phase=DaydreamPhase.DIAGRAM, descriptors=descriptors
        ) as dispatch:
            async with anyio.create_task_group() as tg:
                for kind in kinds:
                # Default-arg capture -- prevents the late-binding closure bug.
                    async def _task(kind_name: str = kind) -> None:
                        async with limiter:
                            try:
                                results[kind_name] = await _run_diagram_kind(
                                    ctx,
                                    kind=kind_name,
                                    eligibility=eligibility,
                                    hunk_ranges=hunk_ranges,
                                    symbols=symbols,
                                    recorder=recorder,
                                    backend=backend,
                                    dispatch=dispatch,
                                )
                            except Exception as exc:  # noqa: BLE001 -- parallel isolation
                                detail = f"{type(exc).__name__}: {exc}"
                                failures[kind_name] = detail
                                results[kind_name] = _diagram_result("failed", detail)

                    tg.start_soon(_task)
            returned_failures = sum(
                result is not None and result.get("status") == "failed"
                for result in results.values()
            )
            if returned_failures:
                status = (
                    LifecycleStatus.FAILED
                    if returned_failures == len(kinds)
                    else LifecycleStatus.PARTIAL
                )
                reason = (
                    LifecycleReasonCode.ALL_CHILDREN_FAILED
                    if returned_failures == len(kinds)
                    else LifecycleReasonCode.SOME_CHILDREN_FAILED
                )
                if dispatch is not None:
                    dispatch.finish(status, reason)

    for kind, result in results.items():
        if result is not None and result.get("status") == "failed":
            failures.setdefault(kind, str(result.get("reason") or "unknown failure"))

    ordered: dict[str, DiagramResult | None] = {kind: results.get(kind) for kind in DIAGRAM_KINDS}
    blocks = render_diagram_blocks(ordered)
    payload: dict[str, Any] = {"eligibility": eligibility.to_dict(), "results": ordered}
    atomic_write_json(diagram_path(dd), payload)
    diagram_markdown_path(dd).write_text(
        f"{blocks}\n" if blocks else "", encoding="utf-8"
    )
    deep_state.diagrams = {
        "blocks": blocks,
        "payload": _diagram_payload_without_mermaid(payload),
        "results": ordered,
    }

    if not kinds:
        phase.finish(LifecycleStatus.SKIPPED, LifecycleReasonCode.NO_ELIGIBLE_WORK)
    elif failures:
        all_failed = len(failures) == len(kinds)
        phase.finish(
            LifecycleStatus.FAILED if all_failed else LifecycleStatus.PARTIAL,
            LifecycleReasonCode.ALL_CHILDREN_FAILED
            if all_failed
            else LifecycleReasonCode.SOME_CHILDREN_FAILED,
        )

    consequence = "the run fails" if mode == "diagram" else "the review continues"
    for kind, detail in sorted(failures.items()):
        print_warning(console, f"Diagram kind {kind} failed ({consequence}): {detail}")
    if blocks and mode != "diagram":
        _apply_diagrams_to_report(ctx, blocks)
    if failures and mode == "diagram":
        return Stop(1)
    return None


async def _step_post_diagram(ctx: FlowContext) -> Stop:
    """Deliver a diagram-only run: findings artifact, or a standalone comment.

    ``--findings-out`` makes this Phase A of the two-phase flow (the artifact
    declares ``kind="diagram"`` and an empty findings list, and the privileged
    Phase B job posts it). Otherwise the comment posts here, and -- mirroring
    ``--comment`` -- an unresolvable PR or a failed POST ends the run with
    exit 1, because the comment was the whole point of the run.
    """
    deep_state = DeepState(ctx.data)
    from daydream.git_ops import GitError
    from daydream.pr_review import (
        _resolve_pr,
        diagram_comment_kinds,
        post_diagram_comment_to_pr,
        render_diagram_comment_body,
    )
    from daydream.runner import _emit_diagram_findings

    diagrams: dict[str, Any] = deep_state.diagrams or {}
    payload: dict[str, Any] = diagrams.get("payload") or {}

    if ctx.config.findings_out is not None:
        return Stop(
            _emit_diagram_findings(
                ctx.work.repo, ctx.config, payload, auth=ctx.github_execution.auth
            )
        )

    try:
        pr = _resolve_pr(
            ctx.work.repo, console, ctx.config.pr_number, auth=ctx.github_execution.auth
        )
    except GitError as exc:
        print_error(console, "Diagram PR Lookup Failed", str(exc))
        return Stop(1)
    if pr is None:
        print_error(
            console,
            "Diagram Comment",
            "no PR resolvable for --diagram-only (pass --pr-number or open a PR "
            "for this branch)",
        )
        return Stop(1)
    url, error = post_diagram_comment_to_pr(
        ctx.work.repo,
        pr,
        body=render_diagram_comment_body(payload),
        kinds=diagram_comment_kinds(payload),
        bot_login=os.environ.get("DAYDREAM_BOT_HANDLE") or None,
        auth=ctx.github_execution.auth,
    )
    if url is None:
        suffix = f" ({error})" if error else ""
        print_error(console, "Diagram Comment Post Failed", f"No comment was posted.{suffix}")
        return Stop(1)
    print_success(console, f"Posted diagram comment: {url}")
    return Stop(0)
