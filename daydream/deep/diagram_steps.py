"""Grounded diagram authoring, repair, report application, and publication."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
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
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.detection import detect_stacks
from daydream.deep.diagram_grounding import RepoSymbols, ground_flowchart, ground_sequence
from daydream.deep.diagram_prompts import build_diagram_repair_prompt
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
from daydream.deep.render import insert_diagrams_section
from daydream.deep.settings import _resolve_config_value
from daydream.extensions import get_registry
from daydream.extensions.api import Stop
from daydream.fanout import run_fanout
from daydream.flows.engine import FlowContext
from daydream.json_utils import atomic_write_json
from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    AdvisoryCandidate,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    SanctionedInputUnavailable,
    fits_inline_diff_budget,
    prepare_sanctioned_inputs,
    sanctioned_transport_for,
    select_advisory_inputs,
    truncate_utf8_to_budget,
)
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.review_budget import review_deadline
from daydream.trajectory import (
    DaydreamPhase,
    LifecycleReasonCode,
    LifecycleStatus,
    dispatch_scope,
    get_current_recorder,
    maybe_fork,
    partial_or_failed_terminal,
    phase_scope,
)
from daydream.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from daydream.run_config import RunConfig
    from daydream.trajectory import DispatchHandle, PhaseScopeHandle, TrajectoryRecorder


@dataclass(frozen=True)
class DiagramSettings:
    """Resolved mode, eligibility thresholds, and service-root globs.

    Empty service roots fall back to Improve's list, then layout inference.
    """

    mode: str
    thresholds: DiagramThresholds
    service_roots: list[str]


def _diagram_mode_for(config: RunConfig, mode: str) -> str:
    """Resolve CLI > file > auto; diagram-only explicitly overrides a file off switch."""
    if config.diagram in DIAGRAM_MODES:
        return str(config.diagram)
    if mode == "diagram":
        return "auto"
    file_config = config.file_config
    file_mode = file_config.diagram_mode if file_config is not None else None
    return file_mode if file_mode in DIAGRAM_MODES else "auto"


def _resolved_diagram_mode(ctx: FlowContext) -> str:
    """Resolve the active mode from the flow state."""
    deep_data = ctx.deep_data()
    return _diagram_mode_for(ctx.config, str(deep_data.get("mode", "loop")))


def _diagram_settings(ctx: FlowContext) -> DiagramSettings:
    """Resolve mode, threshold overrides, and service roots once for this step."""
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


# Models propose evidence-bearing JSON, with at most one repair per kind.
# Only deterministically grounded elements reach the mermaid renderer.


def _diagram_result(
    status: str, reason: str | None, *, advisory: dict[str, Any] | None = None
) -> DiagramResult:
    """Build a no-spec skipped/failed result, retaining its reason and advisory omission."""
    return {
        "status": status,
        "reason": reason,
        "spec_final": None,
        "omit_reasons": [],
        "mermaid": None,
        "advisory": advisory,
    }


def _failed_kind_result(exc: BaseException, advisory: dict[str, Any] | None = None) -> DiagramResult:
    """Project an unexpected kind failure into the shared no-spec failure result."""
    return _diagram_result(
        "failed",
        f"{type(exc).__name__}: {exc}",
        advisory=advisory if advisory and advisory["omitted"] else None,
    )


def _files_by_module(eligibility: Eligibility) -> dict[str, list[str]]:
    """Group the changed code files by module for the sequence prompt."""
    grouped: dict[str, list[str]] = {}
    for path, module in sorted(eligibility.modules.items()):
        grouped.setdefault(module, []).append(path)
    return grouped


def _inline_exploration_text(exploration_dir: Path | None) -> tuple[str | None, str | None]:
    """Read available exploration text for disposable clones within one shared byte budget.

    Scrub summary scaffolding first; dependencies consume the remainder. Unreadable
    files yield None. Truncation markers count toward the budget.
    """
    if exploration_dir is None:
        return None, None
    budget = INLINE_DIFF_BUDGET_BYTES
    marker = "\n[exploration summary truncated]\n"
    try:
        summary: str | None = (exploration_dir / "summary.md").read_text(encoding="utf-8")
    except OSError:
        summary = None
    if summary is not None:
        summary = _scrub_exploration_summary(summary)
        summary = truncate_utf8_to_budget(summary, budget, marker)
        budget = max(budget - len(summary.encode("utf-8")), 0)
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
            dependencies = truncate_utf8_to_budget(dependencies, budget, marker)
    return summary, dependencies


def _scrub_exploration_summary(summary: str) -> str:
    """Remove the re-emitted boundary and table naming sibling artifacts absent from clones."""
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


def _scrub_inline_exploration_summary(
    prepared: PreparedSanctionedInputs | None,
) -> PreparedSanctionedInputs | None:
    """Scrub captured INLINE summaries before rendering, including run_agent's re-render.

    Keep device/inode/size/mtime unchanged so revalidation still attests the source
    file; only its captured text changes.
    """
    if prepared is None or prepared.transport is not SanctionedInputTransport.INLINE:
        return prepared
    scrubbed = tuple(
        replace(item, text=_scrub_exploration_summary(item.text))
        if item.label == "exploration-summary" and item.text is not None
        else item
        for item in prepared.inputs
    )
    return replace(prepared, inputs=scrubbed)


_PRIVATE_ARTIFACT_PLACEHOLDER = "sanctioned artifact storage"


def _redact_private_artifacts(
    prompt: str, diff_path: Path, exploration_dir: Path | None
) -> str:
    """Remove known host-private paths emitted by INLINE extension builders.

    Replace full paths before their parent root. Repository paths are untouched.
    """
    private_paths: tuple[Path | None, ...] = (
        diff_path,
        diff_path.parent / "hunk-index.json",
        exploration_dir,
        diff_path.parent,
    )
    for private in private_paths:
        if private is not None:
            prompt = prompt.replace(str(private), _PRIVATE_ARTIFACT_PLACEHOLDER)
    return prompt


def _diagram_author_prompt(
    ctx: FlowContext,
    kind: str,
    eligibility: Eligibility,
    backend: Any,
    *,
    inline_transport: SanctionedInputTransport | None = None,
) -> str:
    """Build a registry prompt using the resolved sanctioned-input transport.

    INLINE prompts carry the diff and suppress private artifact pointers; EXACT_PATHS
    keeps pointers. Current extension builders accept the inline context arguments;
    known private paths are redacted before INLINE dispatch.
    """
    deep_data = ctx.deep_data()
    diff_path: Path = deep_data["diff_path"]
    inline_diff = _ttt_diff_text(ctx)
    exploration_dir: Path | None = deep_data.get("exploration_dir")
    transport = (
        inline_transport
        if inline_transport is not None
        else sanctioned_transport_for(backend, ctx.work.repo, read_only=True)
    )
    inline = transport is SanctionedInputTransport.INLINE
    builder = get_registry().prompt(
        "diagram_sequence" if kind == "sequence" else "diagram_flowchart"
    )
    inline_kwargs: dict[str, Any]
    if inline:
        # A live artifact session routes the pre-scan through sanctioned inputs
        # instead of inlining it here.
        exploration = _inline_exploration_text(exploration_dir) if ctx.artifacts is None else (None, None)
        inline_kwargs = {
            "exploration_dir": None,
            "clone_mode": True,
            "inline_exploration": exploration[0],
            "inline_dependencies": exploration[1],
        }
    else:
        inline_kwargs = {"exploration_dir": exploration_dir}
    kind_kwargs = (
        {"files_by_module": _files_by_module(eligibility), "schema": SEQUENCE_SPEC_SCHEMA}
        if kind == "sequence"
        else {
            "candidate_roots": [asdict(root) for root in eligibility.candidate_roots],
            "forced": eligibility.flowchart.rule == "forced",
            "schema": FLOWCHART_SPEC_SCHEMA,
        }
    )
    prompt = str(builder(
        diff_path=diff_path,
        inline_diff=inline_diff,
        cwd=ctx.work.repo,
        **kind_kwargs,
        **inline_kwargs,
    ))
    return _redact_private_artifacts(prompt, diff_path, exploration_dir) if inline else prompt


async def _diagram_turn(
    ctx: FlowContext,
    *,
    descriptor: str,
    prompt: str,
    schema: dict[str, Any],
    recorder: "TrajectoryRecorder | None",
    backend: Any,
    dispatch: "DispatchHandle | None",
    continuation: Any = None,
    sanctioned_inputs: PreparedSanctionedInputs | None = None,
    advisory: dict[str, Any] | None = None,
) -> tuple[Any, Any, str | None] | DiagramResult:
    """Run an author/repair turn and return output, continuation, and budget reason.

    SanctionedInputUnavailable propagates to the caller's failure path. Other errors
    fail this kind, retaining any real advisory omission diagnostic.
    """
    try:
        async with maybe_fork(recorder, descriptor, dispatch=dispatch):
            output, token, budget_reason = await run_agent(
                backend,
                ctx.work.repo,
                prompt,
                phase=DaydreamPhase.DIAGRAM,
                output_schema=schema,
                continuation=continuation,
                read_only=True,
                wall_budget_s=DEFAULT_WALL_BUDGET_S,
                deadline=review_deadline(discovery=False),
                tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
                sanctioned_inputs=sanctioned_inputs,
                run_context=ctx.run_context,
            )
    except SanctionedInputUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 -- the kind still fails, keep both facts
        return _failed_kind_result(exc, advisory)
    return output, token, budget_reason


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
    """Author, ground, optionally repair once, then prune and render a kind.

    Author and repair forks run sequentially so recorder ContextVars reset LIFO.
    """
    deep_data = ctx.deep_data()
    schema = SEQUENCE_SPEC_SCHEMA if kind == "sequence" else FLOWCHART_SPEC_SCHEMA

    def _ground(raw: dict[str, Any]) -> Any:
        if kind == "sequence":
            return ground_sequence(
                coerce_sequence_spec(raw),
                repo_root=ctx.work.repo,
                hunk_ranges=hunk_ranges,
                symbols=symbols,
            )
        else:
            return ground_flowchart(
                coerce_flowchart_spec(raw),
                repo_root=ctx.work.repo,
                hunk_ranges=hunk_ranges,
                candidate_roots=eligibility.candidate_roots,
                symbols=symbols,
            )

    diff_path: Path = deep_data["diff_path"]
    exploration_dir = deep_data.get("exploration_dir")
    diagram_diff = _ttt_diff_text(ctx)
    # Bound before the guard: the repair turn and every failure path read both.
    advisory: dict[str, Any] | None = None
    sanctioned_inputs: PreparedSanctionedInputs | None = None

    try:
        # Resolve transport inside the guard: a strict audit-root mismatch must propagate
        # as SanctionedInputUnavailable, not an advisory authoring failure.
        transport = sanctioned_transport_for(backend, ctx.work.repo, read_only=True)
        # Declared in semantic priority: exploration context degrades whole-artifact
        # first, and only pointer transports can carry the host-private diff index.
        candidates: list[AdvisoryCandidate] = []
        if isinstance(exploration_dir, Path):
            candidates += [
                AdvisoryCandidate("exploration-summary", exploration_dir / "summary.md"),
                AdvisoryCandidate("exploration-dependencies", exploration_dir / "dependencies.md"),
                AdvisoryCandidate("exploration-affected-files", exploration_dir / "affected_files.md"),
            ]
        # A disposable clone can read neither host artifact; the prompt inlines the
        # diff itself when it fits the budget, so neither is a candidate on INLINE.
        if transport is SanctionedInputTransport.EXACT_PATHS:
            candidates.append(AdvisoryCandidate("hunk-index", diff_path.parent / "hunk-index.json"))
            if not diagram_diff or not fits_inline_diff_budget(diagram_diff):
                candidates.append(AdvisoryCandidate("diff", diff_path))
        selection = (
            select_advisory_inputs(backend, ctx.work.repo, candidates, read_only=True)
            if ctx.artifacts is not None
            else None
        )
        sanctioned_inputs = (
            prepare_sanctioned_inputs(
                backend,
                ctx.work.repo,
                {
                    label: path
                    for label, path in (selection.selected_paths().items() if selection is not None else ())
                    if path.is_file()
                },
                read_only=True,
            )
            if selection is not None
            else None
        )
        sanctioned_inputs = _scrub_inline_exploration_summary(sanctioned_inputs)
        # The prompt is shaped and rendered to its final, private-path-free form
        # here, so the bytes this call site decides on are the bytes the backend
        # receives; run_agent re-applies the renderer idempotently for revalidation.
        prompt = _diagram_author_prompt(ctx, kind, eligibility, backend, inline_transport=transport)
        if sanctioned_inputs is not None:
            prompt = sanctioned_inputs.render_prompt(prompt)
        advisory = selection.to_dict() if selection is not None else None

        turn = await _diagram_turn(
            ctx,
            descriptor=f"diagram-{kind}",
            prompt=prompt,
            schema=schema,
            recorder=recorder,
            backend=backend,
            dispatch=dispatch,
            sanctioned_inputs=sanctioned_inputs,
            advisory=advisory,
        )
        if not isinstance(turn, tuple):
            return turn
        structured, continuation, budget_reason = turn
    except SanctionedInputUnavailable:
        # A capture/revalidation failure is not an authoring outcome: it must
        # reach the caller's failure path unchanged, without being relabelled
        # as an advisory degradation.
        raise
    except Exception as exc:  # noqa: BLE001 -- the kind still fails, keep both facts
        # Only a real omission is worth carrying on a failure: a kind whose
        # advisory inputs all fit has nothing to report beyond its reason, and
        # ``None`` is the documented "no omission diagnostic" value.
        return _failed_kind_result(exc, advisory)
    if budget_reason:
        # A truncated author turn did not really answer: recording it as an
        # omission would claim the model looked and found nothing to draw.
        return _diagram_result("failed", f"budget exhausted: {budget_reason}", advisory=advisory)
    if not isinstance(structured, dict):
        return _diagram_result("failed", "no structured output produced", advisory=advisory)

    report = _ground(structured)

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
        turn = await _diagram_turn(
            ctx,
            descriptor=f"diagram-{kind}-repair",
            prompt=repair_prompt,
            schema=schema,
            recorder=recorder,
            backend=backend,
            dispatch=dispatch,
            continuation=continuation,
            sanctioned_inputs=sanctioned_inputs,
            advisory=advisory,
        )
        if not isinstance(turn, tuple):
            return turn
        repaired_output, _, repair_budget = turn
        if not repair_budget and isinstance(repaired_output, dict):
            report = _ground(repaired_output)

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
        "spec_final": report.spec_final,
        "omit_reasons": omit_reasons,
        "mermaid": mermaid,
        "advisory": advisory,
    }


def _diagram_payload_without_mermaid(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop rendered mermaid from Phase A payloads; the privileged poster re-renders specs."""
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
    """Idempotently insert diagrams into both reports, preserving supervision and coverage text."""
    deep_data = ctx.deep_data()
    targets = [DeepArtifact.MERGED_REPORT.at(deep_data["dd"])]
    canonical = deep_data.get("merged_report")
    if canonical is not None:
        targets.append(Path(canonical))
    for target in targets:
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8")
        target.write_text(insert_diagrams_section(text, blocks), encoding="utf-8")


async def _step_diagram(ctx: FlowContext) -> Stop | None:
    """Persist eligibility and results even when no kind qualifies.

    Kind failures warn and leave reviews running. Diagram-only failures return 1
    after writing the artifact.
    """
    async with phase_scope(DaydreamPhase.DIAGRAM, stage="diagram") as phase:
        return await _run_diagram_step(ctx, phase=phase)


async def _run_diagram_step(
    ctx: FlowContext, *, phase: "PhaseScopeHandle"
) -> Stop | None:
    """Run the durable diagram step inside its identified phase scope."""
    deep_data = ctx.deep_data()
    settings = _diagram_settings(ctx)
    mode = str(deep_data.get("mode", "loop"))
    target_dir = ctx.work.repo
    dd: Path = deep_data["dd"]

    from daydream.hunk_index import head_side_ranges_by_file, load_hunk_index
    from daydream.run_config import _file_config_or_empty
    from daydream.services import enumerate_services

    changed_files = sorted(str(path) for path in deep_data["changed_files"])
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
        import_graph=(deep_data["import_graph"] if deep_data.get("import_graph") is not None else {}) or {},
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
            async def author(kind_name: str) -> None:
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
                    results[kind_name] = _failed_kind_result(exc)


            await run_fanout(kinds, author, limiter=limiter)
            returned_failures = sum(
                result is not None and result.get("status") == "failed"
                for result in results.values()
            )
            if returned_failures and dispatch is not None:
                dispatch.finish(*partial_or_failed_terminal(returned_failures < len(kinds)))

    for kind, result in results.items():
        if result is not None and result.get("status") == "failed":
            failures.setdefault(kind, str(result.get("reason") or "unknown failure"))


    ordered: dict[str, DiagramResult | None] = {kind: results.get(kind) for kind in DIAGRAM_KINDS}
    blocks = render_diagram_blocks(ordered)
    payload: dict[str, Any] = {"eligibility": eligibility.to_dict(), "results": ordered}
    atomic_write_json(DeepArtifact.DIAGRAM.at(dd), payload)
    DeepArtifact.DIAGRAM_MARKDOWN.at(dd).write_text(
        f"{blocks}\n" if blocks else "", encoding="utf-8"
    )
    deep_data["diagrams"] = {
        "blocks": blocks,
        "payload": _diagram_payload_without_mermaid(payload),
        "results": ordered,
    }

    if not kinds:
        phase.finish(LifecycleStatus.SKIPPED, LifecycleReasonCode.NO_ELIGIBLE_WORK)
    elif failures:
        phase.finish(*partial_or_failed_terminal(len(failures) < len(kinds)))

    consequence = "the run fails" if mode == "diagram" else "the review continues"
    for kind, detail in sorted(failures.items()):
        print_warning(console, f"Diagram kind {kind} failed ({consequence}): {detail}")
    if blocks and mode != "diagram":
        _apply_diagrams_to_report(ctx, blocks)
    if failures and mode == "diagram":
        return Stop(1)
    return None


async def _step_post_diagram(ctx: FlowContext) -> Stop:
    """Emit diagram-only findings for Phase B or post the standalone comment here.

    An unresolved PR or failed POST returns 1 because the comment is the deliverable.
    """
    deep_data = ctx.deep_data()
    from daydream.git_ops import GitError
    from daydream.pr_review import _resolve_pr
    from daydream.reviews.diagrams import diagram_comment_kinds, post_diagram_comment_to_pr, render_diagram_comment_body
    from daydream.runner import _emit_diagram_findings

    diagrams: dict[str, Any] = deep_data.get("diagrams") or {}
    payload: dict[str, Any] = diagrams.get("payload") or {}

    if ctx.config.findings_out is not None:
        from daydream.pr_review import resolve_review_renderers
        from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info

        run_info = render_live_run_info(LiveRunInfoSource(get_current_recorder(), ctx.artifacts))
        if run_info.diagnostic is not None:
            print_warning(console, run_info.diagnostic)
        return Stop(
            _emit_diagram_findings(
                ctx.work.repo, ctx.config, payload, auth=ctx.github_execution.auth,
                run_info=run_info.markdown, renderers=resolve_review_renderers(ctx.registry),
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
