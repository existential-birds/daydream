"""Review for review and fix phases."""

import json
from pathlib import Path
from typing import Any, cast

import anyio

from daydream import agent, config as phase_config, git_ops, review_profile as _rp, ui
from daydream.agent import (
    StructuredOutputFailure,
    resolve_gate,
)
from daydream.artifact_visibility import (
    ArtifactSession,
    ArtifactVisibilityError,
    artifact_session_active,
)
from daydream.backends import (
    Backend,
    effective_fanout_concurrency,
)
from daydream.config import (
    STRUCTURE_STACK_NAME,
)
from daydream.deep.artifacts import (
    deep_dir,
    per_stack_records_path,
    per_stack_review_path,
    write_review_markdown,
)
from daydream.deep.detection import GENERIC_STACK, StackAssignment, base_stack_name
from daydream.deep.records import (
    stamp_record_uids,
)
from daydream.deep.reuse_key import (
    PhaseIdentity,
    blob_map_digest,
    digest_text,
    shard_key_payload,
)
from daydream.deep.reuse_store import ReuseCache
from daydream.deep.review_reuse import ReviewReuseUnit
from daydream.extensions import Registry, get_registry
from daydream.hunk_index import load_hunk_index
from daydream.json_utils import validates_schema
from daydream.phases.inputs import (
    _budgeted_exploration_inputs,
    _inlineable_diff,
    _pointer_dir,
    _prepare_existing_phase_inputs,
    _recipe_for_work,
    append_extended_facts,
)
from daydream.phases.schemas import ALTERNATIVE_REVIEW_SCHEMA, PER_STACK_RECORD_SCHEMA
from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    SanctionedInputTransport,
    fits_inline_diff_budget,
    sanctioned_transport_for,
    truncate_utf8_to_budget,
    uses_diff_reference,
)
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.review_budget import (
    ReviewBudgetExceeded,
    ReviewLimits,
)
from daydream.review_evidence import FinalizationContext
from daydream.review_investigation import ReviewInvestigation
from daydream.review_result import ReasonCode, ReviewCoverage, reason_for_budget, reason_for_exception
from daydream.review_stage_inputs import StageInputFactory
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    load_test_recipe,
)
from daydream.trajectory import (
    DaydreamPhase,
    dispatch_scope,
    finish_partial_or_failed,
    get_current_recorder,
    maybe_fork,
)
from daydream.workspace import WorkContext


@bind_resolved_run_context
async def phase_understand_intent(
    backend: Backend,
    work: WorkContext,
    diff_path: Path,
    log: str,
    branch: str,
    *,
    exploration_dir: Path | None = None,
    pr_description: str | None = None,
    diff_text: str | None = None,
    strategy: str | None = None,
    run_context: RunContext | None = None,
) -> str:
    """Record author intent for later fix prompts."""
    run_context = resolve_run_context(run_context)
    ui.print_phase_hero(agent.console, "LISTEN", ui.phase_subtitle("LISTEN"))
    ui.print_dim(agent.console, f"Model: {backend.model}")

    # Read-only intent uses session-sanctioned exploration inputs. Standalone
    # Codex clones retain bounded inline summary/diff fallback.
    read_only_disposable_clone = getattr(backend, "read_only_disposable_clone", False)
    inline_diff: str | None
    inline_transport = sanctioned_transport_for(backend, work.repo, read_only=True) is SanctionedInputTransport.INLINE
    if inline_transport and diff_text and not fits_inline_diff_budget(diff_text):
        # Clone runs cannot read gitignored artifact pointers. Keep the inline diff
        # self-contained within the shared budget, including its truncation marker.
        inline_diff = truncate_utf8_to_budget(
            diff_text, INLINE_DIFF_BUDGET_BYTES, "\n[diff truncated to fit the prompt budget]\n"
        )
    else:
        inline_diff = None if uses_diff_reference(backend, work.repo, read_only=True) else _inlineable_diff(diff_text)
    session_active = artifact_session_active()
    inline_exploration_summary: str | None = None
    if not session_active and read_only_disposable_clone and exploration_dir is not None:
        try:
            summary_text: str | None = (exploration_dir / "summary.md").read_text(
                encoding="utf-8"
            )
        except OSError:
            summary_text = None
        if summary_text is not None:
            inline_exploration_summary = truncate_utf8_to_budget(
                summary_text, INLINE_DIFF_BUDGET_BYTES, "\n[exploration summary truncated]\n"
            )
    # Exploration is advisory: the shared selector drops inputs that exceed the
    # aggregate INLINE budget. Exact diff capture has its own hard limit; the
    # post-capture transport remains authoritative.
    sanctioned_inputs = _prepare_existing_phase_inputs(
        backend, work,
        {
            "diff": diff_path if inline_diff is None else None,
            **_budgeted_exploration_inputs(
                exploration_dir,
                backend=backend,
                cwd=work.repo,
                read_only=True,
            ),
        },
        capture_without_session=True,
        read_only=True,
    )
    prompt = get_registry().prompt("intent")(
        strategy=strategy if strategy is not None else _rp.build_default_profile().strategies["intent"].content,
        diff_path=str(diff_path),
        branch=branch,
        log=log,
        exploration_dir=_pointer_dir(
            sanctioned_inputs,
            None if not session_active and read_only_disposable_clone else exploration_dir,
        ),
        pr_description=pr_description,
        inline_diff=inline_diff,
        inline_exploration_summary=inline_exploration_summary,
    )
    prompt = append_extended_facts(prompt, _recipe_for_work(work))

    intent_correction = ""
    while True:
        agent.console.print()
        ui.print_info(agent.console, "Agent is analyzing the changes...")

        output, _, budget_reason = await agent.run_agent(
            backend, work.repo, prompt, phase=DaydreamPhase.INTENT,
            review_limits=ReviewLimits(120, 60, 12),
            finalization_context=FinalizationContext(
                task="Describe the intent of the supplied change",
                input_priority=("diff", "exploration-summary"),
                output_semantics="Return concise plain text explaining the problem and proposed behavior. "
                "State unresolved intent explicitly; do not produce a correctness review.",
                supplied_context=(("branch", branch), ("commit log", log),
                                  ("author description", pr_description or ""),
                                  ("author correction", intent_correction),
                                  ("diff", inline_diff or "")),
            ),
            tool_call_budget=phase_config.DEFAULT_TOOL_CALL_BUDGET,
            wall_budget_s=phase_config.REVIEW_WALL_BUDGET_S,
            read_only=True,
            sanctioned_inputs=sanctioned_inputs,
            run_context=run_context,
        )
        if budget_reason is not None:
            raise ReviewBudgetExceeded("Intent analysis", budget_reason, output)
        if not isinstance(output, str) or not output.strip():
            raise ReviewOutputError(output)
        intent_text = output

        agent.console.print()
        # Show the understanding the gate below asks about — the live transcript
        # above may end on tool noise rather than the summary itself.
        ui.print_intent_summary(agent.console, intent_text)
        agent.console.print()
        # Unattended runs accept this read-only understanding, including forced no.
        # Only interactive runs can request a correction; --yes accepts immediately.
        gate = resolve_gate(
            assume=run_context.policy.assume,
            interactive=run_context.policy.interactive,
            safe_default=True,
        )
        if gate is True:
            return intent_text
        if gate is False and not run_context.policy.interactive:
            return intent_text

        response = run_context.choice(
            "Is this understanding correct? [y/provide correction]",
            default="y",
            safe_default="y",
            console=agent.console,
        )

        if response.lower() in ("y", "yes"):
            return intent_text

        intent_correction = response
        # Correction remains read-only. Reuse bounded inline diff when selected;
        # only EXACT_PATHS transport may receive the sanctioned diff_path.
        if inline_diff is not None:
            diff_clause = (
                f"{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
                "Re-examine the codebase and the diff inlined below, and present an "
                "updated understanding of the intent.\n\n"
                f"{inline_diff.rstrip()}\n"
            )
        else:
            diff_clause = (
                f"Re-examine the codebase and the diff at {diff_path}, and present an "
                "updated understanding of the intent.\n"
            )
        prompt = f"""You previously described the intent of these changes as:

{intent_text}

The user corrected your understanding: {response}

{diff_clause}
The diff is the complete review target — do not look up pull requests or invoke any skills
or slash commands; reply with your updated understanding as plain text.

Branch: {branch}

Commit log:
{log}
"""


@bind_resolved_run_context
async def phase_alternative_review(
    backend: Backend,
    work: WorkContext,
    diff_path: Path,
    intent_summary: str,
    *,
    exploration_dir: Path | None = None,
    diff_text: str | None = None,
    strategy: str | None = None,
    run_context: RunContext | None = None,
) -> list[dict[str, Any]]:
    """Use confirmed intent to find concrete implementation problems in a fresh review.

    Return issues with id, title, description, recommendation, severity, and files.
    """
    run_context = resolve_run_context(run_context)
    ui.print_phase_hero(agent.console, "WONDER", ui.phase_subtitle("WONDER"))
    ui.print_dim(agent.console, f"Model: {backend.model}")

    read_only = uses_diff_reference(backend, work.repo, read_only=True)
    inline_diff = None if read_only else _inlineable_diff(diff_text)
    # Apply the shared advisory exploration budget; exact diff limits stay strict.
    sanctioned_inputs = _prepare_existing_phase_inputs(
        backend, work,
        {
            "diff": diff_path if inline_diff is None else None,
            **_budgeted_exploration_inputs(
                exploration_dir,
                backend=backend,
                cwd=work.repo,
                read_only=read_only,
            ),
        },
        capture_without_session=True,
        read_only=read_only,
    )
    prompt = get_registry().prompt("alternatives")(
        strategy=strategy if strategy is not None else _rp.build_default_profile().strategies["alternatives"].content,
        intent_summary=intent_summary,
        diff_path=str(diff_path),
        exploration_dir=_pointer_dir(sanctioned_inputs, exploration_dir),
        inline_diff=inline_diff,
    )
    prompt = append_extended_facts(prompt, _recipe_for_work(work))

    agent.console.print()
    ui.print_info(agent.console, "Agent is evaluating the implementation...")

    result, _, budget_reason = await agent.run_agent(
        backend,
        work.repo,
        prompt,
        output_schema=ALTERNATIVE_REVIEW_SCHEMA,
        require_full_schema=True,
        phase=DaydreamPhase.ALTERNATIVES,
        review_limits=ReviewLimits(300, 90, 24),
        finalization_context=FinalizationContext(
            task="Finalize the assessment of implementation alternatives",
            input_priority=("diff", "exploration-summary"),
            output_semantics="Return issues only for substantiated design failures "
            "or repository convention violations. "
            "An empty issues array is valid.",
            supplied_context=(("confirmed intent", intent_summary), ("diff", inline_diff or "")),
        ),
        tool_call_budget=phase_config.DEFAULT_TOOL_CALL_BUDGET,
        wall_budget_s=phase_config.REVIEW_WALL_BUDGET_S,
        sanctioned_inputs=sanctioned_inputs,
        read_only=read_only,
        run_context=run_context,
    )

    if budget_reason:
        raise ReviewBudgetExceeded("Alternatives", budget_reason, result)

    if not isinstance(result, dict) or not validates_schema(result, ALTERNATIVE_REVIEW_SCHEMA):
        raise ReviewOutputError(result)
    issues = cast(list[dict[str, Any]], result["issues"])

    if issues:
        ui.print_info(agent.console, f"Found {len(issues)} issues")
        ui.print_issues_table(agent.console, issues)
    else:
        ui.print_info(agent.console, "No issues found — the implementation looks good")

    return issues


def valid_record_artifact(
    value: Any, *, scope_id: str, analyzed_revision: dict[str, Any],
) -> bool:
    """Validate persisted provider records, permitting only host identity metadata."""
    if not isinstance(value, dict) or set(value) - {
        "issues", "incomplete", "scope_id", "analyzed_revision", "originating_run_id",
    }:
        return False
    if value.get("scope_id") != scope_id or value.get("analyzed_revision") != analyzed_revision:
        return False
    if not isinstance(value.get("originating_run_id"), str) or not value["originating_run_id"]:
        return False
    if "incomplete" in value and value["incomplete"] is not True:
        return False
    issues = value.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
        return False
    for issue in issues:
        uid = issue.get("uid")
        if not isinstance(uid, str):
            return False
        scope, _, ordinal = uid.rpartition(':')
        if scope != scope_id or not ordinal.isascii() or not ordinal.isdigit() or ordinal.startswith('0'):
            return False
    cleaned = [{key: field for key, field in issue.items() if key != "uid"} for issue in issues]
    return validates_schema({"issues": cleaned}, PER_STACK_RECORD_SCHEMA)


class ReviewOutputError(RuntimeError):
    """Invalid reviewer output with a typed validation reason."""

    def __init__(self, output: Any) -> None:
        self.reason_code = self.reason = ReasonCode(
            output.reason if isinstance(output, StructuredOutputFailure)
            else "missing_output" if output is None or isinstance(output, str) and not output.strip()
            else "malformed_output"
        )
        # Optional content-free diagnostic fragment from the host's schema-aware
        # selection (candidate type + "<validator> at <json_path>"). It composes
        # into the message but never into the typed reason vocabulary, and it still
        # flows through the caller's redaction/bounding before being surfaced.
        detail = getattr(output, "detail", None) if isinstance(output, StructuredOutputFailure) else None
        message = f"{self.reason.value}: reviewer response did not satisfy its schema"
        super().__init__(f"{message} ({detail})" if detail else message)


# Deep-mode: per-stack fan-out


def _read_text_or_none(path: Path | None) -> str | None:
    """Read a text artifact, or ``None`` when it is absent or unreadable."""
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _frontier_files_for_stack(stack: "StackAssignment") -> list[str]:
    """Return the persisted frontier shared by the prompt and reuse key.

    Never recompute it from a fresh graph that could disagree with reviewed inputs.
    """
    return list(stack.frontier_files)


@bind_resolved_run_context
async def phase_per_stack_reviews(
    backend: Backend,
    work: WorkContext,
    stacks: list["StackAssignment"],
    *,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    exploration_dir: Path | None = None,
    diff_text: str | None = None,
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
    strategies: dict[str, str] | None = None,
    registry: Registry | None = None,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
    run_context: RunContext | None = None,
    reuse_cache: ReuseCache | None = None,
    phase_identity: PhaseIdentity | None = None,
    coverage: ReviewCoverage,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Run scoped per-stack reviews under the backend fan-out limit and record each result.

    """
    active_registry = registry if registry is not None else get_registry()
    run_context = resolve_run_context(run_context)
    deep_dir_path = deep_dir(work.repo, session=artifact_session, allow_standalone=allow_standalone)
    recipe_for_prompts = load_test_recipe(deep_dir_path)
    recorder = get_current_recorder()
    if strategies is None:
        defaults = _rp.build_default_profile().strategies
        strategies = {name: defaults[name].content for name in (
            "discovery.per_stack", "discovery.structural", "discovery.generic_fallback",
        )}
    results: dict[str, Path] = {}
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(10, backend)
    )
    prior_commits = git_ops.daydream_commits(work.repo, work.base_branch)
    read_only = (getattr(backend, 'read_only_disposable_clone', False) is True
                 or uses_diff_reference(backend, work.repo, read_only=True))

    hunk_index = load_hunk_index(deep_dir_path.parent)

    structural_records = per_stack_records_path(deep_dir_path, STRUCTURE_STACK_NAME)
    structural_output = per_stack_review_path(deep_dir_path, STRUCTURE_STACK_NAME)
    # A rerun supersedes structural output from any earlier attempt.
    for stale_path in (structural_records, structural_output):
        stale_path.unlink(missing_ok=True)

    dispatch_descriptors = tuple(f"deep-{stack.stack_name}" for stack in stacks)
    async with dispatch_scope(
        recorder,
        phase=DaydreamPhase.DEEP,
        descriptors=dispatch_descriptors,
    ) as dispatch:
        async def _review_stack_impl(stack: "StackAssignment") -> None:
            output_path = per_stack_review_path(deep_dir_path, stack.stack_name)
            base_stack = base_stack_name(stack.stack_name)
            prompt_name, strategy_name = {
                STRUCTURE_STACK_NAME: ("structural", "discovery.structural"),
                GENERIC_STACK: ("generic-fallback", "discovery.generic_fallback"),
            }.get(base_stack, ("per-stack", "discovery.per_stack"))
            from daydream.deep.prompts import (
                build_generic_fallback_prompt,
                build_per_stack_prompt,
                build_structural_prompt,
            )
            builder = active_registry.prompt(prompt_name)
            builtin_builders = (build_generic_fallback_prompt, build_per_stack_prompt, build_structural_prompt)
            bundle_capable = (any(builder is builtin for builtin in builtin_builders)
                              or getattr(builder, 'review_input_bundle', False) is True)
            shared_paths = {"intent": intent_path, "alternatives": alternatives_path if include_alternatives else None}
            if exploration_dir is not None:
                shared_paths.update({"exploration-summary": exploration_dir / "summary.md",
                                     "exploration-affected-files": exploration_dir / "affected_files.md"})
            input_factory = StageInputFactory(
                backend, work, stack, diff_path=diff_path,
                hunk_index_path=diff_path.parent / "hunk-index.json", shared_paths=shared_paths,
                revision=coverage.revision.to_dict(), artifact_session=artifact_session,
                allow_standalone=allow_standalone, read_only=read_only,
                bundle_capable=bundle_capable,
            )
            reuse_unit: ReviewReuseUnit | None = None
            if reuse_cache is not None and phase_identity is not None:
                stack_payload = shard_key_payload(
                    stack_name=stack.stack_name,
                    files=stack.files,
                    frontier_files=_frontier_files_for_stack(stack),
                    docs_only=stack.is_docs_only,
                    diff_path_or_hunks=input_factory.full_diff,
                    hunk_index=hunk_index,
                    exploration_dir=exploration_dir,
                    worktree_root=work.repo,
                    identity=phase_identity,
                    intent_authoritative=intent_authoritative,
                    include_alternatives=include_alternatives,
                    prior_commits=prior_commits,
                    intent_text=_read_text_or_none(intent_path),
                    alternatives_text=(
                        _read_text_or_none(alternatives_path) if include_alternatives else None
                    ),
                )
                if stack.stack_name == STRUCTURE_STACK_NAME:
                    # Structural review reads the recorded whole-diff pointer. Key its file-set
                    # scope and contract, excluding content that serves only as grounding.
                    components = stack_payload["components"]
                    components["hunk_slice"] = digest_text("")
                    components["assigned_blobs"] = blob_map_digest(work.repo, [])
                    components["frontier_blobs"] = blob_map_digest(work.repo, [])
                reuse_unit = ReviewReuseUnit(reuse_cache, f"shard:{stack.stack_name}", phase_identity,
                                             stack_payload, coverage)
                hit = reuse_unit.lookup(deep_dir_path, on_restore_failure=lambda reason: ui.print_warning(
                    agent.console, f"Reuse restore failed for {stack.stack_name}: {reason}"))
                if hit is not None:
                    try:
                        restored = json.loads(per_stack_records_path(deep_dir_path, stack.stack_name).read_text())
                        cache_valid = valid_record_artifact(
                            restored, scope_id=stack.stack_name, analyzed_revision=coverage.revision.to_dict(),
                        ) and restored.get("incomplete") is not True
                    except (OSError, ValueError):
                        cache_valid = False
                    if cache_valid:
                        reuse_unit.record_hit(hit)
                        results[stack.stack_name] = output_path
                        coverage.record_scope(stack.stack_name, "complete")
                        return
                    ui.print_warning(agent.console,
                                     f"Cached records for {stack.stack_name} are invalid; rerunning review")
            per_stack_records_path(deep_dir_path, stack.stack_name).unlink(missing_ok=True)
            base_stack = base_stack_name(stack.stack_name)
            prompt_name, strategy_name = {
                STRUCTURE_STACK_NAME: ("structural", "discovery.structural"),
                GENERIC_STACK: ("generic-fallback", "discovery.generic_fallback"),
            }.get(base_stack, ("per-stack", "discovery.per_stack"))
            prompt_args: dict[str, Any] = {
                "strategy": strategies[strategy_name],
                "files": stack.files,
                "diff_path": diff_path,
                "intent_path": intent_path,
                "alternatives_path": alternatives_path,
                "output_path": output_path,
                "cwd": work.repo,
                "exploration_dir": None,
                "prior_commits": prior_commits,
                "intent_authoritative": intent_authoritative,
                "include_alternatives": include_alternatives,
            }
            # Structural review ranges over the whole repository and therefore
            # keeps the diff pointer. Language and generic scopes can inline hunks.
            if stack.stack_name != STRUCTURE_STACK_NAME:
                prompt_args.update(inline_diff=None, frontier_files=_frontier_files_for_stack(stack))
                if base_stack == GENERIC_STACK:
                    prompt_args["is_docs_only"] = stack.is_docs_only
                else:
                    prompt_args["stack_name"] = stack.stack_name
            def build_stage_prompt(stage: dict[str, Any]) -> str:
                # Invoke the registered builder anew: its kwargs describe this
                # assignment rather than a terminal whole-stack review.
                stage_args = dict(prompt_args)
                from daydream.review_profile import FOLDED_ALTERNATIVES_INSTRUCTION
                stage['folded_alternatives'] = (stack.stack_name == STRUCTURE_STACK_NAME
                                                and builder is build_structural_prompt
                                                and FOLDED_ALTERNATIVES_INSTRUCTION in strategies[strategy_name])
                stage_args["files"] = stage["assigned_files"]
                stage_args["review_stage"] = stage
                paths = input_factory.current_paths
                stage_args.update(
                    diff_path=paths.get("diff", Path("unavailable-stage-diff")),
                    intent_path=paths.get("intent", Path("unavailable-intent")),
                    alternatives_path=paths.get("alternatives", Path("unavailable-alternatives")),
                    output_path=Path("host-owned-review-output"),
                )
                if stack.stack_name != STRUCTURE_STACK_NAME and stage["stage"] == "triage":
                    stage_args["frontier_files"] = []
                prompt = active_registry.prompt(prompt_name)(**stage_args)
                prompt = append_extended_facts(prompt, recipe_for_prompts)
                return prompt + "\n\nHost review stage:\n" + json.dumps(
                    stage_args["review_stage"], ensure_ascii=False,
                )

            stack_name = stack.stack_name
            structured: Any = None
            budget_reason: str | None = None
            async with limiter:
                try:
                    async with maybe_fork(
                        recorder, f"deep-{stack_name}", dispatch=dispatch,
                    ):
                        investigation = ReviewInvestigation(
                            stack, input_factory.full_diff, coverage.revision.to_dict(),
                            assignment_batches=input_factory.assignment_batches,
                        )
                        structured, budget_reason = await investigation.run(
                            backend, work.repo, build_stage_prompt,
                            stage_inputs=input_factory.prepare,
                            wall_budget_s=phase_config.REVIEW_WALL_BUDGET_S,
                            read_only=read_only,
                            run_context=run_context,
                        )
                except Exception as e:  # noqa: BLE001 -- intentionally broad for parallel isolation
                    coverage.record_scope(stack_name, "failed",
                        reasons=(reason_for_exception(e),),
                        diagnostic=f"{type(e).__name__}: reviewer invocation failed ({reason_for_exception(e).value})")
                    return
                if budget_reason:
                    try:
                        reason = ReasonCode(budget_reason)
                    except ValueError:
                        reason = reason_for_budget(budget_reason)
                    status = "incomplete"
                    if investigation.admitted_stages == 0:
                        if budget_reason == "pipeline_budget_exceeded":
                            status = "uncovered"
                        elif investigation.failed_invocation and reason is not ReasonCode.MODEL_BUDGET_EXHAUSTION:
                            status = "failed"
                    coverage.record_scope(stack_name, status, reasons=(reason,),
                                          partial_evidence=investigation.admitted_stages > 0,
                                          diagnostic=(investigation.failure_diagnostic
                                                      or f"review stopped: {reason.value}"))
                    if status == "failed":
                        return
                if not validates_schema(structured, PER_STACK_RECORD_SCHEMA):
                    if not budget_reason:
                        error = ReviewOutputError(structured)
                        coverage.record_scope(stack_name, "failed", reasons=(error.reason,), diagnostic=str(error))
                    return
                issues = [dict(issue) for issue in structured["issues"]]
                # Uids are host-only fields, added after strict model validation.
                stamp_record_uids(issues, stack_name)
                try:
                    per_stack_records_path(deep_dir_path, stack_name).write_text(
                        json.dumps({"issues": issues,
                                    **({"incomplete": True} if budget_reason else {}),
                                    "scope_id": stack_name,
                                        "analyzed_revision": coverage.revision.to_dict(),
                                        "originating_run_id": coverage.run_id},
                                   indent=2)
                    )
                    write_review_markdown(output_path, issues)
                except (OSError, ValueError, TypeError) as exc:
                    coverage.record_scope(stack_name, "failed",
                        reasons=(*coverage.scopes[stack_name]["reason_codes"], ReasonCode.MALFORMED_ARTIFACT),
                        diagnostic=f"{type(exc).__name__}: reviewer artifact publication failed")
                    return
                results[stack_name] = output_path
                if budget_reason is None:
                    coverage.record_scope(stack_name, "complete")
                if reuse_unit is not None and budget_reason is None:
                    records_path = per_stack_records_path(deep_dir_path, stack_name)
                    reuse_unit.store(lambda: {records_path.name: records_path.read_bytes(),
                                             output_path.name: output_path.read_bytes()})

        async def _review_stack(stack: "StackAssignment") -> None:
            try:
                await _review_stack_impl(stack)
            except Exception as exc:  # noqa: BLE001 -- isolate ordinary sibling failures; cancellation propagates
                reason = (ReasonCode.MALFORMED_ARTIFACT
                          if isinstance(exc, (OSError, ValueError, TypeError, ArtifactVisibilityError))
                          else ReasonCode.UNEXPECTED_ANALYSIS_FAILURE)
                coverage.record_scope(stack.stack_name, "failed", reasons=(reason,),
                                      diagnostic=f"{type(exc).__name__}: review scope failed ({reason.value})")

        async with anyio.create_task_group() as tg:
            for stack in stacks:
                tg.start_soon(_review_stack, stack)
        if dispatch is not None and coverage.unfinished_scopes:
            finish_partial_or_failed(dispatch, results)

    failures = coverage.unfinished_scopes
    if failures:
        lines = "\n".join(f"  - {name}: {reason}" for name, reason in sorted(failures.items()))
        ui.print_warning(
            agent.console,
            f"Per-stack reviews failed for {len(failures)} stack(s); "
            "failures will be passed to the merge step.\n" + lines,
        )

    return results, failures
