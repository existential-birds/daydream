"""Testing for review and fix phases."""

import hashlib
import json
import logging
import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from daydream import agent, config as phase_config, git_ops, ui
from daydream.agent import (
    is_environmental_failure,
    resolve_gate,
)
from daydream.artifact_visibility import (
    ArtifactSession,
    artifact_dir_for,
)
from daydream.backends import (
    Backend,
    ContinuationToken,
)
from daydream.extensions import get_registry
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.generated_files import (
    GENERATED_FILES_PROMPT_RULE,
    _changed_untracked_generated_files,
    _restore_untracked_generated_file,
    _snapshot_untracked_generated_files,
    is_generated_file,
    related_manifest_paths,
)
from daydream.git_ops import GitError
from daydream.output_schema import strict_object
from daydream.phases.fix import (
    _backend_concise_fix_prompts,
    _build_fix_scope_clause,
    _build_fix_style_suffix,
    _item_evidence,
)
from daydream.phases.handoff import _emit_failure_handoff
from daydream.phases.inputs import (
    TEST_OUTPUT_TAIL_LINES,
    _render_bash_allowlist,
    _tail_test_output,
    append_extended_facts,
)
from daydream.phases.repair_outcome import classify_repair_outcome
from daydream.phases.test_evidence import (
    RepairAttemptEvidence,
    TestAndHealResult,
    TestAttemptEvidence,
    phase_test_once,
)
from daydream.redaction import redact_text
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    TestRecipe,
)
from daydream.trajectory import (
    DaydreamPhase,
    get_current_recorder,
    maybe_fork,
)
from daydream.workspace import WorkContext

_logger = logging.getLogger(__name__)

# A repair record outlives the process that wrote it, so it carries a bounded
# excerpt of the turn's own partial output rather than the unbounded prose.
_REPAIR_EXCERPT_MAX_CHARS = 2000


def _repair_excerpt(text: str) -> tuple[str, ...]:
    """Return at most one redacted, length-capped excerpt, or nothing for silent output."""
    tail, _truncated = _tail_test_output(text)
    if not tail.strip():
        return ()
    excerpt = redact_text(tail)
    if len(excerpt) > _REPAIR_EXCERPT_MAX_CHARS:
        excerpt = excerpt[-_REPAIR_EXCERPT_MAX_CHARS:]
    return (excerpt,)


def _continuation_ref(token: ContinuationToken | None) -> str | None:
    """Name a continuation without recording it: tokens carry opaque provider data."""
    if token is None:
        return None
    digest = hashlib.sha256(repr(token).encode("utf-8", errors="surrogateescape")).hexdigest()[:16]
    return f"continuation:{digest}"


def _build_fix_prompt(
    test_output: str,
    feedback_items: list[dict[str, Any]] | None = None,
    *,
    repo: Path | None = None,
    concise_mode: bool = False,
) -> str:
    """Combine bounded test output, file/evidence context, and optional concise directives.

    Existing listed files become absolute under repo so initial reads resolve there.
    """
    tail, truncated = _tail_test_output(test_output)
    if truncated:
        # Disclose the drop, and disclose how much: a repair that reads this as
        # the whole failure record will chase a symptom the early lines named.
        output_section = (
            f"Here is the tail of the test output (truncated to its last "
            f"{TEST_OUTPUT_TAIL_LINES} lines; earlier lines were dropped):\n\n{tail}"
        )
    else:
        output_section = f"Here is the test output:\n\n{test_output}"

    parts = [f"The tests failed. {output_section}"]

    if feedback_items:
        files = sorted({item["file"] for item in feedback_items if "file" in item})
        if repo is not None:
            files = [str(repo / f) if (repo / f).is_file() else f for f in files]
        if files:
            file_list = "\n".join(f"- {f}" for f in files)
            # The list names where the findings point; it is not a record of what
            # the previous fix turn changed, and must not read as one.
            parts.append(f"\nFinding target files (not a diff of what changed):\n{file_list}")
        evidence = [value for item in feedback_items if (value := _item_evidence(item))]
        if evidence:
            evidence_list = "\n".join(f"- {value}" for value in evidence)
            parts.append(f"\nEvidence exemplars:\n{evidence_list}")

    if concise_mode:
        parts.append("\nFix the failures.")
    else:
        parts.append("\nAnalyze the failures and fix them.")
    if feedback_items:
        parts.append("Focus on the files listed above.")
        parts.append(
            "Start with the files listed above. Edit authority is narrower than "
            "that list: the authorized edit scope is the only set of paths you may "
            "write, and anything outside it is reachable only by the scope-request "
            "return path, never by editing it and reporting afterwards."
        )

    return "\n".join(parts) + f"\n\n{GENERATED_FILES_PROMPT_RULE}\n" + _build_fix_style_suffix(concise_mode)


def _build_repair_budget_clause(wall_budget_s: float, tool_call_budget: int | None) -> str:
    """State the allowance the host will actually enforce, and nothing more.

    A repair turn is wall-clock bounded; whether it is also tool-call bounded is
    a fact about the resolved configuration, so an uncapped turn says so instead
    of implying a cap that does not exist.
    """
    if tool_call_budget is None:
        tool_line = "no tool-call cap is enforced, so spend the calls the diagnosis needs"
    else:
        tool_line = f"tool-call budget: {tool_call_budget} calls"
    return (
        f"\nTurn budget: {wall_budget_s:g}s of wall clock for this whole turn, and "
        f"{tool_line}. Reaching the wall clock ends the turn mid-work: the partial "
        "edits are kept and recorded, and the turn is not asked to verify them. "
        "Prioritise the most likely root cause and land that fix rather than "
        "covering every candidate.\n"
    )


def _compose_repair_prompt(
    output: str = "",
    feedback_items: list[dict[str, Any]] | None = None,
    *,
    repo: Path | None = None,
    concise_mode: bool = False,
    edit_scope: frozenset[str] = frozenset(),
    wall_budget_s: float = phase_config.DEFAULT_WALL_BUDGET_S,
    tool_call_budget: int | None = phase_config.DEFAULT_TOOL_CALL_BUDGET,
    prompt_body: str | None = None,
) -> str:
    """Assemble the whole repair prompt: findings, edit authority, and real budget.

    One composer, so the enforcement contract (authorized edit scope) and the
    debugging workflow cannot drift into a prompt that contradicts itself.
    ``prompt_body`` carries an extension's override of the ``fix`` prompt; when
    absent the default renderer runs here.
    """
    prompt = prompt_body if prompt_body is not None else _build_fix_prompt(
        output, feedback_items, repo=repo, concise_mode=concise_mode,
    )
    if edit_scope:
        prompt += _build_fix_scope_clause(edit_scope, edit_scope)
    return prompt + _build_repair_budget_clause(wall_budget_s, tool_call_budget)


def _build_setup_investigator_prompt(test_output: str) -> str:
    """Request a read-only JSON diagnosis of test invocation/setup, separate from code failure."""
    tail, truncated = _tail_test_output(test_output)
    if truncated:
        output_section = f"Tail of the failing test output:\n\n{tail}"
    else:
        output_section = f"Failing test output:\n\n{test_output}"

    return (
        "You are a read-only setup-investigator. Your ONLY job is to decide whether "
        "the test command that just failed was the WRONG command to run — not whether "
        "the code under test is broken.\n\n"
        "## Hard Constraints (read-only contract)\n"
        "- You MAY use Read, Grep, and Glob to inspect files.\n"
        "- You MAY use Bash for NON-MUTATING discovery only. Permitted commands: "
        + _render_bash_allowlist() + ".\n"
        "- You MUST NOT run tests, build steps, or installers.\n"
        "- You MUST NOT modify, create, or delete any files.\n"
        "- You MUST NOT invoke Write, Edit, or any file-mutating tool.\n\n"
        "## Files to inspect\n"
        "Look at these to discover the project's canonical test invocation:\n"
        "- `Makefile` (look for `test`, `check`, `ci` targets)\n"
        "- `pyproject.toml` (scripts, tool config)\n"
        "- `package.json` (scripts)\n"
        "- CI configs: `.github/workflows/`, `.circleci/`, `.gitlab-ci.yml`\n"
        "- `CLAUDE.md` and `README*` for documented test commands\n\n"
        f"## Failing invocation\n\n{output_section}\n\n"
        "## Output\n"
        "Return a JSON object matching this schema (and ONLY this JSON, no prose):\n"
        '{"verdict": "correct" | "replace", '
        '"suggested_command": <string or null>, '
        '"reason": <string>}\n\n'
        "- `verdict: \"correct\"` means the command was the right one; failure is "
        "code/test breakage, not invocation error. Set `suggested_command` to null.\n"
        "- `verdict: \"replace\"` means a different command should have been used. "
        "Set `suggested_command` to the exact shell command to run.\n"
        "- `reason` is a one-sentence explanation citing the file/line evidence."
    )


SETUP_INVESTIGATOR_SCHEMA: dict[str, Any] = strict_object({
    "verdict": {"type": "string", "enum": ["correct", "replace"]},
    "suggested_command": {"type": ["string", "null"]},
    "reason": {"type": "string"},
})


def _sanitize_suggested_command(raw: str) -> str:
    """Strip backticks and fold whitespace for confirmation and host-side argv parsing."""
    return " ".join(raw.replace("`", "").split())


@bind_resolved_run_context
async def _run_setup_investigator(
    backend: Backend,
    work: WorkContext,
    test_output: str,
    *,
    run_context: RunContext | None = None,
) -> dict[str, Any] | None:
    """Run a read-only setup-investigator fork and return verdict/command/reason.

    Errors or unparseable output return None, preserving the original retry command.
    """
    run_context = resolve_run_context(run_context)
    recorder = get_current_recorder()
    prompt = _build_setup_investigator_prompt(test_output)

    try:
        async with maybe_fork(recorder, "setup-investigator"):
            result, _, _ = await agent.run_agent(
                backend,
                work.repo,
                prompt,
                output_schema=SETUP_INVESTIGATOR_SCHEMA,
                phase=DaydreamPhase.TEST,
                read_only=True,
                run_context=run_context,
            )
    except Exception:  # the diagnostic and its recorder fork are best-effort
        _logger.debug("setup-investigator failed", exc_info=True)
        return None
    return result if isinstance(result, dict) and "verdict" in result else None


def _reject_test_healing_generated_file_edits(
    repo: Path,
    *,
    snapshot: str | None,
    pre_untracked: set[str],
    pre_untracked_contents: dict[str, bytes] | None = None,
    snapshot_captured: bool = True,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
) -> list[str] | None:
    """Restore generated files, returning ``None`` if any restoration fails."""
    if not snapshot_captured:
        # HEAD is not a safe substitute when capturing the pre-fix state
        # failed: it may discard edits that were present before this pass.
        return []

    ref = snapshot or "HEAD"
    artifact_root = artifact_dir_for(
        repo,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    recovery_dir = artifact_root / "partial-fixes"

    try:
        changed = git_ops.changed_files_against(
            repo, ref, preexisting_untracked=pre_untracked,
        )
    except GitError:
        return []
    tracked_violations: list[str] = []
    paths_to_restore: list[str] = []
    for path in changed:
        try:
            baseline = git_ops.show(repo, ref, path)
        except GitError:
            # Paths absent from the pre-fix ref are newly created and allowed.
            continue
        if not is_generated_file(path, baseline):
            continue
        tracked_violations.append(path)
        paths_to_restore.append(path)

    untracked_baselines = pre_untracked_contents or {}
    untracked_violations = _changed_untracked_generated_files(repo, untracked_baselines)
    direct_violations = [*tracked_violations, *untracked_violations]
    changed_set = set(changed)
    for path in direct_violations:
        for manifest_path in related_manifest_paths(path):
            if manifest_path in changed_set and manifest_path not in paths_to_restore:
                paths_to_restore.append(manifest_path)

    restoration_failed = False
    for path in paths_to_restore:
        try:
            patch = git_ops.diff_worktree_against(repo, ref, [path])
        except GitError as exc:
            ui.print_warning(agent.console, f"Could not save forbidden generated-file edit for '{path}': {exc}")
            patch = ""

        # A generated file created by the healing agent is absent from the
        # pre-fix ref and is therefore allowed.
        if not patch and path not in pre_untracked:
            continue

        try:
            recovery_dir.mkdir(parents=True, exist_ok=True)
            slug = path.replace("/", "-").replace("\\", "-")
            digest = hashlib.sha256(path.encode("utf-8", errors="surrogateescape")).hexdigest()[:12]
            (recovery_dir / f"{slug}-{digest}.patch").write_text(patch, encoding="utf-8")
        except OSError as exc:
            ui.print_warning(agent.console, f"Could not write recovery patch for '{path}': {exc}")

        try:
            git_ops.restore_paths_from_ref(repo, ref, [path])
        except GitError as exc:
            ui.print_warning(agent.console, f"Could not restore generated file '{path}': {exc}")
            restoration_failed = True

    for path in untracked_violations:
        file_path = repo / path
        if file_path.is_file():
            try:
                recovery_dir.mkdir(parents=True, exist_ok=True)
                slug = path.replace("/", "-").replace("\\", "-")
                digest = hashlib.sha256(path.encode("utf-8", errors="surrogateescape")).hexdigest()[:12]
                (recovery_dir / f"{slug}-{digest}.orphan").write_bytes(file_path.read_bytes())
            except OSError as exc:
                ui.print_warning(agent.console, f"Could not save forbidden generated-file edit for '{path}': {exc}")
        try:
            _restore_untracked_generated_file(repo, path, untracked_baselines[path])
        except OSError as exc:
            ui.print_warning(agent.console, f"Could not restore generated file '{path}': {exc}")
            restoration_failed = True

    if direct_violations:
        artifact = artifact_root / "deep" / "generated-file-violations.json"
        try:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps({"violations": direct_violations, "ref": ref}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            ui.print_warning(agent.console, f"Could not record generated-file violations: {exc}")
        if not restoration_failed:
            ui.print_warning(
                agent.console,
                f"Reverted forbidden edits to existing generated files: {', '.join(direct_violations)}. "
                "Add a new migration file instead.",
            )
    return None if restoration_failed else direct_violations


@bind_resolved_run_context
async def phase_test_and_heal(
    backend: Backend,
    work: WorkContext,
    feedback_items: list[dict[str, Any]] | None = None,
    config: Any = None,
    *,
    session_id: str | None = None,
    capture_tree_key: Callable[[], str] | None = None,
    footprint: AuthorizedFixFootprint | None = None,
    repair_backend: Backend | None = None,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
    recipe: TestRecipe | None = None,
    run_context: RunContext | None = None,
) -> TestAndHealResult:
    """Run bound test attempts and offer a bounded authorized heal after failure.

    Two backends serve this phase, because a repair turn is a fix turn: test
    execution and summarization run on ``backend`` (the TEST configuration) while
    the heal turn runs on ``repair_backend`` (the FIX configuration). ``None``
    reuses ``backend`` for both, which is the single-instance behaviour every
    caller that predates the split relies on.
    """
    run_context = resolve_run_context(run_context)
    if session_id is None or capture_tree_key is None or footprint is None:
        raise TypeError(
            "phase_test_and_heal requires session_id, capture_tree_key, and footprint"
        )
    ui.print_phase_hero(agent.console, "AWAKEN", ui.phase_subtitle("AWAKEN"))
    # A repair turn is a fix turn, so it runs on the FIX-configured instance; the
    # repair record names whichever instance actually served it.
    repair_instance = repair_backend if repair_backend is not None else backend
    ui.print_dim(agent.console, f"Model: {backend.model}")
    if repair_instance is not backend:
        ui.print_dim(agent.console, f"Repair model: {repair_instance.model}")

    retries_used = 0
    attempts: list[TestAttemptEvidence] = []
    repairs: list[RepairAttemptEvidence] = []
    job_started = time.monotonic()
    continuation: ContinuationToken | None = None
    # The repair job is host-owned: one identity per heal loop, one execution
    # per repair turn, so a resumed job can tell the two apart.
    job_id = f"repair-{session_id}"
    # Feed redacted host-suite failures through the same environmental/healing
    # gate as agent-run failures on the next iteration.
    host_failure_output: str | None = None

    async def _launch_fix(output: str) -> bool:
        nonlocal retries_used, continuation
        # Snapshot each healing turn because deep's earlier batch guard cannot
        # protect existing generated files from subsequent test-healing edits.
        try:
            snapshot = git_ops.stash_create(work.repo)
            pre_untracked = set(git_ops.list_untracked(work.repo))
            pre_untracked_contents = _snapshot_untracked_generated_files(work.repo, pre_untracked)
            snapshot_captured = True
        except (GitError, OSError) as exc:
            ui.print_warning(agent.console, f"Could not snapshot tree before test-healing fix: {exc}")
            snapshot = None
            pre_untracked = set()
            pre_untracked_contents = {}
            snapshot_captured = False
        wall_budget_s = phase_config.DEFAULT_WALL_BUDGET_S
        tool_call_budget = phase_config.DEFAULT_TOOL_CALL_BUDGET
        fix_prompt = _compose_repair_prompt(
            output, feedback_items,
            repo=work.repo,
            concise_mode=_backend_concise_fix_prompts(repair_instance),
            edit_scope=footprint.run_allowed_paths,
            wall_budget_s=wall_budget_s,
            tool_call_budget=tool_call_budget,
            prompt_body=get_registry().prompt("fix")(
                output, feedback_items, repo=work.repo,
                concise_mode=_backend_concise_fix_prompts(repair_instance),
            ),
        )
        fix_prompt = append_extended_facts(fix_prompt, recipe)
        input_tree_key = capture_tree_key()
        started = time.monotonic()
        partial_output, continuation_token, abort_reason = await agent.run_agent(
            repair_instance, work.repo, fix_prompt, phase=DaydreamPhase.FIX,
            tool_call_budget=tool_call_budget,
            wall_budget_s=wall_budget_s,
            run_context=run_context,
        )
        # The host, not the turn, decides what happened: the abort reason outranks
        # whatever the partial prose claimed. `output` is the failing test output
        # the turn worked from; `partial_output` is the turn's own unfinished text.
        turn_output = partial_output if isinstance(partial_output, str) else ""
        retries_used += 1
        diagnostics: list[str] = []
        # Git-observed paths, never the feedback items' targets: only the tree
        # itself says what the turn actually changed.
        changed: tuple[str, ...] = ()
        if snapshot is not None:
            try:
                changed = tuple(git_ops.changed_paths_z(work.repo, snapshot))
            except (GitError, OSError) as exc:
                # Never abort the repair over a degraded read, and never hide it:
                # the record names the miss so the job can re-derive the paths.
                diagnostics.append(f"changed_paths_unavailable: {exc}")
        repairs.append(RepairAttemptEvidence(
            job_id=job_id,
            execution_id=f"{job_id}:execution:{retries_used}",
            run_id=work.run_id,
            outcome=classify_repair_outcome(abort_reason, turn_output),
            abort_reason=abort_reason,
            backend_name=type(repair_instance).__name__.removesuffix("Backend").lower(),
            model=repair_instance.model,
            execution_elapsed_s=time.monotonic() - started,
            job_elapsed_s=time.monotonic() - job_started,
            input_tree_key=input_tree_key,
            output_tree_key=capture_tree_key(),
            changed_paths=changed,
            focused_evidence=_repair_excerpt(turn_output),
            continuation_ref=_continuation_ref(continuation_token),
            diagnostics=tuple(diagnostics),
        ))
        # Each repair turn still starts a fresh context; the token is recorded for
        # a resuming job, never fed to the next turn of this loop.
        continuation = None
        guard_result = _reject_test_healing_generated_file_edits(
            work.repo,
            snapshot=snapshot,
            snapshot_captured=snapshot_captured,
            pre_untracked=pre_untracked,
            pre_untracked_contents=pre_untracked_contents,
            artifact_session=artifact_session,
            allow_standalone=allow_standalone,
        )
        if abort_reason is not None:
            # An aborted turn left a tree the host cannot vouch for, exactly like a
            # failed restoration: stop here rather than rerun the suite against it.
            # The record above is what a resuming job reads.
            ui.print_warning(
                agent.console,
                f"Test-healing repair turn aborted ({abort_reason}); not rerunning tests "
                "against the unconverged tree.",
            )
            return False
        return guard_result is not None

    while True:
        agent.console.print()
        if retries_used > 0:
            ui.print_info(agent.console, f"Test retry {retries_used}")
        else:
            ui.print_info(agent.console, "Running test suite...")

        if host_failure_output is not None:
            # A host-side run already produced this failure; skip the agent run
            # and feed the same output through the failure gate below.
            output, host_failure_output = host_failure_output, None
            test_passed = False
        else:
            evidence, continuation, output = await phase_test_once(
                backend,
                work,
                config=config,
                session_id=session_id,
                capture_tree_key=capture_tree_key,
                continuation=continuation,
                recipe=recipe,
                run_context=run_context,
            )
            attempts.append(evidence)
            test_passed = evidence.passed

        if test_passed:
            ui.print_success(agent.console, "Tests passed")
            return TestAndHealResult(True, retries_used, True, False, tuple(attempts), tuple(repairs))

        ui.print_warning(agent.console, "Tests may have failed or result is unclear.")

        # An infrastructure failure (DB/cache unreachable) cannot be healed by an
        # agent fix turn, so abort before the heal gate instead of burning a retry.
        if is_environmental_failure(output):
            ui.print_warning(
                agent.console,
                "Test failure looks environmental (infrastructure unavailable); "
                "skipping heal loop.",
            )
            return TestAndHealResult(False, retries_used, False, False, tuple(attempts), tuple(repairs))

        # Unattended defaults abort without mutation. --yes allows one bounded
        # fix/retry, then aborts. Only interactive runs without an assumption show
        # the menu; its default must never create an unattended fix loop.
        decision = resolve_gate(
            assume=run_context.policy.assume,
            interactive=run_context.policy.interactive,
            safe_default=False,
        )
        if decision is False or (decision is True and retries_used > 0):
            ui.print_error(
                agent.console, "Tests failed", "Aborting heal loop (no further auto-retries)",
            )
            await _emit_failure_handoff(
                backend,
                work,
                output,
                offer_clipboard=False,
                artifact_session=artifact_session,
                allow_standalone=allow_standalone,
                run_context=run_context,
            )
            return TestAndHealResult(False, retries_used, False, False, tuple(attempts), tuple(repairs))
        if decision is True:
            # Bounded auto fix-and-retry: launch one fix attempt, then loop.
            agent.console.print()
            ui.print_info(agent.console, "Launching agent to fix test failures (auto)...")
            if not await _launch_fix(output):
                return TestAndHealResult(False, retries_used, False, False, tuple(attempts), tuple(repairs))
            continue

        ui.print_menu(agent.console, "What would you like to do?", [
            ("1", "Retry tests (run again without fixes)"),
            ("2", "Fix and retry (launch agent to fix issues)"),
            ("3", "Ignore and continue (failure still recorded)"),
            ("4", "Abort (exit with failure)"),
        ])

        choice = run_context.choice(
            "Choice",
            default="2",
            safe_default="4",
            assume_yes="2",
            assume_no="4",
            console=agent.console,
        )

        if choice == "1":
            verdict = await _run_setup_investigator(
                backend, work, output, run_context=run_context,
            )

            if verdict is None:
                ui.print_warning(
                    agent.console,
                    "Setup investigator failed; retrying with original command",
                )
            else:
                v = verdict.get("verdict")
                reason = verdict.get("reason", "")
                suggested = verdict.get("suggested_command")
                ui.print_info(agent.console, f"Setup investigator verdict: {v} — {reason}")

                if v == "replace" and isinstance(suggested, str) and suggested.strip():
                    # Show the sanitized command (same transform as the retry prompt)
                    # before asking approval, so the preview matches what gets pinned.
                    sanitized_preview = _sanitize_suggested_command(suggested)
                    ui.print_info(
                        agent.console, f"Suggested command: {sanitized_preview}",
                    )
                    if run_context.confirm(
                        safe_default=False,
                        question="Use suggested command instead?",
                        default="n",
                        console=agent.console,
                    ):
                        # Execute the approved command once host-side; never embed it in an agent prompt.
                        try:
                            cmd = shlex.split(sanitized_preview)
                        except ValueError:
                            # shlex refused it (e.g. unbalanced quotes), so
                            # there is no argv to execute.
                            cmd = []
                        if not cmd:
                            # Sanitization can leave empty argv; retry the suggestion instead of crashing.
                            ui.print_warning(
                                agent.console,
                                "Approved test command was not executable; "
                                "retrying with the original command.",
                            )
                            retries_used += 1
                            continue
                        try:
                            evidence, _, alternate_output = await phase_test_once(
                                backend,
                                work,
                                config=config,
                                session_id=session_id,
                                capture_tree_key=capture_tree_key,
                                command_override=cmd,
                                recipe=recipe,
                                run_context=run_context,
                            )
                        except (OSError, ValueError) as exc:
                            ui.print_warning(agent.console, f"Approved test command could not be run: {exc}")
                            retries_used += 1
                            continue
                        attempts.append(evidence)
                        if evidence.passed:
                            ui.print_success(agent.console, "Tests passed")
                            return TestAndHealResult(
                                True, retries_used, True, False, tuple(attempts), tuple(repairs),
                            )
                        ui.print_warning(agent.console, "Approved test command failed.")
                        host_failure_output = alternate_output
                        continue

            retries_used += 1
            continue

        elif choice == "2":
            agent.console.print()
            ui.print_info(agent.console, "Launching agent to fix test failures...")
            if not await _launch_fix(output):
                return TestAndHealResult(False, retries_used, False, False, tuple(attempts), tuple(repairs))
            continue

        elif choice == "3":
            ui.print_warning(agent.console, "Ignoring test failures, continuing...")
            return TestAndHealResult(False, retries_used, True, True, tuple(attempts), tuple(repairs))

        elif choice == "4":
            ui.print_error(agent.console, "Aborted", "User requested abort")
            await _emit_failure_handoff(
                backend,
                work,
                output,
                offer_clipboard=True,
                artifact_session=artifact_session,
                allow_standalone=allow_standalone,
                run_context=run_context,
            )
            return TestAndHealResult(False, retries_used, False, False, tuple(attempts), tuple(repairs))

        else:
            ui.print_warning(agent.console, f"Invalid choice '{choice}', aborting")
            return TestAndHealResult(False, retries_used, False, False, tuple(attempts), tuple(repairs))
