"""Main orchestration logic for the review and fix loop."""

import contextlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from daydream.agent import (
    MissingSkillError,
    _log_debug,
    console,
    set_debug_log,
    set_model,
    set_quiet_mode,
)
from daydream.backends import Backend, create_backend
from daydream.config import REVIEW_OUTPUT_FILE, REVIEW_SKILLS, SKILL_MAP, ReviewSkillChoice
from daydream.phases import (
    FixResult,
    check_review_file_exists,
    phase_commit_iteration,
    phase_commit_push,
    phase_commit_push_auto,
    phase_fetch_pr_feedback,
    phase_fix,
    phase_parse_feedback,
    phase_respond_pr_feedback,
    phase_review,
    phase_test_and_heal,
    revert_uncommitted_changes,
)
from daydream.ui import (
    SummaryData,
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
    print_iteration_divider,
    print_menu,
    print_phase_hero,
    print_skipped_phases,
    print_success,
    print_summary,
    print_warning,
    prompt_user,
)


@dataclass
class RunConfig:
    """Configuration for a daydream run.

    Attributes:
        target: Target directory path for the review. If None, prompts user.
        skill: Review skill to use ("python", "react", or "elixir"). If None, prompts user.
        model: Claude model to use ("opus", "sonnet", or "haiku"). Default is "opus".
        debug: Enable debug logging to a timestamped file in the target directory.
        cleanup: Remove review output file after completion. If None, prompts user.
        quiet: Suppress verbose output from the agent.
        review_only: Run review phase only without applying fixes.
        start_at: Phase to start at ("review", "parse", "fix", or "test").
        pr_number: GitHub PR number for PR feedback mode. If None, normal mode.
        bot: Bot username whose comments to fetch (e.g. "coderabbitai[bot]").

    """

    target: str | None = None
    skill: str | None = None  # "python", "react", or "elixir"
    model: str | None = None
    debug: bool = False
    cleanup: bool | None = None
    quiet: bool = True
    review_only: bool = False
    start_at: str = "review"
    pr_number: int | None = None
    bot: str | None = None
    backend: str = "claude"
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    loop: bool = False
    max_iterations: int = 5


def _print_missing_skill_error(skill_name: str) -> None:
    """Print error message for missing skill with installation instructions."""
    print_error(console, "Missing Skill", f"Skill '{skill_name}' is not available")

    if skill_name.startswith("beagle"):
        print_info(console, "The Beagle plugin is required but not installed or enabled.")
        console.print()
        print_dim(console, "To install Beagle:")
        print_dim(console, "  1. Open Claude Code in your terminal")
        print_dim(console, "  2. Run: /install-plugin beagle@existential-birds")
        print_dim(console, "  3. Restart Claude Code")
        console.print()
        print_dim(console, "Or enable it manually in ~/.claude/settings.json:")
        print_dim(console, '  "enabledPlugins": {')
        print_dim(console, '    "beagle@existential-birds": true')
        print_dim(console, "  }")
    else:
        print_info(console, f"The plugin providing '{skill_name}' is not installed.")
        print_dim(console, "Check your ~/.claude/settings.json for enabled plugins.")

    console.print()


def _resolve_backend(
    config: RunConfig, phase: str, cache: dict[str, Backend] | None = None
) -> Backend:
    """Get or create the backend for a given phase, respecting per-phase overrides.

    Args:
        config: Run configuration with backend settings.
        phase: Phase name ("review", "fix", or "test").
        cache: Optional dict to cache backends by name. When provided, backends
               are reused if the same backend name is requested multiple times.

    Returns:
        Backend instance for the phase.

    """
    override = getattr(config, f"{phase}_backend", None)
    backend_name = override or config.backend

    if cache is not None:
        if backend_name not in cache:
            cache[backend_name] = create_backend(backend_name, model=config.model)
        return cache[backend_name]

    return create_backend(backend_name, model=config.model)


async def run_pr_feedback(config: RunConfig, target_dir: Path) -> int:
    """Execute the PR feedback flow: fetch, parse, fix, commit, respond.

    Args:
        config: Run configuration with pr_number and bot set.
        target_dir: Resolved target directory path.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    if config.pr_number is None or config.bot is None:
        print_error(console, "Invalid PR config", "--pr and --bot are required for PR feedback mode.")
        return 1

    pr_number = config.pr_number
    bot = config.bot

    backend_cache: dict[str, Backend] = {}
    review_backend = _resolve_backend(config, "review", backend_cache)
    fix_backend = _resolve_backend(config, "fix", backend_cache)

    print_phase_hero(console, "DAYDREAM", phase_subtitle("DAYDREAM"))

    console.print()
    print_info(console, f"PR feedback mode: PR #{pr_number}")
    print_info(console, f"Bot: {bot}")
    print_info(console, f"Target directory: {target_dir}")
    print_info(console, f"Model: {config.model or 'opus'}")
    console.print()

    # Phase 1: Fetch PR feedback
    await phase_fetch_pr_feedback(review_backend, target_dir, pr_number, bot)

    # Phase 2: Parse feedback (reused from normal flow)
    try:
        feedback_items = await phase_parse_feedback(review_backend, target_dir)
    except ValueError:
        print_error(console, "Parse Failed", "Failed to parse PR feedback. Exiting.")
        return 1

    if not feedback_items:
        print_info(console, "No actionable feedback found in PR comments.")
        return 0

    # Phase 3: Fix issues sequentially to avoid concurrent access to a
    # single mutable backend instance.
    results: list[FixResult] = []
    total_items = len(feedback_items)
    for idx, item in enumerate(feedback_items, start=1):
        try:
            await phase_fix(fix_backend, target_dir, item, idx, total_items)
            results.append((item, True, None))
        except Exception as e:
            results.append((item, False, f"{type(e).__name__}: {e}"))

    # If all fixes failed, abort
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    if not successful:
        print_error(
            console,
            "All Fixes Failed",
            f"All {len(failed)} fix(es) failed. Aborting before commit.",
        )
        return 1

    # Phase 4: Commit and push (no user prompt)
    try:
        await phase_commit_push_auto(review_backend, target_dir)
    except Exception as e:
        print_error(console, "Commit/Push Failed", str(e))
        return 1

    # Phase 5: Respond to PR comments
    try:
        await phase_respond_pr_feedback(review_backend, target_dir, pr_number, bot, results)
    except Exception as e:
        print_warning(console, f"Failed to respond to PR comments: {e}")
        print_info(console, "Fixes were already pushed successfully.")

    # Summary
    console.print()
    print_success(
        console,
        f"PR #{pr_number}: {len(successful)} fix(es) applied"
        + (f", {len(failed)} failed" if failed else ""),
    )

    return 0


async def run(config: RunConfig | None = None) -> int:
    """Execute the review and fix loop.

    Args:
        config: Optional configuration. If provided, values skip prompts.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    if config is None:
        config = RunConfig()

    print_phase_hero(console, "DAYDREAM", phase_subtitle("DAYDREAM"))

    # Get target directory (from config or prompt)
    if config.target is not None:
        target_dir = Path(config.target).resolve()
    else:
        target_input = prompt_user(console, "Enter target directory", ".")
        target_dir = Path(target_input).resolve()

    if not target_dir.is_dir():
        print_error(console, "Invalid Path", f"'{target_dir}' is not a valid directory")
        return 1

    # Get review skill (from config or prompt) - not required when starting at "test"
    skill: str | None = None
    if config.start_at != "test":
        if config.skill is not None:
            # Map skill name to full skill path
            if config.skill in SKILL_MAP:
                skill = SKILL_MAP[config.skill]
            elif config.skill in REVIEW_SKILLS.values():
                skill = config.skill
            else:
                print_error(console, "Invalid Skill", f"'{config.skill}' is not a valid skill")
                return 1
        else:
            console.print()
            print_menu(console, "Select review skill", [
                ("1", "Python/FastAPI backend (review-python)"),
                ("2", "React/TypeScript (review-frontend)"),
                ("3", "Elixir/Phoenix (review-elixir)"),
            ])

            skill_choice = prompt_user(console, "Choice", "1")

            # Convert string input to enum for REVIEW_SKILLS lookup
            try:
                skill_enum = ReviewSkillChoice(skill_choice)
            except ValueError:
                print_error(console, "Invalid Choice", f"'{skill_choice}' is not a valid option")
                return 1

            skill = REVIEW_SKILLS[skill_enum]

    # Early validation: check review file exists when starting at parse or fix
    if config.start_at in ("parse", "fix"):
        try:
            check_review_file_exists(target_dir)
        except FileNotFoundError as e:
            print_error(console, "Missing Review File", str(e))
            return 1

    # Set up debug logging if enabled
    debug_log_path: Path | None = None
    if config.debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_path = target_dir / f".review-debug-{timestamp}.log"

    with contextlib.ExitStack() as stack:
        if debug_log_path is not None:
            debug_log_file = stack.enter_context(open(debug_log_path, "w", encoding="utf-8"))
            set_debug_log(debug_log_file)
            stack.callback(set_debug_log, None)
            stack.callback(print_info, console, f"Debug log saved: {debug_log_path}")
            print_info(console, f"Debug log: {debug_log_path}")

        # Get cleanup setting (from config or prompt)
        if config.cleanup is not None:
            cleanup_enabled = config.cleanup
        else:
            cleanup_response = prompt_user(console, "Cleanup review output after completion? [y/N]", "n")
            cleanup_enabled = cleanup_response.lower() in ("y", "yes")

        # Create backends (may differ per-phase if overrides are set)
        backend_cache: dict[str, Backend] = {}
        review_backend = _resolve_backend(config, "review", backend_cache)
        fix_backend = _resolve_backend(config, "fix", backend_cache)
        test_backend = _resolve_backend(config, "test", backend_cache)

        # Set quiet mode: force off for Codex backends since their shell
        # commands are the primary output the user needs to see.
        quiet = config.quiet
        if quiet:
            codex_in_use = config.backend == "codex" or any(
                b == "codex"
                for b in (config.review_backend, config.fix_backend, config.test_backend)
                if b is not None
            )
            if codex_in_use:
                quiet = False
        set_quiet_mode(quiet)
        set_model(config.model or "opus")

        # PR feedback mode: separate flow
        if config.pr_number is not None:
            return await run_pr_feedback(config, target_dir)

        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Model: {config.model or '<backend-default>'}")
        if skill:
            print_info(console, f"Review skill: {skill}")
        if config.review_only:
            print_info(console, "Mode: review-only (skipping fixes)")
        if config.start_at != "review":
            print_skipped_phases(console, config.start_at)
        console.print()

        feedback_items: list[dict[str, Any]] = []
        fixes_applied = 0
        test_retries = 0
        tests_passed = True
        iteration = 0

        async def _run_loop_iteration(
            iteration: int,
        ) -> tuple[list[dict[str, Any]], int, int, bool, bool]:
            """Execute one iteration of the review-parse-fix-test loop.

            Args:
                iteration: Current iteration number (1-based).

            Returns:
                Tuple of (items, fixes_count, retries, tests_passed, should_continue).
                should_continue is False if the loop should break (clean review or test failure).

            Raises:
                MissingSkillError: If the review skill is not available.
                ValueError: If feedback parsing fails.

            """
            if iteration > 1:
                (target_dir / REVIEW_OUTPUT_FILE).unlink(missing_ok=True)
                print_iteration_divider(console, iteration, config.max_iterations)

            # Phase 1: Review
            assert skill is not None, "skill must be set when starting at review phase"
            await phase_review(review_backend, target_dir, skill)

            # Phase 2: Parse feedback
            items = await phase_parse_feedback(review_backend, target_dir)

            if not items:
                print_info(console, f"Clean review on iteration {iteration}")
                return [], 0, 0, True, False  # should_continue=False (clean)

            # Phase 3: Fix
            print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
            fixes_count = 0
            for i, item in enumerate(items, 1):
                await phase_fix(fix_backend, target_dir, item, i, len(items))
                fixes_count += 1

            # Phase 4: Test
            passed, retries = await phase_test_and_heal(test_backend, target_dir)

            if not passed:
                print_warning(console, f"Tests failed on iteration {iteration}, reverting changes")
                if revert_uncommitted_changes(target_dir):
                    print_info(console, "Reverted to last committed state")
                else:
                    print_warning(console, "Failed to revert changes")
                return items, fixes_count, retries, False, False  # should_continue=False (failed)

            # Commit iteration changes so the next review sees a clean tree
            await phase_commit_iteration(fix_backend, target_dir, iteration)

            return items, fixes_count, retries, True, True  # should_continue=True

        if config.loop:
            # --- Loop mode: repeat review-parse-fix-test ---
            while iteration < config.max_iterations:
                iteration += 1

                try:
                    items, fixes_count, retries, passed, should_continue = await _run_loop_iteration(
                        iteration
                    )
                except MissingSkillError as e:
                    _print_missing_skill_error(e.skill_name)
                    return 1
                except ValueError as exc:
                    _log_debug(f"[PHASE2_ERROR] {exc}\n")
                    print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
                    return 1

                feedback_items.extend(items)
                fixes_applied += fixes_count
                test_retries += retries
                tests_passed = passed

                if not should_continue:
                    break

            else:
                # while loop exhausted without break — max iterations reached
                if feedback_items:
                    tests_passed = False
                    print_warning(
                        console,
                        f"Reached max iterations ({config.max_iterations}), "
                        f"{len(feedback_items)} issues found across all iterations",
                    )

        else:
            # --- Single-pass mode (existing behavior) ---

            # Phase 1: Review
            if config.start_at == "review":
                assert skill is not None, "skill must be set when starting at review phase"
                try:
                    await phase_review(review_backend, target_dir, skill)
                except MissingSkillError as e:
                    _print_missing_skill_error(e.skill_name)
                    return 1

            # Phase 2: Parse feedback
            if config.start_at in ("review", "parse", "fix"):
                try:
                    feedback_items = await phase_parse_feedback(review_backend, target_dir)
                except ValueError as exc:
                    _log_debug(f"[PHASE2_ERROR] {exc}\n")
                    print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
                    return 1

            # Review-only exit
            if config.review_only:
                print_summary(
                    console,
                    SummaryData(
                        skill=skill or "N/A",
                        target=str(target_dir),
                        feedback_count=len(feedback_items),
                        fixes_applied=0,
                        test_retries=0,
                        tests_passed=True,
                        review_only=True,
                    ),
                )
                return 0

            # Phase 3: Fix
            if config.start_at in ("review", "parse", "fix"):
                if feedback_items:
                    print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
                    for i, item in enumerate(feedback_items, 1):
                        await phase_fix(fix_backend, target_dir, item, i, len(feedback_items))
                        fixes_applied += 1
                else:
                    print_info(console, "No feedback items found, skipping fix phase")

            # Phase 4: Test
            tests_passed, test_retries = await phase_test_and_heal(test_backend, target_dir)
            iteration = 1

        # Print summary
        print_summary(
            console,
            SummaryData(
                skill=skill or "N/A",
                target=str(target_dir),
                feedback_count=len(feedback_items),
                fixes_applied=fixes_applied,
                test_retries=test_retries,
                tests_passed=tests_passed,
                loop_mode=config.loop,
                iterations_used=iteration if config.loop else 1,
            ),
        )

        # Commit if tests passed
        if tests_passed:
            await phase_commit_push(review_backend, target_dir)

            if cleanup_enabled:
                review_output_path = target_dir / REVIEW_OUTPUT_FILE
                if review_output_path.exists():
                    review_output_path.unlink()
                    print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")

            return 0
        else:
            return 1
