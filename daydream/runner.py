"""Main orchestration logic for the review and fix loop."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from daydream.agent import (
    MissingSkillError,
    console,
    set_debug_log,
    set_model,
    set_quiet_mode,
)
from daydream.config import REVIEW_OUTPUT_FILE, REVIEW_SKILLS, SKILL_MAP, ReviewSkillChoice
from daydream.phases import (
    FixResult,
    check_review_file_exists,
    phase_commit_push,
    phase_commit_push_auto,
    phase_fetch_pr_feedback,
    phase_fix,
    phase_fix_parallel,
    phase_parse_feedback,
    phase_respond_pr_feedback,
    phase_review,
    phase_test_and_heal,
)
from daydream.ui import (
    SummaryData,
    phase_subtitle,
    print_dim,
    print_error,
    print_info,
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
    model: str = "opus"
    debug: bool = False
    cleanup: bool | None = None
    quiet: bool = True
    review_only: bool = False
    start_at: str = "review"
    pr_number: int | None = None
    bot: str | None = None


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


async def run_pr_feedback(config: RunConfig, target_dir: Path) -> int:
    """Execute the PR feedback flow: fetch, parse, fix, commit, respond.

    Args:
        config: Run configuration with pr_number and bot set.
        target_dir: Resolved target directory path.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    print_phase_hero(console, "DAYDREAM", phase_subtitle("DAYDREAM"))

    console.print()
    print_info(console, f"PR feedback mode: PR #{config.pr_number}")
    print_info(console, f"Bot: {config.bot}")
    print_info(console, f"Target directory: {target_dir}")
    print_info(console, f"Model: {config.model}")
    console.print()

    # Phase 1: Fetch PR feedback
    await phase_fetch_pr_feedback(
        target_dir,
        config.pr_number,  # type: ignore[arg-type]
        config.bot,  # type: ignore[arg-type]
    )

    # Phase 2: Parse feedback (reused from normal flow)
    try:
        feedback_items = await phase_parse_feedback(target_dir)
    except ValueError:
        print_error(console, "Parse Failed", "Failed to parse PR feedback. Exiting.")
        return 1

    if not feedback_items:
        print_info(console, "No actionable feedback found in PR comments.")
        return 0

    # Phase 3: Fix in parallel
    results: list[FixResult] = await phase_fix_parallel(target_dir, feedback_items)

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
        await phase_commit_push_auto(target_dir)
    except Exception as e:
        print_error(console, "Commit/Push Failed", str(e))
        return 1

    # Phase 5: Respond to PR comments
    try:
        await phase_respond_pr_feedback(
            target_dir,
            config.pr_number,  # type: ignore[arg-type]
            config.bot,  # type: ignore[arg-type]
            results,
        )
    except Exception as e:
        print_warning(console, f"Failed to respond to PR comments: {e}")
        print_info(console, "Fixes were already pushed successfully.")

    # Summary
    console.print()
    print_success(
        console,
        f"PR #{config.pr_number}: {len(successful)} fix(es) applied"
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
    debug_log_file: TextIO | None = None
    if config.debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_path = target_dir / f".review-debug-{timestamp}.log"
        debug_log_file = open(debug_log_path, "w", encoding="utf-8")  # noqa: SIM115

    try:
        if debug_log_file is not None:
            set_debug_log(debug_log_file)
            print_info(console, f"Debug log: {debug_log_path}")

        # Get cleanup setting (from config or prompt)
        if config.cleanup is not None:
            cleanup_enabled = config.cleanup
        else:
            cleanup_response = prompt_user(console, "Cleanup review output after completion? [y/N]", "n")
            cleanup_enabled = cleanup_response.lower() in ("y", "yes")

        # Set quiet mode and model
        set_quiet_mode(config.quiet)
        set_model(config.model)

        # PR feedback mode: separate flow
        if config.pr_number is not None:
            return await run_pr_feedback(config, target_dir)

        console.print()
        print_info(console, f"Target directory: {target_dir}")
        print_info(console, f"Model: {config.model}")
        if skill:
            print_info(console, f"Review skill: {skill}")
        if config.review_only:
            print_info(console, "Mode: review-only (skipping fixes)")
        if config.start_at != "review":
            print_skipped_phases(console, config.start_at)
        console.print()

        feedback_items: list[dict[str, Any]] = []
        fixes_applied = 0

        # Phase 1: Review (only if starting at review)
        if config.start_at == "review":
            try:
                await phase_review(target_dir, skill)  # type: ignore[arg-type]
            except MissingSkillError as e:
                _print_missing_skill_error(e.skill_name)
                return 1

        # Phase 2: Parse feedback (if starting at review, parse, or fix)
        if config.start_at in ("review", "parse", "fix"):
            try:
                feedback_items = await phase_parse_feedback(target_dir)
            except ValueError:
                print_error(console, "Parse Failed", "Failed to parse feedback. Exiting.")
                return 1

        # If review-only mode, show summary and exit
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

        # Phase 3: Apply fixes (if starting at review, parse, or fix)
        if config.start_at in ("review", "parse", "fix"):
            if feedback_items:
                print_phase_hero(console, "HEAL", phase_subtitle("HEAL"))
                for i, item in enumerate(feedback_items, 1):
                    await phase_fix(target_dir, item, i, len(feedback_items))
                    fixes_applied += 1
            else:
                print_info(console, "No feedback items found, skipping fix phase")

        # Phase 4: Test and heal (always runs unless review_only)
        tests_passed, test_retries = await phase_test_and_heal(target_dir)

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
            ),
        )

        # Prompt for commit if tests passed
        if tests_passed:
            await phase_commit_push(target_dir)

            if cleanup_enabled:
                review_output_path = target_dir / REVIEW_OUTPUT_FILE
                if review_output_path.exists():
                    review_output_path.unlink()
                    print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")

            return 0
        else:
            return 1

    finally:
        if debug_log_file is not None:
            debug_log_file.close()
            set_debug_log(None)
            if debug_log_path:
                print_info(console, f"Debug log saved: {debug_log_path}")
