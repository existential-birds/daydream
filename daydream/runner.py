"""Main orchestration logic for the review and fix loop."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO

from daydream.agent import (
    MissingSkillError,
    console,
    set_debug_log,
    set_model,
    set_quiet_mode,
)
from daydream.config import REVIEW_OUTPUT_FILE, REVIEW_SKILLS, SKILL_MAP, ReviewSkillChoice
from daydream.phases import (
    check_review_file_exists,
    phase_commit_push,
    phase_fix,
    phase_parse_feedback,
    phase_review,
    phase_test_and_heal,
)
from daydream.rlm.errors import ContainerError, HeartbeatFailedError, REPLCrashError
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
    prompt_user,
)


@dataclass
class RunConfig:
    """Configuration for a daydream run.

    Attributes:
        target: Target directory path for the review. If None, prompts user.
        skill: Review skill to use ("python" or "frontend"). If None, prompts user.
        model: Claude model to use ("opus", "sonnet", or "haiku"). Default is "opus".
        debug: Enable debug logging to a timestamped file in the target directory.
        cleanup: Remove review output file after completion. If None, prompts user.
        quiet: Suppress verbose output from the agent.
        review_only: Run review phase only without applying fixes.
        start_at: Phase to start at ("review", "parse", "fix", or "test").
        rlm_mode: Use RLM mode for large codebase review.

    """

    target: str | None = None
    skill: str | None = None  # "python" or "frontend"
    model: str = "opus"
    debug: bool = False
    cleanup: bool | None = None
    quiet: bool = True
    review_only: bool = False
    start_at: str = "review"
    rlm_mode: bool = False


def _print_missing_skill_error(skill_name: str) -> None:
    """Print error message for missing skill with installation instructions."""
    print_error(console, "Missing Skill", f"Skill '{skill_name}' is not available")

    if skill_name.startswith("beagle:"):
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


def _skill_to_languages(skill: str | None) -> list[str]:
    """Map skill name to list of languages.

    Args:
        skill: Skill name ("python" or "frontend").

    Returns:
        List of language names.
    """
    skill_to_languages = {
        "python": ["python"],
        "frontend": ["typescript", "javascript"],
    }
    return skill_to_languages.get(skill or "python", ["python"])


async def run_rlm_review(cwd: Path, languages: list[str]) -> str:
    """Execute RLM-based code review.

    Args:
        cwd: Working directory for the review.
        languages: List of languages to include in review.

    Returns:
        Review output as markdown string.

    Raises:
        REPLCrashError: If REPL process exits unexpectedly.
        HeartbeatFailedError: If REPL stops responding to heartbeats.
        ContainerError: If devcontainer fails to start or crashes.
    """
    from daydream.rlm import load_codebase

    # Load codebase to show stats
    ctx = load_codebase(cwd, languages)
    print_info(console, f"[RLM] Files: {ctx.file_count:,}")
    print_info(console, f"[RLM] Estimated tokens: {ctx.total_tokens:,}")
    console.print()

    if ctx.file_count == 0:
        raise ValueError("No matching files found in target directory")

    # Show largest files
    if ctx.largest_files:
        print_dim(console, "Largest files:")
        for path, tokens in ctx.largest_files[:5]:
            print_dim(console, f"  {path}: {tokens:,} tokens")
        console.print()

    # TODO: Full RLM orchestration implementation
    # This will be expanded to run the full REPL loop with LLM interactions
    print_info(console, "[RLM] Full orchestration not yet implemented")
    print_info(console, "[RLM] Codebase loaded successfully")

    return "# Code Review\n\nRLM review not yet fully implemented."


async def run_standard_review(cwd: Path, skill: str) -> str:
    """Execute standard skill-based code review.

    Args:
        cwd: Working directory for the review.
        skill: Full skill path (e.g., "beagle:review-python").

    Returns:
        Review output as markdown string.
    """
    from daydream.phases import phase_review

    await phase_review(cwd, skill)

    # Read the review output file
    review_output_path = cwd / REVIEW_OUTPUT_FILE
    if review_output_path.exists():
        return review_output_path.read_text()
    return ""


async def run_rlm_review_with_fallback(
    cwd: Path,
    languages: list[str],
    fallback_skill: str,
) -> str:
    """Execute RLM review with graceful fallback to standard review.

    Attempts to run an RLM-based code review first. If RLM mode fails
    due to container or REPL errors, falls back to the standard
    skill-based review.

    Args:
        cwd: Working directory for the review.
        languages: List of languages to include in review.
        fallback_skill: Full skill path to use for fallback (e.g., "beagle:review-python").

    Returns:
        Review output as markdown string.
    """
    try:
        return await run_rlm_review(cwd, languages)
    except (REPLCrashError, HeartbeatFailedError, ContainerError) as e:
        console.print(f"[yellow]RLM mode failed: {e}[/yellow]")
        console.print("[yellow]Falling back to standard skill-based review...[/yellow]")
        console.print()
        return await run_standard_review(cwd, fallback_skill)


async def _run_rlm_mode(config: RunConfig, target_dir: Path) -> int:
    """Execute RLM mode review.

    Args:
        config: Run configuration.
        target_dir: Target directory path.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    languages = _skill_to_languages(config.skill)

    # Determine fallback skill
    fallback_skill = SKILL_MAP.get(config.skill or "python", "beagle:review-python")

    print_info(console, f"[RLM] Mode: reviewing {target_dir}")
    print_info(console, f"[RLM] Languages: {', '.join(languages)}")
    print_info(console, f"[RLM] Model: {config.model}")
    console.print()

    try:
        review_output = await run_rlm_review_with_fallback(
            target_dir,
            languages,
            fallback_skill,
        )

        # Write review output to file if we got content
        if review_output:
            review_output_path = target_dir / REVIEW_OUTPUT_FILE
            review_output_path.write_text(review_output)
            print_success(console, f"Review output written to: {review_output_path}")

        return 0

    except ValueError as e:
        # No files found
        print_error(console, "No Files", str(e))
        return 1
    except MissingSkillError as e:
        # Fallback skill not available
        _print_missing_skill_error(e.skill_name)
        return 1


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

    # RLM mode branch
    if config.rlm_mode:
        return await _run_rlm_mode(config, target_dir)

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
                ("2", "React/TypeScript frontend (review-frontend)"),
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

    try:
        feedback_items: list[dict] = []
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
