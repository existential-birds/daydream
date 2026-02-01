"""Main orchestration logic for the review and fix loop."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from daydream.agent import (
    MissingSkillError,
    console,
    get_debug_log,
    set_debug_log,
    set_quiet_mode,
)
from daydream.config import REVIEW_OUTPUT_FILE, REVIEW_SKILLS
from daydream.phases import (
    phase_commit_push,
    phase_fix,
    phase_parse_feedback,
    phase_review,
    phase_test_and_heal,
)
from daydream.ui import (
    SummaryData,
    print_ascii_header,
    print_dim,
    print_error,
    print_info,
    print_menu,
    print_phase,
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
        debug: Enable debug logging to a timestamped file in the target directory.
        cleanup: Remove review output file after completion. If None, prompts user.
        quiet: Suppress verbose output from the agent.
        review_only: Run review phase only without applying fixes.

    """

    target: str | None = None
    skill: str | None = None  # "python" or "frontend"
    debug: bool = False
    cleanup: bool | None = None
    quiet: bool = True
    review_only: bool = False


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


async def run(config: RunConfig | None = None) -> int:
    """Execute the review and fix loop.

    Args:
        config: Optional configuration. If provided, values skip prompts.

    Returns:
        Exit code (0 for success, 1 for failure)

    """
    if config is None:
        config = RunConfig()

    print_ascii_header(console, "DAYDREAM")

    # Get target directory (from config or prompt)
    if config.target is not None:
        target_dir = Path(config.target).resolve()
    else:
        target_input = prompt_user(console, "Enter target directory", ".")
        target_dir = Path(target_input).resolve()

    if not target_dir.is_dir():
        print_error(console, "Invalid Path", f"'{target_dir}' is not a valid directory")
        return 1

    # Get review skill (from config or prompt)
    if config.skill is not None:
        # Map skill name to full skill path
        skill_map = {"python": "beagle:review-python", "frontend": "beagle:review-frontend"}
        if config.skill in skill_map:
            skill = skill_map[config.skill]
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

        if skill_choice not in REVIEW_SKILLS:
            print_error(console, "Invalid Choice", f"'{skill_choice}' is not a valid option")
            return 1

        skill = REVIEW_SKILLS[skill_choice]

    # Set up debug logging if enabled
    debug_log_path: Path | None = None
    if config.debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_path = target_dir / f".review-debug-{timestamp}.log"
        set_debug_log(open(debug_log_path, "w", encoding="utf-8"))  # noqa: SIM115
        print_info(console, f"Debug log: {debug_log_path}")

    # Get cleanup setting (from config or prompt)
    if config.cleanup is not None:
        cleanup_enabled = config.cleanup
    else:
        cleanup_response = prompt_user(console, "Cleanup review output after completion? [y/N]", "n")
        cleanup_enabled = cleanup_response.lower() in ("y", "yes")

    # Set quiet mode
    set_quiet_mode(config.quiet)

    console.print()
    print_info(console, f"Target directory: {target_dir}")
    print_info(console, f"Review skill: {skill}")
    if config.review_only:
        print_info(console, "Mode: review-only (skipping fixes)")
    console.print()

    try:
        # Phase 1: Review
        try:
            await phase_review(target_dir, skill)
        except MissingSkillError as e:
            _print_missing_skill_error(e.skill_name)
            return 1

        # Phase 2: Parse feedback
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
                    skill=skill,
                    target=str(target_dir),
                    feedback_count=len(feedback_items),
                    fixes_applied=0,
                    test_retries=0,
                    tests_passed=True,
                    review_only=True,
                ),
            )
            return 0

        # Phase 3: Apply fixes
        fixes_applied = 0
        if feedback_items:
            print_phase(console, 3, f"Applying {len(feedback_items)} fixes")
            for i, item in enumerate(feedback_items, 1):
                await phase_fix(target_dir, item, i, len(feedback_items))
                fixes_applied += 1
        else:
            print_info(console, "No feedback items found, skipping fix phase")

        # Phase 4: Test and heal
        tests_passed, test_retries = await phase_test_and_heal(target_dir)

        # Print summary
        print_summary(
            console,
            SummaryData(
                skill=skill,
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
        debug_log = get_debug_log()
        if debug_log is not None:
            debug_log.close()
            set_debug_log(None)
            if debug_log_path:
                print_info(console, f"Debug log saved: {debug_log_path}")
