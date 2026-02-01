"""Phase functions for the review and fix loop."""

from pathlib import Path
from typing import Any

from daydream.agent import (
    console,
    extract_json_from_output,
    run_agent,
)
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.ui import (
    print_dim,
    print_error,
    print_fix_complete,
    print_fix_progress,
    print_info,
    print_menu,
    print_phase,
    print_success,
    print_warning,
    prompt_user,
)


async def phase_review(cwd: Path, skill: str) -> None:
    """Phase 1: Run review skill, write output to .review-output.md.

    Args:
        cwd: Working directory for the review
        skill: The review skill to invoke (e.g., beagle:review-python)

    Returns:
        None

    Raises:
        Exception: If the agent fails to execute the review skill.

    """
    print_phase(console, 1, f"Running review skill: {skill}")

    prompt = f"""/{skill}

Write the full review output to {REVIEW_OUTPUT_FILE} in the project root.
"""

    await run_agent(cwd, prompt)

    output_path = cwd / REVIEW_OUTPUT_FILE
    if output_path.exists():
        print_success(console, f"Review output written to: {output_path}")
    else:
        print_warning(console, "Review output file was not created")


async def phase_parse_feedback(cwd: Path) -> list[dict[str, Any]]:
    """Phase 2: Parse feedback from review output and return validated items.

    Args:
        cwd: Working directory containing the review output

    Returns:
        List of validated feedback items with id, description, file, line

    Raises:
        ValueError: If the agent output cannot be parsed as valid JSON.

    """
    print_phase(console, 2, "Parsing feedback")

    prompt = f"""Read the review output file at {REVIEW_OUTPUT_FILE}.

Extract ONLY actionable issues that need fixing. Skip these sections entirely:
- "Good Patterns" or "Strengths"
- "Summary" sections
- Any positive observations

For each issue found, output a JSON array with this structure:
[
  {{"id": 1, "description": "Brief description of the issue", "file": "path/to/file.py", "line": 42}},
  ...
]

If there are no actionable issues, output an empty array: []

Output ONLY the JSON array, no other text.
"""

    output = await run_agent(cwd, prompt)

    try:
        feedback_items = extract_json_from_output(output)
        print_info(console, f"Found {len(feedback_items)} actionable issues")
        return feedback_items
    except ValueError as e:
        print_error(console, "Parse Error", f"Error parsing feedback: {e}")
        print_dim(console, "Raw output:")
        print_dim(console, "-" * 40)
        print_dim(console, output)
        print_dim(console, "-" * 40)
        raise


async def phase_fix(cwd: Path, item: dict[str, Any], item_num: int, total: int) -> None:
    """Phase 3: Apply a single fix for one feedback item.

    Args:
        cwd: Working directory for the fix
        item: Feedback item containing description, file, and line
        item_num: Current item number (1-indexed)
        total: Total number of items

    Returns:
        None

    """
    description = item.get("description", "No description")
    file_path = item.get("file", "Unknown file")
    line = item.get("line", "Unknown")

    console.print()
    print_fix_progress(console, item_num, total, description)

    prompt = f"""Fix this issue:
{description}

File: {file_path}
Line: {line}

Make the minimal change needed.
"""

    await run_agent(cwd, prompt)
    print_fix_complete(console, item_num, total)


async def phase_test_and_heal(cwd: Path) -> tuple[bool, int]:
    """Phase 4: Run tests and prompt user on failure for action.

    Args:
        cwd: Working directory for running tests

    Returns:
        Tuple of (success: bool, retries_used: int)

    """
    from daydream.agent import _detect_test_success

    print_phase(console, 4, "Running tests")

    retries_used = 0

    while True:
        console.print()
        if retries_used > 0:
            print_info(console, f"Test retry {retries_used}")
        else:
            print_info(console, "Running test suite...")

        prompt = "Run the project's test suite. Report if tests pass or fail."
        output = await run_agent(cwd, prompt)

        test_passed = _detect_test_success(output)

        if test_passed:
            print_success(console, "Tests passed")
            return True, retries_used

        print_warning(console, "Tests may have failed or result is unclear.")
        print_menu(console, "What would you like to do?", [
            ("1", "Retry tests (run again without fixes)"),
            ("2", "Fix and retry (launch agent to fix issues)"),
            ("3", "Ignore and continue (mark as passed)"),
            ("4", "Abort (exit with failure)"),
        ])

        choice = prompt_user(console, "Choice", "2")

        if choice == "1":
            retries_used += 1
            continue

        elif choice == "2":
            console.print()
            print_info(console, "Launching agent to fix test failures...")
            fix_prompt = "The tests failed. Analyze the failures and fix them."
            await run_agent(cwd, fix_prompt)
            retries_used += 1
            continue

        elif choice == "3":
            print_warning(console, "Ignoring test failures, continuing...")
            return True, retries_used

        elif choice == "4":
            print_error(console, "Aborted", "User requested abort")
            return False, retries_used

        else:
            print_warning(console, f"Invalid choice '{choice}', defaulting to fix and retry")
            console.print()
            print_info(console, "Launching agent to fix test failures...")
            fix_prompt = "The tests failed. Analyze the failures and fix them."
            await run_agent(cwd, fix_prompt)
            retries_used += 1
            continue


async def phase_commit_push(cwd: Path) -> None:
    """Prompt user to commit and push changes.

    Args:
        cwd: Working directory for the commit

    Returns:
        None

    """
    response = prompt_user(console, "Commit and push changes? [y/N]", "n")

    if response.lower() in ("y", "yes"):
        console.print()
        print_info(console, "Running commit-push skill...")
        await run_agent(cwd, "/beagle:commit-push")
        print_success(console, "Commit and push complete")
    else:
        print_dim(console, "Skipping commit and push")
