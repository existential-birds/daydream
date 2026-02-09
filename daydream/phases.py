"""Phase functions for the review and fix loop."""

import subprocess
from pathlib import Path
from typing import Any

import anyio

from daydream.agent import (
    _log_debug,
    console,
    detect_test_success,
    run_agent,
)
from daydream.backends import Backend, ContinuationToken
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.ui import (
    ParallelFixPanel,
    phase_subtitle,
    print_dim,
    print_error,
    print_fix_complete,
    print_fix_progress,
    print_info,
    print_menu,
    print_phase_hero,
    print_success,
    print_warning,
    prompt_user,
)

TEST_FIX_PROMPT = "The tests failed. Analyze the failures and fix them."

FEEDBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "description": {"type": "string"},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                },
                "required": ["id", "description", "file", "line"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}

FixResult = tuple[dict[str, Any], bool, str | None]


def _detect_default_branch(cwd: Path) -> str | None:
    """Detect the default branch (main/master) for the repository.

    Returns:
        The default branch name, or None if detection fails.

    """
    # Try the remote HEAD symbolic ref first (most reliable)
    try:
        result = subprocess.run(  # noqa: S603 - arguments are not user-controlled
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        if result.returncode == 0:
            # Output is like "refs/remotes/origin/main"
            return result.stdout.strip().rsplit("/", 1)[-1]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: check if main or master exists locally
    for branch in ("main", "master"):
        try:
            result = subprocess.run(  # noqa: S603 - arguments are not user-controlled
                ["git", "rev-parse", "--verify", branch],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            if result.returncode == 0:
                return branch
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return None


def check_review_file_exists(target_dir: Path) -> None:
    """Check that the review output file exists.

    Args:
        target_dir: Target directory containing the review output.

    Raises:
        FileNotFoundError: If the review output file doesn't exist.

    """
    review_output_path = target_dir / REVIEW_OUTPUT_FILE
    if not review_output_path.exists():
        msg = f"""No review file found.

Expected: {review_output_path}

Run a full review first:
  daydream {target_dir} --python"""
        raise FileNotFoundError(msg)


async def phase_review(backend: Backend, cwd: Path, skill: str) -> None:
    """Phase 1: Run review skill, write output to .review-output.md.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the review
        skill: The review skill to invoke (e.g., beagle:review-python)

    Returns:
        None

    Raises:
        Exception: If the agent fails to execute the review skill.

    """
    print_phase_hero(console, "BREATHE", "\"Be guided by beauty\" —Jim Simons")

    # Use absolute path to prevent model hallucination of paths from training data
    review_output_path = cwd / REVIEW_OUTPUT_FILE
    skill_invocation = backend.format_skill_invocation(skill)

    # Detect the base branch so the agent knows what to diff against.
    # Without this, agents (especially Codex) may run `git diff` with no
    # base ref, which only shows uncommitted changes and misses all
    # committed work on the branch.
    base_branch = _detect_default_branch(cwd)
    if base_branch:
        diff_instruction = (
            f"\nReview the changes on the current branch compared to `{base_branch}`. "
            f"Use `git diff {base_branch}...HEAD` to get the diff.\n"
        )
    else:
        diff_instruction = (
            "\nReview the changes on the current branch compared to the default branch. "
            "Use `git diff main...HEAD` or `git diff master...HEAD` to get the diff.\n"
        )

    prompt = f"""{skill_invocation}
{diff_instruction}
Write the full review output to {review_output_path}.
"""

    await run_agent(backend, cwd, prompt)

    output_path = cwd / REVIEW_OUTPUT_FILE
    if output_path.exists():
        print_success(console, f"Review output written to: {output_path}")
    else:
        print_warning(console, "Review output file was not created")


async def phase_parse_feedback(backend: Backend, cwd: Path) -> list[dict[str, Any]]:
    """Phase 2: Parse feedback from review output and return validated items.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory containing the review output

    Returns:
        List of validated feedback items with id, description, file, line

    Raises:
        ValueError: If the agent output is not a valid list.

    """
    print_phase_hero(console, "REFLECT", phase_subtitle("REFLECT"))

    # Use absolute path to prevent model hallucination of paths from training data
    review_output_path = cwd / REVIEW_OUTPUT_FILE
    prompt = f"""Read the review output file at {review_output_path}.

Extract ONLY actionable issues that need fixing. Skip these sections entirely:
- "Good Patterns" or "Strengths"
- "Summary" sections
- Any positive observations

For each issue found, return a JSON object with this structure:
{{"issues": [
  {{"id": 1, "description": "Brief description of the issue", "file": "path/to/file.py", "line": 42}}
]}}

If there are no actionable issues, return: {{"issues": []}}
"""

    result, _ = await run_agent(backend, cwd, prompt, output_schema=FEEDBACK_SCHEMA)

    if not isinstance(result, dict) or "issues" not in result:
        _log_debug(
            f"[PARSE_FAIL] expected dict with 'issues', "
            f"got {type(result).__name__}: {result!r:.500}\n"
        )
        # When structured output and JSON fallback both fail (e.g. empty
        # response), treat as "no issues" rather than crashing.
        if isinstance(result, str) and not result.strip():
            _log_debug("[PARSE_FALLBACK] empty result, treating as no issues\n")
            print_warning(console, "Agent returned empty response; treating as no actionable issues")
            return []
        raise ValueError(f"Expected dict with 'issues' key, got {type(result)}")

    feedback_items = result["issues"]
    print_info(console, f"Found {len(feedback_items)} actionable issues")
    return feedback_items


async def phase_fix(backend: Backend, cwd: Path, item: dict[str, Any], item_num: int, total: int) -> None:
    """Phase 3: Apply a single fix for one feedback item.

    Args:
        backend: The Backend to execute against.
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

    await run_agent(backend, cwd, prompt)
    print_fix_complete(console, item_num, total)


async def phase_test_and_heal(backend: Backend, cwd: Path) -> tuple[bool, int]:
    """Phase 4: Run tests and prompt user on failure for action.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for running tests

    Returns:
        Tuple of (success: bool, retries_used: int)

    """
    print_phase_hero(console, "AWAKEN", phase_subtitle("AWAKEN"))

    retries_used = 0
    continuation: ContinuationToken | None = None

    while True:
        console.print()
        if retries_used > 0:
            print_info(console, f"Test retry {retries_used}")
        else:
            print_info(console, "Running test suite...")

        prompt = "Run the project's test suite. Report if tests pass or fail."
        output, continuation = await run_agent(backend, cwd, prompt, continuation=continuation)

        test_passed = detect_test_success(output)

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
            _, continuation = await run_agent(backend, cwd, TEST_FIX_PROMPT, continuation=continuation)
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
            _, continuation = await run_agent(backend, cwd, TEST_FIX_PROMPT, continuation=continuation)
            retries_used += 1
            continue


async def phase_commit_push(backend: Backend, cwd: Path) -> None:
    """Prompt user to commit and push changes.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the commit

    Returns:
        None

    """
    response = prompt_user(console, "Commit and push changes? [y/N]", "n")

    if response.lower() in ("y", "yes"):
        console.print()
        print_info(console, "Running commit-push skill...")
        skill_invocation = backend.format_skill_invocation("beagle-core:commit-push")
        await run_agent(backend, cwd, skill_invocation)
        print_success(console, "Commit and push complete")
    else:
        print_dim(console, "Skipping commit and push")


async def phase_fetch_pr_feedback(backend: Backend, cwd: Path, pr_number: int, bot: str) -> None:
    """Fetch PR feedback by invoking the fetch-pr-feedback skill.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the agent
        pr_number: Pull request number to fetch feedback from
        bot: Bot username whose comments to fetch

    Returns:
        None

    Raises:
        Exception: If the agent fails to fetch PR feedback.

    """
    print_phase_hero(console, "LISTEN", phase_subtitle("LISTEN"))

    skill_invocation = backend.format_skill_invocation(
        "beagle-core:fetch-pr-feedback", f"--pr {pr_number} --bot {bot}"
    )

    await run_agent(backend, cwd, skill_invocation)

    output_path = cwd / REVIEW_OUTPUT_FILE
    if output_path.exists():
        print_success(console, f"PR feedback written to: {output_path}")
    else:
        print_warning(console, "PR feedback file was not created")


async def phase_fix_parallel(
    backend: Backend, cwd: Path, feedback_items: list[dict[str, Any]]
) -> list[FixResult]:
    """Apply fixes for all feedback items concurrently using parallel agents.

    Launches one agent per feedback item in a task group. Each agent runs
    independently; individual failures are captured without aborting others.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the fixes
        feedback_items: List of feedback items, each with description, file, line

    Returns:
        List of (item, success, error) tuples for each feedback item

    """
    results: list[FixResult] = []
    limiter = anyio.CapacityLimiter(4)
    panel = ParallelFixPanel(console, feedback_items)
    panel.start()

    async with anyio.create_task_group() as tg:
        for index, item in enumerate(feedback_items):
            description = item.get("description", "No description")
            file_path = item.get("file", "Unknown file")
            line = item.get("line", "Unknown")

            prompt = f"""Fix this issue:
{description}

File: {file_path}
Line: {line}

Make the minimal change needed.
"""

            # Default arguments capture loop variables by value, avoiding late-binding
            # closure issues where all tasks would reference the final loop iteration.
            async def _fix_task(
                task_index: int = index,
                task_item: dict[str, Any] = item,
                task_prompt: str = prompt,
            ) -> None:
                def callback(message: str, i: int = task_index) -> None:
                    panel.update_row(i, message)

                try:
                    async with limiter:
                        await run_agent(backend, cwd, task_prompt, progress_callback=callback)
                    panel.complete_row(task_index)
                    results.append((task_item, True, None))
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    panel.fail_row(task_index, error_msg)
                    results.append((task_item, False, error_msg))

            tg.start_soon(_fix_task)

    panel.finish()

    succeeded = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    if succeeded > 0:
        print_success(console, f"{succeeded} fix(es) applied successfully")
    if failed > 0:
        print_warning(console, f"{failed} fix(es) failed")
    if succeeded == 0 and failed > 0:
        print_error(console, "All fixes failed", "No changes were applied")

    return results


async def phase_commit_push_auto(backend: Backend, cwd: Path) -> None:
    """Automatically commit and push changes without user prompt.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the commit

    Returns:
        None

    """
    console.print()
    print_info(console, "Running commit-push skill...")
    skill_invocation = backend.format_skill_invocation("beagle-core:commit-push")
    await run_agent(backend, cwd, skill_invocation)
    print_success(console, "Commit and push complete")


async def phase_respond_pr_feedback(
    backend: Backend, cwd: Path, pr_number: int, bot: str, results: list[FixResult]
) -> None:
    """Respond to PR feedback with results of applied fixes.

    Filters to successful results only and invokes the respond-pr-feedback
    skill to post replies on the pull request.

    Args:
        backend: The Backend to execute against.
        cwd: Working directory for the agent
        pr_number: Pull request number to respond to
        bot: Bot username to respond as
        results: List of (item, success, error) tuples from phase_fix_parallel

    Returns:
        None

    """
    successful = [(item, ok, err) for item, ok, err in results if ok]

    if not successful:
        print_warning(console, "No successful fixes to report")
        return

    print_info(console, f"Responding to PR #{pr_number} with {len(successful)} fix result(s)...")

    skill_invocation = backend.format_skill_invocation(
        "beagle-core:respond-pr-feedback", f"--pr {pr_number} --bot {bot}"
    )

    await run_agent(backend, cwd, skill_invocation)
    print_success(console, f"Responded to PR #{pr_number} feedback")
