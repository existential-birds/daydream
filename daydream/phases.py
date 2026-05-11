"""Phase functions for the review and fix loop."""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

import daydream
from daydream import git_ops
from daydream.agent import (
    console,
    detect_test_success,
    get_quiet_mode,
    run_agent,
)
from daydream.backends import Backend, ContinuationToken
from daydream.clipboard import clipboard_available, copy_to_clipboard
from daydream.git_ops import BranchNotFoundError, GitError
from daydream.trajectory import (
    DaydreamPhase,
    TrajectoryRecorder,
    get_current_recorder,
    maybe_fork,
)
from daydream.workspace import WorkContext

if TYPE_CHECKING:
    from daydream.deep.detection import StackAssignment
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.ui import (
    ParallelFixPanel,
    phase_subtitle,
    print_dim,
    print_error,
    print_fix_complete,
    print_fix_progress,
    print_info,
    print_issues_table,
    print_menu,
    print_phase_hero,
    print_success,
    print_warning,
    prompt_user,
)

_logger = logging.getLogger(__name__)

TEST_OUTPUT_TAIL_LINES = 100


def _build_fix_prompt(
    test_output: str,
    feedback_items: list[dict[str, Any]] | None = None,
) -> str:
    """Build an enriched prompt for the fix agent with test output and file context.

    Args:
        test_output: Raw test output text.
        feedback_items: Optional list of feedback items with 'file' keys.

    Returns:
        Prompt string with truncated test output and file list.

    """
    lines = test_output.splitlines()
    if len(lines) > TEST_OUTPUT_TAIL_LINES:
        truncated = "\n".join(lines[-TEST_OUTPUT_TAIL_LINES:])
        output_section = f"Here is the tail of the test output:\n\n{truncated}"
    else:
        output_section = f"Here is the test output:\n\n{test_output}"

    parts = [f"The tests failed. {output_section}"]

    if feedback_items:
        files = sorted({item["file"] for item in feedback_items if "file" in item})
        if files:
            file_list = "\n".join(f"- {f}" for f in files)
            parts.append(f"\nFiles modified during the fix phase:\n{file_list}")

    parts.append("\nAnalyze the failures and fix them.")
    if feedback_items:
        parts.append("Focus on the files listed above.")

    return "\n".join(parts)


def _build_setup_investigator_prompt(test_output: str) -> str:
    """Build a read-only diagnostic prompt for the setup-investigator subagent.

    The investigator diagnoses whether the test invocation itself was wrong
    (wrong command, missing setup step, missing env var) — NOT whether the
    code under test is broken. It is strictly read-only.

    Args:
        test_output: Raw failing test output to include verbatim.

    Returns:
        Prompt string demanding a JSON verdict.

    """
    lines = test_output.splitlines()
    if len(lines) > TEST_OUTPUT_TAIL_LINES:
        truncated = "\n".join(lines[-TEST_OUTPUT_TAIL_LINES:])
        output_section = f"Tail of the failing test output:\n\n{truncated}"
    else:
        output_section = f"Failing test output:\n\n{test_output}"

    return (
        "You are a read-only setup-investigator. Your ONLY job is to decide whether "
        "the test command that just failed was the WRONG command to run — not whether "
        "the code under test is broken.\n\n"
        "## Hard Constraints (read-only contract)\n"
        "- You MAY use Read, Grep, and Glob to inspect files.\n"
        "- You MAY use Bash for NON-MUTATING discovery only: `make help`, `npm run` "
        "(no script name — just list scripts), `ls`, `cat`, `pwd`, `git status`.\n"
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


SETUP_INVESTIGATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "replace"]},
        "suggested_command": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "suggested_command", "reason"],
    "additionalProperties": False,
}


async def _run_setup_investigator(
    backend: Backend,
    work: WorkContext,
    test_output: str,
) -> dict[str, Any] | None:
    """Run the read-only setup-investigator subagent and return its verdict.

    Wraps the invocation in ``recorder.fork("setup-investigator")`` when a
    trajectory recorder is active so the diagnostic call is captured as its
    own sub-trajectory. Returns ``None`` on any failure (exception during
    ``run_agent`` or unparseable JSON) so the caller can fall back to the
    original retry command.

    Args:
        backend: The Backend to execute against.
        work: Workspace context; ``work.repo`` is the agent cwd.
        test_output: Raw failing test output to embed in the prompt.

    Returns:
        Parsed verdict dict with keys ``verdict``, ``suggested_command``,
        ``reason`` on success, or ``None`` on any failure.

    """
    recorder = get_current_recorder()
    prompt = _build_setup_investigator_prompt(test_output)

    async def _invoke() -> dict[str, Any] | None:
        try:
            result, _ = await run_agent(
                backend,
                work.repo,
                prompt,
                output_schema=SETUP_INVESTIGATOR_SCHEMA,
                phase=DaydreamPhase.TEST,
            )
        except Exception:  # noqa: BLE001 - investigator failure is non-fatal
            _logger.debug("setup-investigator agent failed", exc_info=True)
            return None

        if isinstance(result, dict) and "verdict" in result:
            return result
        return None

    try:
        async with maybe_fork(recorder, "setup-investigator"):
            return await _invoke()
    except Exception:  # noqa: BLE001 - fork bookkeeping failures are non-fatal
        _logger.debug("setup-investigator fork failed", exc_info=True)
        return None


def _build_failure_summarizer_prompt(
    *,
    test_output: str,
    trajectory_path: Path | None,
    trajectories_dir: Path | None,
    diff_path: Path | None,
    manifest_path: Path | None,
    deep_dir: Path | None,
    changed_files: list[Path],
    has_trajectory: bool,
) -> str:
    """Build a read-only prompt for the failure-summarizer subagent.

    The summarizer is instructed to produce a paste-ready handoff prompt
    (single ``handoff_prompt`` JSON field) that the caller writes verbatim
    to ``handoff.md``. The summarizer is strictly read-only and must
    reference artifacts by absolute path — never embed diffs or
    test-output excerpts.

    Args:
        test_output: Raw failing test output to ground the summary.
        trajectory_path: Live ``trajectory.json`` path if available.
        trajectories_dir: Live ``trajectories/`` directory if available.
        diff_path: Path to ``diff.patch`` if available.
        manifest_path: Path to ``manifest.json`` if available.
        deep_dir: Path to ``deep/`` if available.
        changed_files: Absolute repo paths of files changed in this run.
        has_trajectory: When False the summarizer must include the
            literal ``> Note: trajectory unavailable for this run`` line.

    Returns:
        Prompt string demanding the JSON ``handoff_prompt`` field.
    """
    lines = test_output.splitlines()
    if len(lines) > TEST_OUTPUT_TAIL_LINES:
        truncated = "\n".join(lines[-TEST_OUTPUT_TAIL_LINES:])
        output_section = f"Tail of the failing test output:\n\n{truncated}"
    else:
        output_section = f"Failing test output:\n\n{test_output}"

    artifact_lines: list[str] = []
    if trajectory_path is not None:
        artifact_lines.append(f"- trajectory: {trajectory_path}")
    if trajectories_dir is not None:
        artifact_lines.append(f"- sub-trajectories: {trajectories_dir}")
    if diff_path is not None:
        artifact_lines.append(f"- diff: {diff_path}")
    if manifest_path is not None:
        artifact_lines.append(f"- manifest: {manifest_path}")
    if deep_dir is not None:
        artifact_lines.append(f"- deep artifacts: {deep_dir}")
    artifacts_block = "\n".join(artifact_lines) if artifact_lines else "(none on disk)"

    changed_block = (
        "\n".join(f"- {p}" for p in changed_files) if changed_files else "(none detected)"
    )

    no_trajectory_clause = (
        "" if has_trajectory
        else "\nThe handoff MUST include this literal line verbatim on its own line:\n"
             "    > Note: trajectory unavailable for this run\n"
    )

    return (
        "You are a read-only failure-summarizer. Your job is to write a "
        "paste-ready handoff prompt that another agent will use, in a fresh "
        "session, to propose a fix for the failure daydream just hit.\n\n"
        "## Hard Constraints (read-only contract)\n"
        "- You MAY use Read, Grep, and Glob to inspect the artifacts listed below.\n"
        "- You MAY use Bash for NON-MUTATING discovery only (`ls`, `cat`, `git status`).\n"
        "- You MUST NOT run tests, builds, or installers.\n"
        "- You MUST NOT modify, create, or delete any files.\n"
        "- You MUST NOT invoke Write, Edit, or any file-mutating tool.\n"
        "- You MUST reference artifacts by absolute path. Do NOT embed diff "
        "hunks, trajectory excerpts, or test-output excerpts in the handoff.\n\n"
        "## On-disk artifacts (read these first to ground your summary)\n"
        f"{artifacts_block}\n\n"
        "## Files changed during this daydream run\n"
        f"{changed_block}\n\n"
        f"## Failing test output (for your context only — do NOT paste into handoff)\n\n{output_section}\n\n"
        f"{no_trajectory_clause}"
        "## Handoff prompt template\n"
        "Produce a Markdown document with these sections, in this order:\n\n"
        "1. **Summary** — one paragraph: what daydream attempted, where it "
        "failed (which phase / what gate blocked).\n"
        "2. **Artifacts** — bulleted list of absolute paths from the section "
        "above. No embedded contents.\n"
        "3. **Changed files** — bulleted list of absolute repo paths from the "
        "section above.\n"
        "4. **Instructions for the next agent** — explicit, numbered:\n"
        "   1. Explore the codebase before proposing anything. Read the "
        "artifacts above and use Read/Grep/Glob to build your own model.\n"
        "   2. Propose an architecturally clean, idiomatic solution rooted "
        "in the project's existing patterns.\n"
        "   3. REFUSE to ship inline hacks that only paper over the "
        "symptom (stubbing assertions, skipping tests, hardcoding values to "
        "make a check pass). Root-cause the failure.\n\n"
        "## Output\n"
        "Return a JSON object matching this schema (and ONLY this JSON, no prose):\n"
        '{"handoff_prompt": "<the full Markdown handoff body, ready to paste>"}\n'
    )


FAILURE_SUMMARIZER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "handoff_prompt": {"type": "string"},
    },
    "required": ["handoff_prompt"],
    "additionalProperties": False,
}


def _changed_files(repo: Path) -> list[Path]:
    """Return absolute paths of files changed in *repo* (working tree).

    Delegates to :func:`git_ops.changed_files` for the actual git queries,
    then resolves the repo-relative names to absolute paths.  Returns an
    empty list if git is unavailable or the repo has no commits yet.
    """
    return [(repo / name).resolve() for name in git_ops.changed_files(repo)]


def _resolve_handoff_paths(
    recorder: TrajectoryRecorder | None, work: WorkContext,
) -> tuple[Path, Path | None, Path | None, Path | None, Path | None, Path | None]:
    """Return ``(handoff_path, trajectory_path, trajectories_dir, diff_path, manifest_path, deep_dir)``.

    The handoff file is always anchored on ``work.source`` so it survives
    ephemeral-worktree cleanup. The artifact reference paths point at
    locations that will exist when the next agent reads the handoff:

    * In-place runs: live trajectory subtree under
      ``<source>/.daydream/runs/<session_id>/`` (written by the recorder
      on ``__aexit__`` shortly after this function runs).
    * Ephemeral runs with archiving enabled: archive subtree under
      ``<archive_root>/runs/<session_id>/`` (populated by the on_write
      callback after ``__aexit__``).
    * Ephemeral runs with archiving disabled: live paths, even though
      the worktree will be removed — best-effort, with no persistent
      copy of the artifacts available.

    When *recorder* is ``None``, ``handoff_path`` falls back to
    ``<source>/.daydream/handoff-<ts>.md`` (no session id available) and
    every other path is ``None``.

    Returned artifact paths are not existence-checked: at handoff time
    the trajectory has not been flushed yet and the archive bundle has
    not been copied. The caller treats them as forward references.
    """
    if recorder is None:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")  # noqa: DTZ005 - filename only
        handoff_path = work.source / ".daydream" / f"handoff-{ts}.md"
        return handoff_path, None, None, None, None, None

    handoff_run_dir = work.source / ".daydream" / "runs" / recorder.session_id
    handoff_path = handoff_run_dir / "handoff.md"

    archive_enabled = recorder.on_write is not None
    if work.is_ephemeral and archive_enabled:
        # The ephemeral worktree (and everything under it) will be
        # removed after the recorder exits; the archive callback copies
        # the bundle to <archive_root>/runs/<session_id>/. Reference the
        # archive paths so the handoff stays valid post-cleanup.
        from daydream.archive import get_archive_dir

        artifact_root = get_archive_dir() / "runs" / recorder.session_id
        diff_path = artifact_root / "diff.patch"
        deep_dir = artifact_root / "deep"
    else:
        artifact_root = recorder.target_dir / ".daydream" / "runs" / recorder.session_id
        diff_path = recorder.target_dir / ".daydream" / "diff.patch"
        deep_dir = recorder.target_dir / ".daydream" / "deep"

    trajectory_path = artifact_root / "trajectory.json"
    trajectories_dir = artifact_root / "trajectories"
    manifest_path = artifact_root / "manifest.json"

    return (
        handoff_path,
        trajectory_path,
        trajectories_dir,
        diff_path,
        manifest_path,
        deep_dir,
    )


def _write_handoff(path: Path, body: str) -> None:
    """Write *body* to *path*, creating parent directories as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except OSError:
        _logger.warning("failed to write handoff to %s", path, exc_info=True)


def _build_minimal_handoff(
    *,
    test_output: str,
    trajectory_path: Path | None,
    trajectories_dir: Path | None,
    diff_path: Path | None,
    manifest_path: Path | None,
    deep_dir: Path | None,
    changed_files: list[Path],
    has_trajectory: bool,
) -> str:
    """Build a minimal handoff body without invoking an agent.

    Used when the failure-summarizer subagent fails or no recorder is
    active. Contains the same artifact pointers and instruction block as
    the agent-produced version so the downstream agent gets useful
    context either way.
    """
    lines = test_output.splitlines()
    if len(lines) > TEST_OUTPUT_TAIL_LINES:
        output_section = "\n".join(lines[-TEST_OUTPUT_TAIL_LINES:])
    else:
        output_section = test_output

    artifact_lines: list[str] = []
    if trajectory_path is not None:
        artifact_lines.append(f"- trajectory: {trajectory_path}")
    if trajectories_dir is not None:
        artifact_lines.append(f"- sub-trajectories: {trajectories_dir}")
    if diff_path is not None:
        artifact_lines.append(f"- diff: {diff_path}")
    if manifest_path is not None:
        artifact_lines.append(f"- manifest: {manifest_path}")
    if deep_dir is not None:
        artifact_lines.append(f"- deep artifacts: {deep_dir}")
    artifacts_block = "\n".join(artifact_lines) if artifact_lines else "_(none on disk)_"

    changed_block = (
        "\n".join(f"- {p}" for p in changed_files) if changed_files else "_(none detected)_"
    )

    parts: list[str] = [
        "# Daydream handoff",
        "",
        "## Summary",
        "",
        "Daydream's test phase could not confirm a green run and the user aborted "
        "(option 4). The failure-summarizer subagent did not produce a structured "
        "handoff, so this minimal version was written instead.",
        "",
    ]
    if not has_trajectory:
        parts.append("> Note: trajectory unavailable for this run")
        parts.append("")
    parts.extend([
        "## Artifacts",
        "",
        artifacts_block,
        "",
        "## Changed files",
        "",
        changed_block,
        "",
        "## Failing test output (tail)",
        "",
        "```",
        output_section,
        "```",
        "",
        "## Instructions for the next agent",
        "",
        "1. Explore the codebase before proposing anything. Read the "
        "artifacts above and use Read/Grep/Glob to build your own model.",
        "2. Propose an architecturally clean, idiomatic solution rooted in "
        "the project's existing patterns.",
        "3. REFUSE to ship inline hacks that only paper over the symptom "
        "(stubbing assertions, skipping tests, hardcoding values to make a "
        "check pass). Root-cause the failure.",
        "",
    ])
    return "\n".join(parts)


async def _run_failure_summarizer(
    backend: Backend,
    work: WorkContext,
    test_output: str,
) -> tuple[str, Path]:
    """Run the read-only failure-summarizer and write ``handoff.md``.

    Always writes a handoff file — falls back to ``_build_minimal_handoff``
    when no recorder is active, the summarizer raises, or the agent
    returns an unparseable result. Wraps the invocation in
    ``recorder.fork("failure-summarizer")`` when a recorder is present so
    the diagnostic call is captured as its own sub-trajectory.

    Args:
        backend: Backend used to invoke the summarizer subagent.
        work: Workspace context; ``work.repo`` is the agent cwd.
        test_output: Raw failing test output to ground the summary.

    Returns:
        Tuple ``(handoff_body, handoff_path)``. The body is also what
        was written to disk.
    """
    recorder = get_current_recorder()
    (
        handoff_path,
        trajectory_path,
        trajectories_dir,
        diff_path,
        manifest_path,
        deep_dir,
    ) = _resolve_handoff_paths(recorder, work)

    # Drop a ``.partial`` snapshot of the trajectory before invoking the
    # summarizer. The recorder's ``_write()`` only fires in ``__aexit__``
    # (which happens after this function returns), so without this the
    # main trajectory file would not exist on disk while the handoff is
    # being read. ``write_partial`` is best-effort and never raises.
    if recorder is not None:
        recorder.write_partial()

    has_trajectory = recorder is not None
    changed = _changed_files(work.repo)

    prompt = _build_failure_summarizer_prompt(
        test_output=test_output,
        trajectory_path=trajectory_path,
        trajectories_dir=trajectories_dir,
        diff_path=diff_path,
        manifest_path=manifest_path,
        deep_dir=deep_dir,
        changed_files=changed,
        has_trajectory=has_trajectory,
    )

    async def _invoke() -> str | None:
        try:
            result, _ = await run_agent(
                backend,
                work.repo,
                prompt,
                output_schema=FAILURE_SUMMARIZER_SCHEMA,
                phase=DaydreamPhase.TEST,
            )
        except Exception:  # noqa: BLE001 - summarizer failure is non-fatal
            _logger.debug("failure-summarizer agent failed", exc_info=True)
            return None
        if isinstance(result, dict):
            body = result.get("handoff_prompt")
            if isinstance(body, str) and body.strip():
                return body
        return None

    body: str | None = None
    try:
        async with maybe_fork(recorder, "failure-summarizer"):
            body = await _invoke()
    except Exception:  # noqa: BLE001 - fork bookkeeping failures are non-fatal
        _logger.debug("failure-summarizer fork failed", exc_info=True)
        body = None

    if body is None:
        body = _build_minimal_handoff(
            test_output=test_output,
            trajectory_path=trajectory_path,
            trajectories_dir=trajectories_dir,
            diff_path=diff_path,
            manifest_path=manifest_path,
            deep_dir=deep_dir,
            changed_files=changed,
            has_trajectory=has_trajectory,
        )

    _write_handoff(handoff_path, body)
    return body, handoff_path


def _parse_issue_selection(user_input: str, issues: list[dict[str, Any]]) -> list[int] | None:
    """Parse user's issue selection into a list of issue IDs.

    Args:
        user_input: User input string ("all", "none", "", or comma-separated IDs).
        issues: Full list of issue dicts with "id" keys.

    Returns:
        List of selected issue IDs, or None for explicit skip.

    """
    cleaned = user_input.strip().lower()

    if cleaned in ("none", ""):
        return None

    if cleaned == "all":
        return [issue["id"] for issue in issues]

    valid_ids = {issue["id"] for issue in issues}
    selected = []
    for part in cleaned.split(","):
        part = part.strip()
        if part.isdigit():
            issue_id = int(part)
            if issue_id in valid_ids:
                selected.append(issue_id)

    return selected


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
                    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "rationale": {"type": "string"},
                },
                "required": ["id", "description", "file", "line", "confidence", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}

ALTERNATIVE_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "recommendation": {"type": "string"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "id",
                    "title",
                    "description",
                    "recommendation",
                    "severity",
                    "files",
                    "confidence",
                    "rationale",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "title": {"type": "string"},
                            "changes": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "file": {"type": "string"},
                                        "description": {"type": "string"},
                                        "action": {"type": "string", "enum": ["modify", "create", "delete"]},
                                        "references": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "file": {"type": "string"},
                                                    "symbol": {"type": "string"},
                                                },
                                                "required": ["file", "symbol"],
                                                "additionalProperties": False,
                                            },
                                        },
                                    },
                                    "required": ["file", "description", "action", "references"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["id", "title", "changes"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["summary", "issues"],
            "additionalProperties": False,
        },
    },
    "required": ["plan"],
    "additionalProperties": False,
}


def _confidence_and_convention_instructions() -> str:
    """Prompt language for QUAL-02 confidence, QUAL-03 conventions, QUAL-04 error handling.

    Called by all four phase prompt builders to keep them in lockstep on these
    rules. Returns a markdown section that should be appended after the
    Exploration Context section.

    Returns:
        Markdown-formatted instruction block as a single string.

    """
    return (
        "## Confidence and Convention Rules\n\n"
        "For every issue you report, you MUST set `confidence` and `rationale`:\n"
        "- HIGH: directly verified by a specific entry in the Exploration Context above. "
        "Your rationale MUST name the specific Dependency edge, Convention entry, or "
        "affected file that supports the issue.\n"
        "- MEDIUM: consistent with the Exploration Context but not pinned to a specific entry.\n"
        "- LOW: inferred from the diff alone, no exploration evidence. Your rationale MUST "
        "state 'no exploration evidence'.\n\n"
        "Convention handling has TWO distinct cases — do not conflate them:\n"
        "1. Before proposing a fix, check it against the Codebase Conventions section. "
        "If your fix would violate a convention, DROP IT — do not include it.\n"
        "2. If the reviewed code itself violates a convention, that IS the issue. "
        "flag it as HIGH confidence and cite the convention by name in `rationale`.\n\n"
        "You are reviewing AI-generated code. Be strict. Prefer LOW over MEDIUM when uncertain.\n\n"
        "## Error Handling Semantics (QUAL-04)\n\n"
        "Not all caught-and-logged errors are bugs. Before flagging error handling as\n"
        "an issue, classify the operation's criticality:\n\n"
        "- **Critical path**: The caller NEEDS this result to proceed (e.g., loading\n"
        "  config, connecting to database, parsing user input). Swallowing errors here\n"
        "  IS a bug — flag it.\n"
        "- **Best-effort / diagnostic**: The operation is non-essential (e.g., writing\n"
        "  telemetry, flushing debug traces, updating timestamps, sending analytics).\n"
        "  Logging a warning and continuing is the CORRECT pattern — it prevents a\n"
        "  secondary failure from masking or killing the primary operation.\n\n"
        "A `warn!()` + continue after a non-critical operation is intentional graceful\n"
        "degradation, not a 'silent failure.' Changing it to error propagation (`?`,\n"
        "`return Err`, `unwrap`) would make the system MORE fragile, not less.\n\n"
        "When reporting an error handling issue:\n"
        "- State whether the operation is critical-path or best-effort\n"
        "- If best-effort, explain why propagation would be better than logging\n"
        "- If you cannot articulate why the caller benefits from receiving the error,\n"
        "  DROP the finding\n\n"
        "## Refactoring Recommendations\n\n"
        "Before recommending extraction or deduplication (e.g. 'extract a shared\n"
        "helper', 'consolidate duplicated logic'), check the same directory for\n"
        "existing shared modules (shared.ts, utils.ts, helpers.py, common.go, etc.).\n"
        "If shared utilities already exist, the author likely made a deliberate\n"
        "factoring choice. Refactoring recommendations without evidence that shared\n"
        "code doesn't already exist should be MEDIUM confidence at most. Focus on\n"
        "concrete sub-findings (bugs, correctness issues) rather than structural\n"
        "opinions about code organization."
    )


def _plan_grounding_instructions() -> str:
    """Prompt language for OUTP-01 plan reference grounding.

    Returns:
        Markdown-formatted instruction block as a single string.

    """
    return (
        "## Plan Reference Grounding\n\n"
        "For every change in the plan, populate `references` with `{file, symbol}` entries "
        "drawn ONLY from the Exploration Context above. Do not invent file paths or symbol "
        "names. If you cannot ground a step in the Exploration Context, leave `references` "
        "as an empty array — the renderer will flag ungrounded steps for the user."
    )


def _dependency_impact_instructions() -> str:
    """Prompt language for QUAL-01 cross-file dependency surfacing in review output.

    Returns:
        Markdown-formatted instruction block as a single string.

    """
    return (
        "## Dependency Impact\n\n"
        "Begin your review output with a 'Dependency Impact' section that summarizes the "
        "call-chain analysis from the Exploration Context dependencies above before listing "
        "any issues. When an individual issue's rationale cites a dependency, include the "
        "file:symbol reference inline within that issue."
    )


def _validate_issue(issue: dict[str, Any]) -> None:
    """Validate a parsed feedback issue carries the required confidence/rationale fields.

    Args:
        issue: Issue dict produced by structured-output parsing.

    Raises:
        ValueError: If `confidence` or `rationale` is missing or empty.

    """
    confidence = issue.get("confidence")
    if not confidence or confidence not in ("HIGH", "MEDIUM", "LOW"):
        raise ValueError(f"Issue is missing or has invalid confidence label: {issue!r}")
    rationale = issue.get("rationale")
    if not rationale:
        raise ValueError(f"Issue is missing rationale: {issue!r}")


def _exploration_pointer(exploration_dir: Path | None) -> str:
    """Return a short prompt pointer to exploration files, or empty string."""
    if exploration_dir is None:
        return ""
    return (
        f"Pre-scan exploration results are available in {exploration_dir}/.\n"
        f"Read {exploration_dir}/summary.md for an index of what was found.\n"
        f"Reference individual files as needed during your review — "
        f"do NOT read them all up front.\n"
    )


def _settled_decisions_block(prior_commits: str | None) -> str:
    """Return prompt block marking prior daydream commits as settled, or empty string.

    Args:
        prior_commits: Oneline log of prior daydream commits on this branch.
            When None or empty, returns empty string.

    Returns:
        Prompt block instructing the agent to treat prior commits as settled decisions.

    """
    if not prior_commits:
        return ""
    return (
        "Prior automated-review commits on this branch — treat as settled "
        "decisions unless they introduce bugs or security issues:\n"
        f"{prior_commits}"
    )


def build_review_prompt(
    *,
    skill_invocation: str = "",
    diff_instruction: str = "",
    review_output_path: str = "",
    exploration_dir: Path | None = None,
    prior_commits: str | None = None,
) -> str:
    """Assemble the prompt for `phase_review`.

    Args:
        skill_invocation: Backend-formatted skill invocation string.
        diff_instruction: Diff scope instruction text.
        review_output_path: Absolute path the agent should write its review to.
        exploration_dir: Optional path to exploration output directory.
        prior_commits: Oneline log of prior daydream commits on this branch.
            When present, injected as settled-decisions context.

    Returns:
        Fully assembled prompt string.

    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    settled = _settled_decisions_block(prior_commits)
    if settled:
        parts.append(settled)
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    body = (
        f"{skill_invocation}\n"
        f"{diff_instruction}\n"
        f"Write the full review output to {review_output_path}.\n"
    )
    parts.append(body)
    return "\n".join(parts)


def build_intent_prompt(
    *,
    diff_path: str = "",
    branch: str = "",
    log: str = "",
    exploration_dir: Path | None = None,
) -> str:
    """Assemble the prompt for `phase_understand_intent`.

    Returns:
        Fully assembled prompt string.

    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    body = (
        f"You have full access to explore the codebase. Read the diff file at {diff_path} "
        f"and examine the codebase to understand the intent of these changes. "
        f"Present your understanding concisely — what problem is being solved and how.\n\n"
        f"Branch: {branch}\n\n"
        f"Commit log:\n{log}\n"
    )
    parts.append(body)
    return "\n".join(parts)


def build_alternative_review_prompt(
    *,
    intent_summary: str = "",
    diff_path: str = "",
    exploration_dir: Path | None = None,
) -> str:
    """Assemble the prompt for `phase_alternative_review`.

    Returns:
        Fully assembled prompt string.

    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(_confidence_and_convention_instructions())
    body = (
        f"The intent of this PR has been confirmed as:\n\n"
        f"{intent_summary}\n\n"
        f"Given this intent, explore the codebase and evaluate the implementation "
        f"in the diff at {diff_path}. Would you have done this differently?\n\n"
        f"Return a numbered list of issues covering both architectural alternatives "
        f"and incremental improvements. For each issue, include: a sequential id "
        f"number, a brief title, a description of what's wrong or could be better, "
        f"your recommended alternative, a severity level (high/medium/low), and "
        f"the relevant file paths.\n\n"
        f"If the implementation is solid and you wouldn't change anything, return an empty issues list.\n"
    )
    parts.append(body)
    return "\n".join(parts)


def build_plan_prompt(
    *,
    intent_summary: str = "",
    issues_text: str = "",
    diff_path: str = "",
    exploration_dir: Path | None = None,
) -> str:
    """Assemble the prompt for `phase_generate_plan`.

    Returns:
        Fully assembled prompt string.

    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(_confidence_and_convention_instructions())
    parts.append(_plan_grounding_instructions())
    body = (
        f"The intent of this PR is:\n\n"
        f"{intent_summary}\n\n"
        f"Create a detailed implementation plan for fixing these issues:\n"
        f"{issues_text}\n\n"
        f"For each issue, specify what files to change, what the change should be, "
        f"and why. Make this actionable enough to hand to another developer or agent.\n\n"
        f"The diff is available at {diff_path} for context. Do not invent file paths or symbols.\n"
    )
    parts.append(body)
    return "\n".join(parts)

FixResult = tuple[dict[str, Any], bool, str | None]


def revert_uncommitted_changes(cwd: Path) -> bool:
    """Discard all uncommitted changes (tracked and untracked).

    Used after a failed iteration to restore the last committed state.

    Returns:
        True if revert succeeded, False otherwise.

    """
    try:
        git_ops.checkout_paths(cwd, [Path(".")])
        git_ops.clean_untracked(cwd)
    except GitError as e:
        if not get_quiet_mode():
            print_warning(console, f"Revert failed: {type(e).__name__}: {e}")
        return False
    return True


def _prior_daydream_commits(work: WorkContext) -> str | None:
    """Return oneline log of prior daydream commits on this branch."""
    return git_ops.daydream_commits(work.repo, work.base_branch)


def _detect_default_branch(cwd: Path) -> str | None:
    """Detect the default branch (main/master) for the repository.

    Returns:
        The default branch name, or None if detection fails.

    """
    try:
        return git_ops.default_branch(cwd)
    except (BranchNotFoundError, GitError):
        return None


def _git_diff(cwd: Path, exclude: list[str] | None = None) -> str | None:
    """Get the diff of current branch against the default branch.

    Args:
        cwd: Repository working directory.
        exclude: Optional list of paths to exclude from the diff via git's
            `:(exclude)` magic pathspec. Each entry may be a file or directory.

    Returns:
        The diff output, empty string if no diff, or None if base branch detection fails.

    """
    base_branch = _detect_default_branch(cwd)
    if not base_branch:
        return None
    try:
        return git_ops.diff(cwd, base_branch, exclude=exclude)
    except GitError:
        return None


def _git_log(cwd: Path) -> str:
    """Get the commit log of the current branch since diverging from default branch.

    Returns:
        The log output, or empty string if detection fails.

    """
    base_branch = _detect_default_branch(cwd)
    if not base_branch:
        return ""
    try:
        return git_ops.log(cwd, base_branch)
    except GitError:
        return ""


def _git_branch(cwd: Path) -> str:
    """Get the current branch name.

    Returns:
        The branch name, or empty string if detection fails.

    """
    try:
        name = git_ops.current_branch(cwd)
    except GitError:
        return ""
    return name or ""


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
  daydream {target_dir}"""
        raise FileNotFoundError(msg)


async def phase_review(
    backend: Backend,
    work: WorkContext,
    skill: str,
    *,
    diff_base: str | None = None,
    exploration_dir: Path | None = None,
    exclude: list[str] | None = None,
) -> None:
    """Phase 1: Run review skill, write output to .review-output.md.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for the run; ``work.repo`` is the agent cwd
            and ``work.base_branch`` / ``work.base_sha`` drive the diff
            instruction. Base resolution happens once in ``open_workspace``.
        skill: The review skill to invoke (e.g., beagle:review-python)
        diff_base: Optional commit SHA to diff against. When provided, the review
            covers only changes since that commit (used for incremental reviews
            in loop mode). When None, diffs against the resolved base branch.
        exclude: Optional list of paths the agent should exclude when it runs
            `git diff` itself. Applied via git's `:(exclude)` magic pathspec.

    Returns:
        None

    Raises:
        Exception: If the agent fails to execute the review skill.

    """
    print_phase_hero(console, "BREATHE", "\"Be guided by beauty\" —Jim Simons")

    # Use absolute path to prevent model hallucination of paths from training data
    review_output_path = work.repo / REVIEW_OUTPUT_FILE
    skill_invocation = backend.format_skill_invocation(skill)

    if diff_base:
        # Incremental review: only review changes since the last iteration commit
        diff_instruction = (
            f"\nReview ONLY the changes since commit {diff_base}. "
            f"Use `git diff {diff_base}...HEAD` to get the diff.\n"
        )
    else:
        # Full branch review: hand the agent the workspace's resolved base
        # branch so Codex / Claude don't re-detect (and possibly disagree).
        diff_instruction = (
            f"\nReview the changes on the current branch compared to `{work.base_branch}`. "
            f"Use `git diff {work.base_branch}...HEAD` to get the diff.\n"
        )

    if exclude:
        excluded_paths = ", ".join(exclude)
        pathspec_args = " ".join(f"':(exclude){p.rstrip('/')}'" for p in exclude)
        base_for_example = diff_base or work.base_branch
        ref = f"{base_for_example}...HEAD" if not diff_base else f"{diff_base}...HEAD"
        diff_instruction += (
            f"\nExclude these paths from the diff: {excluded_paths}. "
            f"Use git pathspec magic, e.g. `git diff {ref} -- . {pathspec_args}`.\n"
        )

    prior_commits = _prior_daydream_commits(work)
    prompt = build_review_prompt(
        skill_invocation=skill_invocation,
        diff_instruction=diff_instruction,
        review_output_path=str(review_output_path),
        exploration_dir=exploration_dir,
        prior_commits=prior_commits,
    )

    await run_agent(backend, work.repo, prompt, phase=DaydreamPhase.REVIEW)

    output_path = work.repo / REVIEW_OUTPUT_FILE
    if output_path.exists():
        print_success(console, f"Review output written to: {output_path}")
    else:
        print_warning(console, "Review output file was not created")


async def phase_parse_feedback(
    backend: Backend,
    work: WorkContext,
    *,
    input_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Phase 2: Parse feedback from review output and return validated items.

    Args:
        backend: The Backend to execute against.
        work: Workspace context. ``work.repo`` doubles as the agent's cwd
            and the source of the default review path.
        input_path: Optional explicit path to the review markdown to parse.
            When None (default), reads ``work.repo / REVIEW_OUTPUT_FILE``,
            preserving behavior for single-skill, PR feedback, and TTT flows.
            When provided, reads that path instead — used by deep-mode's
            pre-merge parse stage to iterate per-stack outputs without
            overwriting each other at the shared REVIEW_OUTPUT_FILE location.

    Returns:
        List of validated feedback items with id, description, file, line

    Raises:
        ValueError: If the agent output is not a valid list.

    """
    print_phase_hero(console, "REFLECT", phase_subtitle("REFLECT"))

    # Use absolute path to prevent model hallucination of paths from training data
    review_output_path = input_path if input_path is not None else work.repo / REVIEW_OUTPUT_FILE
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

    result, _ = await run_agent(backend, work.repo, prompt, output_schema=FEEDBACK_SCHEMA, phase=DaydreamPhase.PARSE)

    if not isinstance(result, dict) or "issues" not in result:
        # When structured output and JSON fallback both fail (e.g. empty
        # response), treat as "no issues" rather than crashing.
        if isinstance(result, str) and not result.strip():
            print_warning(console, "Agent returned empty response; treating as no actionable issues")
            return []
        raise ValueError(f"Expected dict with 'issues' key, got {type(result)}")

    feedback_items = result["issues"]
    print_info(console, f"Found {len(feedback_items)} actionable issues")
    return feedback_items


async def phase_fix(
    backend: Backend, work: WorkContext, item: dict[str, Any], item_num: int, total: int,
) -> None:
    """Phase 3: Apply a single fix for one feedback item.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for the fix; ``work.repo`` is the agent cwd.
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

Make the minimal change needed. Do NOT change error handling semantics
(e.g., converting warn-and-continue to error propagation, or vice versa)
unless the issue description specifically explains why the current error
handling strategy is wrong for that code path.
"""

    await run_agent(backend, work.repo, prompt, phase=DaydreamPhase.FIX)
    print_fix_complete(console, item_num, total)


async def phase_test_and_heal(
    backend: Backend,
    work: WorkContext,
    feedback_items: list[dict[str, Any]] | None = None,
) -> tuple[bool, int]:
    """Phase 4: Run tests and prompt user on failure for action.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for running tests; ``work.repo`` is the cwd.
        feedback_items: Optional list of feedback items from the fix phase,
            used to enrich the fix prompt with file context.

    Returns:
        Tuple of (success: bool, retries_used: int)

    """
    print_phase_hero(console, "AWAKEN", phase_subtitle("AWAKEN"))

    retries_used = 0
    continuation: ContinuationToken | None = None
    test_command_override: str | None = None

    while True:
        console.print()
        if retries_used > 0:
            print_info(console, f"Test retry {retries_used}")
        else:
            print_info(console, "Running test suite...")

        if test_command_override:
            # Sanitize LLM-suggested command: collapse to single line,
            # strip control chars that could escape the code fence.
            sanitized_cmd = " ".join(test_command_override.split())
            prompt = (
                "Run this exact test command:\n"
                f"```\n{sanitized_cmd}\n```\n"
                "Report if tests pass or fail."
            )
        else:
            prompt = "Run the project's test suite. Report if tests pass or fail."
        # USE-ONCE RESET: test_command_override is consumed by the prompt
        # construction above (lines ~1314-1324) and must be cleared before
        # run_agent executes so the *next* loop iteration falls back to the
        # default test-suite prompt.  This reset MUST stay between prompt
        # construction and the bottom-of-loop branches that may re-assign
        # test_command_override (choice "1" → setup investigator, line ~1367).
        test_command_override = None

        output, continuation = await run_agent(
            backend, work.repo, prompt, continuation=continuation, phase=DaydreamPhase.TEST,
        )

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
            verdict = await _run_setup_investigator(backend, work, output)

            if verdict is None:
                print_warning(
                    console,
                    "Setup investigator failed; retrying with original command",
                )
            else:
                v = verdict.get("verdict")
                reason = verdict.get("reason", "")
                suggested = verdict.get("suggested_command")
                print_info(console, f"Setup investigator verdict: {v} — {reason}")

                if v == "replace" and isinstance(suggested, str) and suggested.strip():
                    response = prompt_user(
                        console, "Use suggested command instead?", "n",
                    )
                    if response.lower() in ("y", "yes"):
                        test_command_override = suggested.strip()

            retries_used += 1
            continue

        elif choice == "2":
            console.print()
            print_info(console, "Launching agent to fix test failures...")
            fix_prompt = _build_fix_prompt(output, feedback_items)
            _, _ = await run_agent(backend, work.repo, fix_prompt, phase=DaydreamPhase.FIX)
            retries_used += 1
            continuation = None
            continue

        elif choice == "3":
            print_warning(console, "Ignoring test failures, continuing...")
            return True, retries_used

        elif choice == "4":
            print_error(console, "Aborted", "User requested abort")
            body, handoff_path = await _run_failure_summarizer(backend, work, output)
            # Show a short preview — the full handoff can be large enough
            # to push important messages off-screen.
            preview_lines = body.splitlines()
            max_preview = 20
            if len(preview_lines) <= max_preview:
                console.print(body)
            else:
                console.print("\n".join(preview_lines[:max_preview]))
                print_info(
                    console,
                    f"... ({len(preview_lines) - max_preview} more lines, see file below)",
                )
            print_info(console, f"Handoff written: {handoff_path}")

            if clipboard_available():
                response = prompt_user(console, "Copy handoff to clipboard?", "y")
                if response.lower() in ("y", "yes"):
                    if copy_to_clipboard(body):
                        print_success(console, "Handoff copied to clipboard")
                    else:
                        print_warning(
                            console, "Clipboard copy failed; copy manually from path above",
                        )
            else:
                print_info(
                    console, "(clipboard unavailable, copy manually from path above)",
                )

            return False, retries_used

        else:
            print_warning(console, f"Invalid choice '{choice}', aborting")
            return False, retries_used


async def _do_commit(
    backend: Backend,
    work: WorkContext,
    *,
    push: bool = False,
    interactive: bool = False,
    iteration: int | None = None,
    items: list[dict[str, Any]] | None = None,
) -> bool:
    """Stage, commit, and optionally push with daydream trailers.

    Args:
        backend: LLM backend used to run the commit agent.
        work: Current workspace context (repo path, run ID, etc.).
        push: If True, push to the remote after committing.
        interactive: If True, prompt the user for confirmation before
            committing.
        iteration: When set, append an ``Iteration: <n>`` trailer to the
            commit message body.
        items: Optional list of fix dicts (with ``file`` and ``description``
            keys) summarising changes applied in this run; included in the
            agent prompt so the commit message is accurate.

    Returns:
        True if a commit was performed, False if the user declined.

    """
    if interactive:
        response = prompt_user(console, "Commit and push changes? [y/N]", "n")
        if response.lower() not in ("y", "yes"):
            print_dim(console, "Skipping commit and push")
            return False

    iteration_line = f"End the body with: Iteration: {iteration}\n\n" if iteration is not None else ""
    push_line = "Then push to the remote." if push else "Do NOT push. Only commit."

    if items:
        summaries = "\n".join(
            f"- {it.get('file', 'unknown')}: {it.get('description', 'no description')}"
            for it in items
        )
        items_context = (
            "The following fixes were applied in this run — use them to write "
            f"an accurate commit message:\n{summaries}\n\n"
        )
    else:
        items_context = ""

    prompt = (
        "Stage all changes and commit using a conventional commit message. "
        "Review the diff to write a meaningful summary of what was fixed or changed. "
        "Use the format: <type>: <concise summary of changes>\n\n"
        f"{items_context}"
        "Pick the most appropriate type from: fix, refactor, style, perf. "
        "If multiple categories of changes exist, pick the dominant one. "
        "Keep the subject line under 72 characters. "
        "Add a body with bullet points if there are multiple distinct changes. "
        f"{iteration_line}"
        "Add these EXACT git trailers as the last lines of the commit message "
        "(after a blank line following the body):\n\n"
        f"Daydream-Run: {work.run_id}\n"
        f"Daydream-Version: {daydream.__version__}\n\n"
        f"{push_line}"
    )
    try:
        sha_before = git_ops.head_sha(work.repo)
    except GitError:
        sha_before = None

    await run_agent(backend, work.repo, prompt, phase=DaydreamPhase.FIX)

    # --- Post-commit trailer verification ---
    # The agent may omit trailers despite being asked.  Verify and amend if
    # missing so that daydream_commits() can always find them.
    # Only verify when the agent actually created a new commit; otherwise we
    # would silently amend the user's prior commit.
    try:
        sha_after = git_ops.head_sha(work.repo)
    except GitError:
        # No HEAD after the agent run means no commit was created.
        return False

    if sha_after == sha_before:
        return False
    if sha_before is None:
        # Cannot confirm the agent created a new commit — skip trailer
        # verification to avoid amending a pre-existing (non-daydream) commit.
        return True

    expected_trailers = {
        "Daydream-Run": work.run_id,
        "Daydream-Version": daydream.__version__,
    }
    try:
        msg = git_ops.head_commit_message(work.repo)
    except GitError:
        return True

    missing = {k: v for k, v in expected_trailers.items() if f"{k}: {v}" not in msg}
    if missing:
        print_warning(
            console,
            f"Commit missing daydream trailer(s): {', '.join(missing)}; amending",
        )
        try:
            git_ops.amend_trailers(work.repo, missing, message=msg)
        except GitError as exc:
            print_warning(console, f"Failed to amend trailers: {exc}")

    return True


async def phase_commit_push(backend: Backend, work: WorkContext) -> None:
    """Prompt user to commit and push changes.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for the commit.

    """
    console.print()
    print_info(console, "Committing and pushing changes...")
    committed = await _do_commit(backend, work, push=True, interactive=True)
    if committed:
        print_success(console, "Commit and push complete")


async def phase_fetch_pr_feedback(
    backend: Backend, work: WorkContext, pr_number: int, bot: str,
) -> None:
    """Fetch PR feedback by invoking the fetch-pr-feedback skill.

    Args:
        backend: The Backend to execute against.
        work: Workspace context; ``work.repo`` is the agent cwd.
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

    await run_agent(backend, work.repo, skill_invocation, phase=DaydreamPhase.PR_FEEDBACK)

    output_path = work.repo / REVIEW_OUTPUT_FILE
    if output_path.exists():
        print_success(console, f"PR feedback written to: {output_path}")
    else:
        print_warning(console, "PR feedback file was not created")


async def phase_fix_parallel(
    backend: Backend, work: WorkContext, feedback_items: list[dict[str, Any]]
) -> list[FixResult]:
    """Apply fixes for all feedback items concurrently using parallel agents.

    Launches one agent per feedback item in a task group. Each agent runs
    independently; individual failures are captured without aborting others.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for the fixes; ``work.repo`` is the agent cwd.
        feedback_items: List of feedback items, each with description, file, line

    Returns:
        List of (item, success, error) tuples for each feedback item

    """
    recorder = get_current_recorder()
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

Make the minimal change needed. Do NOT change error handling semantics
(e.g., converting warn-and-continue to error propagation, or vice versa)
unless the issue description specifically explains why the current error
handling strategy is wrong for that code path.
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

                async with maybe_fork(recorder, f"fix-{task_index}"):
                    try:
                        async with limiter:
                            await run_agent(
                                backend, work.repo, task_prompt, progress_callback=callback, phase=DaydreamPhase.FIX,
                            )
                        panel.complete_row(task_index)
                        results.append((task_item, True, None))
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {e}"
                        panel.fail_row(task_index, error_msg)
                        results.append((task_item, False, error_msg))

            tg.start_soon(_fix_task)

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)

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


async def phase_commit_iteration(backend: Backend, work: WorkContext, iteration: int) -> None:
    """Commit all changes from the current loop iteration.

    Ensures a clean working tree before the next review iteration starts.
    Does NOT push — the final push happens at the end of the loop.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for the commit; ``work.repo`` is the cwd.
        iteration: Current iteration number (used in commit message)

    """
    print_info(console, f"Committing iteration {iteration} changes...")
    await _do_commit(backend, work, iteration=iteration)
    print_success(console, f"Iteration {iteration} changes committed")


async def phase_commit_push_auto(
    backend: Backend, work: WorkContext, *, items: list[dict[str, Any]] | None = None,
) -> None:
    """Automatically commit and push changes without user prompt.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for the commit.
        items: Optional fix items applied this run; forwarded to the commit
            agent so it can craft an accurate commit message.

    """
    console.print()
    print_info(console, "Committing and pushing changes...")
    await _do_commit(backend, work, push=True, items=items)
    print_success(console, "Commit and push complete")


async def phase_respond_pr_feedback(
    backend: Backend, work: WorkContext, pr_number: int, bot: str, results: list[FixResult]
) -> None:
    """Respond to PR feedback with results of applied fixes.

    Filters to successful results only and invokes the respond-pr-feedback
    skill to post replies on the pull request.

    Args:
        backend: The Backend to execute against.
        work: Workspace context; ``work.repo`` is the agent cwd.
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

    await run_agent(backend, work.repo, skill_invocation, phase=DaydreamPhase.PR_FEEDBACK)
    print_success(console, f"Responded to PR #{pr_number} feedback")


async def phase_understand_intent(
    backend: Backend,
    work: WorkContext,
    diff_path: Path,
    log: str,
    branch: str,
    *,
    exploration_dir: Path | None = None,
) -> str:
    """Phase: Understand the intent of the PR through conversational confirmation.

    The agent examines the diff, commit log, and branch name to understand
    what the PR is trying to accomplish. The user confirms or corrects until
    the understanding is accurate.

    Args:
        backend: The Backend to execute against.
        work: Workspace context; ``work.repo`` is the agent cwd.
        diff_path: Path to the diff file on disk.
        log: Git log output (main..HEAD --oneline).
        branch: Current branch name.

    Returns:
        The confirmed intent summary string.

    """
    print_phase_hero(console, "LISTEN", phase_subtitle("LISTEN"))

    prompt = build_intent_prompt(
        diff_path=str(diff_path),
        branch=branch,
        log=log,
        exploration_dir=exploration_dir,
    )

    while True:
        console.print()
        print_info(console, "Agent is analyzing the changes...")

        output, _ = await run_agent(backend, work.repo, prompt, phase=DaydreamPhase.INTENT)
        intent_text = output if isinstance(output, str) else str(output)

        console.print()
        response = prompt_user(
            console,
            "Is this understanding correct? [y/provide correction]",
            "y",
        )

        if response.lower() in ("y", "yes"):
            return intent_text

        # User provided a correction — build new prompt with context
        prompt = f"""You previously described the intent of these changes as:

{intent_text}

The user corrected your understanding: {response}

Re-examine the codebase and the diff at {diff_path}, and present an updated understanding of the intent.

Branch: {branch}

Commit log:
{log}
"""


async def phase_alternative_review(
    backend: Backend,
    work: WorkContext,
    diff_path: Path,
    intent_summary: str,
    *,
    exploration_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Phase: Evaluate whether there's a better way to implement the PR.

    A fresh agent receives the confirmed intent summary and explores the
    codebase to identify issues — both architectural alternatives and
    incremental improvements.

    Args:
        backend: The Backend to execute against.
        work: Workspace context; ``work.repo`` is the agent cwd.
        diff_path: Path to the diff file on disk.
        intent_summary: Confirmed intent summary from phase_understand_intent.

    Returns:
        List of issue dicts, each with id, title, description, recommendation,
        severity, and files keys.

    """
    print_phase_hero(console, "WONDER", phase_subtitle("WONDER"))

    prompt = build_alternative_review_prompt(
        intent_summary=intent_summary,
        diff_path=str(diff_path),
        exploration_dir=exploration_dir,
    )

    console.print()
    print_info(console, "Agent is evaluating the implementation...")

    result, _ = await run_agent(
        backend, work.repo, prompt, output_schema=ALTERNATIVE_REVIEW_SCHEMA, phase=DaydreamPhase.ALTERNATIVES,
    )

    if isinstance(result, dict) and "issues" in result:
        issues = result["issues"]
    else:
        if not get_quiet_mode():
            print_warning(console, f"TTT review returned unexpected result type: {type(result).__name__}")
        issues = []

    if issues:
        print_info(console, f"Found {len(issues)} issues")
        print_issues_table(console, issues)
    else:
        print_info(console, "No issues found — the implementation looks good")

    return issues


def _write_plan_markdown(
    plan_path: Path,
    plan_data: dict[str, Any],
    intent_summary: str,
    branch: str,
    original_issues: list[dict[str, Any]],
) -> None:
    """Write the plan data as a markdown file.

    Args:
        plan_path: Path to write the markdown file.
        plan_data: Structured plan output from the agent.
        intent_summary: Confirmed intent summary.
        branch: Current branch name.
        original_issues: Full issue list (for severity/recommendation metadata).

    """
    issue_map = {i["id"]: i for i in original_issues}
    plan = plan_data.get("plan", plan_data)  # Handle both wrapped and unwrapped

    lines = [
        "# Implementation Plan",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Branch:** {branch}",
        "",
        "## Intent",
        intent_summary,
        "",
        "## Plan Summary",
        plan.get("summary", "No summary provided."),
        "",
    ]

    for plan_issue in plan.get("issues", []):
        issue_id = plan_issue.get("id", "?")
        title = plan_issue.get("title", "Untitled")
        original = issue_map.get(issue_id, {})

        lines.append(f"## Issue {issue_id}: {title}")
        lines.append(f"**Severity:** {original.get('severity', 'unknown')}")
        if original.get("description"):
            lines.append(f"**Problem:** {original['description']}")
        if original.get("recommendation"):
            lines.append(f"**Recommendation:** {original['recommendation']}")
        lines.append("")

        changes = plan_issue.get("changes", [])
        if changes:
            lines.append("### Changes")
            for change in changes:
                action = change.get("action", "modify")
                file_path = change.get("file", "unknown")
                desc = change.get("description", "")
                lines.append(f"- **{action}** `{file_path}` — {desc}")
            lines.append("")

    plan_path.write_text("\n".join(lines))


async def phase_generate_plan(
    backend: Backend,
    work: WorkContext,
    diff_path: Path,
    intent_summary: str,
    issues: list[dict[str, Any]],
    *,
    exploration_dir: Path | None = None,
    auto_select_all: bool = False,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Phase: Generate an implementation plan for selected issues.

    Prompts the user to select which issues to address, then launches an
    agent to create a detailed plan. Writes the plan as markdown to
    .daydream/plan-{timestamp}.md.

    Args:
        backend: The Backend to execute against.
        work: Workspace context (plan written under ``work.repo``).
        diff_path: Path to the diff file on disk.
        intent_summary: Confirmed intent summary.
        issues: Full list of issues from phase_alternative_review.
        auto_select_all: Skip the interactive prompt and select all issues.

    Returns:
        Tuple of (path to plan file, raw plan dict). Either or both may be None.

    """
    print_phase_hero(console, "ENVISION", phase_subtitle("ENVISION"))

    if auto_select_all:
        selected_ids: list[int] = [i["id"] for i in issues]
    else:
        console.print()
        response = prompt_user(
            console,
            "Create an implementation plan? Enter issue numbers (e.g., 1,3,5) or 'all', or 'none' to skip",
            "all",
        )

        parsed = _parse_issue_selection(response, issues)
        if parsed is None:
            print_dim(console, "Skipping plan generation")
            return None, None
        if not parsed:
            print_warning(console, "No valid issue numbers found in selection")
            return None, None
        selected_ids = parsed

    selected_issues = [i for i in issues if i["id"] in selected_ids]
    issues_text = "\n".join(
        f"- #{i['id']} [{i.get('severity', '?')}] {i.get('title', 'No title')}: "
        f"{i.get('description', '')} → {i.get('recommendation', '')}"
        for i in selected_issues
    )

    prompt = build_plan_prompt(
        intent_summary=intent_summary,
        issues_text=issues_text,
        diff_path=str(diff_path),
        exploration_dir=exploration_dir,
    )

    console.print()
    print_info(console, f"Generating plan for {len(selected_issues)} issue(s)...")

    result, _ = await run_agent(backend, work.repo, prompt, output_schema=PLAN_SCHEMA, phase=DaydreamPhase.PLAN)

    if not isinstance(result, dict):
        if not get_quiet_mode():
            print_warning(
                console,
                f"Failed to generate structured plan; agent returned {type(result).__name__}",
            )
        return None, None

    # Ensure .daydream/ directory exists
    daydream_dir = work.repo / ".daydream"
    daydream_dir.mkdir(exist_ok=True)

    # Write plan file
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    plan_path = daydream_dir / f"plan-{timestamp}.md"

    branch_name = work.head_branch or _git_branch(work.repo)
    _write_plan_markdown(plan_path, result, intent_summary, branch_name, selected_issues)

    print_success(console, f"Plan written to {plan_path}")
    return plan_path, result


# -------------------------
# Deep-mode: per-stack fan-out
# -------------------------


async def phase_per_stack_reviews(
    backend: Backend,
    work: WorkContext,
    stacks: list["StackAssignment"],
    *,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    exploration_dir: Path | None = None,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Run one review agent per detected stack concurrently (D-17).

    Mirrors phase_fix_parallel's capacity-limiter + task-group + default-arg closure
    capture pattern. Per D-38, uses orchestrator-level parallelism -- never passes
    the ``agents`` kwarg (Codex does not support SDK-level sub-agent spawning).

    Args:
        backend: The Backend to execute against.
        work: Workspace context; ``work.repo`` is the agent cwd / repo root.
        stacks: Routed stack assignments (see daydream.deep.detection.detect_stacks).
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        exploration_dir: Optional pre-scan exploration directory.

    Returns:
        Tuple of ``(successes, failures)``:
          - ``successes``: stack_name -> per-stack review output Path for stacks
            that produced a review.
          - ``failures``: stack_name -> "<ExceptionType>: <message>" for stacks
            whose agent raised. Callers MUST surface this to the user and to the
            merge agent so that missing coverage is visible instead of silently
            dropped.

    """
    from daydream.deep.artifacts import deep_dir as _deep_dir
    from daydream.deep.artifacts import per_stack_review_path
    from daydream.deep.prompts import (
        build_generic_fallback_prompt,
        build_per_stack_prompt,
    )

    deep_dir_path = _deep_dir(work.repo)
    recorder = get_current_recorder()
    results: dict[str, Path] = {}
    failures: dict[str, str] = {}
    limiter = anyio.CapacityLimiter(4)
    prior_commits = _prior_daydream_commits(work)

    async with anyio.create_task_group() as tg:
        for stack in stacks:
            output_path = per_stack_review_path(deep_dir_path, stack.stack_name)
            if stack.skill_invocation is None:
                prompt = build_generic_fallback_prompt(
                    files=stack.files,
                    diff_path=diff_path,
                    intent_path=intent_path,
                    alternatives_path=alternatives_path,
                    output_path=output_path,
                    exploration_dir=exploration_dir,
                    is_docs_only=stack.is_docs_only,
                    prior_commits=prior_commits,
                )
            else:
                prompt = build_per_stack_prompt(
                    skill_invocation=stack.skill_invocation,
                    stack_name=stack.stack_name,
                    files=stack.files,
                    diff_path=diff_path,
                    intent_path=intent_path,
                    alternatives_path=alternatives_path,
                    output_path=output_path,
                    exploration_dir=exploration_dir,
                    prior_commits=prior_commits,
                )

            # Default-arg capture -- prevents late-binding closure bug (Pitfall 2).
            async def _task(
                stack_name: str = stack.stack_name,
                task_prompt: str = prompt,
                task_output: Path = output_path,
            ) -> None:
                async with maybe_fork(recorder, f"deep-{stack_name}"):
                    try:
                        async with limiter:
                            await run_agent(backend, work.repo, task_prompt, phase=DaydreamPhase.DEEP)
                        results[stack_name] = task_output
                    except Exception as e:  # noqa: BLE001 -- intentionally broad for parallel isolation
                        reason = f"{type(e).__name__}: {e}"
                        failures[stack_name] = reason
                        print_warning(
                            console,
                            f"Per-stack review for '{stack_name}' failed ({reason}); "
                            "merge report will note this stack as uncovered.",
                        )

            tg.start_soon(_task)

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.DEEP)

    return results, failures


async def phase_cross_stack_merge(
    backend: Backend,
    work: WorkContext,
    *,
    per_stack_records_paths: list[Path],
    intent_path: Path,
    alternatives_path: Path,
    dedup_candidates_path: Path,
    exploration_dir: Path | None = None,
    failed_stacks: dict[str, str] | None = None,
) -> Path:
    """Run the cross-stack merge agent and return the output-report path (D-23..D-27).

    The agent writes the report to ``.daydream/deep/review-output.md`` (same
    directory as per-stack artifacts, which avoids sandbox write restrictions
    that block dotfiles at the repo root). This function then copies the result
    to ``work.repo / REVIEW_OUTPUT_FILE`` for downstream consumers.

    Per D-38, never passes the ``agents`` kwarg (Codex parity).

    Args:
        backend: The Backend to execute against.
        work: Workspace context; report is written under ``work.repo``.
        per_stack_records_paths: Parsed per-stack record JSON paths (D-22 inputs).
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        dedup_candidates_path: Path to dedup-candidates.json (D-27 pre-filter output).
        exploration_dir: Optional pre-scan exploration directory.
        failed_stacks: Optional stack_name -> reason dict for per-stack agents
            that failed. Passed through to the merge prompt so the merged
            report can call out uncovered stacks explicitly.

    Returns:
        Path to the merged report at ``work.repo / REVIEW_OUTPUT_FILE``.

    """
    from daydream.deep.artifacts import deep_dir, merged_report_path
    from daydream.deep.prompts import build_merge_prompt

    canonical_path = work.repo / REVIEW_OUTPUT_FILE
    agent_output_path = merged_report_path(deep_dir(work.repo))

    # Clear stale outputs so a failed merge agent can't leave behind
    # outdated content that downstream stages would silently consume.
    canonical_path.unlink(missing_ok=True)
    agent_output_path.unlink(missing_ok=True)

    prompt = build_merge_prompt(
        per_stack_records_paths=per_stack_records_paths,
        intent_path=intent_path,
        alternatives_path=alternatives_path,
        dedup_candidates_path=dedup_candidates_path,
        output_path=agent_output_path,
        exploration_dir=exploration_dir,
        failed_stacks=failed_stacks,
    )
    print_phase_hero(console, "MERGE", phase_subtitle("MERGE"))
    await run_agent(backend, work.repo, prompt, phase=DaydreamPhase.DEEP)

    # Copy from deep artifact dir to canonical location. The agent writes
    # inside .daydream/deep/ where sandbox restrictions don't apply; Python
    # handles the copy to work.repo/.review-output.md.
    if not agent_output_path.is_file():
        raise FileNotFoundError(f"Expected merged report at {agent_output_path}")
    canonical_path.write_text(agent_output_path.read_text())
    return canonical_path
