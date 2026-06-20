"""Phase functions for the review and fix loop."""

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from rich.text import Text

import daydream
from daydream import git_ops
from daydream.agent import (
    console,
    detect_test_success,
    get_assume,
    get_non_interactive,
    get_quiet_mode,
    resolve_gate,
    resolve_or_prompt,
    run_agent,
)
from daydream.backends import Backend, ContinuationToken
from daydream.backends.claude import READ_ONLY_BASH_ALLOWLIST
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
    phase_subtitle,
    print_dim,
    print_error,
    print_feedback_table,
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

# Generous for a real fix yet bounds a flailing agent that's globbing $HOME after a missed Read.
FIX_MAX_TURNS = 25
_PR_BODY_MAX_CHARS = 8000


def _build_fix_prompt(
    test_output: str,
    feedback_items: list[dict[str, Any]] | None = None,
    *,
    repo: Path | None = None,
) -> str:
    """Build an enriched prompt for the fix agent with test output and file context.

    Args:
        test_output: Raw test output text.
        feedback_items: Optional list of feedback items with 'file' keys.
        repo: Optional repo root; listed files are mapped to absolute paths when
            they exist under it, so the fix agent's first Read hits.

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
        if repo is not None:
            files = [str(repo / f) if (repo / f).is_file() else f for f in files]
        if files:
            file_list = "\n".join(f"- {f}" for f in files)
            parts.append(f"\nFiles modified during the fix phase:\n{file_list}")

    parts.append("\nAnalyze the failures and fix them.")
    if feedback_items:
        parts.append("Focus on the files listed above.")
        parts.append(
            "Start with the files listed above; if a correct fix needs "
            "another file, edit it and say which and why."
        )

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


def _sanitize_suggested_command(raw: str) -> str:
    """Sanitize an LLM-suggested shell command for safe inclusion in a fenced prompt.

    Collapses all whitespace runs to a single space and strips backticks.
    Newlines and backticks are the two characters that can break out of a
    triple-backtick code fence in a downstream prompt; nothing in a legitimate
    test command requires either. The result is suitable for both the
    next-turn ``run_agent`` prompt and for surfacing in the user-facing
    confirmation preview.
    """
    return " ".join(raw.replace("`", "").split())


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
    to ``handoff.md``. The summarizer is strictly read-only and non-mutating
    but MAY use read-only git history commands (``git log``/``show``/``blame``/
    ``diff``) to *verify* any cause/history/blame claim before stating it as
    fact. The handoff separates **Verified facts** (each cited) from
    **Hypotheses (unverified)**, and quotes two tightly-scoped excerpts — the
    failing assertion and the current source at the failing location — while
    full diffs / whole files / trajectory dumps stay banned.

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
        "session, to propose a fix for a failure daydream's test/heal loop "
        "could not clear.\n\n"
        "## Hard Constraints (read-only contract)\n"
        "- You MAY use Read, Grep, and Glob to inspect the artifacts listed below.\n"
        "- You MAY use Bash for NON-MUTATING inspection ONLY. Permitted commands: "
        + ", ".join(f"`{cmd}`" for cmd in READ_ONLY_BASH_ALLOWLIST)
        + ". "
        "These are read-only — they inspect history and never change the repo. Use "
        "them to VERIFY any claim about cause or history before you write it as fact. "
        "Each command MUST be a single, bare invocation — no pipes (`|`), no chaining "
        "(`&&`, `||`, `;`), no subshells (`` ` `` or `$(...)`). The guard hook will "
        "silently deny any command that contains these metacharacters.\n"
        "- You MUST NOT run tests, builds, or installers.\n"
        "- You MUST NOT run any command that writes, stages, commits, checks out, "
        "resets, stashes, or pushes (no `git add/commit/checkout/restore/reset/"
        "stash/push`). You MUST NOT invoke Write, Edit, or any file-mutating tool.\n"
        "- Do NOT embed full diffs, whole files, or trajectory dumps. You MAY — and "
        "in \"Verified facts\" you MUST — quote two tightly-scoped excerpts: (a) the "
        "exact failing assertion / error line(s), and (b) the current source at the "
        "failing location (≤ ~15 lines, cited `path:start-end`). Everything else is "
        "a reference by absolute path.\n\n"
        "## Evidence rule (MANDATORY)\n"
        "Every statement about CAUSE (\"X failed because…\"), HISTORY (\"the run "
        "changed…\", \"this was introduced by…\"), or BLAME (\"daydream appended…\") is "
        "a claim you MUST prove before writing it as fact. To prove a claim: run the "
        "relevant read-only command (`git log -n 10 --oneline`, `git show <sha> -- "
        "<file>`, `git blame -L <line>,<line> -- <file>`, `git diff <base>..HEAD -- "
        "<file>`) or read the named artifact, THEN cite it inline next to the claim, "
        "e.g. `(git blame phases.py:520 → 648a327, an earlier commit, NOT this run)`. "
        "If you cannot prove a claim, it is NOT a verified fact — it goes in "
        "Hypotheses. NEVER attribute a code change to \"the daydream run\" unless "
        "`git log` / `git blame` shows a commit created during this run. A line that "
        "predates the run's first commit was NOT written by the run — say so, with "
        "the blame citation.\n\n"
        "## On-disk artifacts (read these first to ground your summary)\n"
        f"{artifacts_block}\n\n"
        "## Files changed during this daydream run\n"
        f"{changed_block}\n\n"
        "## Failing test output (for your context — quote only the specific failing "
        f"assertion / error line(s) into Verified facts, not the whole tail)\n\n{output_section}\n\n"
        f"{no_trajectory_clause}"
        "## Handoff prompt template\n"
        "Produce a Markdown document with these sections, in this order:\n\n"
        "1. **Summary** — ONE neutral paragraph describing only the directly observed "
        "outcome: the tests did not pass and the heal loop aborted at the test gate. "
        "NO causal claims, NO history, NO blame here — those belong below.\n"
        "2. **Verified facts** — a bulleted list. EVERY item ends with a citation in "
        "parentheses: a command you ran or an artifact path+lines. This section MUST "
        "include, at minimum:\n"
        "   - The exact failing assertion / error line(s), quoted. (cite: test output)\n"
        "   - The current source at the failing location, quoted ≤ 15 lines. "
        "(cite: `path:start-end`, read just now)\n"
        "   - The exit gate that fired (test phase, non-interactive abort or option 4).\n"
        "   - Any history you confirmed with git (what changed in THIS run vs. earlier "
        "commits), each with its `git log`/`blame`/`show`/`diff` citation.\n"
        "   This section must be EVIDENCE-RICH. You are required to actually run the "
        "git/read commands — do not dump everything into Hypotheses to avoid the work.\n"
        "3. **Hypotheses (unverified)** — a bulleted list of candidate causes or "
        "explanations you could NOT prove. Mark each explicitly, e.g. \"UNVERIFIED — "
        "confirm by: <command/check>\". Anything about WHY it failed or WHO introduced "
        "code that git did not confirm goes HERE, never in Verified facts.\n"
        "4. **Artifacts** — bulleted absolute paths from the sections above. No contents.\n"
        "5. **Changed files** — bulleted absolute repo paths from the section above.\n"
        "6. **Instructions for the next agent** — explicit, numbered:\n"
        "   1. Explore the codebase before proposing anything. Treat \"Verified "
        "facts\" as ground truth (citations included); treat \"Hypotheses\" as leads "
        "to confirm or refute FIRST — and do NOT revert or rewrite code on a "
        "Hypothesis alone.\n"
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

    archive_enabled = recorder.on_write is not None
    if work.is_ephemeral and archive_enabled:
        # The ephemeral worktree (and everything under it) will be
        # removed after the recorder exits; the archive callback copies
        # the bundle to <archive_root>/runs/<session_id>/. Write the
        # handoff alongside the archived artifacts so the bundle is
        # self-contained and post-cleanup references stay valid.
        from daydream.archive import get_archive_dir

        artifact_root = get_archive_dir() / "runs" / recorder.session_id
        diff_path = artifact_root / "diff.patch"
        deep_dir = artifact_root / "deep"
    else:
        artifact_root = recorder.target_dir / ".daydream" / "runs" / recorder.session_id
        diff_path = recorder.target_dir / ".daydream" / "diff.patch"
        deep_dir = recorder.target_dir / ".daydream" / "deep"

    handoff_path = artifact_root / "handoff.md"

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


def _write_handoff(path: Path, body: str) -> bool:
    """Write *body* to *path*, creating parent directories as needed.

    Returns ``True`` on success, ``False`` on ``OSError`` (filesystem
    full, permission denied, parent directory cleaned up mid-run, etc.).
    The caller is responsible for surfacing the body inline when this
    returns False so the user does not lose the handoff content.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except OSError:
        _logger.warning("failed to write handoff to %s", path, exc_info=True)
        return False
    return True


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
    active. Mirrors the agent path's **Verified facts** / **Hypotheses
    (unverified)** split so accuracy does not regress on the fallback —
    but it runs with no agent and cannot execute git, so it produces no
    citations and **invents no cause**: the Hypotheses section states the
    cause is UNKNOWN and must be derived by the next agent. The facts it
    *can* assert without an agent (the exit gate, the quoted failing
    output, the git-derived changed-file list) go under Verified facts.
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

    changed_count = len(changed_files)
    changed_fact = (
        f"Files changed in the working tree (git): {changed_count} file(s) — "
        "listed under Changed files below. (cite: git, working-tree status)"
        if changed_count
        else "No working-tree file changes were detected. (cite: git, working-tree status)"
    )

    parts: list[str] = [
        "# Daydream handoff",
        "",
        "## Summary",
        "",
        "Daydream's test phase did not confirm a green run and the heal loop "
        "aborted at the test gate. The failure-summarizer subagent did not produce "
        "a structured handoff, so this minimal version was written instead.",
        "",
    ]
    if not has_trajectory:
        parts.append("> Note: trajectory unavailable for this run")
        parts.append("")
    parts.extend([
        "## Verified facts",
        "",
        "- Tests did not report success; the heal loop aborted at the test gate. "
        "(cite: daydream exit path — non-interactive abort / option 4)",
        f"- {changed_fact}",
        "- Tail of the failing test output, quoted verbatim:",
        "",
        "```",
        output_section,
        "```",
        "",
        "## Hypotheses (unverified)",
        "",
        "- The failure-summarizer agent did not run, so NO causal or historical "
        "analysis was performed: the **cause is UNKNOWN** and must be derived by the "
        "next agent via git (`git log`/`blame`/`show`/`diff`) and Read. Do not assume "
        "a cause — confirm one from evidence first.",
        "",
        "## Artifacts",
        "",
        artifacts_block,
        "",
        "## Changed files",
        "",
        changed_block,
        "",
        "## Instructions for the next agent",
        "",
        "1. Explore the codebase before proposing anything. Treat \"Verified facts\" "
        "as ground truth and \"Hypotheses\" as leads to confirm or refute FIRST; "
        "use Read/Grep/Glob and read-only git to build your own model; do NOT revert "
        "or rewrite code on a hypothesis alone.",
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
) -> tuple[str, Path, bool]:
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
        Tuple ``(handoff_body, handoff_path, written)``. ``written`` is
        ``False`` when the filesystem write failed; callers must surface
        the body inline in that case so the user does not lose it. The
        body is what was written to disk on success.
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
                read_only=True,
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

    written = _write_handoff(handoff_path, body)
    return body, handoff_path, written


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

# Per-stack parse schema (issue #168). Identical to FEEDBACK_SCHEMA but carries a
# required ``severity`` so the scoped Opus arbiter can select high-severity /
# contested findings *before* the merge. The shared FEEDBACK_SCHEMA stays
# severity-free (the shallow loop and PR-feedback parse paths never need it);
# only deep-mode's pre-merge per-stack parse opts into this richer record shape.
#
# Derived from FEEDBACK_SCHEMA to avoid silent drift: we deep-copy the base
# schema and inject the extra ``severity`` field into the items sub-schema.
PER_STACK_RECORD_SCHEMA: dict[str, Any] = copy.deepcopy(FEEDBACK_SCHEMA)
PER_STACK_RECORD_SCHEMA["properties"]["issues"]["items"]["properties"]["severity"] = {
    "type": "string",
    "enum": ["high", "medium", "low"],
}
PER_STACK_RECORD_SCHEMA["properties"]["issues"]["items"]["required"] = [
    "id", "description", "file", "line", "severity", "confidence", "rationale"
]

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

MERGED_ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
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
                    "lens": {"type": "string", "enum": ["per-stack", "cross-stack", "structural"]},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": [
                    "id",
                    "description",
                    "file",
                    "line",
                    "confidence",
                    "rationale",
                    "lens",
                    "severity",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}


def normalize_items(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign fresh contiguous unique integer ids to every item.

    Reassigns each item's ``id`` to a contiguous 1-based sequence regardless of
    incoming numbering, so per-stack, cross-stack, and structural items that
    collide on their original ids end up uniquely keyed. Order and the ``lens``
    field are preserved.

    Args:
        raw: The incoming list of item dicts.

    Returns:
        A new list of item dicts with reassigned ``id`` values.

    Raises:
        ValueError: If ``raw`` is not a list.
    """
    if not isinstance(raw, list):
        raise ValueError(f"normalize_items expected a list, got {type(raw).__name__}")
    normalized: list[dict[str, Any]] = []
    for new_id, item in enumerate(raw, start=1):
        normalized.append({**item, "id": new_id})
    return normalized


# Canonical severity ordering shared by the deep fix loop and the shallow fix
# loop. Defined here (next to normalize_items / MERGED_ITEMS_SCHEMA) so both
# callers can import a single helper rather than duplicate the map.
_SEVERITY_RANK: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


def severity_sorted(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable-sort canonical items by severity (high < medium < low)."""
    return sorted(items, key=lambda it: _SEVERITY_RANK.get(it.get("severity") or "", 1))


def group_items_by_file(items: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Partition fix items into per-file groups, preserving input order.

    Shared by the parallel fix loop: distinct files become distinct groups that
    can run concurrently, while items targeting the same file stay together so
    they run serially (no read-modify-write races). Group emission order is the
    first-appearance order of each file; within-group order is input order, so a
    ``severity_sorted`` input yields severity-ordered groups. Items with a
    missing/None file bucket into a single ``"<no-file>"`` group (cannot prove
    disjoint -> serialize for safety). Pure: no I/O, no mutation of inputs.

    Args:
        items: Canonical fix items (each a dict with at least an optional
            ``"file"`` key).

    Returns:
        Ordered list of ``(file_key, items_for_file)`` tuples, where
        *file_key* is the file path string or ``"<no-file>"`` for items
        lacking a file, and *items_for_file* preserves the input order of
        items assigned to that file.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = item.get("file") or "<no-file>"
        grouped.setdefault(key, []).append(item)
    return list(grouped.items())


RECOMMENDATION_VERDICTS_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "integer"},
                    "verdict": {"type": "string",
                                "enum": ["consistent", "contradicts", "uncertain"]},
                    "evidence": {"type": "string"},
                    "unverified_assumptions": {"type": "array",
                                               "items": {"type": "string"}},
                },
                "required": ["issue_id", "verdict", "evidence",
                             "unverified_assumptions"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["verdicts"],
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
    pr_description: str | None = None,
) -> str:
    """Assemble the prompt for `phase_understand_intent`.

    Args:
        diff_path: Path to the diff file the agent should read.
        branch: Branch name under review.
        log: Commit log for the branch.
        exploration_dir: Optional pre-scan exploration directory pointer.
        pr_description: Optional author-supplied pull-request description. When
            present (non-empty after strip), an authoritative-intent section is
            prepended ahead of the diff-reading instructions. When ``None`` or
            empty, the prompt is byte-identical to the no-PR-body case.

    Returns:
        Fully assembled prompt string.

    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    if pr_description and pr_description.strip():
        body_text = pr_description.strip()
        if len(body_text) > _PR_BODY_MAX_CHARS:
            body_text = body_text[:_PR_BODY_MAX_CHARS] + "\n[PR description truncated]"
        parts.append(
            "The author supplied the following pull-request description. Treat this "
            "author-stated intent as AUTHORITATIVE: where the description and the "
            "intent you would infer from the diff conflict, the description outranks "
            "the diff. Crucially, when the description says something is deliberate but "
            "the diff appears to contradict it — a near-1.0 ratio that looks inert, a "
            "guard that looks like a no-op, a pass-through that looks unfinished — that "
            "is a deliberate design decision to preserve, NOT a defect to surface or "
            "'complete'.\n\n"
            "Pull request description:\n"
            "<pr_description>\n"
            f"{body_text.replace('</pr_description>', '<\\/pr_description>')}\n"
            "</pr_description>\n"
        )
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
    print_dim(console, f"Model: {backend.model}")

    # Use absolute path to prevent model hallucination of paths from training data
    review_output_path = work.repo / REVIEW_OUTPUT_FILE
    skill_invocation = backend.format_skill_invocation(skill)

    if diff_base:
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
    output_schema: dict[str, Any] | None = None,
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
        output_schema: Optional structured-output schema. Defaults to
            ``FEEDBACK_SCHEMA``. Deep-mode's pre-merge per-stack parse passes
            ``PER_STACK_RECORD_SCHEMA`` so each record carries ``severity`` for
            the scoped Opus arbiter (issue #168). When the schema requires a
            ``severity`` field, the prompt instructs the agent to extract it.

    Returns:
        List of validated feedback items with id, description, file, line

    Raises:
        ValueError: If the agent output is not a valid list.

    """
    print_phase_hero(console, "REFLECT", phase_subtitle("REFLECT"))
    print_dim(console, f"Model: {backend.model}")

    schema = output_schema if output_schema is not None else FEEDBACK_SCHEMA
    wants_severity = "severity" in (
        schema.get("properties", {})
        .get("issues", {})
        .get("items", {})
        .get("properties", {})
    )
    severity_field = ', "severity": "high|medium|low"' if wants_severity else ""
    severity_hint = (
        "\nAlso set a `severity` of high | medium | low for each issue, taken from the "
        "review's own severity/priority label. Default to high when the review gives "
        "no explicit severity.\n"
        if wants_severity
        else ""
    )

    # Use absolute path to prevent model hallucination of paths from training data
    review_output_path = input_path if input_path is not None else work.repo / REVIEW_OUTPUT_FILE
    prompt = f"""Read the review output file at {review_output_path}.

Extract ONLY actionable issues that need fixing. Skip these sections entirely:
- "Good Patterns" or "Strengths"
- "Summary" sections
- Any positive observations
{severity_hint}
For each issue found, return a JSON object with this structure:
{{"issues": [
  {{"id": 1, "description": "Brief description of the issue", "file": "path/to/file.py", "line": 42{severity_field}}}
]}}

If there are no actionable issues, return: {{"issues": []}}
"""

    result, _ = await run_agent(backend, work.repo, prompt, output_schema=schema, phase=DaydreamPhase.PARSE)

    if not isinstance(result, dict) or "issues" not in result:
        # When structured output and JSON fallback both fail (e.g. empty
        # response), treat as "no issues" rather than crashing.
        if isinstance(result, str) and not result.strip():
            print_warning(console, "Agent returned empty response; treating as no actionable issues")
            return []
        raise ValueError(f"Expected dict with 'issues' key, got {type(result)}")

    feedback_items = result["issues"]
    issue_count = len(feedback_items)
    print_info(console, f"Found {issue_count} actionable {'issue' if issue_count == 1 else 'issues'}")
    if feedback_items:
        print_feedback_table(console, feedback_items)
    return feedback_items


def _coerce_verdicts_payload(value: Any) -> dict[str, Any]:
    """Normalize a raw verifier result into a ``{"verdicts": [dict, ...]}`` payload.

    Accepts any candidate (dict, JSON-parsed value, or other) and returns a
    payload whose ``verdicts`` key is guaranteed to be a list of dicts.
    Non-dict entries inside the list are dropped rather than rejecting the
    whole payload so partial agent output is still usable downstream.
    """
    if not isinstance(value, dict):
        return {"verdicts": []}
    raw = value.get("verdicts")
    if not isinstance(raw, list):
        return {"verdicts": []}
    return {"verdicts": [entry for entry in raw if isinstance(entry, dict)]}


async def phase_verify_recommendations(
    backend: Backend,
    work: WorkContext,
    *,
    merged_items_path: Path,
    deep_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Audit each non-structural item's recommendation against the codebase.

    Loads the canonical merged item list (the single source of truth produced
    by the cross-stack merge), filters to the language lenses
    (``per-stack`` / ``cross-stack``), and runs a read-only verifier subagent
    that decides, for every such item, whether the recommendation is
    ``consistent`` with trait/interface specs and sibling implementations,
    ``contradicts`` them, or is ``uncertain`` from the codebase alone.
    Writes the result as ``recommendation-verdicts.json`` inside
    ``deep_dir``. The fix gate reads the file and inlines per-item verdicts
    into the ``phase_fix`` prompt; verdicts are advisory and keyed by the
    canonical ``issue_id``.

    Mirrors ``_run_setup_investigator`` in shape: small read-only contract
    encoded in the verifier prompt, compact JSON schema, single ``run_agent``
    call. Trajectory observability is handled by ``run_agent``'s recorder
    integration via ``phase=DaydreamPhase.VERIFY``; no explicit ``fork`` at
    this layer.

    Args:
        backend: The Backend to execute against.
        work: Workspace context; ``work.repo`` is the verifier's cwd and the
            repository root passed into the prompt.
        merged_items_path: Path on disk to the canonical ``merged-items.json``.
            Items tagged ``lens="structural"`` are filtered out in Python
            before the prompt is built (see filter site below); only the
            language-lens items are rendered for the verifier.
        deep_dir: The ``.daydream/deep/`` artifacts directory for this run.
            The verdicts JSON is written inside it via ``verdicts_path``.

    Returns:
        Tuple of (path, payload) where path is the verdicts JSON file written
        inside ``deep_dir`` and payload is the already-parsed dict. The file
        always exists on successful return; on parse failure, missing agent
        output, or an empty filtered item list, ``{"verdicts": []}`` is
        written so downstream code does not need to handle a missing file.

    """
    # Late imports avoid circular dependency with daydream.deep (which imports
    # from daydream.phases). Same pattern used by phase_per_stack_reviews and
    # phase_cross_stack_merge above.
    from daydream.deep.artifacts import verdicts_path
    from daydream.deep.prompts import build_verification_prompt

    output_path = verdicts_path(deep_dir)

    items: list[dict[str, Any]] = json.loads(merged_items_path.read_text()).get("items", [])
    # DELIBERATE: structural items get no verifier verdict (plan Assumption 2 --
    # structural is validated at review time by review-structure's G3 evidence
    # gates and protected at fix time by the contract-wins guard, so the
    # interface-conformance verifier does not apply to it).
    verifiable = [i for i in items if i.get("lens") in ("per-stack", "cross-stack")]

    if not verifiable:
        empty_payload: dict[str, Any] = {"verdicts": []}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(empty_payload, indent=2))
        return output_path, empty_payload

    prompt = build_verification_prompt(
        items=verifiable,
        target_dir=work.repo,
        output_path=output_path,
    )

    result, _ = await run_agent(
        backend,
        work.repo,
        prompt,
        output_schema=RECOMMENDATION_VERDICTS_SCHEMA,
        max_turns=25,
        phase=DaydreamPhase.VERIFY,
    )

    candidate: Any = result
    if isinstance(result, str):
        try:
            candidate = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            candidate = None
    payload = _coerce_verdicts_payload(candidate)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path, payload


async def phase_fix(
    backend: Backend,
    work: WorkContext,
    item: dict[str, Any],
    item_num: int,
    total: int,
    *,
    console_lock: anyio.Lock | None = None,
    intent_path: Path | None = None,
) -> None:
    """Phase 3: Apply a single fix for one feedback item.

    Args:
        backend: The Backend to execute against.
        work: Workspace context for the fix; ``work.repo`` is the agent cwd.
        item: Feedback item containing description, file, and line
        item_num: Current item number (1-indexed)
        total: Total number of items
        console_lock: Optional lock to serialize console writes across
            concurrent callers.  Pass the same lock to every concurrent
            ``phase_fix`` invocation; leave ``None`` for serial callers.
        intent_path: Optional path to the confirmed author-intent file. When
            present and readable, its text is injected as authoritative intent
            with a rule forbidding fixes that undo a deliberate decision. The
            read is best-effort enrichment: a missing or unreadable file is
            skipped silently so an intent-read failure can never block the fix.

    Returns:
        None

    """
    description = item.get("description", "No description")
    file_path = item.get("file", "Unknown file")
    resolved = work.repo / file_path
    file_ref = str(resolved) if resolved.is_file() else file_path
    line = item.get("line", "Unknown")

    async with (console_lock if console_lock is not None else anyio.Lock()):
        console.print()
        print_fix_progress(console, item_num, total, description)

    prompt = f"""Fix this issue:
{description}

File: {file_ref}
Line: {line}

Make the minimal change needed. Do NOT change error handling semantics
(e.g., converting warn-and-continue to error propagation, or vice versa)
unless the issue description specifically explains why the current error
handling strategy is wrong for that code path.

Anchor the change to what this finding names — the file/symbol/line above. Do
NOT make gratuitous edits to adjacent fields, keys, or functions the fix does
not require; naming one issue is not license to "tidy" its neighbours. If a
correct fix genuinely requires an edit the finding didn't name — a caller that
must change in step, or a file the review step missed — make it, but name and
justify each out-of-scope edit in your commit message rather than expanding
silently. If the change balloons far beyond the named site, stop and report.

If this finding conflicts with an explicit in-code contract — a JSON schema, a
type signature, or a comment documenting intent — the contract wins. Do not
override documented intent to satisfy the finding; note the conflict in your
commit message (or report inability to fix). Treat low/medium-confidence
findings with extra skepticism here.
"""

    # Best-effort: inject the confirmed author intent so the fixer won't undo a
    # deliberate decision. Guarded on .exists() and wrapped so an intent-read
    # failure NEVER blocks or crashes a fix (deliberate deviation from strict
    # propagation). A read failure skips the block; it is never coerced into a
    # fake intent string.
    if intent_path is not None and intent_path.exists():
        try:
            confirmed_intent = intent_path.read_text()
        except OSError:
            confirmed_intent = None
        if confirmed_intent and confirmed_intent.strip():
            prompt += (
                "\nCONFIRMED AUTHOR INTENT for this change (authoritative):\n"
                f"{confirmed_intent.strip()}\n\n"
                "This confirmed intent is the highest-priority authority: it outranks both "
                "the in-code-contract rule above and the finding itself. "
                "If applying this fix would undo, revert, or contradict a decision the "
                "confirmed intent describes as deliberate, do NOT apply it. Report the "
                "conflict in your commit message (or report inability to fix) instead of "
                "overriding the author's deliberate choice.\n"
            )

    verifier_verdict = item.get("verifier_verdict")
    if verifier_verdict:
        evidence = item.get("evidence", "")
        prompt += f"\nVerifier verdict: {verifier_verdict}. Evidence: {evidence}.\n"
        assumptions = item.get("unverified_assumptions") or []
        if isinstance(assumptions, list) and assumptions:
            joined = "; ".join(str(a) for a in assumptions)
            prompt += f"Unverified assumptions: {joined}.\n"
        if verifier_verdict == "contradicts":
            prompt += (
                "\nDo NOT apply the recommendation literally if it contradicts the cited spec.\n"
                "Explain the conflict in your commit message and choose a fix that preserves\n"
                "the spec, or stop and report inability to fix.\n"
            )
        elif verifier_verdict == "uncertain":
            prompt += (
                "\nThe verifier could not confirm whether this recommendation is correct.\n"
                "Proceed cautiously: apply the minimal fix and note the uncertainty in your\n"
                "commit message.\n"
            )

    if console_lock is not None:
        # Concurrent path: suppress the Live/LiveToolPanelRegistry renderer in
        # run_agent so multiple concurrent agents don't each start their own
        # Rich Live context on the shared console (which garbles output).
        # The callback serializes progress lines through the shared lock.
        async def _cb(text: Text) -> None:
            async with console_lock:
                console.print(text)

        await run_agent(
            backend, work.repo, prompt,
            phase=DaydreamPhase.FIX, max_turns=FIX_MAX_TURNS,
            progress_callback=_cb,
        )
    else:
        await run_agent(backend, work.repo, prompt, phase=DaydreamPhase.FIX, max_turns=FIX_MAX_TURNS)
    async with (console_lock if console_lock is not None else anyio.Lock()):
        print_fix_complete(console, item_num, total)


async def phase_fix_parallel(
    backend: Backend,
    work: WorkContext,
    items: list[dict[str, Any]],
    *,
    limiter_size: int = 4,
    intent_path: Path | None = None,
) -> dict[str, str]:
    """Phase 3 (parallel): Apply fixes file-partitioned and concurrently.

    Items are grouped by ``file`` (preserving the caller's severity ordering).
    Each file-group becomes one task that applies its items serially, while
    distinct files run concurrently under an ``anyio.CapacityLimiter``. Same-file
    serialization prevents concurrent writes to the *same named file*; it does
    not guarantee disjoint edits if an agent touches files other than the one
    named in the item's ``file`` key. Commit stays serial and after.

    Args:
        backend: The Backend to execute against (shared across tasks).
        work: Workspace context for the fixes; ``work.repo`` is the agent cwd.
        items: Feedback items, already severity-sorted by the caller.
        limiter_size: Max number of file-groups to fix concurrently.
        intent_path: Optional confirmed-intent file forwarded unchanged to each
            ``phase_fix`` call so every fix carries the deliberate-intent guard.

    Returns:
        ``failures``: file -> "<ExceptionType>: <message>" for file-groups whose
        fix raised. Empty dict on full success. Callers MUST surface this to the
        user so that uncommitted failures are visible instead of silently dropped.

    """
    raw_groups = group_items_by_file(items)
    # Assign stable 1-based counters by pairing each item with its number
    # directly, avoiding fragile id()-keyed dicts whose keys are memory
    # addresses and can collide if dicts are reallocated between loops.
    counter = 0
    groups_numbered: list[tuple[str, list[tuple[dict[str, Any], int]]]] = []
    for file_key, group_items in raw_groups:
        numbered: list[tuple[dict[str, Any], int]] = []
        for item in group_items:
            counter += 1
            numbered.append((item, counter))
        groups_numbered.append((file_key, numbered))

    recorder = get_current_recorder()
    failures: dict[str, str] = {}
    _failures_lock = anyio.Lock()
    limiter = anyio.CapacityLimiter(limiter_size)
    _console_lock = anyio.Lock()
    total = len(items)

    async with anyio.create_task_group() as tg:
        for file_key, numbered_items in groups_numbered:
            # Default-arg capture -- prevents late-binding closure bug (Pitfall 2).
            async def _task(
                fkey: str = file_key,
                grp: list[tuple[dict[str, Any], int]] = numbered_items,
            ) -> None:
                _fkey_slug = fkey.replace("/", "-").replace("\\", "-")
                async with limiter:
                    async with maybe_fork(recorder, f"fix-{_fkey_slug}"):
                        try:
                            for item, item_num in grp:
                                await phase_fix(
                                    backend, work, item, item_num, total,
                                    console_lock=_console_lock,
                                    intent_path=intent_path,
                                )
                        except Exception as e:  # noqa: BLE001 -- intentionally broad for parallel isolation
                            reason = f"{type(e).__name__}: {e}"
                            async with _failures_lock:
                                failures[fkey] = reason
                            async with _console_lock:
                                print_warning(
                                    console,
                                    f"Fixes for '{fkey}' failed ({reason}); other fixes applied "
                                    "but this file's changes are left uncommitted.",
                                )

            tg.start_soon(_task)

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)

    return failures


async def _emit_failure_handoff(
    backend: Backend,
    work: WorkContext,
    output: str,
    *,
    offer_clipboard: bool,
) -> None:
    """Run the failure summarizer, display the handoff body, and optionally
    offer to copy it to the clipboard.

    Args:
        backend: Backend used to invoke the summarizer subagent.
        work: Workspace context passed through to ``_run_failure_summarizer``.
        output: Raw failing test output to ground the summary.
        offer_clipboard: When ``True``, prompt the user to copy the handoff to
            the clipboard (interactive mode).  When ``False``, skip the prompt
            (non-interactive mode).
    """
    body, handoff_path, handoff_written = await _run_failure_summarizer(
        backend, work, output,
    )
    if handoff_written:
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
    else:
        print_warning(
            console,
            f"Failed to write handoff to {handoff_path}; "
            "printing inline so it is not lost:",
        )
        console.print(body)

    if offer_clipboard:
        if clipboard_available():
            if resolve_or_prompt(
                assume=get_assume(),
                interactive=not get_non_interactive(),
                safe_default=False,
                question="Copy handoff to clipboard?",
                default="y",
            ):
                if copy_to_clipboard(body):
                    print_success(console, "Handoff copied to clipboard")
                else:
                    recovery = (
                        "copy manually from path above"
                        if handoff_written
                        else "copy manually from the inline output above"
                    )
                    print_warning(
                        console, f"Clipboard copy failed; {recovery}",
                    )
        else:
            recovery = (
                "copy manually from path above"
                if handoff_written
                else "copy manually from the inline output above"
            )
            print_info(
                console, f"(clipboard unavailable, {recovery})",
            )


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
    print_dim(console, f"Model: {backend.model}")

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
            sanitized_cmd = _sanitize_suggested_command(test_command_override)
            prompt = (
                "Run this exact test command:\n"
                f"```\n{sanitized_cmd}\n```\n"
                "Report if tests pass or fail."
            )
        else:
            prompt = "Run the project's test suite. Report if tests pass or fail."
        # Use-once: clear after consuming above so the next iteration falls back
        # to the default prompt. Must stay before the bottom-of-loop branches that
        # may re-assign test_command_override (choice "1" → setup investigator).
        test_command_override = None

        output, continuation = await run_agent(
            backend, work.repo, prompt, continuation=continuation, phase=DaydreamPhase.TEST,
        )

        test_passed = detect_test_success(output)

        if test_passed:
            print_success(console, "Tests passed")
            return True, retries_used

        print_warning(console, "Tests may have failed or result is unclear.")

        # Test-heal retry gate across the two interaction axes. With no human at
        # the keyboard, the menu's default "2" (fix-and-retry) would launch an
        # unbounded, mutating fix loop with nothing to stop it -- so the
        # unattended safe default is to abort (choice-"4" semantics: surface the
        # failure honestly with no mutation). ``--yes`` opts into a SINGLE bounded
        # auto fix-and-retry (choice "2"); after that one attempt it falls through
        # to abort so the loop still terminates. Only an interactive run with no
        # assumption shows the menu.
        decision = resolve_gate(
            assume=get_assume(),
            interactive=not get_non_interactive(),
            safe_default=False,
        )
        if decision is False or (decision is True and retries_used > 0):
            print_error(
                console, "Tests failed", "Aborting heal loop (no further auto-retries)",
            )
            await _emit_failure_handoff(backend, work, output, offer_clipboard=False)
            return False, retries_used
        if decision is True:
            # Bounded auto fix-and-retry: launch one fix attempt, then loop.
            console.print()
            print_info(console, "Launching agent to fix test failures (auto)...")
            fix_prompt = _build_fix_prompt(output, feedback_items, repo=work.repo)
            _, _ = await run_agent(
                backend, work.repo, fix_prompt, phase=DaydreamPhase.FIX, max_turns=FIX_MAX_TURNS,
            )
            retries_used += 1
            continuation = None
            continue

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
                    # Show the sanitized command (same transform as the retry prompt)
                    # before asking approval, so the preview matches what gets pinned.
                    sanitized_preview = _sanitize_suggested_command(suggested)
                    print_info(
                        console, f"Suggested command: {sanitized_preview}",
                    )
                    if resolve_or_prompt(
                        assume=get_assume(),
                        interactive=not get_non_interactive(),
                        safe_default=False,
                        question="Use suggested command instead?",
                        default="n",
                    ):
                        test_command_override = sanitized_preview

            retries_used += 1
            continue

        elif choice == "2":
            console.print()
            print_info(console, "Launching agent to fix test failures...")
            fix_prompt = _build_fix_prompt(output, feedback_items, repo=work.repo)
            _, _ = await run_agent(
                backend, work.repo, fix_prompt, phase=DaydreamPhase.FIX, max_turns=FIX_MAX_TURNS,
            )
            retries_used += 1
            continuation = None
            continue

        elif choice == "3":
            print_warning(console, "Ignoring test failures, continuing...")
            return True, retries_used

        elif choice == "4":
            print_error(console, "Aborted", "User requested abort")
            await _emit_failure_handoff(backend, work, output, offer_clipboard=True)
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
        # Commit/push gate across the two interaction axes. ``--yes`` commits
        # without prompting; an unattended run with no assumption declines
        # (safe_default=False — the interactive default is decline); otherwise
        # prompt. Routed here (not at the caller) so every interactive commit
        # path honours both axes.
        decision = resolve_or_prompt(
            assume=get_assume(),
            interactive=not get_non_interactive(),
            safe_default=False,
            question="Commit and push changes? [y/N]",
            default="n",
        )
        if not decision:
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

    # Post-commit trailer verification: the agent may omit trailers, so amend if
    # missing (daydream_commits() relies on them). Only when a new commit was
    # created — otherwise we would silently amend the user's prior commit.
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
    print_dim(console, f"Model: {backend.model}")

    skill_invocation = backend.format_skill_invocation(
        "beagle-core:fetch-pr-feedback", f"--pr {pr_number} --bot {bot}"
    )

    await run_agent(backend, work.repo, skill_invocation, phase=DaydreamPhase.PR_FEEDBACK)

    output_path = work.repo / REVIEW_OUTPUT_FILE
    if output_path.exists():
        print_success(console, f"PR feedback written to: {output_path}")
    else:
        print_warning(console, "PR feedback file was not created")


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
        results: List of (item, success, error) tuples, one per applied fix

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
    pr_description: str | None = None,
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
        exploration_dir: Optional directory of pre-scan exploration context.
        pr_description: Optional author-written PR description body. When
            present, it is threaded into the INITIAL intent proposal as the
            authoritative statement of intent. It is deliberately NOT
            re-injected into the interactive correction-loop rebuild: once a
            human supplies a correction, that correction is the higher
            authority.

    Returns:
        The confirmed intent summary string.

    """
    print_phase_hero(console, "LISTEN", phase_subtitle("LISTEN"))
    print_dim(console, f"Model: {backend.model}")

    prompt = build_intent_prompt(
        diff_path=str(diff_path),
        branch=branch,
        log=log,
        exploration_dir=exploration_dir,
        pr_description=pr_description,
    )

    while True:
        console.print()
        print_info(console, "Agent is analyzing the changes...")

        output, _ = await run_agent(backend, work.repo, prompt, phase=DaydreamPhase.INTENT)
        intent_text = output if isinstance(output, str) else str(output)

        console.print()
        # Confirm-or-correct gate. ``--yes`` and unattended runs accept the
        # understanding as-is and proceed (this read step is non-mutating, so the
        # safe unattended outcome is to continue, not to block); only an
        # interactive run with no assumption may offer a correction. A forced
        # "no" declines the current understanding and falls through so the user
        # can supply a correction interactively — but only when interactive;
        # a forced "no" in a non-interactive run also proceeds rather than
        # blocking on stdin.
        gate = resolve_gate(
            assume=get_assume(),
            interactive=not get_non_interactive(),
            safe_default=True,
        )
        if gate is True:
            return intent_text
        if gate is False and get_non_interactive():
            return intent_text

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
    print_dim(console, f"Model: {backend.model}")

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
    print_dim(console, f"Model: {backend.model}")

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


# Deep-mode: per-stack fan-out


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

    Uses a capacity-limiter + task-group + default-arg closure capture pattern.
    Per D-38, uses orchestrator-level parallelism -- never passes the ``agents``
    kwarg (Codex does not support SDK-level sub-agent spawning).

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
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep import prompts as _prompts
    from daydream.deep.artifacts import deep_dir as _deep_dir
    from daydream.deep.artifacts import per_stack_review_path

    deep_dir_path = _deep_dir(work.repo)
    recorder = get_current_recorder()
    results: dict[str, Path] = {}
    failures: dict[str, str] = {}
    limiter = anyio.CapacityLimiter(4)
    prior_commits = _prior_daydream_commits(work)

    async with anyio.create_task_group() as tg:
        for stack in stacks:
            output_path = per_stack_review_path(deep_dir_path, stack.stack_name)
            if stack.stack_name == STRUCTURE_STACK_NAME:
                prompt = _prompts.build_structural_prompt(
                    files=stack.files,
                    diff_path=diff_path,
                    intent_path=intent_path,
                    alternatives_path=alternatives_path,
                    output_path=output_path,
                    exploration_dir=exploration_dir,
                    prior_commits=prior_commits,
                )
            elif stack.skill_invocation is None:
                prompt = _prompts.build_generic_fallback_prompt(
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
                prompt = _prompts.build_per_stack_prompt(
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
                        failures[stack_name] = f"{type(e).__name__}: {e}"

            tg.start_soon(_task)

    if failures:
        lines = "\n".join(f"  - {name}: {reason}" for name, reason in sorted(failures.items()))
        print_warning(
            console,
            f"Per-stack reviews failed for {len(failures)} stack(s); "
            "failures will be passed to the merge step.\n" + lines,
        )

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.DEEP)

    return results, failures


ARBITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "arb_id": {"type": "integer"},
                    "keep": {"type": "boolean"},
                    "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]},
                    "description": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["arb_id", "keep", "severity", "confidence", "description", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["findings"],
    "additionalProperties": False,
}


async def phase_arbiter_review(
    backend: Backend,
    work: WorkContext,
    *,
    selected_records: list[dict[str, Any]],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    exploration_dir: Path | None = None,
) -> dict[int, dict[str, Any]]:
    """Re-review high-severity / contested per-stack findings with the arbiter (#168).

    Runs a single heavyweight (Opus by default) agent over only the findings the
    cheaper per-stack reviewers flagged as high-severity or contested. The
    arbiter adjudicates -- confirming, re-ranking, sharpening, or rejecting each
    -- but never discovers new findings. The result re-keys onto the input by
    ``arb_id`` so the caller can revise (keep), drop (explicit ``keep:false``), or
    retain-unchanged (missing verdict) the originating records before the
    cross-stack merge.

    Args:
        backend: The Backend to execute against (resolved via phase ``arbiter``).
        work: Workspace context; ``work.repo`` is the agent cwd.
        selected_records: Per-stack records selected for arbitration. Each is
            tagged with a fresh 1-based ``arb_id`` before being written to the
            arbiter input artifact; the originals are left untouched.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        exploration_dir: Optional pre-scan exploration directory.

    Returns:
        Mapping of ``arb_id`` -> adjudicated finding dict with keys ``keep``,
        ``severity``, ``confidence``, ``description``, ``rationale``. A missing
        ``arb_id`` (the agent dropped or truncated a row) is fail-open: the caller
        retains the original record unchanged with a warning, since arbitration
        targets are the high-severity / contested findings worth protecting.

    """
    from daydream.deep.artifacts import arbiter_input_path, deep_dir
    from daydream.deep.prompts import build_arbiter_prompt

    print_phase_hero(console, "ARBITRATE", phase_subtitle("ARBITRATE"))
    print_dim(console, f"Model: {backend.model}")
    print_info(console, f"Arbitrating {len(selected_records)} high-severity/contested finding(s)")

    dd = deep_dir(work.repo)
    input_path = arbiter_input_path(dd)
    arbiter_input = [
        {
            "arb_id": i,
            "file": rec.get("file"),
            "line": rec.get("line"),
            "severity": rec.get("severity"),
            "confidence": rec.get("confidence"),
            "description": rec.get("description"),
            "rationale": rec.get("rationale"),
        }
        for i, rec in enumerate(selected_records, start=1)
    ]
    input_path.write_text(json.dumps(arbiter_input, indent=2))

    prompt = build_arbiter_prompt(
        arbiter_input_path=input_path,
        diff_path=diff_path,
        intent_path=intent_path,
        alternatives_path=alternatives_path,
        exploration_dir=exploration_dir,
    )
    result, _ = await run_agent(backend, work.repo, prompt, output_schema=ARBITER_SCHEMA, phase=DaydreamPhase.DEEP)

    if not isinstance(result, dict) or not isinstance(result.get("findings"), list):
        raise ValueError(f"Arbiter returned no findings list (got {type(result).__name__})")

    verdicts: dict[int, dict[str, Any]] = {}
    for finding in result["findings"]:
        arb_id = finding.get("arb_id")
        if isinstance(arb_id, int):
            verdicts[arb_id] = finding
    kept = sum(1 for v in verdicts.values() if v.get("keep"))
    print_info(console, f"Arbiter: kept {kept}, dropped {len(verdicts) - kept}")
    return verdicts


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
    structural_records_path: Path | None = None,
) -> Path:
    """Run the cross-stack merge agent and return the merged-report path (D-23..D-27).

    The merge agent returns a schema-validated item list (``MERGED_ITEMS_SCHEMA``)
    covering per-stack and cross-stack findings, each tagged with ``lens``.
    Structural records (from ``structural_records_path``) are appended to that
    list in Python, tagged ``lens="structural"`` -- never requested via prose,
    so the structural lens cannot be silently dropped by the agent. The combined
    list is normalized (fresh unique ids), written as the canonical
    ``merged-items.json``, and rendered to ``review-output.md`` (single source of
    truth → markdown). The markdown is written inside ``.daydream/deep/`` (which
    avoids sandbox write restrictions that block dotfiles at the repo root) and
    then copied to ``work.repo / REVIEW_OUTPUT_FILE`` for downstream consumers.

    Per D-38, never passes the ``agents`` kwarg (Codex parity).

    Args:
        backend: The Backend to execute against.
        work: Workspace context; report is written under ``work.repo``.
        per_stack_records_paths: Parsed per-stack record JSON paths (D-22 inputs).
            Must NOT include the structural meta-stack records file -- callers
            partition that out and pass it via ``structural_records_path``.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        dedup_candidates_path: Path to dedup-candidates.json (D-27 pre-filter output).
        exploration_dir: Optional pre-scan exploration directory.
        failed_stacks: Optional stack_name -> reason dict for per-stack agents
            that failed. Passed through to the merge prompt so the merged
            report can call out uncovered stacks explicitly.
        structural_records_path: Optional path to the parsed structural
            meta-stack records JSON. When provided, its findings are appended to
            the canonical item list tagged ``lens="structural"`` (high severity
            by construction -- the structural lens carries different convictions
            and is not deduplicated against the language stacks). ``None`` when
            the structural reviewer did not run (docs-only diff, empty diff).

    Returns:
        Path to the rendered merged report at ``work.repo / REVIEW_OUTPUT_FILE``.

    Raises:
        ValueError: If the merge agent returns empty or schema-invalid output
            (no silent ``[]`` fallback that would mask a broken merge).

    """
    from daydream.deep.artifacts import deep_dir, merged_items_path, merged_report_path
    from daydream.deep.prompts import build_merge_prompt
    from daydream.deep.render import render_report

    dd = deep_dir(work.repo)
    canonical_path = work.repo / REVIEW_OUTPUT_FILE
    report_path = merged_report_path(dd)
    items_path = merged_items_path(dd)

    # Clear stale outputs so a failed merge agent can't leave behind
    # outdated content that downstream stages would silently consume.
    canonical_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    items_path.unlink(missing_ok=True)

    prompt = build_merge_prompt(
        per_stack_records_paths=per_stack_records_paths,
        intent_path=intent_path,
        alternatives_path=alternatives_path,
        dedup_candidates_path=dedup_candidates_path,
        output_path=report_path,
        exploration_dir=exploration_dir,
        failed_stacks=failed_stacks,
        structural_records_path=structural_records_path,
    )
    print_phase_hero(console, "MERGE", phase_subtitle("MERGE"))
    print_dim(console, f"Model: {backend.model}")
    result, _ = await run_agent(
        backend, work.repo, prompt, output_schema=MERGED_ITEMS_SCHEMA, phase=DaydreamPhase.DEEP
    )

    # Fail loudly on empty/invalid output -- a silent [] would hide a broken
    # merge and ship an empty report downstream.
    if not isinstance(result, dict) or not isinstance(result.get("items"), list):
        raise ValueError(f"Cross-stack merge returned no item list (got {type(result).__name__})")
    agent_items: list[dict[str, Any]] = result["items"]

    # Append structural findings in Python, tagged lens="structural". They are
    # parsed FEEDBACK_SCHEMA records ({id, description, file, line}) and carry no
    # confidence/severity, so default both to HIGH/high -- the structural lens is
    # high-conviction by construction and must not be demoted at sort time.
    structural_items: list[dict[str, Any]] = []
    if structural_records_path is not None and structural_records_path.is_file():
        try:
            structural_records = json.loads(structural_records_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # Structural records come from a prior agent run that may have emitted
            # malformed output; degrade to none rather than crash the merge.
            print_warning(console, f"Skipping malformed structural records: {type(exc).__name__}: {exc}")
            structural_records = []
        for rec in structural_records:
            structural_items.append(
                {
                    **rec,
                    "lens": "structural",
                    "confidence": rec.get("confidence", "HIGH"),
                    "severity": rec.get("severity", "high"),
                }
            )

    items = normalize_items(agent_items + structural_items)
    items_path.write_text(json.dumps({"items": items}, indent=2))
    print_info(console, f"Merged into {len(items)} items")

    # Render the human report FROM the canonical items, then copy from the deep
    # artifact dir to the canonical location (the deep dir avoids sandbox write
    # restrictions on repo-root dotfiles).
    report_path.write_text(render_report(items))
    canonical_path.write_text(report_path.read_text())
    return canonical_path
