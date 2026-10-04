"""Handoff for review and fix phases."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio

from daydream import agent, git_ops, ui
from daydream.artifact_visibility import (
    ArtifactSession,
    ArtifactVisibilityError,
    artifact_dir_for,
    artifact_session_active,
)
from daydream.backends import (
    Backend,
)
from daydream.clipboard import clipboard_available, copy_to_clipboard
from daydream.output_schema import strict_object
from daydream.phases.inputs import _prepare_existing_phase_inputs, _render_bash_allowlist, _tail_test_output
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.trajectory import (
    DaydreamPhase,
    TrajectoryRecorder,
    get_current_recorder,
    maybe_fork,
    partial_document_path,
    run_directory,
    run_document_path,
    siblings_directory,
)
from daydream.workspace import WorkContext

_logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class HandoffArtifacts:
    """Durable public links for a failed run; readability is decided separately."""

    trajectory: Path | None = None
    trajectories: Path | None = None
    diff: Path | None = None
    manifest: Path | None = None
    deep: Path | None = None

    def render(self, *, empty: str) -> str:
        links = (
            ("trajectory", self.trajectory), ("sub-trajectories", self.trajectories),
            ("diff", self.diff), ("manifest", self.manifest), ("deep artifacts", self.deep),
        )
        return "\n".join(f"- {label}: {path}" for label, path in links if path is not None) or empty

    def inputs(self) -> dict[str, Path]:
        """Name only concrete evidence files; directory links never grant read authority."""
        paths = {
            "trajectory-partial": None if self.trajectory is None else partial_document_path(self.trajectory),
            "diff": self.diff,
            "manifest": self.manifest,
            **{label: None if self.deep is None else self.deep / f"{label}.json"
               for label in ("merged-items", "test-verdict", "fix-failures")},
        }
        return {label: path for label, path in paths.items() if path is not None}


def _build_failure_summarizer_prompt(
    *,
    test_output: str,
    artifacts: HandoffArtifacts,
    changed_files: list[Path],
    has_trajectory: bool,
    durable_changed_files: list[Path] | None = None,
    governed_input_labels: tuple[str, ...] | None = None,
) -> str:
    """Build a grounded handoff prompt from the durable run artifacts."""
    tail, truncated = _tail_test_output(test_output)
    if truncated:
        output_section = f"Tail of the failing test output:\n\n{tail}"
    else:
        output_section = f"Failing test output:\n\n{test_output}"

    artifacts_block = artifacts.render(empty="(none will be published)")

    def _bullets(paths: list[Path]) -> str:
        return "\n".join(f"- {p}" for p in paths) if paths else "(none detected)"

    readable_changed_block = _bullets(changed_files)
    durable_changed_block = _bullets(
        changed_files if durable_changed_files is None else durable_changed_files
    )
    if governed_input_labels is None:
        artifact_read_clause = (
            "- You MAY use Read, Grep, and Glob to inspect the artifacts listed below.\n"
        )
        artifacts_heading = "## On-disk artifacts (read these first to ground your summary)\n"
    else:
        artifact_read_clause = (
            "- You MAY use Read only on the exact sanctioned artifact files appended "
            f"below under these logical labels: {', '.join(governed_input_labels)}. Do not "
            "enumerate their parent directories, and do not copy their private paths into "
            "the handoff.\n"
            if governed_input_labels
            else "- No artifact file is sanctioned as readable during this turn. The "
            "public paths below are future links only; do not inspect them or their "
            "parent directories.\n"
        )
        artifacts_heading = "## Future handoff links (not readable evidence during this turn)\n"

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
        f"{artifact_read_clause}"
        "- You MAY use Bash for NON-MUTATING inspection ONLY. Permitted commands: "
        + _render_bash_allowlist()
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
        f"{artifacts_heading}"
        f"{artifacts_block}\n\n"
        "## Files readable in the current model workspace\n"
        f"{readable_changed_block}\n\n"
        "Use the following durable public paths only in the handoff's Changed files "
        "section; do not serialize current-workspace or sanctioned private paths:\n"
        f"{durable_changed_block}\n\n"
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


FAILURE_SUMMARIZER_SCHEMA: dict[str, Any] = strict_object({
    "handoff_prompt": {"type": "string"},
})


def _changed_files(repo: Path) -> list[Path]:
    """Join validated Git-relative changed names to repo without resolving leaves.

    Unavailable Git or a repository without commits returns no paths.
    """
    paths: list[Path] = []
    for name in git_ops.changed_files(repo):
        parts = name.split("/")
        if "\0" in name or Path(name).is_absolute() or any(p in ("", ".", "..") for p in parts):
            _logger.warning("skipping unsafe changed-file name: %r", name)
            continue
        paths.append(repo.joinpath(*parts))
    return paths


def _require_handoff_session(
    work: WorkContext,
    artifact_session: ArtifactSession | None,
    *,
    allow_standalone: bool,
) -> None:
    """Fail closed when a strict handoff accessor has no bound session."""
    if artifact_session is None:
        if not allow_standalone:
            # Strict callers must supply the session even if a standalone
            # archive would otherwise make a durable reference available.
            artifact_dir_for(work.repo, session=None, allow_standalone=False)
        if artifact_session_active():
            raise ArtifactVisibilityError(
                "an explicit artifact session is required for handoff routing"
            )


def _resolve_handoff_paths(
    recorder: TrajectoryRecorder | None,
    work: WorkContext,
    *,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
) -> tuple[Path, HandoffArtifacts]:
    """Resolve durable public handoff artifacts from the active run."""
    _require_handoff_session(work, artifact_session, allow_standalone=allow_standalone)

    if recorder is None:
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")  # noqa: DTZ005 - filename only
        unbound_target = work.source if work.is_ephemeral else work.repo
        live_handoff = artifact_dir_for(
            work.repo if artifact_session is not None else unbound_target,
            session=artifact_session,
            allow_standalone=allow_standalone,
        ) / f"handoff-{ts}.md"
        handoff_path = (
            artifact_session.durable_path_for(live_handoff, repo=work.repo)
            if artifact_session is not None
            else live_handoff
        )
        return handoff_path, HandoffArtifacts()

    if artifact_session is not None:
        live_daydream_dir = artifact_dir_for(
            work.repo,
            session=artifact_session,
            allow_standalone=False,
        )
        live_artifact_root = run_directory(live_daydream_dir, recorder.session_id)
        artifact_root = artifact_session.durable_path_for(live_artifact_root, repo=work.repo)
        diff_path = artifact_session.durable_path_for(
            live_daydream_dir / "diff.patch", repo=work.repo,
        )
        deep_dir = artifact_session.durable_path_for(live_daydream_dir / "deep", repo=work.repo)
    elif work.is_ephemeral:
        # Standalone execution has no runtime publication owner. Its disposable
        # workspace and an optional recorder callback cannot promise durable
        # evidence. Keep the handoff in the source without inventing archive links.
        source_daydream = artifact_dir_for(work.source, session=None, allow_standalone=allow_standalone)
        return run_directory(source_daydream, recorder.session_id) / "handoff.md", HandoffArtifacts()
    else:
        daydream_dir = artifact_dir_for(
            recorder.target_dir,
            session=artifact_session,
            allow_standalone=allow_standalone,
        )
        artifact_root = run_directory(daydream_dir, recorder.session_id)
        diff_path = daydream_dir / "diff.patch"
        deep_dir = daydream_dir / "deep"

    trajectory_path = run_document_path(artifact_root)
    if artifact_session is not None:
        trajectory_path = artifact_session.durable_path_for(recorder.path, repo=work.repo)

    return artifact_root / "handoff.md", HandoffArtifacts(
        trajectory=trajectory_path,
        trajectories=siblings_directory(artifact_root),
        diff=diff_path,
        # Runtime publication does not produce an archive manifest. Optional
        # persistence cannot promise one before it has actually succeeded.
        manifest=(artifact_root / "manifest.json") if (
            artifact_session is None and (artifact_root / "manifest.json").is_file()
        ) else None,
        deep=deep_dir,
    )


def _handoff_write_path(
    handoff_reference: Path,
    work: WorkContext,
    *,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
) -> Path:
    """Return the private active-session destination for a public handoff ref."""
    _require_handoff_session(work, artifact_session, allow_standalone=allow_standalone)
    if artifact_session is None:
        return handoff_reference
    return artifact_session.live_path_for(handoff_reference, repo=work.repo)


def _write_handoff(path: Path, body: str) -> bool:
    """Create parents and write the body; return False on OSError so callers display it inline."""
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
    artifacts: HandoffArtifacts,
    changed_files: list[Path],
    has_trajectory: bool,
) -> str:
    """Build a no-agent fallback separating verified facts from unknown causes.

    Quote test output and known changed files; invent no citations or hypothesis.
    """
    output_section, _ = _tail_test_output(test_output)

    artifacts_block = artifacts.render(empty="_(none on disk)_")

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


@bind_resolved_run_context
async def _run_failure_summarizer(
    backend: Backend,
    work: WorkContext,
    test_output: str,
    *,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
    run_context: RunContext | None = None,
) -> tuple[str, Path, bool]:
    """Run a read-only fork or use the minimal fallback, then attempt to write handoff.md.

    No recorder, agent errors, or unusable output trigger fallback. Return body, path,
    and written status; callers must display the body inline when writing fails.
    """
    run_context = resolve_run_context(run_context)
    recorder = get_current_recorder()
    handoff_path, artifacts = _resolve_handoff_paths(
        recorder,
        work,
        artifact_session=artifact_session,
        allow_standalone=allow_standalone,
    )

    # The recorder has not exited yet. Persist a best-effort partial trajectory
    # for the summarizer before the main trajectory is written.
    if recorder is not None:
        recorder.write_partial()

    has_trajectory = artifacts.trajectory is not None
    active_session = artifact_session is not None
    changed_live = _changed_files(work.repo)
    possible_inputs = artifacts.inputs()
    private_runtime_paths: tuple[Path, ...] = ()
    if active_session:
        assert artifact_session is not None
        changed_for_model = [path.relative_to(work.repo) for path in changed_live]
        durable_changed = [work.source / name for name in changed_for_model]
        private_runtime_paths = (
            artifact_session.layout.artifact_runtime_root,
            artifact_session.layout.operational_workspaces_root,
        ) + ((work.repo,) if work.repo != work.source else ())

        possible_inputs = {
            label: artifact_session.live_path_for(path, repo=work.repo) for label, path in possible_inputs.items()
        }
        if recorder is not None:
            possible_inputs["trajectory-partial"] = partial_document_path(recorder.path)
    else:
        changed_for_model = durable_changed = changed_live
    readable_inputs = {label: path for label, path in possible_inputs.items() if path.is_file()}

    prompt = _build_failure_summarizer_prompt(
        test_output=test_output,
        artifacts=artifacts,
        changed_files=changed_for_model,
        has_trajectory=has_trajectory,
        durable_changed_files=durable_changed,
        governed_input_labels=(tuple(sorted(readable_inputs)) if active_session else None),
    )

    body: str | None = None
    try:
        async with maybe_fork(recorder, "failure-summarizer"):
            sanctioned_inputs = _prepare_existing_phase_inputs(backend, work, readable_inputs, read_only=True)
            result, _, _ = await agent.run_agent(
                backend, work.repo, prompt,
                output_schema=FAILURE_SUMMARIZER_SCHEMA,
                phase=DaydreamPhase.TEST,
                read_only=True,
                sanctioned_inputs=sanctioned_inputs,
                run_context=run_context,
            )
            candidate = result.get("handoff_prompt") if isinstance(result, dict) else None
            if isinstance(candidate, str) and candidate.strip():
                if any(str(path) in candidate for path in private_runtime_paths):
                    _logger.warning(
                        "failure-summarizer output contained a private runtime path; using the deterministic handoff"
                    )
                else:
                    body = candidate
    except Exception:  # the diagnostic and its recorder fork are best-effort
        _logger.debug("failure-summarizer failed", exc_info=True)
        body = None

    if body is None:
        fallback_output = test_output
        if any(str(path) in fallback_output for path in private_runtime_paths):
            fallback_output = "Test output omitted because it contained a private runtime path."
        body = _build_minimal_handoff(
            test_output=fallback_output,
            artifacts=artifacts,
            changed_files=durable_changed,
            has_trajectory=has_trajectory,
        )

    written = _write_handoff(
        _handoff_write_path(
            handoff_path,
            work,
            artifact_session=artifact_session,
            allow_standalone=allow_standalone,
        ),
        body,
    )
    return body, handoff_path, written


@bind_resolved_run_context
async def _emit_failure_handoff(
    backend: Backend,
    work: WorkContext,
    output: str,
    *,
    offer_clipboard: bool,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
    run_context: RunContext | None = None,
) -> None:
    """Summarize and display test failure; offer clipboard copying only when requested."""
    run_context = resolve_run_context(run_context)
    body, handoff_path, handoff_written = await _run_failure_summarizer(
        backend,
        work,
        output,
        artifact_session=artifact_session,
        allow_standalone=allow_standalone,
        run_context=run_context,
    )
    if handoff_written:
        preview_lines = body.splitlines()
        max_preview = 20
        if len(preview_lines) <= max_preview:
            agent.console.print(body)
        else:
            agent.console.print("\n".join(preview_lines[:max_preview]))
            ui.print_info(
                agent.console,
                f"... ({len(preview_lines) - max_preview} more lines, see file below)",
            )
        ui.print_info(agent.console, f"Handoff written: {handoff_path}")
    else:
        ui.print_warning(
            agent.console,
            f"Failed to write handoff to {handoff_path}; "
            "printing inline so it is not lost:",
        )
        agent.console.print(body)

    if offer_clipboard:
        if clipboard_available():
            if run_context.confirm(
                safe_default=False,
                question="Copy handoff to clipboard?",
                default="y",
                console=agent.console,
            ):
                if await anyio.to_thread.run_sync(copy_to_clipboard, body):
                    ui.print_success(agent.console, "Handoff copied to clipboard")
                else:
                    recovery = (
                        "copy manually from path above"
                        if handoff_written
                        else "copy manually from the inline output above"
                    )
                    ui.print_warning(
                        agent.console, f"Clipboard copy failed; {recovery}",
                    )
        else:
            recovery = (
                "copy manually from path above"
                if handoff_written
                else "copy manually from the inline output above"
            )
            ui.print_info(
                agent.console, f"(clipboard unavailable, {recovery})",
            )
