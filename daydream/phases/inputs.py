"""Inputs for review and fix phases."""

from collections.abc import Mapping
from pathlib import Path

from daydream import git_ops
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    artifact_dir_for,
    artifact_session_active,
)
from daydream.backends import (
    Backend,
)
from daydream.backends.claude import READ_ONLY_BASH_ALLOWLIST
from daydream.git_ops import BranchNotFoundError, GitError
from daydream.prompt_budget import (
    AdvisoryCandidate,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    fits_inline_diff_budget,
    prepare_sanctioned_inputs,
    select_advisory_inputs,
    uses_diff_reference,
)
from daydream.prompts.grounding import render_test_recipe_block
from daydream.test_execution import (
    TestRecipe,
    load_test_recipe,
)
from daydream.workspace import WorkContext

_EXPLORATION_PHASE_INPUTS = {
    "exploration-summary": "summary.md",
    "exploration-affected-files": "affected_files.md",
}


def _prepare_existing_phase_inputs(
    backend: Backend,
    work: WorkContext,
    inputs: Mapping[str, Path | None],
    *,
    exploration_dir: Path | None = None,
    read_only: bool = False,
    capture_without_session: bool = False,
) -> PreparedSanctionedInputs | None:
    """Capture existing named inputs, shared exploration files, and bounded-review finalization inputs."""
    if not artifact_session_active() and not capture_without_session:
        return None
    captured = dict(inputs)
    if exploration_dir is not None:
        captured.update(
            (label, exploration_dir / name) for label, name in _EXPLORATION_PHASE_INPUTS.items()
        )
    return prepare_sanctioned_inputs(
        backend,
        work.repo,
        {label: path for label, path in captured.items() if path is not None and (
            path.is_file() or label == "diff" and uses_diff_reference(backend, work.repo, read_only=read_only)
        )},
        read_only=read_only,
    )


def _pointer_dir(
    prepared: PreparedSanctionedInputs | None, exploration_dir: Path | None
) -> Path | None:
    """Drop a prompt's exploration pointer when the inputs travel inline."""
    if prepared is not None and prepared.transport is SanctionedInputTransport.INLINE:
        return None
    return exploration_dir


def _budgeted_exploration_inputs(
    exploration_dir: Path | None,
    *,
    backend: Backend,
    cwd: Path,
    read_only: bool = False,
) -> dict[str, Path | None]:
    """Admit whole exploration artifacts through the shared advisory budget, summary first.

    Return None for omitted inputs so budgeting and capture use the same policy.
    """
    labels = tuple(_EXPLORATION_PHASE_INPUTS)
    if exploration_dir is None:
        return dict.fromkeys(labels)
    candidates = [
        AdvisoryCandidate(label, exploration_dir / _EXPLORATION_PHASE_INPUTS[label])
        for label in labels
    ]
    selection = select_advisory_inputs(backend, cwd, candidates, read_only=read_only)
    admitted = selection.selected_paths()
    return {label: admitted.get(label) for label in labels}


TEST_OUTPUT_TAIL_LINES = 100

_PR_BODY_MAX_CHARS = 8000


def _tail_test_output(test_output: str) -> tuple[str, bool]:
    """Return ``(text, truncated)``: the last TEST_OUTPUT_TAIL_LINES lines when longer, else the full output."""
    lines = test_output.splitlines()
    if len(lines) > TEST_OUTPUT_TAIL_LINES:
        return "\n".join(lines[-TEST_OUTPUT_TAIL_LINES:]), True
    return test_output, False



def _render_bash_allowlist() -> str:
    """Render the backend-enforced read-only Bash allowlist identically for every prompt."""
    return ", ".join(f"`{cmd}`" for cmd in READ_ONLY_BASH_ALLOWLIST)


def _inlineable_diff(diff_text: str | None) -> str | None:
    """Inline an under-budget diff (including empty text); return None when absent or oversized."""
    if diff_text is None:
        return None
    if not fits_inline_diff_budget(diff_text):
        return None
    return diff_text


def _detect_default_branch(cwd: Path) -> str | None:
    """Return the repository default branch, or None when detection fails."""
    try:
        return git_ops.default_branch(cwd)
    except (BranchNotFoundError, GitError):
        return None


def _git_log(cwd: Path) -> str:
    """Return the current branch log since divergence, or empty text when detection fails."""
    base_branch = _detect_default_branch(cwd)
    if not base_branch:
        return ""
    try:
        return git_ops.log(cwd, base_branch)
    except GitError:
        return ""


def _git_branch(cwd: Path) -> str:
    """Return the current branch name, or empty text when detection fails."""
    try:
        name = git_ops.current_branch(cwd)
    except GitError:
        return ""
    return name or ""


def _recipe_for_work(work: WorkContext) -> TestRecipe | None:
    """Read the routed run recipe, falling back to standalone .daydream/deep.

    Missing, unreadable, malformed, or stale recipes return None.
    """
    try:
        deep = artifact_dir_for(work.repo, allow_standalone=True) / "deep"
    except ArtifactVisibilityError:
        return None
    return load_test_recipe(deep)


def append_extended_facts(prompt: str, recipe: TestRecipe | None) -> str:
    """Append the host-owned resolved-test-recipe block to *prompt*.

    Applied by the host *after* the extension-overridable prompt builder
    returns, so a fork's prompt override cannot drop the facts. ``None``
    leaves the prompt byte-identical.
    """
    if recipe is None:
        return prompt
    return f"{prompt}\n\n{render_test_recipe_block(recipe)}"
