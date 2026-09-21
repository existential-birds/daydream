"""Tests for the dependency-neutral prompt-size policy."""

from pathlib import Path


def test_inline_diff_budget_uses_utf8_bytes() -> None:
    """The shared policy accepts its boundary and rejects oversized text."""
    from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES, fits_inline_diff_budget

    assert fits_inline_diff_budget("x" * INLINE_DIFF_BUDGET_BYTES)
    assert not fits_inline_diff_budget("x" * (INLINE_DIFF_BUDGET_BYTES + 1))
    assert not fits_inline_diff_budget("あ" * (INLINE_DIFF_BUDGET_BYTES // 2))


def test_exact_phase_artifacts_do_not_restrict_scoped_repository_reads(tmp_path: Path) -> None:
    from daydream.prompt_budget import PreparedSanctionedInput, PreparedSanctionedInputs, SanctionedInputTransport

    artifact = tmp_path / "intent.md"
    inputs = PreparedSanctionedInputs(
        SanctionedInputTransport.EXACT_PATHS,
        (PreparedSanctionedInput("intent", artifact, None, "digest", 1, 2, 3, 4),),
        object(), tmp_path, True,
    )
    prompt = inputs.render_prompt("Review src/app.py")
    assert "repository source reads" in prompt
    assert prompt.endswith(f"- intent: {artifact}")
    assert inputs.render_prompt(prompt) == prompt
