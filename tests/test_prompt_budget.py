"""Tests for the dependency-neutral prompt-size policy."""

import os
from pathlib import Path

import pytest

from daydream.backends.pi import PiBackend
from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    PreparedSanctionedInput,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    SanctionedInputUnavailable,
    fits_inline_diff_budget,
    prepare_sanctioned_inputs,
)


def test_inline_diff_budget_uses_utf8_bytes() -> None:
    assert fits_inline_diff_budget("x" * INLINE_DIFF_BUDGET_BYTES)
    assert not fits_inline_diff_budget("x" * (INLINE_DIFF_BUDGET_BYTES + 1))
    assert not fits_inline_diff_budget("あ" * (INLINE_DIFF_BUDGET_BYTES // 2))

def test_exact_phase_artifacts_are_appended_without_rewriting_the_prompt(tmp_path: Path) -> None:

    artifact = tmp_path / "intent.md"
    inputs = PreparedSanctionedInputs(SanctionedInputTransport.EXACT_PATHS,
        (PreparedSanctionedInput("intent", artifact, None, "digest", 1, 2, 3, 4),), object(), tmp_path, True,
    )
    prompt = inputs.render_prompt("Review src/app.py")
    assert prompt.startswith("Review src/app.py")
    assert prompt.endswith(f"- intent: {artifact}")
    assert inputs.render_prompt(prompt) == prompt

def test_pi_diff_reference_has_separate_admission_and_no_finalization_capture(tmp_path: Path) -> None:
    diff = tmp_path / "diff.patch"
    diff.write_text("UNIQUE_DIFF_SENTINEL\n" + "x" * 3_690_129)
    intent = tmp_path / "intent.md"
    intent.write_text("Confirmed intent")
    backend = PiBackend(model="fixture")
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"diff": diff, "intent": intent}, read_only=True)
    assert str(diff) in prepared.render()
    assert "UNIQUE_DIFF_SENTINEL" not in prepared.render()
    assert prepared.inputs[0].text is None
    prepared.revalidate(backend, tmp_path, True)
    context = prepared.finalization_text(backend, tmp_path, True)
    assert "Confirmed intent" in context
    assert "UNIQUE_DIFF_SENTINEL" not in context

@pytest.mark.parametrize("change", ["mutation", "same_stat", "replace", "symlink", "missing", "utf8",
                                    "backend", "cwd", "mode"])
def test_pi_diff_reference_rejects_changed_content_identity_or_binding(tmp_path: Path, change: str) -> None:
    diff = tmp_path / "diff.patch"
    diff.write_text("before")
    backend = PiBackend(model="fixture")
    prepared = prepare_sanctioned_inputs(backend, tmp_path, {"diff": diff}, read_only=True)
    metadata = diff.stat()
    cwd = tmp_path
    mode = True
    if change in {"mutation", "same_stat"}:
        diff.write_text("AFTER!")
        if change == "same_stat":
            os.utime(diff, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    elif change == "replace":
        other = tmp_path / "other"
        other.write_text("before")
        other.replace(diff)
    elif change == "symlink":
        other = tmp_path / "other"
        diff.rename(other)
        diff.symlink_to(other)
    elif change == "missing":
        diff.unlink()
    elif change == "utf8":
        diff.write_bytes(b"\xff")
    elif change == "backend":
        backend = PiBackend(model="fixture")
    elif change == "cwd":
        cwd = tmp_path / "other"
        cwd.mkdir()
    else:
        mode = False
    with pytest.raises(SanctionedInputUnavailable):
        prepared.revalidate(backend, cwd, mode)
    with pytest.raises(SanctionedInputUnavailable):
        prepared.finalization_text(backend, cwd, mode)

@pytest.mark.parametrize("pi, inline, label", [(False, False, "diff"), (True, True, "diff"), (True, False, "intent")])
def test_diff_reference_policy_preserves_other_input_limits(tmp_path: Path, pi: bool, inline: bool, label: str,
) -> None:
    class IsolatedPi(PiBackend):
        sandbox = True
    backend = IsolatedPi(model="fixture") if inline else PiBackend(model="fixture") if pi else object()
    path = tmp_path / "input.txt"
    path.write_text("x" * 3_690_129)
    with pytest.raises(SanctionedInputUnavailable):
        prepare_sanctioned_inputs(backend, tmp_path, {label: path}, read_only=True)
