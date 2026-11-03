"""Tests for the dependency-neutral prompt-size policy."""

from pathlib import Path

from daydream.backends.pi import PiBackend
from daydream.prompt_budget import (
    prepare_sanctioned_inputs,
)


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
