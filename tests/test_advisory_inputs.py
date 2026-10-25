from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    AdvisoryCandidate,
    SanctionedInputUnavailable,
    prepare_sanctioned_inputs,
    select_advisory_inputs,
    truncate_utf8_to_budget,
)

_MARKER = "\n[diff truncated to fit the prompt budget]\n"

def test_text_exactly_at_the_cap_minus_marker_is_unchanged() -> None:
    text = "x" * (INLINE_DIFF_BUDGET_BYTES - len(_MARKER.encode("utf-8")))
    assert truncate_utf8_to_budget(text, INLINE_DIFF_BUDGET_BYTES, _MARKER) == text

def test_truncation_is_byte_correct_for_multibyte_content() -> None:
    # "é" is 2 UTF-8 bytes: character-index slicing would emit 2× the budget here.
    result = truncate_utf8_to_budget("é" * INLINE_DIFF_BUDGET_BYTES, INLINE_DIFF_BUDGET_BYTES, _MARKER)
    assert len(result.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
    assert "\ufffd" not in result

def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size, encoding="utf-8")
    return path


def _inline_backend() -> SimpleNamespace:
    return SimpleNamespace(read_only_disposable_clone=True, model="fake")

@pytest.mark.parametrize('allowance', [12 * 1024, 64 * 1024])
def test_admitted_set_fits_capture_order_when_a_bigger_label_sorts_first(tmp_path: Path, allowance: int) -> None:
    # "1-large" sorts before "9-small", so the selector's declared order and the
    # capture routine's sorted order disagree. Both fit the emitted inline budget.
    large = _write(tmp_path / "large.md", allowance - 2_000)
    small = _write(tmp_path / "small.md", 190)
    selection = select_advisory_inputs(
        _inline_backend(), tmp_path, [AdvisoryCandidate("9-small", small), AdvisoryCandidate("1-large", large)],
        read_only=True, inline_max_bytes=allowance,
    )
    prepared = prepare_sanctioned_inputs(_inline_backend(), tmp_path, selection.selected_paths(), read_only=True,
                                         inline_max_bytes=allowance)
    assert [item.label for item in prepared.inputs] == ["1-large", "9-small"]
    assert len(prepared.render().encode()) <= allowance
    assert selection.allowance_bytes == allowance

@pytest.mark.parametrize('allowance', [12 * 1024, 64 * 1024])
def test_selected_candidate_that_grows_before_capture_still_fails_closed(tmp_path: Path, allowance: int) -> None:
    path = _write(tmp_path / "summary.md", 100)
    selection = select_advisory_inputs(
        _inline_backend(), tmp_path, [AdvisoryCandidate("exploration-summary", path)], read_only=True,
        inline_max_bytes=allowance,
    )
    _write(path, allowance + 1)
    with pytest.raises(SanctionedInputUnavailable):
        prepare_sanctioned_inputs(_inline_backend(), tmp_path, selection.selected_paths(), read_only=True,
                                  inline_max_bytes=allowance)
