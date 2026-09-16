from pathlib import Path
from types import SimpleNamespace

import pytest

from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES,
    AdvisoryCandidate,
    PreparedSanctionedInput,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    SanctionedInputUnavailable,
    inline_section_emitted_bytes,
    prepare_sanctioned_inputs,
    select_advisory_inputs,
    truncate_utf8_to_budget,
)

_MARKER = "\n[diff truncated to fit the prompt budget]\n"


def test_text_within_budget_is_returned_unchanged() -> None:
    assert truncate_utf8_to_budget("abc", INLINE_DIFF_BUDGET_BYTES, _MARKER) == "abc"


def test_text_exactly_at_the_cap_minus_marker_is_unchanged() -> None:
    text = "x" * (INLINE_DIFF_BUDGET_BYTES - len(_MARKER.encode("utf-8")))
    assert truncate_utf8_to_budget(text, INLINE_DIFF_BUDGET_BYTES, _MARKER) == text


def test_truncated_text_keeps_the_marker_inside_the_budget() -> None:
    result = truncate_utf8_to_budget("x" * (INLINE_DIFF_BUDGET_BYTES * 2), INLINE_DIFF_BUDGET_BYTES, _MARKER)
    assert result.endswith(_MARKER)
    assert len(result.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES


def test_truncation_is_byte_correct_for_multibyte_content() -> None:
    # "é" is 2 UTF-8 bytes: character-index slicing would emit 2× the budget here.
    result = truncate_utf8_to_budget("é" * INLINE_DIFF_BUDGET_BYTES, INLINE_DIFF_BUDGET_BYTES, _MARKER)
    assert len(result.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
    assert "\ufffd" not in result


def test_inline_section_bytes_match_the_real_renderer() -> None:
    def _item(label: str, text: str) -> PreparedSanctionedInput:
        return PreparedSanctionedInput(
            label=label, path=Path("/nowhere"), text=text, sha256="0" * 64,
            device=0, inode=0, size=len(text.encode("utf-8")), mtime_ns=0,
        )

    items = (_item("exploration-summary", "s" * 614), _item("exploration-dependencies", "é" * 11))
    prepared = PreparedSanctionedInputs(
        transport=SanctionedInputTransport.INLINE,
        inputs=items,
        backend_identity=object(),
        cwd=Path("/nowhere"),
        read_only=True,
    )
    assert inline_section_emitted_bytes([(item.label, item.size) for item in items]) == len(
        prepared.render().encode("utf-8")
    )


def _write(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x" * size, encoding="utf-8")
    return path


def _inline_backend() -> SimpleNamespace:
    return SimpleNamespace(read_only_disposable_clone=True, model="fake")


def test_over_budget_advisory_candidate_is_omitted_not_raised(tmp_path: Path) -> None:
    candidates = [
        AdvisoryCandidate("exploration-summary", _write(tmp_path / "s.md", 614)),
        AdvisoryCandidate("exploration-dependencies", _write(tmp_path / "d.md", 4_306)),
        AdvisoryCandidate("exploration-affected-files", _write(tmp_path / "a.md", 23_684)),
    ]
    selection = select_advisory_inputs(_inline_backend(), tmp_path, candidates, read_only=True)
    assert list(selection.selected_paths()) == ["exploration-summary", "exploration-dependencies"]
    assert [(o.label, o.size, o.reason) for o in selection.omitted] == [
        ("exploration-affected-files", 23_684, "exceeds-byte-budget")
    ]
    assert selection.admitted_bytes == 4_920


def test_admitted_set_fits_capture_order_when_a_bigger_label_sorts_first(tmp_path: Path) -> None:
    # "1-large" sorts before "9-small", so the selector's declared order and the
    # capture routine's sorted order disagree. Both fit the emitted inline budget.
    large = _write(tmp_path / "large.md", SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES - 2_000)
    small = _write(tmp_path / "small.md", 190)
    selection = select_advisory_inputs(
        _inline_backend(), tmp_path,
        [AdvisoryCandidate("9-small", small), AdvisoryCandidate("1-large", large)],
        read_only=True,
    )
    prepared = prepare_sanctioned_inputs(
        _inline_backend(), tmp_path, selection.selected_paths(), read_only=True
    )
    assert [item.label for item in prepared.inputs] == ["1-large", "9-small"]


def test_the_admitted_section_fits_the_inline_byte_budget(tmp_path: Path) -> None:
    # At this size the first candidate's *rendered* section is 12,287 bytes; the
    # label/tag/separator overhead is what makes the second candidate overflow.
    at_cap = _write(tmp_path / "at-cap.md", SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES - 95)
    one_more = _write(tmp_path / "one-more.md", 1)
    selection = select_advisory_inputs(
        _inline_backend(), tmp_path,
        [AdvisoryCandidate("a", at_cap), AdvisoryCandidate("b", one_more)],
        read_only=True,
    )
    prepared = prepare_sanctioned_inputs(
        _inline_backend(), tmp_path, selection.selected_paths(), read_only=True
    )
    assert len(prepared.render().encode("utf-8")) <= SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES
    assert [o.label for o in selection.omitted] == ["b"]


def test_exact_paths_allows_a_large_file_within_todays_limits(tmp_path: Path) -> None:
    big = _write(tmp_path / "diff.patch", 900_000)          # < 1 MiB per-file limit
    over = _write(tmp_path / "over.patch", 1_048_577)       # > 1 MiB per-file limit
    selection = select_advisory_inputs(
        SimpleNamespace(model="fake"), tmp_path,
        [AdvisoryCandidate("diff", big), AdvisoryCandidate("hunk-index", over)],
        read_only=True,
    )
    assert list(selection.selected_paths()) == ["diff"]
    assert [o.reason for o in selection.omitted] == ["exceeds-file-limit"]


def test_missing_advisory_candidate_is_omitted_as_unavailable(tmp_path: Path) -> None:
    selection = select_advisory_inputs(
        _inline_backend(), tmp_path,
        [AdvisoryCandidate("exploration-summary", tmp_path / "absent.md")],
        read_only=True,
    )
    assert selection.selected_paths() == {}
    assert [(o.label, o.reason) for o in selection.omitted] == [("exploration-summary", "unavailable")]


def test_selected_candidate_that_grows_before_capture_still_fails_closed(tmp_path: Path) -> None:
    path = _write(tmp_path / "summary.md", 100)
    selection = select_advisory_inputs(
        _inline_backend(), tmp_path, [AdvisoryCandidate("exploration-summary", path)], read_only=True
    )
    _write(path, SANCTIONED_INLINE_INPUT_AGGREGATE_MAX_BYTES + 1)
    with pytest.raises(SanctionedInputUnavailable):
        prepare_sanctioned_inputs(_inline_backend(), tmp_path, selection.selected_paths(), read_only=True)
