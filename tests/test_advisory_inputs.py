from pathlib import Path

from daydream.prompt_budget import (
    INLINE_DIFF_BUDGET_BYTES,
    PreparedSanctionedInput,
    PreparedSanctionedInputs,
    SanctionedInputTransport,
    inline_section_emitted_bytes,
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
