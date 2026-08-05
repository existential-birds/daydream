"""Dependency-neutral prompt-size policy shared by review prompt builders."""

# Upper bound for inlined diff text. Above this bound, prompts retain an
# on-disk diff pointer rather than embedding the diff.
INLINE_DIFF_BUDGET_BYTES = 12_288


def fits_inline_diff_budget(text: str) -> bool:
    """Whether ``text`` fits the UTF-8 byte budget for an inlined diff."""
    return len(text.encode("utf-8")) <= INLINE_DIFF_BUDGET_BYTES
