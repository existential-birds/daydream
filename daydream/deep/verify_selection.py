"""Host-side, pure inputs for the recommendation-verifier selection decision.

This module is the single home of the deterministic, fail-open reader the
verify-selection predicate consumes: :func:`changed_text_at` answers "what text
did the diff add at (or around) the line this finding cites?" from the run's own
``diff.patch`` text. It walks the unified diff once, tracks the current
post-state file (the ``+++ b/`` header, unquoted exactly as
``daydream/hunk_index.py`` unquotes it) and the new-side line counter, and
returns the newline-joined text of the ``+`` content lines in the hunk covering
``line`` for ``file``. A cited line that is not itself an added line yields the
empty string, as does a missing file, a malformed or empty diff, or a
non-positive line number — the classifier reads ``""`` as "no changed-line
signal", never as a skip signal (Pattern B, fail-open).

The reader is deliberately range-free: a cited line the hunk index snapped is
already reflected in ``item["line"]`` before this runs, so importing the index
for ranges would double-apply the snap. No other artifact is read, no
randomness or time is consulted, and the function is total: the only failure
mode is unparseable input, which is the documented ``""`` answer.
"""

from __future__ import annotations

from daydream.hunk_index import _HUNK_HEADER, _header_path


def changed_text_at(diff_text: str, file: str, line: object) -> str:
    """Return the added text of the hunk a cited ``file``/``line`` falls in.

    Walks ``diff_text`` once, resolving the current post-state path from each
    ``+++`` header and the new-side line counter from each hunk header. When
    ``line`` is one of the added (``+``) lines of ``file``, returns the
    newline-joined text of every added line in that same hunk (a hunk's added
    text is the unit the classifier reads); otherwise returns ``""``.

    Args:
        diff_text: Unified-diff text, as written to ``diff.patch``.
        file: Repo-relative path to look up.
        line: New-side line number the finding cites.

    Returns:
        The added lines' text, or ``""`` when there is no changed-line signal.
    """
    if not isinstance(line, int) or isinstance(line, bool) or line <= 0:
        return ""
    if not file:
        return ""

    # new_line -> (hunk ordinal, added text), collected only for the queried file.
    added_by_line: dict[int, tuple[int, str]] = {}
    current_file: str | None = None
    current_hunk: int | None = None
    hunk_count = 0
    new_line = 0
    prev_old_header = False
    for raw in diff_text.splitlines():
        if raw.startswith(("--- ", '--- "')):
            prev_old_header = True
            continue
        if raw.startswith("+++ ") and prev_old_header:
            prev_old_header = False
            current_file = _header_path(raw)
            current_hunk = None
            new_line = 0
            continue
        prev_old_header = False
        header = _HUNK_HEADER.match(raw)
        if raw.startswith("@@") and header:
            new_start = int(header.group(3))
            new_count = int(header.group(4)) if header.group(4) else 1
            new_line = new_start
            if new_count == 0:
                # Empty new-side range (pure deletion): no added lines to read.
                current_hunk = None
                continue
            current_hunk = hunk_count
            hunk_count += 1
        elif current_file == file and raw.startswith("+"):
            if current_hunk is not None:
                added_by_line[new_line] = (current_hunk, raw[1:])
            new_line += 1
        elif raw.startswith("-"):
            continue
        elif raw.startswith(" "):
            new_line += 1

    hit = added_by_line.get(line)
    if hit is None:
        return ""
    wanted_hunk = hit[0]
    return "\n".join(
        text
        for added_line, (hunk, text) in sorted(added_by_line.items())
        if hunk == wanted_hunk
    )
