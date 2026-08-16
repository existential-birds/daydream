"""ASCII normalization for smart quotes on the fix-apply path (#687).

Fix agents sometimes write typographic smart quotes (``”`` ``“`` ``’`` ``‘``)
into code comments and string literals. Left alone, those bytes land in the
committed tree and every later review re-surfaces the same typographic finding.
This module provides a deterministic, backend-agnostic scrub: a pure
``str.translate`` transform over the four smart-quote code points plus a
changed-file driver that rewrites only the changed-file set in place.
"""

from collections.abc import Iterable
from pathlib import Path

from daydream.generated_files import is_generated_file

# U+201C LEFT DOUBLE QUOTATION MARK / U+201D RIGHT DOUBLE QUOTATION MARK -> "
# U+2018 LEFT SINGLE QUOTATION MARK / U+2019 RIGHT SINGLE QUOTATION MARK -> '
_SMART_QUOTE_TABLE = str.maketrans(
    {"\u201C": '"', "\u201D": '"', "\u2018": "'", "\u2019": "'"}
)


def normalize_smart_quotes(text: str) -> str:
    """Replace the four smart-quote code points with ASCII straight quotes.

    Pure transform: ``str.translate`` over a fixed mapping, byte-identical for
    any input that already contains none of the four code points. Never
    reformats or reflows the text.
    """
    return text.translate(_SMART_QUOTE_TABLE)


def scrub_smart_quotes_changed_files(repo: Path, changed_files: Iterable[str]) -> list[str]:
    """Rewrite smart quotes to ASCII in every changed, non-generated text file.

    Best-effort normalization, never a gate: a missing file, a read failure
    (``OSError``), or a non-UTF-8 (binary/undecodable) file skips that file and
    the scrub continues — it must never abort a run or block a commit. Files
    ``is_generated_file`` classifies as generated (glob patterns plus
    ``@generated`` / ``DO NOT EDIT`` header markers) and anything under
    ``.daydream/`` are excluded.

    Args:
        repo: The git working directory (repo-relative paths resolve against it).
        changed_files: Repo-relative paths the fix pass edited.

    Returns:
        The repo-relative paths whose bytes were rewritten (sorted by the
        caller's responsibility; order here follows the input order).
    """
    scrubbed: list[str] = []
    for path in changed_files:
        if path.startswith(".daydream/"):
            continue
        file_path = repo / path
        try:
            content = file_path.read_bytes()
        except OSError:
            # Missing or unreadable: skip and continue.
            continue
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            # Binary / non-UTF-8 file: skip and continue.
            continue
        if is_generated_file(path, decoded):
            continue
        normalized = normalize_smart_quotes(decoded)
        if normalized != decoded:
            file_path.write_bytes(normalized.encode("utf-8"))
            scrubbed.append(path)
    return scrubbed
