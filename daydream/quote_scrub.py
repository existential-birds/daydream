"""ASCII normalization for smart quotes on the fix-apply path (#687).

Fix agents sometimes write typographic smart quotes (``”`` ``“`` ``’`` ``‘``)
into code comments and string literals. Left alone, those bytes land in the
committed tree and every later review re-surfaces the same typographic finding.
This module provides a deterministic, backend-agnostic scrub: a pure
``str.translate`` transform over the four smart-quote code points plus a
changed-file driver that rewrites only the lines the fix pass added, in place.
"""

import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path

from daydream.generated_files import is_generated_file
from daydream.git_ops import GitError, diff_worktree_against

# U+201C LEFT DOUBLE QUOTATION MARK / U+201D RIGHT DOUBLE QUOTATION MARK -> "
# U+2018 LEFT SINGLE QUOTATION MARK / U+2019 RIGHT SINGLE QUOTATION MARK -> '
_SMART_QUOTE_TABLE = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})

# Extended header lines git emits in place of hunks (binary/rename/mode-only
# diffs). Their presence marks the output as git-structured even without ``+++``
# file headers; anything else in a header-less diff is an external diff driver's
# output, which cannot be attributed.
_NON_HUNK_DIFF_LINES = (
    "diff --git ",
    "index ",
    "Binary files ",
    "--- ",
    '--- "',
    "similarity index ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "old mode ",
    "new mode ",
    "deleted file mode ",
    "new file mode ",
    "\\ No newline at end of file",
)


def normalize_smart_quotes(text: str) -> str:
    """Replace the four smart-quote code points with ASCII straight quotes.

    Pure transform: ``str.translate`` over a fixed mapping, byte-identical for
    any input that already contains none of the four code points. Never
    reformats or reflows the text.
    """
    return text.translate(_SMART_QUOTE_TABLE)


def _attribution_unusable(diff_text: str) -> bool:
    """True when a non-empty *diff_text* carries no parseable unified-diff structure.

    A working-tree diff restricted to real paths must contain ``+++`` file
    headers; their absence means the text came from an external diff driver
    rather than git's unified format, so added-line attribution is impossible —
    and falling back to whole-file normalization would rewrite baseline smart
    quotes in tracked files. Binary-only, rename-only, and mode-change-only
    diffs (git's extended header lines, no hunks) are exempt: their files fail
    UTF-8 decoding or carry no added lines, and untracked siblings still
    normalize whole-file.
    """
    lines = diff_text.splitlines()
    if not any(line.strip() for line in lines):
        return False
    if any(line.startswith("+++") for line in lines):
        return False
    return not all(not line or line.startswith(_NON_HUNK_DIFF_LINES) for line in lines)


def _added_line_numbers(diff_text: str) -> dict[str, set[int]]:
    """Map repo-relative paths in a working-tree diff to the new-file line
    numbers the diff adds.

    Parses ``git diff <ref>`` unified-diff output: within each ``+++`` file
    section it tracks the new-file line counter and records every ``+`` line.
    Context lines advance the counter; deletions (``-``) and header lines do
    not, matching how git numbers the new file. Every file present in the diff
    is a key (mapping to the possibly-empty set of added lines); files absent
    from the diff (untracked new files) are not keys at all.

    A ``+++`` line is treated as a file header only when the previous line was
    its ``--- `` counterpart (git always emits the pair adjacently), so an added
    line whose content starts with ``++ b/`` — rendered identically to a header
    — is parsed as content and cannot re-key the current file.

    Delegates to the shared unified-diff parser in ``daydream.hunk_index``
    (``added_line_numbers(parse_hunks(...))``) so quote_scrub, pr_review and
    coverage all count from the same source and cannot drift.
    """
    from daydream.hunk_index import added_line_numbers, parse_hunks

    return added_line_numbers(parse_hunks(diff_text))
def _normalize_added_lines(text: str, added: set[int]) -> str:
    """Normalize smart quotes only on the 1-based new-file lines in *added*.

    Splits on ``\\n`` (matching git's line accounting; a trailing ``\\r`` in
    CRLF files rides along as line content) and rejoins unchanged, so every
    byte outside an added line is preserved exactly.
    """
    parts = text.split("\n")
    return "\n".join(normalize_smart_quotes(part) if idx in added else part for idx, part in enumerate(parts, start=1))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* without ever exposing a truncated file.

    ``Path.write_bytes`` opens ``wb`` (truncate-then-write), so a mid-write
    failure (ENOSPC, ...) leaves the source file truncated. Instead, write to
    a sibling temp file, preserve the original file's permission bits, and
    ``os.replace`` it into place — the original bytes survive any write
    failure and the final swap is atomic. The temp file is cleaned up on any
    failure.

    Raises:
        OSError: On any write failure (after cleaning up the temp file).
    """
    if path.is_symlink():
        # The driver reads through the link (read_bytes/stat follow it), so
        # os.replace here would swap the link's directory entry for a regular
        # file and destroy the symlink. Write to the link target instead.
        path = path.resolve()
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        try:
            mode = stat.S_IMODE(path.stat().st_mode)
        except OSError:
            mode = None
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
        if mode is not None:
            os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def scrub_smart_quotes_changed_files(
    repo: Path,
    changed_files: Iterable[str],
    *,
    pre_fix_ref: str | None = None,
) -> list[str]:
    """Rewrite agent-written smart quotes to ASCII in changed, non-generated files.

    Best-effort normalization, never a gate: a missing file, a read or write
    failure (``OSError``), or a non-UTF-8 (binary/undecodable) file skips that
    file and the scrub continues — it must never abort a run or block a commit.
    Files ``is_generated_file`` classifies as generated (glob patterns plus
    ``@generated`` / ``DO NOT EDIT`` header markers) and anything under
    ``.daydream/`` are excluded.

    Only lines the fix pass added are rewritten, attributed from the working-
    tree diff against *pre_fix_ref*: pre-existing smart quotes in baseline
    string literals, doc examples, or fixture data are never touched, and the
    rewrite can never turn an untouched single-quoted literal into a syntax
    error. A path absent from that diff is a newly created (untracked) file,
    every line of which is agent-authored, so it is normalized in full. When
    *pre_fix_ref* is None no attribution is attempted and the whole file is
    normalized — production callers always pass it. Writes are atomic (sibling
    temp file + ``os.replace``) so a mid-write failure can never leave a
    truncated source file.

    Args:
        repo: The git working directory (repo-relative paths resolve against it).
        changed_files: Repo-relative paths the fix pass edited.
        pre_fix_ref: Base ref the fix pass edited against; the working-tree
            diff against it attributes the agent-added lines.

    Returns:
        The repo-relative paths whose bytes were rewritten (sorted by the
        caller's responsibility; order here follows the input order).

    Raises:
        GitError: If the attribution diff cannot be computed — including when
            the diff output contains non-UTF-8 bytes (a changed file with
            binary/latin-1 content) and cannot be decoded; callers treat this
            as fail-open (degrade to a warning, never abort a run).
    """
    changed = list(changed_files)
    if pre_fix_ref is None:
        added_lines: dict[str, set[int]] | None = None
    else:
        try:
            diff_text = diff_worktree_against(repo, pre_fix_ref, changed)
        except UnicodeDecodeError as exc:
            # A changed file with non-UTF-8 content makes the attribution diff
            # undecodable. The diff cannot be computed: degrade to the
            # documented GitError fail-open path instead of crashing the run.
            raise GitError(
                f"attribution diff against {pre_fix_ref} is not valid UTF-8: {exc}"
            ) from exc
        if _attribution_unusable(diff_text):
            # Not unified-diff output (external diff driver, ...): attribution
            # is impossible, and whole-file normalization would rewrite baseline
            # smart quotes in tracked files. Fail open through the caller's
            # GitError guard instead.
            raise GitError(
                "attribution diff for smart-quote scrub is not unified-diff output "
                f"(external diff driver?): {diff_text[:120]!r}"
            )
        added_lines = _added_line_numbers(diff_text)
    scrubbed: list[str] = []
    for path in changed:
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
        if added_lines is None:
            normalized = normalize_smart_quotes(decoded)
        else:
            added = added_lines.get(path)
            if added is None:
                # Absent from the attribution diff: a newly created (untracked)
                # file, so every line is agent-authored.
                normalized = normalize_smart_quotes(decoded)
            else:
                normalized = _normalize_added_lines(decoded, added)
        if normalized != decoded:
            try:
                _atomic_write_bytes(file_path, normalized.encode("utf-8"))
            except OSError:
                # Write failure (read-only fs, ENOSPC, permissions, ...): skip
                # and continue — never abort the run.
                continue
            scrubbed.append(path)
    return scrubbed

