"""Normalize four smart-quote characters on agent-added lines in changed source files."""

import stat
from collections.abc import Iterable
from pathlib import Path

from daydream.generated_files import is_generated_file
from daydream.git_ops import GitError, diff_worktree_against
from daydream.json_utils import atomic_write_bytes

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
    """Translate four smart quotes to ASCII without reflowing or changing other text."""
    return text.translate(_SMART_QUOTE_TABLE)


def _attribution_unusable(diff_text: str) -> bool:
    """Reject nonempty external diff output that cannot attribute added lines.

    Git binary/rename/mode-only headers are valid exemptions. Missing attribution
    must never widen normalization to baseline lines in tracked files."""
    lines = diff_text.splitlines()
    if not any(line.strip() for line in lines):
        return False
    if any(line.startswith("+++") for line in lines):
        return False
    return not all(not line or line.startswith(_NON_HUNK_DIFF_LINES) for line in lines)


def _normalize_added_lines(text: str, added: set[int]) -> str:
    """Normalize only the 1-based added lines; preserve other bytes and CRLF endings."""
    parts = text.split("\n")
    return "\n".join(normalize_smart_quotes(part) if idx in added else part for idx, part in enumerate(parts, start=1))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace content while retaining permissions and symlink targets.

    A sibling temp prevents truncation on write failure; parent fsync persists the
    rename. OSError propagates after best-effort temp cleanup."""
    if path.is_symlink():
        # The driver reads through the link (read_bytes/stat follow it), so
        # os.replace here would swap the link's directory entry for a regular
        # file and destroy the symlink. Write to the link target instead.
        path = path.resolve()
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        mode = None
    atomic_write_bytes(path, data, fsync=True, dir_fsync=True, mode=mode)


def scrub_smart_quotes_changed_files(
    repo: Path,
    changed_files: Iterable[str],
    *,
    pre_fix_ref: str | None = None,
) -> list[str]:
    """Normalize agent-added lines, returning rewritten paths in input order.

    Skip generated/.daydream files and per-file I/O or UTF-8 failures. With pre_fix_ref,
    attribute tracked additions from the diff; paths absent from it are normalized
    whole as untracked files. Without a ref, normalize whole files. Production passes
    a ref. Unusable/non-UTF-8 attribution raises GitError for the caller’s warning
    path. Atomic writes preserve source bytes on pre-publication failure."""
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
        from daydream.hunk_index import added_line_numbers, parse_hunks

        added_lines = added_line_numbers(parse_hunks(diff_text))
    scrubbed: list[str] = []
    for path in changed:
        if path.startswith(".daydream/"):
            continue
        file_path = repo / path
        try:
            decoded = file_path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if is_generated_file(path, decoded):
            continue
        added = added_lines.get(path) if added_lines is not None else None
        normalized = normalize_smart_quotes(decoded) if added is None else _normalize_added_lines(decoded, added)
        if normalized != decoded:
            try:
                _atomic_write_bytes(file_path, normalized.encode("utf-8"))
            except OSError:
                # Write failure (read-only fs, ENOSPC, permissions, ...): skip
                # and continue — never abort the run.
                continue
            scrubbed.append(path)
    return scrubbed

