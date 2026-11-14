"""Deterministic language-stack shards named <stack>#<index>.

The structure meta-stack retains its original object and all changed files;
it is neither split nor counted against the fan-out cap.
"""

from __future__ import annotations

from daydream.deep.diff import iter_diff_blocks


def file_change_bytes(diff: str) -> dict[str, int]:
    """Map changed paths to UTF-8 block sizes; the first block wins.

    Files absent from the map still receive a one-byte shard weight.
    """
    sizes: dict[str, int] = {}
    for path, block in iter_diff_blocks(diff):
        sizes.setdefault(path, len(block.encode("utf-8")))
    return sizes


