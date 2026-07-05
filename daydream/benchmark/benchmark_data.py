"""Load, inject, and save the withmartian ``benchmark_data.json`` corpus.

The corpus is a JSON object keyed by golden upstream PR URL. Each entry carries
``golden_comments`` (the recall denominator) and ``reviews`` — a flat list of
per-tool review objects scanned/filtered by the ``tool`` field. The daydream
reviewer is not present upstream; it is injected here as a synthetic review so
the existing scoring steps pick it up automatically. The injected/queried tool
label is configurable (defaulting to ``DAYDREAM_TOOL``) so distinct reviewer
backends can coexist under separate labels (e.g. ``daydream-glm``).

See ``research/benchmark-pipeline.md`` §1 for the full shape.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

DAYDREAM_TOOL = "daydream"


def load_benchmark_data(path: str | Path) -> dict[str, Any]:
    """Load the benchmark corpus from ``path``.

    Returns:
        The parsed top-level dict, keyed by golden PR URL.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_benchmark_data(path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write the benchmark corpus to ``path``.

    Writes to a sibling temp file then ``os.replace`` to avoid truncating the
    committed corpus on a partial write. Uses ``indent=2`` and
    ``ensure_ascii=False`` and preserves dict key order.

    A ``benchmark_data.json.lock`` file serialises concurrent writers within
    the same OS (``fcntl.LOCK_EX``). Two ``daydream bench`` processes sharing
    the same ``--benchmark-repo`` path would otherwise silently clobber each
    other's injections: both read the old corpus, both write a partial sweep,
    and the second ``os.replace`` wins. The lock does not protect across
    machines or network filesystems — run one bench sweep per benchmark repo
    at a time (see ``docs/benchmark.md``).
    """
    dest = Path(path)
    lock_path = dest.parent / f"{dest.name}.lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dest.parent,
            prefix=f"{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tmp = tf.name
            tf.write(json.dumps(data, indent=2, ensure_ascii=False))
        os.replace(tmp, dest)
        fcntl.flock(lf, fcntl.LOCK_UN)


def has_daydream_review(entry: dict[str, Any], *, tool: str = DAYDREAM_TOOL) -> bool:
    """Report whether a corpus entry already carries a synthetic daydream review."""
    return any(review.get("tool") == tool for review in entry.get("reviews", []))


def inject_daydream_review(
    data: dict[str, Any],
    golden_url: str,
    comments: list[dict[str, Any]],
    *,
    force: bool,
    tool: str = DAYDREAM_TOOL,
) -> bool:
    """Inject a synthetic daydream review into one golden PR entry.

    Args:
        data: The benchmark corpus dict (mutated in place).
        force: When a daydream review already exists, replace its
            ``review_comments`` instead of leaving it untouched.

    Returns:
        ``True`` if the corpus was modified, ``False`` if an existing daydream
        review was left in place (idempotent no-op).

    Raises:
        KeyError: If ``golden_url`` is not present in ``data``.
    """
    entry = data[golden_url]
    reviews = entry["reviews"]

    for review in reviews:
        if review.get("tool") == tool:
            if not force:
                return False
            review["review_comments"] = comments
            return True

    reviews.append(
        {
            "tool": tool,
            "repo_name": tool,
            "pr_url": golden_url,
            "review_comments": comments,
        }
    )
    return True
