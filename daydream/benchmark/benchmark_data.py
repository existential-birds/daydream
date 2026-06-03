"""Load, inject, and save the withmartian ``benchmark_data.json`` corpus.

The corpus is a JSON object keyed by golden upstream PR URL. Each entry carries
``golden_comments`` (the recall denominator) and ``reviews`` — a flat list of
per-tool review objects scanned/filtered by the ``tool`` field. ``daydream`` is
not present upstream; it is injected here as a synthetic review so the existing
scoring steps pick it up automatically.

See ``research/benchmark-pipeline.md`` §1 for the full shape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DAYDREAM_TOOL = "daydream"


def load_benchmark_data(path: str | Path) -> dict[str, Any]:
    """Load the benchmark corpus from ``path``.

    Args:
        path: Filesystem path to ``benchmark_data.json``.

    Returns:
        The parsed top-level dict, keyed by golden PR URL.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_benchmark_data(path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write the benchmark corpus to ``path``.

    Writes to a sibling temp file then ``os.replace`` to avoid truncating the
    committed corpus on a partial write. Uses ``indent=2`` and
    ``ensure_ascii=False`` and preserves dict key order.

    Args:
        path: Destination path for ``benchmark_data.json``.
        data: The corpus dict to serialize.
    """
    dest = Path(path)
    tmp = dest.with_name(f"{dest.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, dest)


def inject_daydream_review(
    data: dict[str, Any],
    golden_url: str,
    comments: list[dict[str, Any]],
    *,
    force: bool,
) -> bool:
    """Inject a synthetic ``daydream`` review into one golden PR entry.

    Args:
        data: The benchmark corpus dict (mutated in place).
        golden_url: The golden upstream PR URL key.
        comments: ``review_comments`` payload for the daydream review.
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
        if review.get("tool") == DAYDREAM_TOOL:
            if not force:
                return False
            review["review_comments"] = comments
            return True

    reviews.append(
        {
            "tool": DAYDREAM_TOOL,
            "repo_name": DAYDREAM_TOOL,
            "pr_url": golden_url,
            "review_comments": comments,
        }
    )
    return True
