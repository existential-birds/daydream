#!/usr/bin/env -S uv run --script
"""Self-contained corpus micro-metric aggregation for the Harbor verifier image.

Stdlib only, never imports ``daydream``: the aggregation contract is loaded at
startup from the canonical ``verifier_core.py`` colocated next to this file in
the compiled stage root (written by the build). Reads one JSONL line per task
— a reward dict or ``null`` (unscored infrastructure failure) — and writes the
pooled micro metrics to the ``-o`` path. Invocation matches Harbor 0.22's
``uv run metric.py -i <rewards.jsonl> -o <metric.json>``; a missing input
raises ``FileNotFoundError`` (uncaught -> nonzero exit, no output written —
fail-closed, never partial). A missing or unloadable ``verifier_core.py``
fails closed the same way, never with a placeholder result.
"""
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable


def _load_aggregate_metrics() -> Callable[[list[dict[str, object] | None]], dict[str, float | int]]:
    """Load ``aggregate_metrics`` from the colocated canonical ``verifier_core.py``.

    The module must be registered in ``sys.modules`` *before* ``exec_module``:
    ``@dataclass`` decoration inside the canonical module resolves
    ``cls.__module__`` through ``sys.modules``, and would otherwise raise
    ``AttributeError``. The registration is removed again in ``finally``, so a
    later bare import of ``verifier_core`` anywhere in the same process cannot
    silently resolve to this compiled copy.
    """
    path = Path(__file__).resolve().parent / "verifier_core.py"
    spec = importlib.util.spec_from_file_location("verifier_core", path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"cannot load canonical verifier core from {path}")
    module = importlib.util.module_from_spec(spec)
    registered = "verifier_core" not in sys.modules
    sys.modules.setdefault("verifier_core", module)
    try:
        spec.loader.exec_module(module)
    finally:
        if registered:
            sys.modules.pop("verifier_core", None)
    return module.aggregate_metrics  # type: ignore[no-any-return]


aggregate_metrics = _load_aggregate_metrics()


def aggregate_rewards_file(path: str) -> dict[str, float | int]:
    """Read a JSONL rewards file (reward dict or null per line) and aggregate.

    A malformed/non-dict line is an unscored infrastructure ``None`` row, never
    an aborting exception. An empty file aggregates to ``task_count == 0`` /
    score 1.0.
    """
    rows: list[dict[str, object] | None] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            rows.append(None)
            continue
        rows.append(obj if isinstance(obj, dict) else None)
    return aggregate_metrics(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate Harbor verifier rewards JSONL into pooled micro metrics.")
    parser.add_argument("-i", "--input", required=True, help="JSONL rewards input (one reward dict or null per task)")
    parser.add_argument("-o", "--output", required=True, help="JSON metric output path (written atomically)")
    args = parser.parse_args(argv)
    result = aggregate_rewards_file(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.parent / f".{out.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
