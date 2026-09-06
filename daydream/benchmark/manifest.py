"""One strict, bounded-error reader for benchmark manifests."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.benchmark import schema, storage


@dataclass(frozen=True)
class LoadedBenchmarkManifest:
    raw: dict[str, Any]
    model: schema.BenchmarkManifest


def load_benchmark_manifest(
    root: Path, *, canonicalize_case_order: bool = False
) -> LoadedBenchmarkManifest:
    """Read once, optionally canonicalize case rows, and model-validate."""
    root = Path(root)
    try:
        loaded = storage.load_yaml_strict(root / "benchmark.yaml")
        raw = copy.deepcopy(loaded)
        if canonicalize_case_order and "cases" in raw:
            rows = raw["cases"]
            if not isinstance(rows, list):
                raise TypeError("cases must be a list")
            validated = [schema.CaseIndexEntry.model_validate(row) for row in rows]
            paired = sorted(
                zip(validated, rows, strict=True),
                key=lambda pair: (
                    pair[0].pr_number,
                    schema.head_sha_from_case_id(pair[0].case_id),
                    pair[0].case_id,
                ),
            )
            raw["cases"] = [row for _model, row in paired]
        model = schema.BenchmarkManifest.model_validate(raw)
        raw_ids = [row["case_id"] for row in raw.get("cases", [])]
        if raw_ids != [row.case_id for row in model.cases]:
            raise ValueError("raw/model case order mismatch")
        return LoadedBenchmarkManifest(raw=raw, model=model)
    except (storage.WorkspaceCorrupt, OSError, ValueError, TypeError) as exc:
        raise storage.WorkspaceCorrupt(f"{root}: invalid benchmark.yaml") from exc
