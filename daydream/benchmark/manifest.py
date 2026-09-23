"""One strict, bounded-error reader for benchmark manifests."""

from __future__ import annotations

from pathlib import Path

from daydream.benchmark import schema, storage


def load_benchmark_manifest(
    root: Path, *, canonicalize_case_order: bool = False
) -> schema.BenchmarkManifest:
    """Read once, optionally canonicalize case rows, and model-validate."""
    root = Path(root)
    try:
        loaded = storage.load_yaml_strict(root / "benchmark.yaml")
        if canonicalize_case_order and "cases" in loaded:
            rows = loaded["cases"]
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
            loaded["cases"] = [row for _model, row in paired]
        return schema.BenchmarkManifest.model_validate(loaded)
    except (storage.WorkspaceCorrupt, OSError, ValueError, TypeError) as exc:
        raise storage.WorkspaceCorrupt(f"{root}: invalid benchmark.yaml") from exc
