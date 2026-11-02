"""Task: projector emits schema-distinct process-trace and task-only records
behind the ``emit_process_traces`` flag (default off preserves outcome projection
behavior: non-decisive findings land in the adjudication report only)."""

from pathlib import Path
from typing import Any

from daydream.training.corpus_projection.projector import build_frozen_corpus
from tests.harness.record_projection import projection_config, seed_projection_store


def _build(tmp_path: Path, **kw: Any) -> tuple[Path, dict[str, Any]]:
    store = seed_projection_store(tmp_path, dispositions=("accepted", "ambiguous"))
    config = projection_config(store, tmp_path, **kw)
    return config.out_dir, build_frozen_corpus(config)
