"""Build 50 training records through the immutable store and offline projector."""
from __future__ import annotations

import shutil
from pathlib import Path

from daydream.dataset import LocalRecordStore
from daydream.training.corpus_projection.projector import build_frozen_corpus
from tests.harness.record_projection import add_projection_run, projection_config

SALT = "issue-1081-fixture-salt"
HOLDOUT_RATE = 0.2
VAL_RATE = 0.2


def build_projection_50(tmp_path: Path) -> Path:
    store = LocalRecordStore(tmp_path / "projection-fixture" / "records")
    for index in range(23):
        add_projection_run(store, run_id=f"sess-gold-{index:02d}", dispositions=("accepted", "rejected"),
                           repo_slug=f"owner/repo-{index % 3}")
    for index in range(2):
        add_projection_run(store, run_id=f"sess-amb-{index}", dispositions=("ambiguous",))
    config = projection_config(store, tmp_path / "projection-fixture", salt=SALT, holdout_rate=HOLDOUT_RATE,
                               val_rate=VAL_RATE, emit_process_traces=True)
    result = build_frozen_corpus(config)
    assert result["total"] == 50
    return config.out_dir


def main() -> None:
    """Build in scratch space and copy content to --out; the loader digest depends on bytes, not
    mtimes.
    """
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
        help="Destination projection directory (e.g. tests/fixtures/training/projection-50)",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        proj_dir = build_projection_50(Path(tmp).resolve())
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.copytree(proj_dir, args.out)
    print(f"committed projection fixture written to {args.out}")


if __name__ == "__main__":
    main()


