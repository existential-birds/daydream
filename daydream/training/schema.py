"""Schema constants for training JSONL records.

The committed JSONSchema artifact at `daydream/training/schema/record-schema.json`
is the canonical contract downstream consumers pin against, and the
curation-manifest schema at `daydream/training/schema/curation-manifest.json`
pins the admission ledger.

Bump rule (per plan §8):
- Adding a non-required field: no version bump.
- Removing a field, renaming, changing types, changing semantics: ship a new
  JSONSchema artifact under a new neutral name and update every consumer in the
  same change set (greenfield: no production training run has ever completed,
  so there are no old consumers to keep alive).
"""

from pathlib import Path

TRAINING_SCHEMA_VERSION: str = "1"

TRAINING_RECORD_SCHEMA_PATH: Path = Path(__file__).parent / "schema" / "record-schema.json"
