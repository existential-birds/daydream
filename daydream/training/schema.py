"""Schema version constants for training JSONL records.

Mirrors `MANIFEST_SCHEMA_VERSION` in `daydream.archive.manifest`. The committed
JSONSchema artifact at `daydream/training/schema/v1.json` is the canonical
contract downstream consumers pin against.

Bump rule (per plan §8):
- Adding a non-required field: no version bump.
- Removing a field, renaming, changing types, changing semantics: bump to the
  next integer, ship the new JSONSchema artifact alongside the old one, and
  leave existing versioned artifacts in place for old consumers.

v2 (`corpus-v2`, per-finding records) ships alongside v1; v1 is never modified.
"""

from pathlib import Path

TRAINING_SCHEMA_VERSION: str = "1"
TRAINING_SCHEMA_V1_PATH: Path = Path(__file__).parent / "schema" / "v1.json"

TRAINING_SCHEMA_V2_VERSION: str = "2"
TRAINING_SCHEMA_V2_PATH: Path = Path(__file__).parent / "schema" / "v2.json"
