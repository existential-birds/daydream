# Reward calibration

`daydream corpus calibrate-reward --help` owns flags. Calibration is offline
analysis of candidate weights, never permission to mutate live reward defaults.

Input is a frozen `corpus build` projection: `corpus.jsonl`, `lineage.json`,
and `SHA256SUMS`, plus gold labels and breakdowns keyed by `record_id`.
The loader validates digests, versions, unique identities, split assignments,
repository provenance, and C5 exclusions before computing statistics.
Stage-0 score inputs also require their matching model digest.

Use [the calibration fixture](../tests/fixtures/training/calibration/) for a
complete example and [calibration.py](../daydream/training/calibration.py) for
the wire contract. Identical inputs and seed produce identical artifact bytes;
outputs pin input digests and versions. Existing run IDs refuse replacement.
