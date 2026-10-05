# Pin dataset snapshot membership

Saved dataset snapshots identify an exact, immutable selection of run records and observations, retaining the shard digests and temporal cutoffs used for that selection. Later feedback or corrected judgments belong to a new snapshot so that an earlier analysis or training experiment can be repeated with the same evidence. This preserves reproducibility while allowing the collected evidence to grow.
