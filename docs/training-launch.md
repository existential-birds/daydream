# Training

Read before changing or launching the staged pipeline. Production training has
not completed; fixture validation establishes pipeline behavior, not model quality
or GPU memory/cost estimates. Recipes and defaults live in [rl/train](../rl/train/)
and [training](../daydream/training/); RL work also requires
[its package guidance](../rl/daydream_review/README.md) and `make rl-check`.

```bash
daydream corpus build --store RECORD_STORE --snapshot-id SNAPSHOT_ID --out PROJECTION_DIR/corpus.jsonl
daydream train --projection PROJECTION_DIR --out OUT_DIR --dry-run
```

Use the dry run to validate inputs before a real launch. The stage order is
reward construction/gating → SFT cold-start → deterministic RFT → online GRPO.
The projection is the only training input. Preserve its `_SUCCESS` marker,
lineage, split digests, original base/head/diff, and benchmark exclusions.
Load through `training.stacks.load_v2_projection`; revalidate split and corpus
identity on every load/resume. Never silently re-split or train on drifted data.

Capture and sealed RL share the intrinsic scoring owner; harvest adds outcome
evidence separately. Read [scoring](../daydream/training/reward.py) for the current
reward/version and [calibration](calibration.md) before changing weights.
Keep posterior cost separate. Pin model, data/split digests, recipe, hardware,
wall time, and cost in the actual run manifest; fixture measurements do not
establish comparative quality or production launch sizing.
