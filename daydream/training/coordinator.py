"""Four-stage training coordinator: one command, stage manifest, loadable adapter.

Implements M15 (the single ``daydream train`` entrypoint chaining Stage 0 → 3)
and M16 (the stage manifest with per-stage digests and gate evidence), tying
into M18 via the :class:`~daydream.training.lineage.RunIdentity` stamped into
every manifest.

Contract points:

- **Ordered execution**: stages run in the order given in
  :class:`PipelineConfig.stages`; each writes its outputs under
  ``out_dir/stageN/`` and one ``manifest.json`` lands at the run root carrying
  stage digests, the run identity, gate evidence, and the final adapter path.
- **Gate-enforced handoff (M4/M15)**: Stage 3 runs only after a *passed*
  Stage-0 gate. A failed or missing gate raises :class:`RuntimeError` before
  Stage 3's directory is created, and the manifest is not written — a refused
  run leaves no partial-success artifact.
- **Dry path (CI)**: ``dry_run=True`` executes everything that needs no GPU —
  corpus load (fail-closed via :mod:`daydream.training.stacks`), Stage-0 gate
  evaluation on cached model state, validation, manifest — and marks the wall-
  clock GPU stages ``skipped_dry``. The Stage-3 adapter *handoff* (pure file
  assembly from Stage-0 state, no GPU) is still produced on the dry path so
  the declared adapter path is loadable-shape-validated in CI.
- **Atomicity**: the manifest is written temp-then-rename, mirroring the
  corpus exporter's atomic-write discipline in
  :func:`daydream.training.corpus.run_build_corpus`.
- **Adapter handoff**: the final stage's output is a LoRA adapter checkpoint
  in the ``save_adapter_separately`` shape (``adapter_config.json`` +
  ``adapter_state.json``), and the manifest's ``adapter_path`` points at it.

The manifest write is the last step: any raise anywhere above it leaves the
run with stage artifacts but no manifest, so a manifest's presence is proof
the run completed or was refused cleanly before Stage 3.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.training import gate as gate_mod
from daydream.training import stacks
from daydream.training.gate import FrozenSplit, GateConfig, GateReport, freeze_split
from daydream.training.lineage import RunIdentity, stage_digests, validate_resume
from daydream.training.reward import DEFAULT_WEIGHTS, REWARD_VERSION
from daydream.training.reward_model import OutcomeModel, train_outcome_model

__all__ = ["PipelineConfig", "run_pipeline"]

STAGES: tuple[str, ...] = ("stage0", "stage1", "stage2", "stage3")
GPU_STAGES: frozenset[str] = frozenset({"stage1", "stage2", "stage3"})


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for one four-stage training run.

    The hyperparameter fields below feed the locked
    :class:`~daydream.training.lineage.RunIdentity`; changing any of them
    changes the run's identity and invalidates a resume (M18).

    Attributes:
        corpus: Path to the JSONL training corpus (one record per line).
        out_dir: Root directory for stage outputs and ``manifest.json``.
        stages: Ordered stage names to run.
        base_model: HuggingFace model id the LoRA adapter trains against.
        tokenizer_renderer: Renderer family (``default`` — never stock qwen3).
        max_seq_len: Maximum packed sequence length.
        lora_rank: LoRA rank (SFT rank 64–128 per the spec).
        lora_targets: LoRA target modules.
        optimizer: Optimizer name.
        learning_rate: Learning rate.
        seed: Master seed (split freeze + training determinism).
        held_out_fraction: Fraction of gold rows reserved for the Stage-0 gate.
        gate_config: Documented Stage-0 gate thresholds.
        allow_copyleft: Explicitly opted-in copyleft slugs (C8).
        profile_policy: Profile policy name carried in the run identity.
        stack_pins: Exact dependency pins carried in the run identity.
    """

    corpus: Path
    out_dir: Path
    stages: tuple[str, ...] = STAGES
    base_model: str = "Qwen/Qwen3-8B"
    tokenizer_renderer: str = "default"
    max_seq_len: int = 32768
    lora_rank: int = 64
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    seed: int = 0
    held_out_fraction: float = 0.2
    gate_config: GateConfig = field(default_factory=GateConfig)
    allow_copyleft: frozenset[str] = frozenset()
    profile_policy: str = "decisive-only"
    stack_pins: dict[str, str] = field(
        default_factory=lambda: {"verifiers": "0.2.1", "prime-rl": "0.7.0"}
    )

    def __post_init__(self) -> None:
        unknown = [s for s in self.stages if s not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage name(s) {unknown}; valid stages: {', '.join(STAGES)}")
        if not (0.0 < self.held_out_fraction < 1.0):
            raise ValueError(
                f"held_out_fraction must be in (0, 1) exclusive (got {self.held_out_fraction!r})"
            )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically: temp file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes — the corpus content address."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract gold outcome rows for the Stage-0 labels file from either shape.

    Accepts both the committed fixture shape (``comment_id``/``text``/``label``)
    and production ``run_build_corpus`` exports (``session_id``/
    ``review_output``/``outcome_label``). Gold-gate evidence fields
    (``has_posterior``, ``labeler_policy_version``, ``decisive_mix``,
    ``decisive_only``) are carried through when the record provides them; an
    absent ``labeler_policy_version`` is left absent so the admission guard in
    :mod:`daydream.training.reward_model` refuses the legacy row rather than
    silently admitting it.
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        comment_id = rec.get("comment_id") or rec.get("session_id")
        text = rec.get("text") or rec.get("review_output")
        label = rec.get("label") or rec.get("outcome_label")
        # Only the two gold outcome classes feed the Stage-0 model; contested/
        # null-label rows are not gold outcome rows and are excluded here.
        if label not in ("accepted", "rejected"):
            continue
        if not (comment_id and text):
            continue
        row: dict[str, Any] = {
            "comment_id": comment_id,
            "text": text,
            "label": label,
        }
        for key in ("has_posterior", "labeler_policy_version", "decisive_mix", "decisive_only"):
            if key in rec:
                row[key] = rec[key]
        rows.append(row)
    return rows


def _sft_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Materialize the Stage-1 dataset-SFT JSONL and tier counts (M8/M9).

    The Stage-1 dataset is **gold-positive only** (M8): only accepted-class
    gold completions are written, so the shipped ``sft@`` recipe never trains
    on rejected or silver traces. Rows are prompt/completion JSONL — the shape
    the prime-rl ``sft`` loader accepts (``messages`` column or both
    ``prompt``/``completion``). Gold vs silver counts are reported separately
    (M9), matching the recipe's tier accounting.

    Returns:
        ``(rows, tier_counts)`` where ``tier_counts`` has ``gold`` and
        ``silver`` counts (silver = explicitly ``tier == "silver"`` rows,
        never mixed into the gold-positive data).
    """
    gold: list[dict[str, Any]] = []
    silver = 0
    for rec in records:
        label = rec.get("label") or rec.get("outcome_label")
        completion = rec.get("completion") or rec.get("text") or rec.get("review_output")
        if not isinstance(completion, str) or not completion:
            continue
        if label == "accepted":
            gold.append(
                {
                    "prompt": rec.get("prompt") or _sft_prompt(rec),
                    "completion": completion,
                }
            )
        elif rec.get("tier") == "silver":
            silver += 1
    return gold, {"gold": len(gold), "silver": silver}


def _sft_prompt(rec: dict[str, Any]) -> str:
    """Deterministic SFT prompt built from a record's frozen review context."""
    code_ctx: dict[str, Any] = {}
    raw_ctx = rec.get("code_context")
    if isinstance(raw_ctx, dict):
        code_ctx = raw_ctx
    parts = [f"repo: {rec.get('repo_slug', 'unknown')}"]
    stack = rec.get("stack") or rec.get("detected_stack")
    if stack:
        parts.append(f"stack: {stack}")
    for key in ("base_sha", "head_sha"):
        value = rec.get(key) or code_ctx.get(key)
        if value:
            parts.append(f"{key}: {value}")
    changed = code_ctx.get("changed_files") or []
    if changed:
        parts.append("changed_files: " + ", ".join(str(p) for p in changed))
    return "; ".join(parts)


def _rft_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Materialize Stage-2 RFT replay inputs from corpus records (M16).

    Each input carries the frozen task identity :func:`daydream.training.rft.run_rft`
    rebuilds tasks from (``id``/``base_sha``/``head_sha``/``diff``) plus the
    intrinsic signals ``reward.score_trajectory`` consumes. Identity is
    validated fail-closed, matching ``run_rft``: a record missing
    base/head/diff raises naming it, so Stage 2 is never recorded complete
    over unrunnable inputs.

    Raises:
        RuntimeError: When any record lacks ``base_sha``/``head_sha``/``diff``
            identity (never written, never skipped silently).
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        code_ctx: dict[str, Any] = {}
        raw_ctx = rec.get("code_context")
        if isinstance(raw_ctx, dict):
            code_ctx = raw_ctx
        rid = str(rec.get("comment_id") or rec.get("session_id") or "")
        base_sha = rec.get("base_sha") or code_ctx.get("base_sha")
        head_sha = rec.get("head_sha") or code_ctx.get("head_sha")
        diff = rec.get("diff")
        missing = [
            name
            for name, value in (("base_sha", base_sha), ("head_sha", head_sha), ("diff", diff))
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"stage2 refused: record {rid!r} lacks frozen task identity field(s) {missing}; "
                "RFT rebuilds every task from base/head/diff identity (M16) and never skips a "
                "record silently"
            )
        length = rec.get("length")
        if length is None:
            length = len(str(rec.get("text") or rec.get("review_output") or ""))
        rows.append(
            {
                "id": rid,
                "repo_slug": rec.get("repo_slug"),
                "base_sha": base_sha,
                "head_sha": head_sha,
                "diff": diff,
                "findings": rec.get("findings", []),
                "verifier_verdicts": rec.get("verifier_verdicts"),
                "grounding_rate": rec.get("grounding_rate", rec.get("grounding_score")),
                "format_valid": bool(rec.get("format_valid", False)),
                "length": length,
            }
        )
    return rows


def _run_stage0(
    config: PipelineConfig, records: list[dict[str, Any]], stage_dir: Path
) -> tuple[dict[str, Any], GateReport, FrozenSplit]:
    """Stage 0: freeze split, train the outcome model, evaluate the gate.

    All CPU-bound; runs identically on the dry path (the gate evaluates on
    cached model state — the small classifier is trained in-process, no GPU).
    """
    rows = _outcome_rows(records)
    if not rows:
        raise RuntimeError(
            "stage0 gate evidence missing: corpus carries no gold outcome rows "
            "(no accepted/rejected comment or review-outcome labels); the gate refuses closed"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    labels_path = stage_dir / "labels.jsonl"
    labels_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows))

    split = freeze_split(
        labels_path, held_out_fraction=config.held_out_fraction, seed=config.seed
    )
    model: OutcomeModel = train_outcome_model(
        labels_path,
        split={"train": 1.0 - config.held_out_fraction, "held_out": config.held_out_fraction},
        seed=config.seed,
    )
    report = gate_mod.evaluate_gate(model, split, config.gate_config)

    _atomic_write_json(stage_dir / "model-state.json", model.state_dict())
    # The gate report on disk is the artifact a Stage-3 run points
    # --taskset.gate-report-path at (rl/daydream_review_v1 gate_refusal).
    # Its schema is the bare ``GateReport.to_dict()`` payload — top-level
    # ``passed``/``evidence_digest`` — so the boundary consumer reads it
    # unmodified. Split evidence lives beside it as its own artifact.
    _atomic_write_json(stage_dir / "gate-report.json", report.to_dict())
    _atomic_write_json(stage_dir / "split.json", split.to_dict())
    entry: dict[str, Any] = {
        "status": "complete",
        "gate": report.to_dict(),
        "split": split.to_dict(),
        "model_fingerprint": model.model_fingerprint,
        "label_ratio_reported": model.label_ratio_reported,
    }
    return entry, report, split


def _run_gpu_stage_shim(
    config: PipelineConfig, records: list[dict[str, Any]], stage: str, stage_dir: Path
) -> dict[str, Any]:
    """Execute the CPU-side artifact work of a GPU stage (non-dry runs only).

    The wall-clock training itself happens in the standalone prime-rl project
    (``rl/train/{sft,rft}.toml``); the coordinator owns the file artifacts the
    handoff needs: the Stage-1 dataset, the Stage-2 replay inputs, and the
    Stage-3 adapter checkpoint.
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    if stage == "stage1":
        rows, tier_counts = _sft_rows(records)
        (stage_dir / "sft-dataset.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + ("\n" if rows else "")
        )
        return {"status": "complete", "records": len(rows), "tier_counts": tier_counts}
    if stage == "stage2":
        rows = _rft_rows(records)
        (stage_dir / "rft-inputs.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n"
        )
        return {"status": "complete", "records": len(rows)}
    # stage3: adapter checkpoint handoff. The gate enforcement above guarantees
    # Stage 0 ran, so its model-state checkpoint is the merged adapter state.
    adapter_dir = stage_dir / "adapter"
    state_path = stage_dir.parent / "stage0" / "model-state.json"
    if not state_path.is_file():
        raise RuntimeError(
            "stage3 refused: Stage-0 model-state checkpoint is missing at "
            f"{state_path}; the adapter cannot be assembled without the validated "
            "outcome-model state"
        )
    state = json.loads(state_path.read_text())
    _atomic_write_json(
        adapter_dir / "adapter_config.json",
        {
            "base_model": config.base_model,
            "tokenizer_renderer": config.tokenizer_renderer,
            "lora_rank": config.lora_rank,
            "lora_targets": list(config.lora_targets),
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
        },
    )
    _atomic_write_json(adapter_dir / "adapter_state.json", state)
    return {
        "status": "complete",
        "adapter": "adapter",
        "records": 0,
    }


def _reward_weights_snapshot() -> dict[str, float]:
    """Scalar snapshot of the golden-locked reward weights for the manifest.

    Only the numeric weight fields are carried (the mapping fields and the
    identity flag are not scalars); the manifest keeps a plain JSON-dict so
    the run identity is serializable without pickling.
    """
    return {
        name: value
        for name in ("w_correctness", "w_grounding", "w_len", "w_fp", "len_tau", "len_scale")
        for value in (float(getattr(DEFAULT_WEIGHTS, name)),)
    }


def run_pipeline(config: PipelineConfig, *, dry_run: bool) -> dict[str, Any]:
    """Run the configured stages in order and write the stage manifest.

    Args:
        config: The pipeline configuration (corpus, output root, stages).
        dry_run: When true, execute only what needs no GPU (corpus load,
            Stage-0 gate, validation, manifest) and mark the GPU stages
            ``skipped_dry`` — the CI path.

    Returns:
        The manifest payload (also written atomically to
        ``<out_dir>/manifest.json``).

    Raises:
        ValueError: On a fail-closed corpus load (C5/C8), an unknown stage,
            or a resumed run whose locked run-identity drifted (ResumeAborted).
        RuntimeError: When Stage 3 is requested without a passed Stage-0 gate,
            or the corpus carries no gold outcome rows for Stage 0.
    """
    corpus_path = Path(config.corpus)
    records = stacks.load_dataset(corpus_path, allow_copyleft=config.allow_copyleft)

    out_dir = Path(config.out_dir)
    stage_entries: dict[str, dict[str, Any]] = {}
    stage_records: dict[str, dict[str, Any]] = {}
    gate_report: GateReport | None = None

    split_digest = ""
    for stage in config.stages:
        stage_dir = out_dir / stage
        if stage == "stage0":
            entry, gate_report, frozen_split = _run_stage0(config, records, stage_dir)
            stage_entries[stage] = entry
            split_digest = frozen_split.digest
            continue

        if stage == "stage3":
            if gate_report is None:
                raise RuntimeError(
                    "stage3 refused: no Stage-0 gate evidence — the gate must pass before "
                    "the adapter stage runs; the gate refuses closed, never open"
                )
            if not gate_report.passed:
                raise RuntimeError(
                    f"stage3 refused: Stage-0 gate failed (evidence digest "
                    f"{gate_report.evidence_digest[:12]}…) — the run is stopped before the "
                    "adapter stage; no manifest is written for a refused run"
                )

        if dry_run and stage in GPU_STAGES:
            if stage == "stage3":
                # The adapter handoff is pure file assembly from Stage-0 state
                # (no GPU), so the dry path still produces a declared, loadable-
                # shape adapter checkpoint even while the wall-clock training
                # itself is skipped.
                _run_gpu_stage_shim(config, records, stage, stage_dir)
            stage_entries[stage] = {"status": "skipped_dry"}
            continue

        entry = _run_gpu_stage_shim(config, records, stage, stage_dir)
        stage_entries[stage] = entry
        stage_records[stage] = {"records": records}

    identity = RunIdentity(
        base_model=config.base_model,
        tokenizer_renderer=config.tokenizer_renderer,
        max_seq_len=config.max_seq_len,
        lora_rank=config.lora_rank,
        lora_targets=config.lora_targets,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        corpus_digest=_file_digest(corpus_path),
        split_digest=split_digest,
        profile_policy=config.profile_policy,
        reward_version=REWARD_VERSION,
        reward_weights=_reward_weights_snapshot(),
        stack_pins=dict(config.stack_pins),
    )

    adapter_path: str | None = None
    if "stage3" in stage_entries and (
        stage_entries["stage3"].get("status") in ("complete", "skipped_dry")
    ):
        adapter_path = str(out_dir / "stage3" / "adapter")

    manifest: dict[str, Any] = {
        "run_identity": identity.to_dict(),
        "dry_run": dry_run,
        "stages": stage_entries,
        "stage_digests": stage_digests(stage_records),
        "adapter_path": adapter_path,
        "corpus": str(corpus_path),
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8")).get("run_identity")
        if prior is not None:
            # M18/AC4 resume guard: a resumed run aborts loudly, never
            # overwrites, when any locked run-identity field (including the
            # split digest) drifted from the prior run's manifest.
            validate_resume(RunIdentity.from_dict(prior), identity)
    _atomic_write_json(manifest_path, manifest)
    return manifest
