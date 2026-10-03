"""Run ordered training stages and atomically publish their manifest.

The projection loader enforces admission and frozen splits. Stage 0 trains and
evaluates on that split; Stage 3 requires its passed gate before creating any
adapter directory. Dry runs skip GPU training but still assemble the adapter
from Stage-0 state, preserving a loadable handoff for CI.

Resume validates locked identity before stage writes, then checks the split
digest after Stage 0. The manifest is written last; failures may leave stage
artifacts but cannot publish a new successful manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from daydream.json_utils import atomic_write_json
from daydream.training import gate as gate_mod
from daydream.training.gate import FrozenSplit, GateConfig, GateReport
from daydream.training.lineage import ResumeAborted, RunIdentity, stage_digests, validate_resume
from daydream.training.reward import DEFAULT_WEIGHTS, REWARD_VERSION
from daydream.training.reward_model import OutcomeModel, train_outcome_model
from daydream.training.rft import _reconstruct_task
from daydream.training.stacks import V2Projection, load_v2_projection

__all__ = ["PipelineConfig", "run_pipeline"]

STAGES: tuple[str, ...] = ("stage0", "stage1", "stage2", "stage3")


@dataclass(frozen=True)
class PipelineConfig:
    """Training inputs and hyperparameters, locked into RunIdentity for resume.

    Only frozen projection directories are accepted. The default renderer must
    remain compatible with the training recipes; stage order is caller-supplied.
    """

    out_dir: Path
    projection: Path | None = None
    stages: tuple[str, ...] = STAGES
    base_model: str = "Qwen/Qwen3-8B"
    tokenizer_renderer: str = "default"
    max_seq_len: int = 32768
    lora_rank: int = 64
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    optimizer: str = "adamw"
    learning_rate: float = 1e-5
    seed: int = 0
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
        if self.projection is None:
            raise ValueError(
                "no projection input: PipelineConfig requires projection=<frozen projection dir>"
            )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically via the shared crash-safe primitive."""
    atomic_write_json(path, payload, sort_keys=True)


def _outcome_rows(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Capture gold labels in file order and their pinned training partitions.

    Preserve admission evidence and promote lineage.labeler_policy_version only
    when absent at top level. Missing policy stays missing so model admission
    refuses it; missing text or identity on a labeled row raises RuntimeError.
    """
    rows: list[dict[str, Any]] = []
    partitions: dict[str, list[dict[str, Any]]] = {
        split: [] for split in ("train", "validation", "holdout")
    }
    for rec in records:
        comment_id = rec.get("session_id")
        text = rec.get("finding_text")
        label = rec.get("outcome_label")
        # Only the two gold outcome classes feed the Stage-0 model; contested/
        # null-label rows are not gold outcome rows and are excluded here.
        if label not in ("accepted", "rejected"):
            continue
        if not (comment_id and text):
            raise RuntimeError(
                f"stage0 refused: projection gold record "
                f"{rec.get('record_id') or comment_id!r} has outcome_label "
                f"{label!r} but no finding_text — a gold record without its "
                "localized finding text is a broken projection, not an empty row"
            )
        row: dict[str, Any] = {
            "comment_id": comment_id,
            "text": text,
            "label": label,
        }
        for key in ("has_posterior", "labeler_policy_version", "decisive_mix", "decisive_only"):
            if key in rec:
                row[key] = rec[key]
        if "labeler_policy_version" not in row:
            lineage_obj = rec.get("lineage")
            if isinstance(lineage_obj, dict) and lineage_obj.get("labeler_policy_version") is not None:
                row["labeler_policy_version"] = lineage_obj["labeler_policy_version"]
        rows.append(row)
        split = _record_views(rec)[1]["split"]
        partitions[split].append(row)
    return rows, partitions


def _sft_rows(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build prompt/completion rows from accepted findings only.

    Count silver rows separately without training on them. Empty or non-string
    finding text is excluded from either count.
    """
    silver = 0
    gold: list[dict[str, Any]] = []
    for rec in records:
        label = rec.get("outcome_label")
        completion = rec.get("finding_text")
        if not isinstance(completion, str) or not completion:
            continue
        if label == "accepted":
            row = {
                "prompt": rec.get("prompt") or _sft_prompt(rec),
                "completion": completion,
            }
            gold.append(row)
        elif rec.get("tier") == "silver":
            silver += 1
    return gold, {"gold": len(gold), "silver": silver}


def _record_views(
    rec: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a record's v2 ``task_identity``/``lineage`` views."""

    def view(key: str) -> dict[str, Any]:
        value = rec.get(key)
        return value if isinstance(value, dict) else {}

    return view("task_identity"), view("lineage")


def _sft_prompt(rec: dict[str, Any]) -> str:
    """Build a deterministic prompt from frozen task identity, lineage, and stack."""
    identity, lineage_obj = _record_views(rec)
    repo_slug = identity.get("repo_slug") or lineage_obj.get("repo_slug") or "unknown"
    parts = [f"repo: {repo_slug}"]
    stack = rec.get("stack") or rec.get("detected_stack")
    if stack:
        parts.append(f"stack: {stack}")
    for key in ("base_sha", "head_sha"):
        value = identity.get(key)
        if value:
            parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _rft_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build replay inputs from frozen repo/base/head/diff identity, refusing missing fields.

    Both SHAs use the replay's shared full-hex validator. Validated projection rows
    are format-valid; adjudicated outcomes map to the shared verdict vocabulary.
    Absent verifier evidence stays unknown, and missing length uses finding text.
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        identity, lineage_obj = _record_views(rec)
        row = {
            "id": str(rec.get("session_id") or ""),
            "repo_slug": identity.get("repo_slug") or lineage_obj.get("repo_slug"),
            "base_sha": identity.get("base_sha"),
            "head_sha": identity.get("head_sha"),
            "diff": rec.get("diff"),
        }
        try:
            _reconstruct_task(row)
        except ValueError as exc:
            raise RuntimeError(f"stage2 refused: {exc}") from exc
        # Projection records carry no archived verifier-verdicts file; the
        # record's own adjudicated outcome is its capture-time judgment.
        # Map it onto the shared verdict vocabulary (the labels
        # score_trajectory's verdict_map consumes) so the replay reads a
        # real correctness axis instead of flooring every candidate at a
        # 0.0 composite. format_valid is True: a frozen v2 record is
        # admission/shape/drift-validated, so the v1 bronze-parse failure
        # floor cannot apply here.
        disposition = rec.get("outcome_label") or rec.get("disposition")
        verdict = {
            "accepted": "consistent",
            "rejected": "contradicts",
            "ambiguous": "uncertain",
            "unanswered": "uncertain",
            "missing": "uncertain",
        }.get(str(disposition))
        verifier_verdicts: Any = (
            [{"verdict": verdict}] if verdict is not None else rec.get("verifier_verdicts")
        )
        format_valid = True
        length = rec.get("length")
        if length is None:
            length = len(str(rec.get("finding_text") or ""))
        row.update(
            findings=rec.get("findings", []),
            verifier_verdicts=verifier_verdicts,
            format_valid=format_valid,
            length=length,
        )
        rows.append(row)
    return rows


def _run_stage0(
    config: PipelineConfig,
    stage_dir: Path,
    *,
    projection: V2Projection,
) -> tuple[dict[str, Any], GateReport, FrozenSplit]:
    """Train and evaluate the CPU outcome model on the projection's frozen split.

    Dry and ordinary runs perform identical Stage-0 work.
    """
    rows, partitions = _outcome_rows(projection.records)
    if not rows:
        raise RuntimeError(
            "stage0 gate evidence missing: corpus carries no gold outcome rows "
            "(no accepted/rejected comment or review-outcome labels); the gate refuses closed"
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    labels_path = stage_dir / "labels.jsonl"
    labels_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows))

    if not partitions["holdout"]:
        raise RuntimeError(
            "stage0 refused: the projection frozen split has no gold outcome rows in "
            "its holdout split; the gate would evaluate against nothing and refuses closed"
        )
    split = gate_mod._build_frozen_split(
        labels_path,
        train_rows=[*partitions["train"], *partitions["validation"]],
        held_out_rows=partitions["holdout"],
        seed=config.seed,
        held_out_fraction=float(cast(float, projection.lineage["holdout_rate"])),
    )
    model: OutcomeModel = train_outcome_model(
        labels_path,
        split=split,
        seed=config.seed,
    )
    report = gate_mod.evaluate_gate(model, split, config.gate_config)

    _atomic_write_json(stage_dir / "model-state.json", model.state_dict())
    # The gate report on disk is the artifact a Stage-3 run points
    # --taskset.gate-report-path at (rl/daydream_review gate_refusal).
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


def _run_stage_artifacts(
    config: PipelineConfig, records: list[dict[str, Any]], stage: str, stage_dir: Path
) -> dict[str, Any]:
    """Write the SFT dataset, RFT replay inputs, or adapter checkpoint.

    Actual GPU training lives in rl/train/{sft,rft}.toml. Adapter assembly also
    runs on the dry path, using the validated Stage-0 model state.
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
    """Capture numeric reward weights for JSON lineage, excluding mappings and flags."""
    return {
        name: float(getattr(DEFAULT_WEIGHTS, name))
        for name in ("w_len", "w_fp", "len_tau", "len_scale")
    }


def _make_identity(config: PipelineConfig, corpus_digest: str, split_digest: str) -> RunIdentity:
    """Build the locked run identity stamped into the manifest (M18)."""
    return RunIdentity(
        base_model=config.base_model,
        tokenizer_renderer=config.tokenizer_renderer,
        max_seq_len=config.max_seq_len,
        lora_rank=config.lora_rank,
        lora_targets=config.lora_targets,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        corpus_digest=corpus_digest,
        split_digest=split_digest,
        profile_policy=config.profile_policy,
        reward_version=REWARD_VERSION,
        reward_weights=_reward_weights_snapshot(),
        stack_pins=dict(config.stack_pins),
    )


def run_pipeline(config: PipelineConfig, *, dry_run: bool) -> dict[str, Any]:
    """Run stages in order and atomically publish the completed manifest.

    Projection/admission or resume drift raises ValueError; missing gold evidence
    or an absent/failed Stage-0 gate for Stage 3 raises RuntimeError. Dry runs
    retain Stage 0 and adapter assembly while marking GPU stages skipped_dry.
    """
    # PipelineConfig.__post_init__ enforces a projection input; the assert keeps
    # the invariant documented and narrows the type for mypy.
    assert config.projection is not None
    # Frozen projection directory: the v2 loader re-applies the C5/C8 and
    # split-drift gates, and the directory-level digest replaces the
    # single-file corpus digest in the run identity.
    projection_path = config.projection
    corpus_path = Path(projection_path)
    projection = load_v2_projection(projection_path, allow_copyleft=config.allow_copyleft)
    records = list(projection.records)
    corpus_digest = projection.digest

    out_dir = Path(config.out_dir)
    stage_entries: dict[str, dict[str, Any]] = {}
    gate_report: GateReport | None = None

    # M18/AC4 resume guard, hoisted ahead of the stage loop: the prior
    # manifest is read and every locked field EXCEPT split_digest (unknown
    # until Stage 0 freezes the split) is compared before any stage runs, so
    # a drifted re-run aborts before it can overwrite the prior run's stage
    # artifacts. split_digest is rechecked after Stage 0 below.
    manifest_path = out_dir / "manifest.json"
    prior_identity: RunIdentity | None = None
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8")).get("run_identity")
        if prior is not None:
            prior_identity = RunIdentity.from_dict(prior)
    if prior_identity is not None:
        # The candidate identity stands in for the not-yet-frozen split so
        # the comparison covers every other locked field (M18 lists each
        # drifted field); the split itself is rechecked post-Stage-0.
        validate_resume(prior_identity, _make_identity(config, corpus_digest, prior_identity.split_digest))

    split_digest = ""
    for stage in config.stages:
        stage_dir = out_dir / stage
        if stage == "stage0":
            entry, gate_report, frozen_split = _run_stage0(
                config, stage_dir, projection=projection
            )
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

        # Adapter assembly is CPU-only and must remain loadable on dry runs.
        entry = (
            _run_stage_artifacts(config, records, stage, stage_dir)
            if not dry_run or stage == "stage3" else {}
        )
        stage_entries[stage] = {"status": "skipped_dry"} if dry_run else entry

    # The split is frozen only by Stage 0, so its digest is rechecked against
    # the prior run here; every other locked field was compared pre-loop.
    if prior_identity is not None and split_digest != prior_identity.split_digest:
        raise ResumeAborted(
            "resume aborted: locked run-identity field split_digest differs from the prior run: "
            f"{prior_identity.split_digest!r} -> {split_digest!r}"
        )
    identity = _make_identity(config, corpus_digest, split_digest)

    adapter_path: str | None = None
    if "stage3" in stage_entries and (
        stage_entries["stage3"].get("status") in ("complete", "skipped_dry")
    ):
        adapter_path = str(out_dir / "stage3" / "adapter")

    manifest: dict[str, Any] = {
        "run_identity": identity.to_dict(),
        "dry_run": dry_run,
        "stages": stage_entries,
        "stage_digests": stage_digests({stage: {"records": records} for stage in stage_entries}),
        "adapter_path": adapter_path,
        "corpus": str(corpus_path),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest
