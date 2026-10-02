"""Render validated trajectories as PR run summaries; a separate filesystem adapter
tolerates missing/corrupt files. Group by daydream_phase, prefer recorded costs, and
synthesize missing costs from model prices. Unpriced models show a dash plus a footnote.
Pure rendering errors propagate; standalone filesystem rendering falls back.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import daydream
from daydream.atif import Step, Trajectory
from daydream.pricing import ModelPrice, compute_cost_from_totals, load_user_prices, resolve_prices
from daydream.timeutil import parse_iso_timestamp
from daydream.trajectory.invocation import _GENERIC_MODEL_LABELS

# Display labels use serialized DaydreamPhase values.
_PHASE_LABELS: dict[str, str] = {
    "review": "Review",
    "parse": "Parse Feedback",
    "fix": "Fix",
    "test": "Test & Heal",
    "intent": "Understand Intent",
    "alternatives": "Alternatives",
    "deep": "Deep Review",
    "merge": "Merge Findings",
    "exploration": "Exploration",
    "verify": "Verify Recommendations",
    "recon": "Reconnaissance",
    "audit": "Audit",
    "vet": "Vet Findings",
    "plan_write": "Write Plans",
    "diagram": "Diagram",
    # Host-side (non-agent) operations (issue #726).
    "test-execution": "Test Execution",
    "hook-run": "Pre-push Hook",
    "commit": "Commit",
    "push": "Push",
    "remote-ci": "Remote CI",
}

FALLBACK_NOTE = "*run details unavailable*"


def _format_duration(seconds: float | None) -> str:
    """Format elapsed seconds, using a dash for unavailable timing."""
    if seconds is None:
        return "—"
    if seconds < 1:
        return "<1s"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    h, remainder = divmod(total, 3600)
    m = remainder // 60
    return f"{h}h {m}m" if m else f"{h}h"


@dataclass
class _PhaseAgg:
    """Phase metrics and model attribution. Any unpriced step marks the entire cost
    unknown; all step sources contribute timing, while agent steps contribute usage.
    """

    phase_key: str
    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cost_unknown: bool = False
    models: set[str] = field(default_factory=set)
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    @property
    def duration_s(self) -> float | None:
        if self.first_timestamp is None or self.last_timestamp is None:
            return None
        return (parse_iso_timestamp(self.last_timestamp) - parse_iso_timestamp(self.first_timestamp)).total_seconds()


@dataclass
class _RunAgg:
    """Whole-run rollup aggregates with per-phase breakdown."""

    phases: dict[str, _PhaseAgg] = field(default_factory=dict)
    unknown_models: set[str] = field(default_factory=set)

    def rollup(self) -> _PhaseAgg:
        """Reduce phases into the same metric shape used by each table row."""
        phases = self.phases.values()
        firsts = [p.first_timestamp for p in phases if p.first_timestamp]
        lasts = [p.last_timestamp for p in phases if p.last_timestamp]
        return _PhaseAgg(
            phase_key="",
            steps=sum(p.steps for p in phases),
            tool_calls=sum(p.tool_calls for p in phases),
            input_tokens=sum(p.input_tokens for p in phases),
            cached_tokens=sum(p.cached_tokens for p in phases),
            output_tokens=sum(p.output_tokens for p in phases),
            cost_usd=sum(p.cost_usd for p in phases),
            cost_unknown=any(p.cost_unknown for p in phases),
            models=set().union(*(p.models for p in phases)),
            first_timestamp=min(firsts) if firsts else None,
            last_timestamp=max(lasts) if lasts else None,
        )


def render_run_info(
    trajectories: Sequence[Trajectory],
    *,
    prices: dict[str, ModelPrice] | None = None,
) -> str:
    """Render validated trajectories with built-in or explicitly supplied effective prices.
    Empty/no-phase input returns fallback markdown. This value-only path performs no
    user-price I/O and propagates unexpected errors to its orchestration provider.
    """
    if not trajectories:
        return _render_fallback()
    if prices is None:
        prices = resolve_prices()
    agg = _aggregate(trajectories, prices)
    if not agg.phases:
        return _render_fallback()
    return _render(agg)


def render_run_info_block(trajectory_paths: Sequence[Path]) -> str:
    """Load trajectory files independently and render with effective user prices. Never
    raise: missing/malformed inputs or rendering failure produce fallback markdown. The
    caller owns the outer details shell; output includes the version footer.
    """
    try:
        trajectories = _load_trajectories(trajectory_paths)
        if not trajectories:
            return _render_fallback()
        prices = resolve_prices(load_user_prices())
        return render_run_info(trajectories, prices=prices)
    except Exception:  # noqa: BLE001 - K8/M9: comment must always post
        return _render_fallback()


def _load_trajectories(paths: Sequence[Path]) -> list[Trajectory]:
    """Parse each trajectory independently, skipping failed files so surviving forks still
    contribute.
    """
    out: list[Trajectory] = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            out.append(Trajectory.model_validate(data))
        except Exception:  # noqa: BLE001 - per-file resilience
            continue
    return out


def _aggregate(trajectories: Sequence[Trajectory], prices: dict[str, ModelPrice]) -> _RunAgg:
    """Aggregate phase-tagged steps, honoring ATIF root-model fallback. Generic backend
    labels do not enter displayed model sets, but still reach pricing to mark unknown
    costs.
    """
    agg = _RunAgg()
    for traj in trajectories:
        agent_default_model = traj.agent.model_name
        for step in traj.steps:
            phase_key = _phase_key_of(step)
            if phase_key is None:
                continue
            phase = agg.phases.setdefault(phase_key, _PhaseAgg(phase_key))
            # Track timestamps from ALL sources (user + agent) for latency.
            if step.timestamp:
                if phase.first_timestamp is None or parse_iso_timestamp(
                    step.timestamp
                ) < parse_iso_timestamp(phase.first_timestamp):
                    phase.first_timestamp = step.timestamp
                if phase.last_timestamp is None or parse_iso_timestamp(
                    step.timestamp
                ) > parse_iso_timestamp(phase.last_timestamp):
                    phase.last_timestamp = step.timestamp
            if step.source != "agent":
                continue
            phase.steps += 1
            phase.tool_calls += len(step.tool_calls or [])
            effective_model = step.model_name or agent_default_model
            if effective_model and effective_model not in _GENERIC_MODEL_LABELS:
                phase.models.add(effective_model)
            _accumulate_metrics(agg, phase, step, fallback_model=agent_default_model, prices=prices)
    return agg


def _phase_key_of(step: Step) -> str | None:
    extra = step.extra or {}
    val = extra.get("daydream_phase")
    return val if isinstance(val, str) else None


def _accumulate_metrics(
    agg: _RunAgg,
    phase: _PhaseAgg,
    step: Step,
    *,
    fallback_model: str | None = None,
    prices: dict[str, ModelPrice],
) -> None:
    """Clamp token counts and cached-as-subset once, then prefer recorded cost. Otherwise
    price the explicit/root model at the step date; missing/unpriced models mark the
    phase cost unknown.
    """
    metrics = step.metrics
    if metrics is None:
        return
    prompt = max(metrics.prompt_tokens or 0, 0)
    completion = max(metrics.completion_tokens or 0, 0)
    cached_raw = max(metrics.cached_tokens or 0, 0)
    # defensive guard: backends emit cached ≤ prompt (cache reads folded into the
    # total); clamp protects against malformed/legacy metrics
    cached = min(cached_raw, prompt)
    phase.input_tokens += prompt
    phase.cached_tokens += cached
    phase.output_tokens += completion

    if metrics.cost_usd is not None:
        phase.cost_usd += metrics.cost_usd
        return

    model = step.model_name if step.model_name is not None else fallback_model
    if model is None:
        # Step has no model attribution and no SDK-provided cost: cannot
        # price. Mark unknown so the phase row degrades to '—'.
        phase.cost_unknown = True
        return
    synth = compute_cost_from_totals(
        model,
        total_input_tokens=prompt,
        cached_input_tokens=cached,
        output_tokens=completion,
        prices=prices,
        effective_date=parse_iso_timestamp(step.timestamp).date() if step.timestamp else None,
    )
    if synth is None:
        phase.cost_unknown = True
        agg.unknown_models.add(model)
        return
    phase.cost_usd += synth


def _render(agg: _RunAgg) -> str:
    """Compose the rollup, the per-phase table, optional footnote, and footer."""
    rollup = agg.rollup()
    lines = _render_rollup(rollup)
    lines.append("")
    lines.extend(_render_phase_table(agg))
    if rollup.cost_unknown and agg.unknown_models:
        lines.append("")
        lines.append(_render_unknown_models_note(agg))
    lines.append("")
    lines.append(_version_footer())
    return "\n".join(lines)


def _render_rollup(agg: _PhaseAgg) -> list[str]:
    """Render the visible run rollup."""
    return [
        f"- **Model:** {_model_label(agg.models, mixed="mixed — see breakdown")}",
        f"- **Cost:** {_rollup_cost(agg)}",
        f"- **Tokens:** {_rollup_tokens(agg)}",
        f"- **Steps / tool calls:** {_format_int(agg.steps)} / {_format_int(agg.tool_calls)}",
        f"- **Duration:** {_format_duration(agg.duration_s)}",
    ]


def _version_footer() -> str:
    """Render the ``<sub>Generated by daydream vX.Y.Z</sub>`` footer line."""
    return f"<sub>Generated by daydream v{daydream.__version__}</sub>"


def _model_label(models: set[str], *, mixed: str = "mixed") -> str:
    if not models:
        return "unknown"
    if len(models) > 1:
        return mixed
    return next(iter(models))


def _rollup_cost(agg: _PhaseAgg) -> str:
    if agg.cost_unknown:
        return "—"  # M6
    return _format_cost(agg.cost_usd)


def _rollup_tokens(agg: _PhaseAgg) -> str:
    """Render input/cache/output counts; omit cache details when input or cached counts are
    zero.
    """
    inp = agg.input_tokens
    cached = agg.cached_tokens
    out = agg.output_tokens
    plain = f"{_format_int(inp)} in → {_format_int(out)} out"
    if inp <= 0 or cached <= 0:
        return plain
    pct = _format_cache_hit_pct(inp, cached)
    return f"{_format_int(inp)} in ({_format_int(cached)} cached, {pct} hit) → {_format_int(out)} out"


def _render_phase_table(agg: _RunAgg) -> list[str]:
    """Render phases in first-encounter order inside a collapsed details block."""
    rows: list[str] = [
        "<details><summary>Per-phase breakdown</summary>",
        "",
        "| Phase | Model | Tools | Input (cached) | Output | Cost | Latency |",
        "|---|---|---|---|---|---|---|",
    ]
    # Forks restart step ids, so preserve first encounter rather than sorting ids.
    for phase in agg.phases.values():
        rows.append(_render_phase_row(phase))
    rows.append("")
    rows.append("</details>")
    return rows


def _render_phase_row(phase: _PhaseAgg) -> str:
    label = _PHASE_LABELS.get(phase.phase_key, phase.phase_key.replace("_", " ").title())
    model_cell = _model_label(phase.models)
    cost_cell = _rollup_cost(phase)
    pct = _format_cache_hit_pct(phase.input_tokens, phase.cached_tokens)
    if pct is not None and phase.cached_tokens > 0:
        input_cell = f"{_format_int(phase.input_tokens)} ({pct})"
    else:
        input_cell = _format_int(phase.input_tokens)
    latency_cell = _format_duration(phase.duration_s)
    return (
        f"| {label} | {model_cell} | {_format_int(phase.tool_calls)} | "
        f"{input_cell} | {_format_int(phase.output_tokens)} | "
        f"{cost_cell} | {latency_cell} |"
    )


def _render_unknown_models_note(agg: _RunAgg) -> str:
    """Name unpriced models in a deterministic footnote."""
    names = sorted(agg.unknown_models)
    if len(names) == 1:
        return f"<sub>Cost unavailable: model `{names[0]}` is not in the price table.</sub>"
    joined = ", ".join(f"`{n}`" for n in names)
    return f"<sub>Cost unavailable: models {joined} are not in the price table.</sub>"


def _render_fallback() -> str:
    """Return unavailable-details text plus the version footer."""
    return f"{FALLBACK_NOTE}\n\n{_version_footer()}"


def _format_int(n: int) -> str:
    """Clamp negative counts to zero and add thousands separators."""
    return f"{max(n, 0):,}"


def _format_cost(cost: float) -> str:
    """Clamp negative costs; render positive sub-cent costs as <$0.01, otherwise two
    decimals.
    """
    if cost < 0:
        cost = 0.0
    if cost > 0 and cost < 0.01:
        return "<$0.01"
    return f"${cost:.2f}"


def _format_cache_hit_pct(input_tokens: int, cached_tokens: int) -> str | None:
    """Return a rounded cache percentage clamped to 0..100, or None when input is absent."""
    if input_tokens <= 0:
        return None
    pct = round(100 * cached_tokens / input_tokens)
    return f"{min(max(pct, 0), 100)}%"


__all__ = [
    "FALLBACK_NOTE",
    "render_run_info",
    "render_run_info_block",
]
