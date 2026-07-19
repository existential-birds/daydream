"""Pure renderer for the enriched daydream PR-summary comment block.

Refs #65.

The single public surface is :func:`render_run_info_block`. Everything else
in this module is private (leading underscore) and exists to keep the
renderer pure: read-only filesystem access on the trajectory paths, no
network, no global state, no logging side effects.

Architectural notes:

- ATIF construction lives in ``daydream.trajectory`` (D-19 module-bloat ban).
  Here we only *consume* the ATIF Pydantic models — the renderer parses
  trajectory JSON via :meth:`Trajectory.model_validate` and walks
  :attr:`Trajectory.steps`.
- Phase grouping uses ``Step.extra['daydream_phase']`` (a string key, since
  the value is loaded from JSON, not an in-memory ``DaydreamPhase`` enum
  member). Display labels come from :data:`_PHASE_LABELS`.
- Cost source: when a step's ``Metrics.cost_usd`` is set (Claude SDK does
  this), it is used verbatim. When ``cost_usd`` is ``None`` (Codex), the
  synthesized value from :func:`daydream.pricing.compute_cost_from_totals` is used
  (reverses project decision D-16). Unknown models render ``—`` plus a
  footnote.
- Failure mode: every entry point catches Exception and returns the
  fallback block ('run details unavailable' + version footer) so the
  comment always posts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import daydream
from daydream.atif import Step, Trajectory
from daydream.pricing import ModelPrice, compute_cost_from_totals, load_user_prices, resolve_prices
from daydream.timeutil import parse_iso_timestamp

# Display labels for each phase key used in Step.extra['daydream_phase'].
# Keys are the string values of daydream.trajectory.DaydreamPhase. Defined
# here (not in trajectory.py) because this is a display concern — see
# .beagle/concepts/enriched-pr-comment/phase-labels-decision.md.
_PHASE_LABELS: dict[str, str] = {
    "review": "Review",
    "parse": "Parse Feedback",
    "fix": "Fix",
    "test": "Test & Heal",
    "intent": "Understand Intent",
    "alternatives": "Alternatives",
    "pr_feedback": "PR Feedback",
    "deep": "Deep Review",
    "exploration": "Exploration",
    "verify": "Verify Recommendations",
}

FALLBACK_NOTE = "*run details unavailable*"

# Generic backend labels that pre-date a real SDK model id arriving on the
# event stream. Defense-in-depth: trajectory.TrajectoryRecorder upgrades
# these as soon as the first MetricsEvent / CostEvent surfaces a real id,
# so a generic label here means the run never observed a real model name.
_GENERIC_MODEL_LABELS: frozenset[str] = frozenset({"claude", "codex", ""})


def _parse_ts(ts: str) -> datetime:
    """Parse an ATIF ISO 8601 timestamp (Z-suffix) to a timezone-aware datetime."""
    return parse_iso_timestamp(ts)


def _format_duration(seconds: float | None) -> str:
    """Format a duration for display in the PR comment.

    Returns '—' when timing data is unavailable.
    """
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
    """Per-phase running totals.

    ``models`` is a set so we can detect mixed-model phases (rare — usually
    one model per phase, but the shape supports it). ``cost_unknown`` flips
    true if any step in this phase ran on a model we cannot price (no
    ``cost_usd`` from the backend AND not in MODEL_PRICES); the table cell
    then renders ``—`` per M6.

    Attributes:
        phase_key: Identifier for the phase these totals belong to.
        steps: Number of steps recorded in this phase.
        tool_calls: Number of tool calls made during this phase.
        input_tokens: Total input tokens consumed by this phase.
        cached_tokens: Total cached input tokens consumed by this phase.
        output_tokens: Total output tokens produced by this phase.
        cost_usd: Accumulated cost in USD for this phase.
        cost_unknown: True if any step ran on a model that cannot be priced.
        models: Set of model names that ran steps in this phase.
        first_timestamp: ISO-8601 timestamp of the earliest step in this phase.
        last_timestamp: ISO-8601 timestamp of the latest step in this phase.
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
        return (_parse_ts(self.last_timestamp) - _parse_ts(self.first_timestamp)).total_seconds()


@dataclass
class _RunAgg:
    """Whole-run rollup aggregates with per-phase breakdown."""

    phases: dict[str, _PhaseAgg] = field(default_factory=dict)
    unknown_models: set[str] = field(default_factory=set)

    @property
    def total_steps(self) -> int:
        return sum(p.steps for p in self.phases.values())

    @property
    def total_tools(self) -> int:
        return sum(p.tool_calls for p in self.phases.values())

    @property
    def total_input(self) -> int:
        return sum(p.input_tokens for p in self.phases.values())

    @property
    def total_cached(self) -> int:
        return sum(p.cached_tokens for p in self.phases.values())

    @property
    def total_output(self) -> int:
        return sum(p.output_tokens for p in self.phases.values())

    @property
    def total_cost(self) -> float:
        return sum(p.cost_usd for p in self.phases.values())

    @property
    def any_cost_unknown(self) -> bool:
        return any(p.cost_unknown for p in self.phases.values())

    @property
    def all_models(self) -> set[str]:
        all_m: set[str] = set()
        for p in self.phases.values():
            all_m.update(p.models)
        return all_m

    @property
    def total_duration_s(self) -> float | None:
        firsts = [p.first_timestamp for p in self.phases.values() if p.first_timestamp]
        lasts = [p.last_timestamp for p in self.phases.values() if p.last_timestamp]
        if not firsts or not lasts:
            return None
        return (_parse_ts(max(lasts)) - _parse_ts(min(firsts))).total_seconds()


def render_run_info_block(trajectory_paths: list[Path]) -> str:
    """Render the enriched run-info markdown block for the PR comment.

    Pure function. Reads the given trajectory files from disk, parses
    them, aggregates metrics, and returns the markdown to embed in the
    PR summary comment. Never raises — on any error, returns the
    fallback block ('run details unavailable' + version footer).

    Args:
        trajectory_paths: Filesystem paths to ATIF v1.7 trajectory JSON
            files. May be empty (deep-mode parent + sibling forks). Each
            file is parsed independently; metrics are summed across them.

    Returns:
        A markdown string suitable for embedding inside the existing
        ``<details>ℹ️ Review info</details>`` shell. No outer ``<details>``
        — the caller owns the shell. The string ends with a
        ``<sub>Generated by daydream vX.Y.Z</sub>`` footer.
    """
    try:
        if not trajectory_paths:
            return _render_fallback()
        trajectories = _load_trajectories(trajectory_paths)
        if not trajectories:
            return _render_fallback()
        # Build the effective price table ONCE (built-ins + user overrides
        # from prices.toml). Inside the try so any load failure degrades to
        # the fallback block rather than escaping the never-raises contract.
        prices = resolve_prices(load_user_prices())
        agg = _aggregate(trajectories, prices)
        if not agg.phases:
            return _render_fallback()
        return _render(agg)
    except Exception:  # noqa: BLE001 - K8/M9: comment must always post
        return _render_fallback()


def _load_trajectories(paths: list[Path]) -> list[Trajectory]:
    """Parse each path as an ATIF Trajectory; skip files that don't parse.

    A single missing or malformed file does not poison the whole run
    rollup — we render whatever we can. The outer ``render_run_info_block``
    catches every Exception, but per-file resilience here means a deep-run
    where one fork's trajectory failed to write still produces a useful
    summary from the other forks.
    """
    out: list[Trajectory] = []
    for p in paths:
        try:
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            out.append(Trajectory.model_validate(data))
        except Exception:  # noqa: BLE001 - per-file resilience
            continue
    return out


def _aggregate(trajectories: list[Trajectory], prices: dict[str, ModelPrice]) -> _RunAgg:
    """Walk every step in every trajectory, summing into per-phase rollups.

    ATIF v1.7 spec: ``Step.model_name`` omission implies the model defined in
    the root-level agent config (``Trajectory.agent.model_name``). We honor
    that by computing ``effective_model = step.model_name or
    traj.agent.model_name`` for each step. Generic backend labels (see
    :data:`_GENERIC_MODEL_LABELS`) are never added to ``phase.models`` —
    downstream rendering would relabel them as ``unknown`` anyway, so they
    must not pollute the per-phase model set. The fallback is still threaded
    into :func:`_accumulate_metrics` so the cost path can attempt pricing
    (and fall through to ``cost_unknown`` when the label is generic).

    Args:
        prices: Effective model price table (built-ins merged with user
            overrides) threaded into :func:`_accumulate_metrics` for cost
            synthesis.
    """
    agg = _RunAgg()
    for traj in trajectories:
        agent_default_model = traj.agent.model_name
        for step in traj.steps:
            phase_key = _phase_key_of(step)
            if phase_key is None:
                continue
            phase = _ensure_phase(agg, phase_key)
            # Track timestamps from ALL sources (user + agent) for latency.
            if step.timestamp:
                if phase.first_timestamp is None or _parse_ts(step.timestamp) < _parse_ts(phase.first_timestamp):
                    phase.first_timestamp = step.timestamp
                if phase.last_timestamp is None or _parse_ts(step.timestamp) > _parse_ts(phase.last_timestamp):
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


def _ensure_phase(agg: _RunAgg, phase_key: str) -> _PhaseAgg:
    if phase_key not in agg.phases:
        agg.phases[phase_key] = _PhaseAgg(phase_key=phase_key)
    return agg.phases[phase_key]


def _accumulate_metrics(
    agg: _RunAgg,
    phase: _PhaseAgg,
    step: Step,
    *,
    fallback_model: str | None = None,
    prices: dict[str, ModelPrice],
) -> None:
    """Add this step's token + cost contribution into the phase aggregate.

    Token clamp (single source of truth): per the ATIF Metrics docstring,
    ``cached_tokens`` is a SUBSET of ``prompt_tokens`` (not additive), and
    all token counts are non-negative. We clamp once at the top so
    aggregates and synthesized costs share the same clean values.

    Cost rule (M5/M6/C5):

    - If ``Metrics.cost_usd`` is present, use it verbatim — Claude SDK
      surfaces real billed cost, no need to synthesize.
    - Else if model is in :data:`daydream.pricing.MODEL_PRICES`, synthesize
      cost from token counts via ``compute_cost_from_totals`` (which derives
      the uncached input count from the clamped totals).
    - Else mark ``phase.cost_unknown`` and remember the model name for the
      footnote.

    Per ATIF v1.7 (``Step.model_name`` field docs), an omitted step model
    implies the root-level :attr:`Agent.model_name`. Callers thread that
    value in via ``fallback_model`` so cost synthesis can land on the
    intended model when the step itself doesn't carry an explicit id.

    Args:
        agg: Whole-run rollup; ``unknown_models`` is updated when a model
            cannot be priced.
        fallback_model: Model id to use when ``step.model_name`` is omitted
            (the root-level :attr:`Agent.model_name`).
        prices: Effective model price table (built-ins merged with user
            overrides) passed to :func:`daydream.pricing.compute_cost_from_totals`
            for cost synthesis.
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
        effective_date=_parse_ts(step.timestamp).date() if step.timestamp else None,
    )
    if synth is None:
        phase.cost_unknown = True
        agg.unknown_models.add(model)
        return
    phase.cost_usd += synth


def _render(agg: _RunAgg) -> str:
    """Compose the rollup, the per-phase table, optional footnote, and footer."""
    lines: list[str] = []
    lines.extend(_render_rollup(agg))
    lines.append("")
    lines.extend(_render_phase_table(agg))
    if agg.any_cost_unknown and agg.unknown_models:
        lines.append("")
        lines.append(_render_unknown_models_note(agg))
    lines.append("")
    lines.append(_version_footer())
    return "\n".join(lines)


def _render_rollup(agg: _RunAgg) -> list[str]:
    """Visible rollup (M1) — Mode line removed per the summarize cleanup."""
    return [
        f"- **Model:** {_rollup_model(agg)}",
        f"- **Cost:** {_rollup_cost(agg)}",
        f"- **Tokens:** {_rollup_tokens(agg)}",
        f"- **Steps / tool calls:** {_format_int(agg.total_steps)} / {_format_int(agg.total_tools)}",
        f"- **Duration:** {_format_duration(agg.total_duration_s)}",
    ]


def _version_footer() -> str:
    """Render the ``<sub>Generated by daydream vX.Y.Z</sub>`` footer line."""
    return f"<sub>Generated by daydream v{daydream.__version__}</sub>"


def _rollup_model(agg: _RunAgg) -> str:
    models = agg.all_models
    if not models:
        return "unknown"
    if len(models) > 1:
        return "mixed — see breakdown"  # M7
    only = next(iter(models))
    if only in _GENERIC_MODEL_LABELS:
        return "unknown"
    return only


def _rollup_cost(agg: _RunAgg) -> str:
    if agg.any_cost_unknown:
        return "—"  # M6
    return _format_cost(agg.total_cost)


def _rollup_tokens(agg: _RunAgg) -> str:
    """Render the tokens segment of the rollup (M8/M10).

    Format examples:
      ``33,600 in (22,600 cached, 67% hit) → 6,900 out``
      ``800 in → 200 out``  (cache hit ratio omitted when input == 0 OR
      cached == 0; a 0% hit ratio adds noise, not signal.)
    """
    inp = agg.total_input
    cached = agg.total_cached
    out = agg.total_output
    if inp <= 0:
        return f"{_format_int(inp)} in → {_format_int(out)} out"
    pct = _format_cache_hit_pct(inp, cached)
    if cached > 0 and pct is not None:
        return (
            f"{_format_int(inp)} in ({_format_int(cached)} cached, {pct} hit) "
            f"→ {_format_int(out)} out"
        )
    return f"{_format_int(inp)} in → {_format_int(out)} out"


def _render_phase_table(agg: _RunAgg) -> list[str]:
    """Per-phase breakdown inside a collapsed `<details>` block (M2)."""
    rows: list[str] = [
        "<details><summary>Per-phase breakdown</summary>",
        "",
        "| Phase | Model | Tools | Input (cached) | Output | Cost | Latency |",
        "|---|---|---|---|---|---|---|",
    ]
    # Preserve dict insertion order: _ensure_phase inserts each phase on
    # first encounter, so dict order already matches traversal order. Per-
    # file step_id restarts at 1 in deep-mode forks, so sorting by step_id
    # would mix fork phases ahead of later-numbered parent phases.
    ordered = list(agg.phases.values())
    for phase in ordered:
        rows.append(_render_phase_row(phase))
    rows.append("")
    rows.append("</details>")
    return rows


def _render_phase_row(phase: _PhaseAgg) -> str:
    label = _PHASE_LABELS.get(phase.phase_key, phase.phase_key.replace("_", " ").title())
    if not phase.models:
        model_cell = "unknown"
    elif len(phase.models) > 1:
        model_cell = "mixed"
    else:
        only = next(iter(phase.models))
        model_cell = "unknown" if only in _GENERIC_MODEL_LABELS else only
    cost_cell = "—" if phase.cost_unknown else _format_cost(phase.cost_usd)
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
    """Footnote naming each unpriced model (M6)."""
    names = sorted(agg.unknown_models)
    if len(names) == 1:
        return f"<sub>Cost unavailable: model `{names[0]}` is not in the price table.</sub>"
    joined = ", ".join(f"`{n}`" for n in names)
    return f"<sub>Cost unavailable: models {joined} are not in the price table.</sub>"


def _render_fallback() -> str:
    """M9: degrade to a 'run details unavailable' note plus version footer."""
    return f"{FALLBACK_NOTE}\n\n{_version_footer()}"


def _format_int(n: int) -> str:
    """M10: thousand separators on values >=1,000.

    Negative inputs are clamped to 0 — token counts are by definition
    non-negative; a negative value implies a corrupt trajectory and we
    prefer ``0`` over a confusing ``-3,400`` in the user-facing table.
    """
    if n < 0:
        n = 0
    if n >= 1_000:
        return f"{n:,}"
    return str(n)


def _format_cost(cost: float) -> str:
    """M10: cost <$0.01 renders as ``<$0.01``; otherwise ``$X.XX``.

    Costs are aggregated as floats; values like 0.005 should not render as
    ``$0.01`` (overstates) nor ``$0.00`` (understates). The ``<$0.01``
    sentinel matches the spec example.
    """
    if cost < 0:
        cost = 0.0
    if cost > 0 and cost < 0.01:
        return "<$0.01"
    return f"${cost:.2f}"


def _format_cache_hit_pct(input_tokens: int, cached_tokens: int) -> str | None:
    """M10 trailing rule: cache hit ratio omitted when input tokens = 0.

    Returns a formatted percentage like ``"67%"`` or ``None`` to signal
    'omit'. We also clamp the ratio to ``[0, 100]`` because trajectories
    occasionally double-count cached tokens vs. prompt tokens during a
    metrics race (the ratio shouldn't render as ``113%`` even if the
    underlying numbers say so).
    """
    if input_tokens <= 0:
        return None
    pct = round(100 * cached_tokens / input_tokens)
    if pct < 0:
        pct = 0
    if pct > 100:
        pct = 100
    return f"{pct}%"


__all__ = [
    "FALLBACK_NOTE",
    "render_run_info_block",
]
