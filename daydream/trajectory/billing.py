"""Host-owned usage totals: one generation charge, unknown never zero, forks folded once.

Each evidence document owns one :class:`UsageLedger`. Invocations charge the
ledger of the document they record into; a completed fork folds its ledger into
its parent once, so the root ledger is the run's whole total. Cached tokens stay a
subset of prompt tokens and are never added to them. A dimension that was never
observed stays ``None``; an observed zero stays zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Generic backend labels replaced by the first native model name.
GENERIC_MODEL_LABELS: frozenset[str] = frozenset({"claude", "codex", "osprey", "unknown", ""})

_DIMENSIONS = (
    ("prompt", "prompt_tokens", "total_prompt_tokens"),
    ("completion", "completion_tokens", "total_completion_tokens"),
    ("cached", "cached_tokens", "total_cached_tokens"),
    ("cost", "cost_usd", "total_cost_usd"),
)


@dataclass
class UsageLedger:
    """Running usage for one document, with fork folding tracked for its scope label."""

    prompt: int | None = None
    completion: int | None = None
    cached: int | None = None
    cost: float | None = None
    folded_forks: bool = False

    def charge(self, *, prompt_tokens: int | None, completion_tokens: int | None,
               cached_tokens: int | None, cost_usd: float | None) -> None:
        """Add each observed dimension; an absent dimension changes nothing."""
        for attribute, value in (("prompt", prompt_tokens), ("completion", completion_tokens),
                                 ("cached", cached_tokens), ("cost", cost_usd)):
            if value is not None:
                setattr(self, attribute, (getattr(self, attribute) or 0) + value)

    def fold(self, child: UsageLedger) -> None:
        """Fold one completed fork's total into this ledger exactly once."""
        self.charge(prompt_tokens=child.prompt, completion_tokens=child.completion,
                    cached_tokens=child.cached, cost_usd=child.cost)
        self.folded_forks = True

    def final_metrics(self, total_steps: int) -> dict[str, Any]:
        """Project the document totals; folded fork totals carry an explicit scope label."""
        metrics: dict[str, Any] = {
            total: getattr(self, attribute)
            for attribute, _usage, total in _DIMENSIONS if getattr(self, attribute) is not None
        }
        metrics["total_steps"] = total_steps
        if self.folded_forks:
            # Token and cost totals include successful forks; total_steps stays
            # this document's own count. Cached tokens remain a subset of prompt.
            metrics["extra"] = {
                "daydream_metric_scope": "whole_run_including_forks",
                "total_steps_scope": "local_trajectory",
            }
        return metrics

    def usage(self) -> dict[str, Any]:
        """The run usage handed to embedders, without elapsed time."""
        usage: dict[str, Any] = {
            name: getattr(self, attribute)
            for attribute, name, _total in _DIMENSIONS if getattr(self, attribute) is not None
        }
        if "prompt_tokens" in usage and "completion_tokens" in usage:
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return usage


def run_usage() -> dict[str, Any]:
    """Return the active run's whole usage (root ledger, forks folded once), or ``{}``."""
    from daydream.trajectory.context import get_current_recorder

    recorder = get_current_recorder()
    while recorder is not None and recorder.parent is not None:
        recorder = recorder.parent
    return {} if recorder is None else recorder.usage.usage()
