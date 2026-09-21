"""Scalar precedence and resume settings shared by deep composition and stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from daydream.deep.detection import StackAssignment
    from daydream.runner import RunConfig


def _resolve_config_value[T: (int, float, bool)](config: RunConfig, attr: str, default: T) -> T:
    """Resolve a scalar setting: ``RunConfig`` attr (when present) > file config > default.

    Precedence mirrors ``_resolve_backend`` / ``_resolved_model`` at
    ``runner.py:295-326``. Uses ``is not None`` checks rather than truthiness
    so a configured value of ``0`` / ``0.0`` is honored.
    """
    value = getattr(config, attr, None)
    if value is not None:
        return cast(T, value)  # T-typed attr read off a config object; scalar-validated by the callers
    file_config = config.file_config
    if file_config is not None:
        value = getattr(file_config, attr, None)
        if value is not None:
            return cast(T, value)
    return default


def _resolve_opt_in(config: RunConfig, attr: str) -> bool:
    """Resolve a boolean opt-in by truthiness: ``RunConfig`` attr > file config > ``False``."""
    if getattr(config, attr, False):
        return True
    file_config = config.file_config
    return bool(file_config is not None and getattr(file_config, attr, False))


def fresh_ttt(config: RunConfig) -> bool:
    """Whether fresh intent and alternatives run for this resume point."""
    return config.start_at not in ("per-stack", "merge", "fix")


def fold_default_alternatives(stacks: list[StackAssignment], strategy: str) -> bool:
    """The scheduled structural reviewer owns the packaged design-review lens.

    Compare effective content, not provenance: partial profiles inherit defaults,
    while any distinct custom alternatives policy keeps its independent pass.
    This is independent of resume state so re-reviews retain the folded duty.
    """
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.review_profile import build_default_profile

    return (
        any(stack.stack_name == STRUCTURE_STACK_NAME for stack in stacks)
        and strategy == build_default_profile().strategies["alternatives"].content
    )
