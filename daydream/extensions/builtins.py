"""Built-in registry seed.

``register_builtins(registry)`` seeds the registry with everything daydream
does today: built-in skill slots, prompt names, and flow definitions. It grows
across Tasks 5-15 of the extension-seam plan; for now it seeds nothing.

Uses only function-local late imports (import-cycle guard): this module must
not import from ``daydream.runner`` or ``daydream.phases`` at module level.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daydream.extensions.registry import Registry


def register_builtins(registry: Registry) -> None:
    """Seed ``registry`` with daydream's built-in phases, flows, skills, and prompts."""
