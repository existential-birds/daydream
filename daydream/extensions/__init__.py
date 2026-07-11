"""Extension seam facade.

The public API a ``daydream_ext`` package (and daydream itself) programs
against: the versioned contract types from ``api`` and the ``Registry``.
"""

from daydream.extensions.api import (
    EXTENSION_API_VERSION,
    BreakLoop,
    ExtensionError,
    ExtensionVersionError,
    FlowStep,
    LoopGroup,
    StackRule,
    Stop,
    ToolDecision,
    ToolSupervisor,
    UnresolvedExtensionError,
)
from daydream.extensions.loader import build_registry, get_registry, set_registry
from daydream.extensions.registry import Registry

__all__ = [
    "EXTENSION_API_VERSION",
    "BreakLoop",
    "ExtensionError",
    "ExtensionVersionError",
    "FlowStep",
    "LoopGroup",
    "Registry",
    "StackRule",
    "Stop",
    "ToolDecision",
    "ToolSupervisor",
    "UnresolvedExtensionError",
    "build_registry",
    "get_registry",
    "set_registry",
]
