"""Extension seam facade.

The public API a ``daydream_ext`` package (and daydream itself) programs
against: the versioned contract types from ``api`` and the ``Registry``.
"""

from daydream.executors import (
    ArtifactEnvelope,
    ExecutionRef,
    ExecutionSnapshot,
    ExecutionStatus,
    ExecutorCapability,
    ExecutorError,
    ExecutorJob,
    LocalExecutor,
    ReviewExecutor,
    ScriptedExecutor,
)
from daydream.extensions.api import (
    DAYDREAM_SERVICE_V1,
    EXTENSION_API_VERSION,
    MIN_SUPPORTED_DAYDREAM_SERVICE_V1,
    MIN_SUPPORTED_EXTENSION_API_VERSION,
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
    "ArtifactEnvelope",
    "DAYDREAM_SERVICE_V1",
    "EXTENSION_API_VERSION",
    "BreakLoop",
    "ExecutionRef",
    "ExecutionSnapshot",
    "ExecutionStatus",
    "ExecutorCapability",
    "ExecutorError",
    "ExecutorJob",
    "ExtensionError",
    "ExtensionVersionError",
    "FlowStep",
    "LocalExecutor",
    "LoopGroup",
    "MIN_SUPPORTED_DAYDREAM_SERVICE_V1",
    "MIN_SUPPORTED_EXTENSION_API_VERSION",
    "Registry",
    "ReviewExecutor",
    "ScriptedExecutor",
    "StackRule",
    "Stop",
    "ToolDecision",
    "ToolSupervisor",
    "UnresolvedExtensionError",
    "build_registry",
    "get_registry",
    "set_registry",
]
