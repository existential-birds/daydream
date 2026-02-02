"""RLM (Recursive Language Model) code review module.

This module provides capabilities for reviewing large codebases (1M+ tokens)
using sandboxed Python REPL execution with sub-LLM orchestration.
"""

from daydream.rlm.container import ContainerConfig, DevContainer, find_devcontainer_config
from daydream.rlm.environment import (
    FileInfo,
    FinalAnswer,
    RepoContext,
    Service,
    build_repl_namespace,
)
from daydream.rlm.errors import (
    ContainerError,
    HeartbeatFailedError,
    REPLCrashError,
    REPLTimeoutError,
    RLMError,
)
from daydream.rlm.runner import RLMConfig, RLMRunner, load_codebase

__all__ = [
    # Errors
    "RLMError",
    "REPLCrashError",
    "REPLTimeoutError",
    "HeartbeatFailedError",
    "ContainerError",
    # Environment
    "FileInfo",
    "Service",
    "RepoContext",
    "FinalAnswer",
    "build_repl_namespace",
    # Container
    "ContainerConfig",
    "DevContainer",
    "find_devcontainer_config",
    # Runner
    "RLMConfig",
    "RLMRunner",
    "load_codebase",
]
