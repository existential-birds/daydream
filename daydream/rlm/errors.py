"""Error types for the RLM module."""


class RLMError(Exception):
    """Base class for RLM errors."""


class REPLCrashError(RLMError):
    """REPL process exited unexpectedly."""


class REPLTimeoutError(RLMError):
    """Code execution exceeded timeout."""


class HeartbeatFailedError(RLMError):
    """REPL stopped responding to heartbeats."""


class ContainerError(RLMError):
    """Devcontainer failed to start or crashed."""
