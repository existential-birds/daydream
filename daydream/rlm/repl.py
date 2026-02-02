# daydream/rlm/repl.py
"""REPL process manager for executing model-generated code.

This module manages the Python REPL process that executes code generated
by the model, handling IPC communication and output capture.
"""

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from typing import Callable

from daydream.config import RLM_OUTPUT_TRUNCATION_LIMIT
from daydream.rlm.environment import (
    FinalAnswer,
    RepoContext,
    build_repl_namespace,
)


class StreamingStringIO(io.StringIO):
    """StringIO that streams writes to a callback in real-time."""

    def __init__(self, on_write: Callable[[str], None] | None = None):
        """Initialize streaming StringIO.

        Args:
            on_write: Optional callback called on each write with the text.
        """
        super().__init__()
        self._on_write = on_write

    def write(self, s: str) -> int:
        """Write string and notify callback."""
        if s and self._on_write is not None:
            self._on_write(s)
        return super().write(s)


@dataclass
class ExecuteResult:
    """Result of executing code in the REPL.

    Attributes:
        output: Captured stdout from execution.
        error: Error message if execution failed, None otherwise.
        final_answer: Final answer if FINAL/FINAL_VAR was called, None otherwise.
    """

    output: str
    error: str | None
    final_answer: str | None

    @property
    def is_error(self) -> bool:
        """Check if execution resulted in an error."""
        return self.error is not None

    @property
    def is_final(self) -> bool:
        """Check if execution produced a final answer."""
        return self.final_answer is not None


class REPLProcess:
    """Manages execution of Python code in a sandboxed namespace.

    This class provides the core REPL functionality, executing code
    in an isolated namespace with access to the codebase and LLM functions.
    """

    def __init__(
        self,
        context: RepoContext,
        llm_callback: Callable[[str, str], str],
        llm_parallel_callback: Callable[[list[str], str], list[str]] | None = None,
    ):
        """Initialize REPL process.

        Args:
            context: Repository context with codebase data.
            llm_callback: Function to handle llm_query calls.
            llm_parallel_callback: Function to handle parallel queries.
        """
        self.context = context
        self.llm_callback = llm_callback
        self.llm_parallel_callback = llm_parallel_callback
        self._namespace: dict | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if REPL is initialized and running."""
        return self._running

    def start(self) -> None:
        """Initialize the REPL namespace."""
        self._namespace = build_repl_namespace(
            self.context,
            llm_query_fn=self.llm_callback,
            llm_query_parallel_fn=self.llm_parallel_callback,
        )
        self._running = True

    def stop(self) -> None:
        """Clean up the REPL."""
        self._namespace = None
        self._running = False

    def execute(
        self,
        code: str,
        on_output: Callable[[str], None] | None = None,
    ) -> ExecuteResult:
        """Execute Python code in the REPL namespace.

        Args:
            code: Python code to execute.
            on_output: Optional callback for real-time output streaming.

        Returns:
            ExecuteResult with output, error, or final answer.
        """
        if not self._running or self._namespace is None:
            return ExecuteResult(
                output="",
                error="REPL is not running",
                final_answer=None,
            )

        stdout_capture = StreamingStringIO(on_write=on_output)
        stderr_capture = StreamingStringIO(on_write=on_output)

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self._namespace, self._namespace)

            output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # Combine stdout and stderr
            if stderr_output:
                output = output + stderr_output

            # Truncate if too long
            if len(output) > RLM_OUTPUT_TRUNCATION_LIMIT:
                output = (
                    output[:RLM_OUTPUT_TRUNCATION_LIMIT]
                    + "\n[truncated - use llm_query to analyze large outputs]"
                )

            return ExecuteResult(output=output, error=None, final_answer=None)

        except FinalAnswer as fa:
            # Model called FINAL() or FINAL_VAR()
            output = stdout_capture.getvalue()
            return ExecuteResult(
                output=output,
                error=None,
                final_answer=fa.answer,
            )

        except SystemExit as se:
            output = stdout_capture.getvalue()
            return ExecuteResult(
                output=output,
                error=f"SystemExit: {se}",
                final_answer=None,
            )

        except Exception:
            # Capture full traceback
            output = stdout_capture.getvalue()
            tb = traceback.format_exc()
            return ExecuteResult(
                output=output,
                error=tb,
                final_answer=None,
            )

    def __enter__(self) -> "REPLProcess":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
