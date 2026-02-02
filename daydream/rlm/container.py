# daydream/rlm/container.py
"""Devcontainer management for sandboxed REPL execution.

This module handles starting, stopping, and executing commands in
devcontainers for secure code execution.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.config import RLM_CONTAINER_STARTUP_TIMEOUT
from daydream.rlm.errors import ContainerError


@dataclass
class ContainerConfig:
    """Configuration for devcontainer setup.

    Attributes:
        workspace_path: Path to the workspace/repository to mount.
        container_id: Existing container ID to use (if already running).
        mount_readonly: Whether to mount workspace as read-only.
    """

    workspace_path: str
    container_id: str | None = None
    mount_readonly: bool = True


def find_devcontainer_config(workspace_path: Path) -> Path | None:
    """Find devcontainer.json in workspace.

    Args:
        workspace_path: Path to search for .devcontainer directory.

    Returns:
        Path to devcontainer.json if found, None otherwise.
    """
    devcontainer_dir = workspace_path / ".devcontainer"
    config_file = devcontainer_dir / "devcontainer.json"
    if config_file.exists():
        return config_file
    return None


class DevContainer:
    """Manages a devcontainer for sandboxed execution.

    Provides methods to start, stop, and execute commands in a
    devcontainer environment.
    """

    def __init__(self, config: ContainerConfig):
        """Initialize DevContainer with configuration.

        Args:
            config: Container configuration.
        """
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self.container_id: str | None = config.container_id

    @property
    def is_running(self) -> bool:
        """Check if container is running."""
        return self.container_id is not None

    async def start(self, timeout: float = RLM_CONTAINER_STARTUP_TIMEOUT) -> None:
        """Start the devcontainer.

        Args:
            timeout: Maximum time to wait for container to start.

        Raises:
            ContainerError: If container fails to start.
        """
        if self.is_running:
            return

        workspace = Path(self.config.workspace_path)

        # Check for devcontainer config
        config_path = find_devcontainer_config(workspace)
        if config_path is None:
            raise ContainerError(
                f"No .devcontainer/devcontainer.json found in {workspace}"
            )

        try:
            # Start devcontainer using CLI
            proc = await asyncio.create_subprocess_exec(
                "devcontainer",
                "up",
                "--workspace-folder",
                str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            if proc.returncode != 0:
                raise ContainerError(
                    f"devcontainer up failed: {stderr.decode()}"
                )

            # Parse container ID from output
            output = json.loads(stdout.decode())
            self.container_id = output.get("containerId")

            if not self.container_id:
                raise ContainerError("No container ID in devcontainer output")

        except asyncio.TimeoutError:
            raise ContainerError(f"Container startup timed out after {timeout}s")
        except json.JSONDecodeError as e:
            raise ContainerError(f"Failed to parse devcontainer output: {e}")

    async def stop(self) -> None:
        """Stop the devcontainer."""
        if not self.is_running or self.container_id is None:
            return

        container_id = self.container_id
        try:
            proc = await asyncio.create_subprocess_exec(
                "devcontainer",
                "stop",
                "--container-id",
                container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        finally:
            self.container_id = None

    async def exec_command(
        self,
        command: list[str],
        stdin: asyncio.StreamReader | None = None,
    ) -> tuple[asyncio.StreamWriter, asyncio.StreamReader, asyncio.StreamReader]:
        """Execute a command in the container.

        Args:
            command: Command and arguments to execute.
            stdin: Optional stdin stream to connect.

        Returns:
            Tuple of (stdin_writer, stdout_reader, stderr_reader).

        Raises:
            RuntimeError: If container is not running.
        """
        if not self.is_running or self.container_id is None:
            raise RuntimeError("Container is not running")

        # Build exec command
        exec_cmd: list[str] = [
            "devcontainer",
            "exec",
            "--container-id",
            self.container_id,
            "-i",  # Keep stdin attached
            *command,
        ]

        self.process = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # These will not be None when PIPE is specified
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None

        return self.process.stdin, self.process.stdout, self.process.stderr

    async def __aenter__(self) -> "DevContainer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
