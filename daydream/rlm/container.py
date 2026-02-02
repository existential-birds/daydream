# daydream/rlm/container.py
"""Devcontainer management using trailofbits/claude-code-devcontainer.

This module wraps the `devc` CLI from trailofbits/claude-code-devcontainer
for sandboxed Python execution.
"""

import asyncio
from pathlib import Path
from typing import Any

from daydream.rlm.errors import ContainerError


class DevContainer:
    """Wrapper around trailofbits/claude-code-devcontainer.

    Uses the `devc` CLI for container lifecycle management.
    """

    def __init__(self, workspace_path: Path):
        """Initialize DevContainer.

        Args:
            workspace_path: Path to the workspace to mount in container.
        """
        self.workspace_path = workspace_path
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if container is running."""
        return self._running

    async def _run_devc(self, args: list[str], timeout: float = 120.0) -> tuple[str, str, int]:
        """Run a devc CLI command.

        Args:
            args: Arguments to pass to devc.
            timeout: Maximum time to wait for command.

        Returns:
            Tuple of (stdout, stderr, returncode).

        Raises:
            ContainerError: If command fails or times out.
        """
        cmd = ["devc", *args]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            return stdout.decode(), stderr.decode(), proc.returncode or 0
        except asyncio.TimeoutError:
            raise ContainerError(f"devc command timed out after {timeout}s: {' '.join(cmd)}")
        except FileNotFoundError:
            raise ContainerError(
                "devc CLI not found. Install with: "
                "git clone https://github.com/trailofbits/claude-code-devcontainer ~/.claude-devcontainer && "
                "~/.claude-devcontainer/install.sh self-install"
            )

    async def start(self) -> None:
        """Start the devcontainer.

        Raises:
            ContainerError: If container fails to start.
        """
        if self._running:
            return

        # Ensure .devcontainer config exists
        devcontainer_dir = self.workspace_path / ".devcontainer"
        if not devcontainer_dir.exists():
            stdout, stderr, rc = await self._run_devc(
                ["template", str(self.workspace_path)]
            )
            if rc != 0:
                raise ContainerError(f"devc template failed: {stderr}")

        # Start container
        stdout, stderr, rc = await self._run_devc(["up"])
        if rc != 0:
            raise ContainerError(f"devc up failed: {stderr}")

        self._running = True

    async def stop(self) -> None:
        """Stop the devcontainer."""
        if not self._running:
            return

        await self._run_devc(["down"])
        self._running = False

    async def exec_python(self, code: str) -> tuple[str, str, int]:
        """Execute Python code in the container.

        Args:
            code: Python code to execute.

        Returns:
            Tuple of (stdout, stderr, returncode).

        Raises:
            RuntimeError: If container is not running.
        """
        if not self._running:
            raise RuntimeError("Container is not running")

        proc = await asyncio.create_subprocess_exec(
            "devcontainer", "exec",
            "--workspace-folder", str(self.workspace_path),
            "python", "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode(), stderr.decode(), proc.returncode or 0

    async def __aenter__(self) -> "DevContainer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
