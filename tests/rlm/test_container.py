# tests/rlm/test_container.py
"""Tests for devcontainer management using devc CLI."""

from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest

from daydream.rlm.container import DevContainer
from daydream.rlm.errors import ContainerError


class TestDevContainer:
    """Tests for DevContainer class."""

    def test_init(self, tmp_path):
        """DevContainer should initialize with workspace path."""
        dc = DevContainer(tmp_path)
        assert dc.workspace_path == tmp_path
        assert dc.is_running is False

    def test_not_running_initially(self, tmp_path):
        """DevContainer should not be running initially."""
        dc = DevContainer(tmp_path)
        assert dc.is_running is False

    @pytest.mark.asyncio
    async def test_exec_python_requires_running(self, tmp_path):
        """exec_python should raise if container not running."""
        dc = DevContainer(tmp_path)
        with pytest.raises(RuntimeError, match="not running"):
            await dc.exec_python("print('hello')")

    @pytest.mark.asyncio
    async def test_start_creates_devcontainer_config(self, tmp_path):
        """start should call devc template if .devcontainer doesn't exist."""
        dc = DevContainer(tmp_path)

        with patch.object(dc, "_run_devc", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("", "", 0)
            await dc.start()

            # Should call template first (no .devcontainer), then up
            assert mock_run.call_count == 2
            mock_run.assert_any_call(["template", str(tmp_path)])
            mock_run.assert_any_call(["up"])
            assert dc.is_running is True

    @pytest.mark.asyncio
    async def test_start_skips_template_if_config_exists(self, tmp_path):
        """start should skip devc template if .devcontainer exists."""
        devcontainer_dir = tmp_path / ".devcontainer"
        devcontainer_dir.mkdir()

        dc = DevContainer(tmp_path)

        with patch.object(dc, "_run_devc", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("", "", 0)
            await dc.start()

            # Should only call up (not template)
            mock_run.assert_called_once_with(["up"])
            assert dc.is_running is True

    @pytest.mark.asyncio
    async def test_start_raises_on_failure(self, tmp_path):
        """start should raise ContainerError on devc failure."""
        devcontainer_dir = tmp_path / ".devcontainer"
        devcontainer_dir.mkdir()

        dc = DevContainer(tmp_path)

        with patch.object(dc, "_run_devc", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("", "error message", 1)

            with pytest.raises(ContainerError, match="devc up failed"):
                await dc.start()

    @pytest.mark.asyncio
    async def test_stop(self, tmp_path):
        """stop should call devc down."""
        dc = DevContainer(tmp_path)
        dc._running = True  # Simulate running state

        with patch.object(dc, "_run_devc", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("", "", 0)
            await dc.stop()

            mock_run.assert_called_once_with(["down"])
            assert dc.is_running is False

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_running(self, tmp_path):
        """stop should be a no-op if not running."""
        dc = DevContainer(tmp_path)

        with patch.object(dc, "_run_devc", new_callable=AsyncMock) as mock_run:
            await dc.stop()
            mock_run.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager(self, tmp_path):
        """DevContainer should work as async context manager."""
        devcontainer_dir = tmp_path / ".devcontainer"
        devcontainer_dir.mkdir()

        with patch.object(DevContainer, "_run_devc", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ("", "", 0)

            async with DevContainer(tmp_path) as dc:
                assert dc.is_running is True

            assert dc.is_running is False
