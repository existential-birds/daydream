# tests/rlm/test_container.py
"""Tests for devcontainer management."""

import pytest

from daydream.rlm.container import (
    ContainerConfig,
    DevContainer,
    find_devcontainer_config,
)


class TestContainerConfig:
    """Tests for ContainerConfig dataclass."""

    def test_container_config_defaults(self):
        """ContainerConfig should have sensible defaults."""
        cfg = ContainerConfig(workspace_path="/repo")
        assert cfg.workspace_path == "/repo"
        assert cfg.container_id is None
        assert cfg.mount_readonly is True

    def test_container_config_custom(self):
        """ContainerConfig should accept custom values."""
        cfg = ContainerConfig(
            workspace_path="/repo",
            container_id="abc123",
            mount_readonly=False,
        )
        assert cfg.container_id == "abc123"
        assert cfg.mount_readonly is False


class TestFindDevcontainerConfig:
    """Tests for find_devcontainer_config function."""

    def test_find_devcontainer_config_not_found(self, tmp_path):
        """Should return None if no .devcontainer found."""
        result = find_devcontainer_config(tmp_path)
        assert result is None

    def test_find_devcontainer_config_found(self, tmp_path):
        """Should return path to devcontainer.json if found."""
        devcontainer_dir = tmp_path / ".devcontainer"
        devcontainer_dir.mkdir()
        config_file = devcontainer_dir / "devcontainer.json"
        config_file.write_text('{"name": "test"}')

        result = find_devcontainer_config(tmp_path)
        assert result == config_file


class TestDevContainer:
    """Tests for DevContainer class."""

    def test_devcontainer_init(self):
        """DevContainer should initialize with config."""
        cfg = ContainerConfig(workspace_path="/repo")
        dc = DevContainer(cfg)
        assert dc.config == cfg
        assert dc.process is None
        assert dc.container_id is None

    def test_devcontainer_not_running_initially(self):
        """DevContainer should not be running initially."""
        cfg = ContainerConfig(workspace_path="/repo")
        dc = DevContainer(cfg)
        assert dc.is_running is False

    @pytest.mark.asyncio
    async def test_devcontainer_exec_requires_running(self):
        """exec_command should raise if container not running."""
        cfg = ContainerConfig(workspace_path="/repo")
        dc = DevContainer(cfg)
        with pytest.raises(RuntimeError, match="not running"):
            await dc.exec_command(["echo", "test"])
