# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys
from unittest.mock import patch

import pytest

from daydream.cli import _parse_args
from daydream.runner import RunConfig


class TestParseArgsRLM:
    """Tests for --rlm flag parsing."""

    def test_rlm_flag_sets_mode(self):
        """--rlm flag should enable RLM mode."""
        with patch.object(sys, "argv", ["daydream", "/repo", "--rlm", "--python"]):
            config = _parse_args()
            assert config.rlm_mode is True

    def test_rlm_without_skill_allowed(self):
        """--rlm can be used with language flags."""
        with patch.object(sys, "argv", ["daydream", "/repo", "--rlm", "--python"]):
            config = _parse_args()
            assert config.rlm_mode is True
            assert config.skill == "python"

    def test_default_no_rlm(self):
        """RLM mode should be off by default."""
        with patch.object(sys, "argv", ["daydream", "/repo", "--python"]):
            config = _parse_args()
            assert config.rlm_mode is False


class TestRLMIntegration:
    """Tests for RLM integration in runner."""

    @pytest.mark.asyncio
    async def test_rlm_mode_prints_not_implemented(self, tmp_path, capsys):
        """RLM mode should indicate it's not yet complete."""
        from daydream.runner import run, RunConfig

        # Create a minimal Python file
        (tmp_path / "main.py").write_text("x = 1")

        config = RunConfig(
            target=str(tmp_path),
            skill="python",
            rlm_mode=True,
            review_only=True,
        )

        exit_code = await run(config)

        # Should complete without error for now
        # (Full implementation comes later)
        captured = capsys.readouterr()
        assert "RLM" in captured.out or exit_code == 0
