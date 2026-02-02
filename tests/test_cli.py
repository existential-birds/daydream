# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys
from unittest.mock import patch

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
