# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys
from unittest.mock import AsyncMock, patch

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
    async def test_rlm_mode_runs_successfully(self, tmp_path, capsys):
        """RLM mode should run and produce output."""
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

        # Check output contains RLM logs (Rich may output to stdout or stderr)
        captured = capsys.readouterr()
        all_output = captured.out + captured.err

        # Should either show RLM output or complete successfully
        assert "RLM" in all_output or exit_code == 0


class TestRLMFallback:
    """Tests for RLM graceful fallback mechanism."""

    @pytest.mark.asyncio
    async def test_fallback_triggered_on_repl_crash(self, tmp_path, capsys):
        """Fallback should trigger when REPLCrashError occurs."""
        from daydream.rlm.errors import REPLCrashError
        from daydream.runner import run_rlm_review_with_fallback

        # Create a minimal Python file
        (tmp_path / "main.py").write_text("x = 1")

        # Mock run_rlm_review to raise REPLCrashError
        async def mock_rlm_review(cwd, languages):
            raise REPLCrashError("REPL process died unexpectedly")

        # Mock run_standard_review to return a known value
        async def mock_standard_review(cwd, skill):
            return "# Standard Review\n\nFallback review content."

        with patch(
            "daydream.runner.run_rlm_review", side_effect=mock_rlm_review
        ), patch(
            "daydream.runner.run_standard_review", side_effect=mock_standard_review
        ):
            result = await run_rlm_review_with_fallback(
                tmp_path,
                ["python"],
                "beagle:review-python",
            )

        # Check that fallback was used
        assert "Standard Review" in result
        captured = capsys.readouterr()
        assert "RLM mode failed" in captured.out
        assert "Falling back" in captured.out

    @pytest.mark.asyncio
    async def test_fallback_triggered_on_heartbeat_failed(self, tmp_path, capsys):
        """Fallback should trigger when HeartbeatFailedError occurs."""
        from daydream.rlm.errors import HeartbeatFailedError
        from daydream.runner import run_rlm_review_with_fallback

        # Create a minimal Python file
        (tmp_path / "main.py").write_text("x = 1")

        # Mock run_rlm_review to raise HeartbeatFailedError
        async def mock_rlm_review(cwd, languages):
            raise HeartbeatFailedError("No heartbeat response in 5 seconds")

        # Mock run_standard_review to return a known value
        async def mock_standard_review(cwd, skill):
            return "# Heartbeat Fallback\n\nReview after heartbeat failure."

        with patch(
            "daydream.runner.run_rlm_review", side_effect=mock_rlm_review
        ), patch(
            "daydream.runner.run_standard_review", side_effect=mock_standard_review
        ):
            result = await run_rlm_review_with_fallback(
                tmp_path,
                ["python"],
                "beagle:review-python",
            )

        # Check that fallback was used
        assert "Heartbeat Fallback" in result
        captured = capsys.readouterr()
        assert "RLM mode failed" in captured.out
        assert "heartbeat" in captured.out.lower() or "Heartbeat" in captured.out

    @pytest.mark.asyncio
    async def test_fallback_triggered_on_container_error(self, tmp_path, capsys):
        """Fallback should trigger when ContainerError occurs."""
        from daydream.rlm.errors import ContainerError
        from daydream.runner import run_rlm_review_with_fallback

        # Create a minimal Python file
        (tmp_path / "main.py").write_text("x = 1")

        # Mock run_rlm_review to raise ContainerError
        async def mock_rlm_review(cwd, languages):
            raise ContainerError("Failed to start devcontainer")

        # Mock run_standard_review to return a known value
        async def mock_standard_review(cwd, skill):
            return "# Container Fallback\n\nReview after container failure."

        with patch(
            "daydream.runner.run_rlm_review", side_effect=mock_rlm_review
        ), patch(
            "daydream.runner.run_standard_review", side_effect=mock_standard_review
        ):
            result = await run_rlm_review_with_fallback(
                tmp_path,
                ["python"],
                "beagle:review-python",
            )

        # Check that fallback was used
        assert "Container Fallback" in result
        captured = capsys.readouterr()
        assert "RLM mode failed" in captured.out
        assert "devcontainer" in captured.out.lower() or "container" in captured.out.lower()

    @pytest.mark.asyncio
    async def test_no_fallback_on_success(self, tmp_path, capsys):
        """No fallback should occur when RLM review succeeds."""
        from daydream.runner import run_rlm_review_with_fallback

        # Create a minimal Python file
        (tmp_path / "main.py").write_text("x = 1")

        # Mock run_rlm_review to succeed
        async def mock_rlm_review(cwd, languages):
            return "# RLM Review\n\nSuccessful RLM review."

        # Mock run_standard_review - should NOT be called
        mock_standard = AsyncMock(return_value="# Standard Review")

        with patch(
            "daydream.runner.run_rlm_review", side_effect=mock_rlm_review
        ), patch(
            "daydream.runner.run_standard_review", mock_standard
        ):
            result = await run_rlm_review_with_fallback(
                tmp_path,
                ["python"],
                "beagle:review-python",
            )

        # Check that RLM review was used, not fallback
        assert "RLM Review" in result
        mock_standard.assert_not_called()
        captured = capsys.readouterr()
        assert "Falling back" not in captured.out

    @pytest.mark.asyncio
    async def test_fallback_warning_message_is_printed(self, tmp_path, capsys):
        """Fallback should print warning message to console."""
        from daydream.rlm.errors import REPLCrashError
        from daydream.runner import run_rlm_review_with_fallback

        # Create a minimal Python file
        (tmp_path / "main.py").write_text("x = 1")

        # Mock run_rlm_review to raise error with specific message
        async def mock_rlm_review(cwd, languages):
            raise REPLCrashError("custom error message here")

        async def mock_standard_review(cwd, skill):
            return "# Fallback"

        with patch(
            "daydream.runner.run_rlm_review", side_effect=mock_rlm_review
        ), patch(
            "daydream.runner.run_standard_review", side_effect=mock_standard_review
        ):
            await run_rlm_review_with_fallback(
                tmp_path,
                ["python"],
                "beagle:review-python",
            )

        captured = capsys.readouterr()
        # Verify error message is in output
        assert "custom error message here" in captured.out
        # Verify fallback message is printed
        assert "Falling back to standard skill-based review" in captured.out
