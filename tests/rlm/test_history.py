"""Tests for conversation history management."""

import pytest
from daydream.rlm.history import ConversationHistory, Exchange


class TestExchange:
    """Tests for Exchange dataclass."""

    def test_token_estimate(self):
        """Token estimate should be ~4 chars per token."""
        ex = Exchange(iteration=1, code="x" * 100, output="y" * 100, error=None)
        # 200 chars = ~50 tokens
        assert 40 <= ex.token_estimate() <= 60

    def test_token_estimate_with_error(self):
        """Token estimate should include error text."""
        ex = Exchange(iteration=1, code="x" * 40, output="y" * 40, error="z" * 40)
        # 120 chars = ~30 tokens
        assert 25 <= ex.token_estimate() <= 35


class TestConversationHistory:
    """Tests for ConversationHistory class."""

    def test_add_exchange(self):
        """Should add exchanges to history."""
        history = ConversationHistory()
        history.add_exchange(1, "x=1", "ok")
        assert len(history._exchanges) == 1
        assert history._exchanges[0].iteration == 1

    def test_format_empty(self):
        """Empty history should return empty string."""
        history = ConversationHistory()
        assert history.format_for_prompt() == ""

    def test_format_single_exchange(self):
        """Single exchange should format correctly."""
        history = ConversationHistory()
        history.add_exchange(1, "print(1)", "1")
        formatted = history.format_for_prompt()
        assert "Iteration 1" in formatted
        assert "print(1)" in formatted
        assert "1" in formatted

    def test_format_respects_recent_count(self):
        """Should only include recent_count most recent exchanges."""
        history = ConversationHistory(recent_count=2)
        for i in range(5):
            history.add_exchange(i+1, f"x={i}", str(i))
        formatted = history.format_for_prompt()
        # Should have iterations 4 and 5 (most recent 2)
        assert "Iteration 4" in formatted
        assert "Iteration 5" in formatted
        # Should NOT have iteration 1 or 2
        assert "Iteration 1" not in formatted
        assert "Iteration 2" not in formatted

    def test_clear(self):
        """Clear should reset all history."""
        history = ConversationHistory()
        history.add_exchange(1, "x=1", "1")
        history._summaries.append("summary")
        history.clear()
        assert len(history._exchanges) == 0
        assert len(history._summaries) == 0

    def test_token_budget_enforcement(self):
        """Should truncate content to stay within token budget."""
        history = ConversationHistory(
            recent_count=3,
            max_history_tokens=500,  # Small budget
        )
        # Add exchanges with large content
        for i in range(5):
            history.add_exchange(
                iteration=i+1,
                code="x = 1\n" * 100,  # ~600 chars each
                output="output " * 50,  # ~350 chars each
            )

        formatted = history.format_for_prompt()
        estimated_tokens = history._estimate_tokens(formatted)

        # Should be within budget (with some tolerance)
        assert estimated_tokens <= 600  # Allow small overage

    def test_progressive_truncation(self):
        """Should progressively truncate under budget pressure."""
        history = ConversationHistory(
            recent_count=5,
            max_history_tokens=200,  # Very small budget
        )
        for i in range(5):
            history.add_exchange(i+1, "code" * 50, "out" * 50)

        formatted = history.format_for_prompt()
        # Should still produce output, just truncated
        assert "Iteration" in formatted
        assert len(formatted) < 1500  # Reasonably bounded

    def test_includes_error_in_format(self):
        """Should include error text in formatted output."""
        history = ConversationHistory()
        history.add_exchange(1, "1/0", "", error="ZeroDivisionError")
        formatted = history.format_for_prompt()
        assert "ZeroDivisionError" in formatted

    def test_truncates_long_code(self):
        """Should truncate long code blocks."""
        history = ConversationHistory()
        long_code = "x = 1\n" * 200  # Very long
        history.add_exchange(1, long_code, "ok")
        formatted = history.format_for_prompt()
        # Should have truncation indicator
        assert "..." in formatted
        # Should not have all the code
        assert len(formatted) < len(long_code)

    def test_truncates_long_output(self):
        """Should truncate long output."""
        history = ConversationHistory()
        long_output = "result " * 200  # Very long
        history.add_exchange(1, "x=1", long_output)
        formatted = history.format_for_prompt()
        # Should have truncation indicator
        assert "..." in formatted
