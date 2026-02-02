"""Conversation history management for RLM runner."""

from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class Exchange:
    """A single iteration exchange."""
    iteration: int
    code: str
    output: str
    error: str | None = None

    def token_estimate(self) -> int:
        """Estimate tokens (~4 chars per token)."""
        content = self.code + (self.output or "") + (self.error or "")
        return len(content) // 4


@dataclass
class ConversationHistory:
    """Manages conversation history with token budget enforcement.

    Keeps recent exchanges in full detail and enforces a token budget
    by progressively truncating content when the limit is exceeded.
    """

    recent_count: int = 3
    max_history_tokens: int = 12000
    llm_callback: Callable[[str], Awaitable[str]] | None = None

    _exchanges: list[Exchange] = field(default_factory=list)
    _summaries: list[str] = field(default_factory=list)

    def add_exchange(
        self,
        iteration: int,
        code: str,
        output: str,
        error: str | None = None,
    ) -> None:
        """Add a new exchange to history."""
        self._exchanges.append(Exchange(
            iteration=iteration,
            code=code,
            output=output,
            error=error,
        ))

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token)."""
        return len(text) // 4

    def format_for_prompt(self) -> str:
        """Format history for inclusion in continuation prompt.

        Always keeps the NEWEST exchanges (most recent iterations).
        Enforces max_history_tokens by progressively truncating:
        1. First, truncate output/code previews more aggressively
        2. Then, reduce number of exchanges (dropping oldest first)
        3. Finally, drop summaries if needed
        """
        if not self._exchanges:
            return ""

        # Start with generous limits, reduce if over budget
        code_limit = 400
        output_limit = 200
        effective_recent = self.recent_count

        for attempt in range(3):  # Max 3 attempts to fit budget
            parts = ["## Conversation History"]

            # Include summaries (oldest context, dropped first under pressure)
            if self._summaries and attempt < 2:  # Drop summaries on last attempt
                parts.append("\n### Earlier Analysis")
                for summary in self._summaries:
                    parts.append(f"- {summary}")

            # Include recent exchanges - ALWAYS from the END (newest first)
            # e.g., with 10 exchanges and effective_recent=3, we get [7, 8, 9]
            recent_start = max(0, len(self._exchanges) - effective_recent)
            recent = self._exchanges[recent_start:]  # Always newest N exchanges

            if recent:
                parts.append("\n### Recent Iterations")
                for ex in recent:
                    parts.append(f"\n**Iteration {ex.iteration}**")
                    code_preview = ex.code[:code_limit]
                    if len(ex.code) > code_limit:
                        code_preview += "..."
                    parts.append(f"```python\n{code_preview}\n```")
                    if ex.output:
                        out_preview = ex.output[:output_limit]
                        if len(ex.output) > output_limit:
                            out_preview += "..."
                        parts.append(f"Output: `{out_preview}`")
                    if ex.error:
                        parts.append(f"Error: {ex.error[:100]}")

            result = "\n".join(parts)
            estimated_tokens = self._estimate_tokens(result)

            if estimated_tokens <= self.max_history_tokens:
                return result

            # Over budget - reduce limits for next attempt
            code_limit = code_limit // 2
            output_limit = output_limit // 2
            if attempt == 1:
                effective_recent = max(1, effective_recent - 1)

        # Final fallback: return truncated result
        return result[:self.max_history_tokens * 4]  # ~max_tokens chars

    def clear(self) -> None:
        """Reset all history state."""
        self._exchanges.clear()
        self._summaries.clear()
