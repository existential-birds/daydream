"""Self-contained isolated Harbor verifier for the Harbor verifier image.

Stdlib + httpx only. Never imports daydream: this file must run unchanged
inside a compiled-grade verifier image that has no daydream wheel. It wires a
bounded per-pair judge prompt, two isolated external judge clients (Anthropic
Messages + OpenAI-compatible) behind one ``complete_json`` seam, strict verdict
parsing, a shared retry/redirect/timeout policy, a concurrency-10 runner, and
``run_verifier`` which writes ``reward.json`` / ``reward-details.json``
atomically.
"""

from __future__ import annotations

from typing import Any, Protocol

import verifier_core


class _AsyncHttpClient(Protocol):
    """An ``httpx.AsyncClient``-shaped seam for the injected fake clients."""

    async def post(
        self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float
    ) -> Any:
        """POST ``json`` to ``url`` and return an httpx-like response object."""


_VERIFIER_THRESHOLD = verifier_core.CONFIDENCE_THRESHOLD


def main() -> None:
    ...


if __name__ == "__main__":
    main()