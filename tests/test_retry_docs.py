"""Documentation contract for the retry-recovery settings (issue #734, part of PR 2).

Pins the operator-facing README section: the config key, its env counterparts,
the interaction with the invocation deadline and the fix file-group budget, and
the sanctioned ``0`` value. Pure text assertions — no run, backend, or clock.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_readme_documents_the_retry_settings_and_their_interaction() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Retry recovery", 1)[1].split("\n### ", 1)[0]

    for token in ("retry_recovery_allowance_s", "300", "DAYDREAM_PI_RETRY_ATTEMPTS",
                  "DAYDREAM_PI_RETRY_RECOVERY_ALLOWANCE_S", "group_max_wall_s",
                  "additional to the invocation deadline", "clamped",
                  "first retryable failure", "never caps", "0"):
        assert token in section, token
