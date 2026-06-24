#!/usr/bin/env python3
"""Tests for replay._is_transient.

Standalone (stdlib unittest) — the bench scripts are not wired into the package
pytest suite. Run directly:  python3 test_replay.py
"""

from __future__ import annotations

import unittest

from replay import _is_transient

# A real `PiError: terminated` Backend-Execution-Error panel as captured in
# daydream's stdout when the z.ai/GLM provider drops the streaming connection.
PIERROR_TERMINATED_STDOUT = """\
Reviewing changed files...

╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ PiError: terminated
╚═══════════════════════════════════════════════════════════════════════════

╔═ ⚠️  Fatal Error ═════════════════════════════════════════════════════════
║ terminated
╚═══════════════════════════════════════════════════════════════════════════
"""

ECONNRESET_STDOUT = """\
╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ PiError: read ECONNRESET
╚═══════════════════════════════════════════════════════════════════════════
"""

SOCKET_HANG_UP_STDOUT = """\
╔═ ⚠️  Fatal Error ═════════════════════════════════════════════════════════
║ socket hang up
╚═══════════════════════════════════════════════════════════════════════════
"""

OVERLOADED_429_STDOUT = """\
╔═ ⚠️  Backend Execution Error ═════════════════════════════════════════════
║ 429 service may be temporarily overloaded, please try again later
╚═══════════════════════════════════════════════════════════════════════════
"""

# A clean, successful review with NO error. Deliberately mentions "error" in
# benign prose ("error handling", "ECONNRESET") to guard against over-broad
# matching that would retry a perfectly good review forever.
CLEAN_SUCCESS_STDOUT = """\
Review complete. 2 findings written to findings.json.

  - [medium] The new fetch() call has no error handling; a dropped
    connection (e.g. ECONNRESET) would crash the worker. Wrap it in
    try/except and log the failure.
  - [low] The "terminated" status string is hardcoded; prefer an enum.

Done.
"""


class IsTransientTest(unittest.TestCase):
    def test_pierror_terminated_is_transient(self) -> None:
        # Regression guard: the stream-drop that motivated this fix MUST retry.
        self.assertTrue(_is_transient(PIERROR_TERMINATED_STDOUT))

    def test_econnreset_is_transient(self) -> None:
        self.assertTrue(_is_transient(ECONNRESET_STDOUT))

    def test_socket_hang_up_is_transient(self) -> None:
        self.assertTrue(_is_transient(SOCKET_HANG_UP_STDOUT))

    def test_429_overload_still_transient(self) -> None:
        self.assertTrue(_is_transient(OVERLOADED_429_STDOUT))

    def test_clean_success_is_not_transient(self) -> None:
        # Mentions "error"/"ECONNRESET"/"terminated" in prose but did NOT error.
        self.assertFalse(_is_transient(CLEAN_SUCCESS_STDOUT))

    def test_empty_is_not_transient(self) -> None:
        self.assertFalse(_is_transient(""))


if __name__ == "__main__":
    unittest.main()
