"""Drive redrive_post's real CLI and canonical posting path with fake gh at the boundary.

Report parsing, classification, payload construction, and posting delegation stay real.
"""

from __future__ import annotations

import sys

import pytest

import scripts.redrive_post as redrive


def _run_cli(*argv: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["redrive_post", *argv])
    redrive.main()
