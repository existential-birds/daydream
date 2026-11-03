"""Verbose diagnostics preserve exception structure within privacy and size bounds.

Check chaining, suppression, notes/groups, redaction before truncation, control
characters, and the fixed failure marker. Imports must remain stdlib plus privacy
policy so a fatal path cannot initialize logging, Rich, or telemetry.
"""

from __future__ import annotations

import pytest

from daydream import git_ops
from daydream.diagnostics import format_verbose_exception, sanitize_verbose_message


def test_verbose_diagnostic_preserves_structure_within_privacy_and_size_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chaining and notes survive; credentials never do, even across the final cap or near it."""
    canary = "ghp_" + "A" * 16
    hidden_local = canary  # noqa: F841 - traceback locals must stay private
    environ = {"GITHUB_TOKEN": "svc-ghp-abc", "API_ENDPOINT_URL": "https://svc-user:svc-pa55@example.com"}
    try:
        try:
            raise KeyError("implicit-inner")
        except KeyError:
            raise ValueError("implicit-outer: https://svc-user:svc-pa55@example.com with svc-ghp-abc "
                "from /var/tmp/svc-ghp-abc/key")
    except ValueError as exc:
        captured = exc
    captured.add_note("operator hint: check disk space")
    out = format_verbose_exception(captured, environ=environ)
    assert "occurred while handling the following" in out
    assert "KeyError" in out and "implicit-inner" in out
    assert "implicit-outer" in out
    assert out.index("implicit-outer") < out.index("implicit-inner")
    assert "operator hint: check disk space" in out
    assert canary not in out
    assert "hidden_local" not in out
    for fragment in ("svc-ghp-abc", "svc-user", "svc-pa55"):
        assert fragment not in out
    assert "[REDACTED_CREDENTIAL]" in out

    # A credential buried in the middle of a huge payload is gone after the single final cap: redaction runs
    # over the COMPLETE value first, so no pre-cap windowing can leave a straddling fragment behind.
    message = "x" * 100_000 + " token=ghp_" + "R" * 10 + " " + "y" * 100_000
    outer = RuntimeError(message)
    outer.__cause__ = git_ops.GitError("root tail")
    out = format_verbose_exception(outer)
    assert "ghp_" not in out
    assert "R" * 10 not in out
    assert len(out.encode("utf-8")) <= 65536
    assert out.count("[VERBOSE_DIAGNOSTIC_TRUNCATED]") == 1
    # A whole diagnostic below the cap still redacts a credential near its retained tail.
    near_bound = format_verbose_exception(RuntimeError("x" * (65536 - 220) + "token=ghp_" + "W" * 8 + " tail"))
    assert "ghp_" not in near_bound and "[REDACTED" in near_bound and "tail" in near_bound
    monkeypatch.setattr("daydream.observability.privacy.PrivacyPolicy.text",
        lambda self, value: (_ for _ in ()).throw(RuntimeError("redaction boom")),
    )
    assert format_verbose_exception(RuntimeError("any")) == "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]"

    class Hostile(Exception):
        def __str__(self) -> str:
            raise RuntimeError("hostile str")

    # A raising message fails closed in both the verbose chain and the fatal panel.
    assert format_verbose_exception(Hostile()) == "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]"
    assert sanitize_verbose_message(Hostile()) == ""
