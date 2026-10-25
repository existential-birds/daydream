"""Verbose diagnostics preserve exception structure within privacy and size bounds.

Check chaining, suppression, notes/groups, redaction before truncation, control
characters, and the fixed failure marker. Imports must remain stdlib plus privacy
policy so a fatal path cannot initialize logging, Rich, or telemetry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream import git_ops
from daydream.diagnostics import format_verbose_exception


def test_implicit_context_shown_when_not_suppressed() -> None:
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

def test_suppressed_context_absent() -> None:
    try:
        try:
            raise KeyError("implicit-inner")
        except KeyError:
            raise ValueError("implicit-outer") from None
    except ValueError as exc:
        captured = exc
    out = format_verbose_exception(captured)
    assert "implicit-inner" not in out
    assert "During handling" not in out

def test_exception_group_width_capped_and_recognizable() -> None:
    group = ExceptionGroup("fan-out failed", [ValueError(f"child-{i}") for i in range(10)],)
    out = format_verbose_exception(group)
    assert "fan-out failed" in out
    assert "child-0" in out
    assert "child-8" not in out
    assert "child-9" not in out
    assert "more" in out



def test_mixed_width_diagnostic_is_capped_with_single_marker() -> None:
    outer = RuntimeError("HEAD-SENTINEL " + "xé漢😀" * 50_000)
    inner = git_ops.GitError("root cause tail: git reflog expired")
    outer.__cause__ = inner
    out = format_verbose_exception(outer)
    assert len(out.encode("utf-8")) <= 65536
    assert out.count("[VERBOSE_DIAGNOSTIC_TRUNCATED]") == 1
    assert out.startswith("Traceback") or "HEAD-SENTINEL" in out
    assert "root cause tail" in out
    assert out.index("HEAD-SENTINEL") < out.index("[VERBOSE_DIAGNOSTIC_TRUNCATED]")
    assert out.index("[VERBOSE_DIAGNOSTIC_TRUNCATED]") < out.index("root cause tail")

def test_formatter_failure_returns_only_unavailable_marker(monkeypatch: pytest.MonkeyPatch,) -> None:
    monkeypatch.setattr("daydream.observability.privacy.PrivacyPolicy.text",
        lambda self, value: (_ for _ in ()).throw(RuntimeError("redaction boom")),
    )
    assert format_verbose_exception(RuntimeError("any")) == "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]"


def test_middle_credential_never_survives_full_value_redaction_and_cap() -> None:
    """A credential buried in the middle of a huge payload is gone after the single final cap: redaction runs over
    the COMPLETE value first, so no pre-cap windowing can leave a straddling fragment behind."""
    message = "x" * 100_000 + " token=ghp_" + "R" * 10 + " " + "y" * 100_000
    outer = RuntimeError(message)
    outer.__cause__ = git_ops.GitError("root tail")
    out = format_verbose_exception(outer)
    assert "ghp_" not in out
    assert "R" * 10 not in out
    assert len(out.encode("utf-8")) <= 65536
    assert out.count("[VERBOSE_DIAGNOSTIC_TRUNCATED]") == 1

def test_credential_near_size_bound_is_redacted() -> None:
    """A whole diagnostic below the cap still redacts a credential near its tail."""
    message = "x" * (65536 - 220) + "token=ghp_" + "W" * 8 + " tail"
    out = format_verbose_exception(RuntimeError(message))
    assert "ghp_" not in out
    assert "[REDACTED" in out
    assert "tail" in out

def test_diagnostics_module_avoids_banned_apis_and_imports() -> None:
    src = (Path(__file__).resolve().parents[1] / "daydream" / "diagnostics.py").read_text()
    for token in (
        "basicConfig", "print_exception", "import logging", "from logging", "import rich", "from rich", "opentelemetry",
        "from daydream.backends", "from daydream.trajectory", "_redact_windows", "_REDACTION_WINDOW",
    ):
        assert token not in src, f"diagnostics.py must not use {token!r}"
