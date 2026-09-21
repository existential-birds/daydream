"""Tests for the verbose fatal-diagnostic formatter (``daydream.diagnostics``).

Pins the observable contract of :func:`format_verbose_exception`: chaining
order, context suppression, notes, exception-group caps, local-variable
privacy, credential redaction (before any size bound), control-character
neutralization, the head + single-marker + root-cause-tail bound, and the
fail-closed ``[VERBOSE_DIAGNOSTIC_UNAVAILABLE]`` marker. The module must stay
import-light (stdlib + ``PrivacyPolicy`` only) so a fatal path can never pull
in logging/rich/telemetry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream import git_ops
from daydream.diagnostics import format_verbose_exception


def test_explicit_cause_printed_in_python_order() -> None:
    inner = git_ops.GitError("isolation probe failure")
    outer = ValueError("outer boom")
    outer.__cause__ = inner
    out = format_verbose_exception(outer)
    assert "ValueError" in out and "outer boom" in out
    assert "GitError" in out and "isolation probe failure" in out
    assert "directly caused by the following exception" in out
    assert out.index("outer boom") < out.index("isolation probe failure")
    assert out.index("directly caused by the following exception") < out.index(
        "isolation probe failure"
    )


def test_implicit_context_shown_when_not_suppressed() -> None:
    try:
        try:
            raise KeyError("implicit-inner")
        except KeyError:
            raise ValueError("implicit-outer")
    except ValueError as exc:
        captured = exc
    out = format_verbose_exception(captured)
    assert "occurred while handling the following" in out
    assert "KeyError" in out and "implicit-inner" in out
    assert "implicit-outer" in out
    assert out.index("implicit-outer") < out.index("implicit-inner")


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


def test_exception_notes_remain_visible() -> None:
    exc = ValueError("boom")
    exc.add_note("operator hint: check disk space")
    out = format_verbose_exception(exc)
    assert "operator hint: check disk space" in out


def test_exception_group_width_capped_and_recognizable() -> None:
    group = ExceptionGroup(
        "fan-out failed",
        [ValueError(f"child-{i}") for i in range(10)],
    )
    out = format_verbose_exception(group)
    assert "fan-out failed" in out
    assert "child-0" in out
    assert "child-8" not in out
    assert "child-9" not in out
    assert "more" in out


def test_local_variables_never_captured() -> None:
    canary = "ghp_" + "A" * 16

    def boom() -> None:
        hidden_local = canary  # noqa: F841 - the point is that this local never leaks
        raise RuntimeError("locals stay private")

    try:
        boom()
    except RuntimeError as exc:
        captured = exc
    out = format_verbose_exception(captured)
    assert canary not in out
    assert "hidden_local" not in out


def test_credential_shaped_strings_redacted() -> None:
    sentinel = "ghp_" + "x" * 16
    outer = RuntimeError(f"command failed near token={sentinel}")
    inner = git_ops.GitError("config: token=opaque-test-12345")
    outer.__cause__ = inner
    out = format_verbose_exception(outer)
    assert sentinel not in out
    assert "opaque-test-12345" not in out
    assert "[REDACTED" in out


def test_environment_literal_secrets_and_username_paths_redacted() -> None:
    environ = {
        "GITHUB_TOKEN": "svc-ghp-abc",
        "API_ENDPOINT_URL": "https://svc-user:svc-pa55@example.com",
    }
    message = (
        "fetching https://svc-user:svc-pa55@example.com with "
        "svc-ghp-abc from /var/tmp/svc-ghp-abc/key"
    )
    out = format_verbose_exception(RuntimeError(message), environ=environ)
    for fragment in ("svc-ghp-abc", "svc-user", "svc-pa55"):
        assert fragment not in out
    assert "[REDACTED_CREDENTIAL]" in out


def test_terminal_control_characters_neutralized() -> None:
    evil = "\x1b[31mRED\x1b[0m\rBEEP\x07\tTAB"
    out = format_verbose_exception(RuntimeError(f"status: {evil}"))
    assert "\x1b" not in out
    assert "\r" not in out
    assert "\x07" not in out
    assert "\n" in out
    assert "\t" in out


def test_redaction_precedes_head_tail_bound() -> None:
    padding = "x" * (65536 - 60)
    sentinel = "ghp_" + "y" * 8 + "z" * 10
    message = padding + " " + sentinel + " tail"
    out = format_verbose_exception(RuntimeError(message))
    assert "ghp_" not in out
    assert "[REDACTED" in out
    assert out.rstrip().endswith("tail") or "tail" in out


def test_large_diagnostics_capped_with_single_marker() -> None:
    outer = RuntimeError("HEAD-SENTINEL " + "x" * 200_000)
    inner = git_ops.GitError("root cause tail: git reflog expired")
    outer.__cause__ = inner
    out = format_verbose_exception(outer)
    assert len(out.encode("utf-8")) <= 65536
    assert out.count("[VERBOSE_DIAGNOSTIC_TRUNCATED]") == 1
    assert out.startswith("Traceback") or "HEAD-SENTINEL" in out
    assert "root cause tail" in out
    assert out.index("HEAD-SENTINEL") < out.index("[VERBOSE_DIAGNOSTIC_TRUNCATED]")
    assert out.index("[VERBOSE_DIAGNOSTIC_TRUNCATED]") < out.index("root cause tail")


def test_formatter_failure_returns_only_unavailable_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "daydream.diagnostics.PrivacyPolicy.text",
        lambda self, value: (_ for _ in ()).throw(RuntimeError("redaction boom")),
    )
    assert format_verbose_exception(RuntimeError("any")) == "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]"


def test_unstringifiable_exception_fails_closed() -> None:
    class Evil(Exception):
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    assert format_verbose_exception(Evil("x")) == "[VERBOSE_DIAGNOSTIC_UNAVAILABLE]"


def test_middle_credential_never_survives_full_value_redaction_and_cap() -> None:
    """A credential buried in the middle of a huge payload is gone after the
    single final cap: redaction runs over the COMPLETE value first, so no
    pre-cap windowing can leave a straddling fragment behind."""
    message = "x" * 100_000 + " token=ghp_" + "R" * 10 + " " + "y" * 100_000
    outer = RuntimeError(message)
    outer.__cause__ = git_ops.GitError("root tail")
    out = format_verbose_exception(outer)
    assert "ghp_" not in out
    assert "R" * 10 not in out
    assert len(out.encode("utf-8")) <= 65536
    assert out.count("[VERBOSE_DIAGNOSTIC_TRUNCATED]") == 1


def test_credential_straddling_final_cap_still_redacted() -> None:
    """A credential near the very end of the payload sits inside the retained
    root-cause tail after the final cap; it must be redacted because redaction
    ran on the complete pre-cap value, never on sliced windows."""
    message = "x" * (65536 - 220) + "token=ghp_" + "W" * 8 + " tail"
    out = format_verbose_exception(RuntimeError(message))
    assert "ghp_" not in out
    assert "[REDACTED" in out
    assert "tail" in out


def test_diagnostics_module_avoids_banned_apis_and_imports() -> None:
    src = (Path(__file__).resolve().parents[1] / "daydream" / "diagnostics.py").read_text()
    for token in (
        "basicConfig", "print_exception", "import logging", "from logging",
        "import rich", "from rich", "opentelemetry",
        "from daydream.backends", "from daydream.trajectory",
        "_redact_windows", "_REDACTION_WINDOW",
    ):
        assert token not in src, f"diagnostics.py must not use {token!r}"
