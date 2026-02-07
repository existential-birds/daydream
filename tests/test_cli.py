# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys

import pytest

from daydream.cli import _parse_args


def test_default_backend_is_claude(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.backend == "claude"


def test_backend_flag_codex(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--backend", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_backend_short_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "-b", "codex"])
    config = _parse_args()
    assert config.backend == "codex"


def test_model_default_is_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python"])
    config = _parse_args()
    assert config.model is None


def test_model_explicit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--model", "sonnet"])
    config = _parse_args()
    assert config.model == "sonnet"


def test_invalid_backend_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/project", "--python", "--backend", "invalid"])
    with pytest.raises(SystemExit):
        _parse_args()


def test_review_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--backend", "claude", "--review-backend", "codex",
    ])
    config = _parse_args()
    assert config.backend == "claude"
    assert config.review_backend == "codex"


def test_fix_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--fix-backend", "codex",
    ])
    config = _parse_args()
    assert config.fix_backend == "codex"


def test_test_backend_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "daydream", "/tmp/project", "--python",
        "--test-backend", "codex",
    ])
    config = _parse_args()
    assert config.test_backend == "codex"
