"""Tests for daydream.training.exclusion helpers (C5 + C8 enforcement)."""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.training.exclusion import (
    is_copyleft,
    is_excluded,
    load_exclusion_list,
)

SEED_EXCLUSION_REPOS = {
    "getsentry/sentry",
    "grafana/grafana",
    "calcom/cal.com",
    "discourse/discourse",
    "keycloak/keycloak",
}


def test_load_exclusion_list_contains_seed_repos() -> None:
    excluded = load_exclusion_list()
    assert SEED_EXCLUSION_REPOS.issubset(excluded)


def test_load_exclusion_list_ignores_blank_and_comment_lines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmp_file = tmp_path / "test_excl.txt"
    tmp_file.write_text(
        "# leading comment\n"
        "foo/bar\n"
        "\n"
        "baz/qux\n"
        "#another comment\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("daydream.training.exclusion.EXCLUSION_PATH", tmp_file)
    assert load_exclusion_list() == frozenset({"foo/bar", "baz/qux"})


def test_is_excluded_returns_false_for_none() -> None:
    assert is_excluded(None) is False


def test_is_excluded_returns_true_for_seed_repo() -> None:
    assert is_excluded("getsentry/sentry") is True


def test_is_copyleft_opt_in_overrides_skip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmp_file = tmp_path / "test_copyleft.txt"
    tmp_file.write_text("gnu/coreutils\n", encoding="utf-8")
    monkeypatch.setattr("daydream.training.exclusion.COPYLEFT_PATH", tmp_file)
    assert is_copyleft("gnu/coreutils", frozenset()) is True
    assert is_copyleft("gnu/coreutils", frozenset({"gnu/coreutils"})) is False


def test_is_copyleft_returns_false_for_none() -> None:
    assert is_copyleft(None, frozenset()) is False
