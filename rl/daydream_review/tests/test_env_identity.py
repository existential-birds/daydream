"""#1093: the Stage-3 environment is neutrally named and decoupled from the legacy loader."""

from __future__ import annotations

import importlib

import pytest


def test_package_neutral_name() -> None:
    import daydream_review

    assert "daydream_review_v1" not in daydream_review.__file__


def test_task_identity_neutral() -> None:
    from daydream_review.taskset import DEFAULT_TASKSET_ID

    assert DEFAULT_TASKSET_ID == "daydream-review"
    assert "v1" not in DEFAULT_TASKSET_ID


def test_legacy_loader_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream_review.corpus")
