"""#1093 naming gate: no project-owned v1/v2 names in importable surfaces."""

import importlib

import pytest


def test_canonical_projection_package_imports_under_neutral_name() -> None:
    m = importlib.import_module("daydream.training.corpus_projection")
    assert hasattr(m, "run_build_corpus_v2") is False  # renamed in this task
    assert hasattr(m, "build_frozen_corpus")
    assert hasattr(m, "BuildFrozenCorpusConfig")


def test_old_package_name_gone() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream.training.corpus_v2")


def test_rubric_module_neutral() -> None:
    importlib.import_module("daydream.training.rubric")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream.training.rubric_v2")
