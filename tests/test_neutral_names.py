"""#1093 naming gate: no project-owned v1/v2 names in importable surfaces."""

import importlib
import inspect

import pytest


def test_stacks_neutral_surface() -> None:
    import daydream.training.stacks as stacks

    assert hasattr(stacks, "load_v2_projection")
    assert not hasattr(stacks, "load_dataset")  # legacy v1 loader gone
    assert not hasattr(stacks, "legacy_policy")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream.training.stacks_v2")


def test_coordinator_uses_neutral_stacks() -> None:
    import daydream.training.coordinator as coord

    assert "stacks_v2" not in inspect.getsource(coord)


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
