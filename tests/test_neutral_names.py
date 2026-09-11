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


def test_schema_files_neutral() -> None:
    from pathlib import Path

    import daydream.training

    schema_dir = Path(daydream.training.__file__).parent / "schema"
    assert (schema_dir / "record-schema.json").is_file()
    assert not (schema_dir / "v1.json").exists()
    assert not (schema_dir / "v2.json").exists()
    assert (schema_dir / "curation-manifest.json").is_file()
    assert not (schema_dir / "curation-manifest-v1.json").exists()


def test_versioned_test_files_gone() -> None:
    from pathlib import Path

    tests_dir = Path(__file__).parent
    for stale in (
        "test_corpus_v2.py",
        "test_stacks_v2_load.py",
        "test_training_contract_v1_v2.py",
        "test_training_coordinator_v2.py",
        "test_training_rft_v2_sha.py",
        "test_training_rubric_v2.py",
    ):
        assert not (tests_dir / stale).exists()


def test_rubric_module_neutral() -> None:
    importlib.import_module("daydream.training.rubric")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("daydream.training.rubric_v2")
