"""Tests for the four-stage training coordinator's input contract (#1093).

The legacy v1 ``corpus`` input was removed (#1093); ``PipelineConfig`` takes
exactly one input, ``projection`` — a frozen projection directory. The stage
behavior itself is exercised against the projection loader in
``tests/test_training_coordinator_v2.py`` and the committed projection fixture
suite.
"""

from pathlib import Path
from typing import Any

import pytest

from daydream.training.coordinator import PipelineConfig


def test_cli_verb_wired(cli_runner: Any) -> None:
    r = cli_runner.invoke(["train", "--help"])
    assert r.exit_code == 0


def test_pipeline_config_rejects_legacy_corpus_kwarg() -> None:
    """#1093: the v1 `corpus` input is gone; the canonical input is `projection`."""

    with pytest.raises(TypeError):
        PipelineConfig(out_dir=Path("/tmp/x"), corpus=Path("/tmp/corpus.jsonl"))  # type: ignore[call-arg]


def test_pipeline_config_projection_only() -> None:
    """#1093: `projection` is the one and only pipeline input (package renamed in this task)."""

    cfg = PipelineConfig(out_dir=Path("/tmp/x"), projection=Path("/tmp/proj"))
    assert cfg.projection == Path("/tmp/proj")


def test_pipeline_config_requires_projection() -> None:

    with pytest.raises(ValueError, match="projection"):
        PipelineConfig(out_dir=Path("/tmp/x"))


def test_train_cli_rejects_legacy_corpus_flag(cli_runner: Any) -> None:
    """#1093: `--corpus` is gone from the train parser; `--projection` replaced it."""
    r = cli_runner.invoke(["train", "--corpus", "x.jsonl", "--out", str(Path("/tmp/train-out"))])
    assert r.exit_code != 0
