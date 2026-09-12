"""The committed projection fixture must pass the canonical loader gates and
drive the CI training dry run (issue #1093, task 9)."""

from pathlib import Path


def test_committed_projection_fixture_passes_loader_gates(tmp_path: Path) -> None:
    """The committed fixture must load via the canonical projection loader."""
    from daydream.training.stacks import load_v2_projection

    projection = load_v2_projection(
        Path("tests/fixtures/training/projection-50"), allow_copyleft=frozenset()
    )
    assert len(projection.records) > 0
    assert projection.digest  # directory-level digest present


def test_pipeline_dry_run_over_committed_fixture(tmp_path: Path) -> None:
    from daydream.training.coordinator import PipelineConfig, run_pipeline

    manifest = run_pipeline(
        PipelineConfig(
            projection=Path("tests/fixtures/training/projection-50"), out_dir=tmp_path
        ),
        dry_run=True,
    )
    assert manifest["stages"]["stage0"]["status"] == "complete"
    assert manifest["stages"]["stage0"]["gate"]["passed"] is True
