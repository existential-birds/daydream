# tests/test_atif_vendor_smoke.py
"""Smoke test for the vendored ATIF foundation (Phase 1).

Verifies VEND-01..05 success criteria:
- Models import cleanly from daydream.atif
- Validator accepts every Terminus-2 + OpenHands golden fixture
- Validator rejects the deliberately-broken negative fixture
- validate() accepts a dict (not just a path), per CONTEXT.md D-08

Phase 5 (TEST-04) replaces this with a parametrized test_atif_models.py.
"""

import json
from pathlib import Path

import pytest

from daydream.atif import Trajectory, TrajectoryValidator, validate

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "atif_golden"


def test_models_import_cleanly() -> None:
    """VEND-01: top-level public surface is importable."""
    from daydream.atif.models import (
        Agent,
        FinalMetrics,
        Metrics,
        ObservationResult,
        Step,
        ToolCall,
        Trajectory,
    )
    # Exercise __all__: simple no-op references silence flake8 F401.
    assert all(
        cls.__module__.startswith("daydream.atif.models")
        for cls in (Agent, FinalMetrics, Metrics, ObservationResult, Step, ToolCall, Trajectory)
    )


def _golden_paths() -> list[Path]:
    return sorted(p for p in GOLDEN_DIR.rglob("*.json") if "_invalid" not in p.parts)


@pytest.mark.parametrize("golden_path", _golden_paths(), ids=lambda p: p.name)
def test_golden_fixtures_validate(golden_path: Path) -> None:
    """VEND-05 + D-09: every Terminus-2 (v1.6) and OpenHands (v1.5) golden validates."""
    assert validate(golden_path) is True


def test_invalid_fixture_rejected() -> None:
    """D-13: deliberately-broken fixture fails validation."""
    invalid_path = GOLDEN_DIR / "_invalid" / "non-sequential-step-id.json"
    validator = TrajectoryValidator()
    assert validator.validate(invalid_path) is False
    # Surface the specific error category for diagnostic clarity.
    assert any("step_id" in err.lower() for err in validator.errors), validator.errors


def test_validate_via_dict_roundtrip() -> None:
    """D-08: programmatic validate() accepts a dict, not just a path."""
    sample_path = GOLDEN_DIR / "terminus2" / "hello-world-invalid-json.trajectory.json"
    data = json.loads(sample_path.read_text())
    # validate_images=False: in-memory dict has no filesystem anchor.
    assert validate(data, validate_images=False) is True
    # Round-trip through Trajectory model proves model_validate works.
    Trajectory.model_validate(data)
