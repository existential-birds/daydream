"""M8/M10 gates: real-path CLI smoke over the committed fixture, replay determinism, no-default-mutation."""

import json
from pathlib import Path

from daydream.cli import main

FIXTURE = Path(__file__).parent / "fixtures" / "training" / "calibration"


def _run_main(argv: list[str]) -> int:
    try:
        main([*argv])
    except SystemExit as exc:  # pragma: no cover - main raises SystemExit(0) on success paths
        return exc.code if isinstance(exc.code, int) else 0
    return 0


def test_fixture_run_produces_valid_replayable_artifact(tmp_path: Path) -> None:
    """Real-path: cli.main over the checked-in fixture, twice."""
    out1, out2 = tmp_path / "r1", tmp_path / "r2"
    argv = [
        "corpus",
        "calibrate-reward",
        "--corpus-dir",
        str(FIXTURE / "corpus"),
        "--gold-labels",
        str(FIXTURE / "gold.json"),
        "--breakdowns",
        str(FIXTURE / "breakdowns.json"),
        "--run-id",
        "fixture-run",
        "--seed",
        "42",
        "--candidate",
        "w_fp=0.1,0.2,0.3",
        "--grid-points",
        "5",
        "--bootstrap-resamples",
        "200",
        "--out",
    ]
    assert _run_main([*argv, str(out1)]) == 0
    art = json.loads((out1 / "calibration.json").read_text())
    assert art["schema_version"] == "calibration-artifact-v1"
    assert art["stage0_analysis"]["status"] in {"unavailable", "ok"}  # explicit, never missing
    assert _run_main([*argv, str(out2)]) == 0
    assert (out1 / "calibration.json").read_bytes() == (out2 / "calibration.json").read_bytes()  # AC 1


def test_no_production_default_mutation() -> None:
    src = Path("daydream/training/calibration.py").read_text()
    assert "DEFAULT_WEIGHTS" not in src  # M10: never imported-and-mutated
    # reward.py itself untouched:
    assert Path("daydream/training/reward.py").read_text().count("DEFAULT_WEIGHTS =") == 1
