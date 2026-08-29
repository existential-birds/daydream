"""Launch documentation carries real measured values, never placeholders (M21, M23)."""

import re
from pathlib import Path

DOC = Path(__file__).parents[2] / "docs" / "training-launch.md"


def test_no_placeholders_in_launch_doc() -> None:
    text = DOC.read_text()
    assert not re.search(r"<[A-Za-z][^>]*>|TBD|PLACEHOLDER|example\.com", text)  # M21: placeholders do not satisfy


def test_names_corpora_splits_model_hardware_walltime_costs() -> None:
    text = DOC.read_text().lower()
    for section in ("corpus", "split", "model", "hardware", "wall time", "cost"):
        assert section in text
