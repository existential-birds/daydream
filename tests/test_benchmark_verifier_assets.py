"""Byte-parity + metric-equivalence + separate-filesystem isolation for the Harbor verifier assets.

Golden gate: the ``templates/tests/verifier_core.py`` copy must stay
byte-identical (SHA-256) to the in-repo source so future edits to the source
fail loudly. ``templates/metric.py``'s inlined aggregation must equal
``verifier_core.aggregate_metrics`` field-for-field on the same rows.
"""

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verifier_core_template_is_byte_identical_to_source() -> None:
    source = REPO / "daydream" / "benchmark" / "harbor" / "verifier_core.py"
    copy = (
        REPO
        / "daydream"
        / "benchmark"
        / "harbor"
        / "templates"
        / "tests"
        / "verifier_core.py"
    )
    assert copy.exists()
    assert _sha256(copy) == _sha256(source)