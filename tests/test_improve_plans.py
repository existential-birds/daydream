import json
from pathlib import Path

from daydream.improve.plans import load_rejections, record_rejections


def test_load_rejections_returns_empty_for_absent_or_malformed_file(
    tmp_path: Path,
) -> None:
    plans_dir = tmp_path / "daydream_plans"
    assert load_rejections(plans_dir) == {}

    plans_dir.mkdir()
    (plans_dir / "rejected.json").write_text("{not json")
    assert load_rejections(plans_dir) == {}


def test_record_rejections_appends_and_loads_by_fingerprint(
    tmp_path: Path,
) -> None:
    plans_dir = tmp_path / "daydream_plans"
    first = {
        "fingerprint": "abc",
        "title": "Phantom N+1",
        "path": "apps/catalog/api.py",
        "reason": "No query loop exists.",
        "rejected_at_sha": "123",
    }
    second = {
        "fingerprint": "def",
        "title": "By-design behavior",
        "path": "apps/billing/api.py",
        "reason": "Documented behavior.",
        "rejected_at_sha": "456",
    }

    record_rejections(plans_dir, [first])
    record_rejections(plans_dir, [second])

    assert load_rejections(plans_dir) == {"abc": first, "def": second}
    envelope = json.loads((plans_dir / "rejected.json").read_text())
    assert envelope == {
        "schema_version": 1,
        "rejected": [first, second],
    }
