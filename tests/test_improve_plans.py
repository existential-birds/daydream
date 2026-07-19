import json
from pathlib import Path

from daydream.improve.plans import (
    load_rejections,
    record_rejections,
    render_plan,
    write_plans,
)


def _finding() -> dict[str, object]:
    return {
        "fingerprint": "fp-fix-n-plus-one",
        "title": "Fix N+1 queries",
        "category": "performance",
        "path": "apps/catalog/api.py",
        "body": "The endpoint issues one query per catalog item.",
        "impact": "HIGH",
        "effort": "M",
        "risk": "MED",
        "confidence": "HIGH",
        "evidence": ["apps/catalog/api.py:12"],
    }


def _sel(*, slug: str) -> dict[str, object]:
    finding = {**_finding(), "fingerprint": f"fp-{slug}", "title": slug}
    return {
        "finding": finding,
        "slug": slug,
        "title": slug,
        "priority": "P1",
        "depends_on": [],
        "markdown": "## Steps\n\n### Step 1\n\nMake the change.",
    }


def test_rendered_plan_carries_every_template_section() -> None:
    text = render_plan(
        _finding(),
        body_markdown="## Steps\n...",
        planned_at="abc1234",
        number=1,
        slug="fix-n-plus-one",
        commands={"Tests": "uv run pytest"},
    )
    for section in (
        "## Status",
        "## Why this matters",
        "## Current state",
        "## Commands you will need",
        "## Scope",
        "## Steps",
        "## Test plan",
        "## Done criteria",
        "## STOP conditions",
    ):
        assert section in text
    assert "abc1234" in text and "Drift check" in text


def test_index_reconcile_keeps_numbering_monotonic(tmp_path: Path) -> None:
    write_plans(
        tmp_path / "daydream_plans",
        [_sel(slug="a")],
        planned_at="abc1234",
    )
    write_plans(
        tmp_path / "daydream_plans",
        [_sel(slug="b")],
        planned_at="def5678",
    )
    names = sorted(
        path.name
        for path in (tmp_path / "daydream_plans").glob("[0-9]*.md")
    )
    assert names == ["001-a.md", "002-b.md"]


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
