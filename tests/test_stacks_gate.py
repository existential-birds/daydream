"""Structural consumption gate for ``load_dataset_v2`` (issue #1080).

Enters from the production entrypoint (``load_dataset_v2``) over real
projection directories on the real filesystem and asserts observable
outcomes: the loader must structurally require per-record repo identity and
license decisions, and re-run the C5/C8 gates fail-closed over the loaded
records — no kwarg can suppress C5, and C8 is admitted only via the exact
``allow_copyleft`` opt-in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.archive.hydrate_rules import (
    REASON_CODE_C5_EXCLUDED_REPO,
    REASON_CODE_C8_COPYLEFT_UNOPTED,
)
from daydream.training.stacks import load_dataset_v2


def _record(**overrides: object) -> dict[str, object]:
    """A minimal v2 record in the shape the projector emits: repo identity
    and an immutable license decision nested under ``lineage``."""
    record: dict[str, object] = {
        "schema_version": "2",
        "record_id": "rec-0001",
        "tier": "gold",
        "lineage": {
            "repo_slug": "owner/repo",
            "license_decision": {
                "status": "admitted",
                "repo_slug": "owner/repo",
                "reason_code": None,
            },
        },
    }
    record.update(overrides)
    # Convenience: a repo_slug override must propagate into the lineage's
    # identity and decision stamp, not land as a foreign top-level key.
    if "repo_slug" in overrides:
        lineage = cast(dict[str, object], record["lineage"])
        lineage["repo_slug"] = overrides["repo_slug"]
        decision = cast(dict[str, object], lineage["license_decision"])
        decision["repo_slug"] = overrides["repo_slug"]
        # The override must not remain as a foreign top-level key.
        record.pop("repo_slug")
    return record


def _write_projection(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    out = tmp_path / "proj"
    out.mkdir(exist_ok=True)
    (out / "_SUCCESS").write_text("ok\n")
    (out / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    for name in ("validation.jsonl", "holdout.jsonl"):
        (out / name).write_text("", encoding="utf-8")
    return out


def _load_records(out: Path) -> list[dict[str, Any]]:
    lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _strip(tmp_path: Path, *path: str) -> None:
    out = tmp_path / "proj"
    records = _load_records(out)
    for record in records:
        node: Any = record
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    (out / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


def _set(tmp_path: Path, dotted: str, value: object) -> None:
    out = tmp_path / "proj"
    records = _load_records(out)
    keys = dotted.split(".")
    for record in records:
        node: Any = record
        for key in keys[:-1]:
            node = node[key]
        node[keys[-1]] = value
    (out / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )


def test_load_v2_raises_on_stripped_repo_slug(tmp_path: Path) -> None:
    _write_projection(tmp_path, [_record()])
    _strip(tmp_path, "lineage", "repo_slug")
    with pytest.raises(ValueError, match="repo_slug"):
        load_dataset_v2(tmp_path / "proj")


def test_load_v2_raises_on_stripped_license_decision(tmp_path: Path) -> None:
    _write_projection(tmp_path, [_record()])
    _strip(tmp_path, "lineage", "license_decision")
    with pytest.raises(ValueError, match="license_decision"):
        load_dataset_v2(tmp_path / "proj")


def test_load_v2_raises_on_unknown_license_status(tmp_path: Path) -> None:
    _write_projection(tmp_path, [_record()])
    _set(tmp_path, "lineage.license_decision.status", "maybe")
    with pytest.raises(ValueError, match="license_decision"):
        load_dataset_v2(tmp_path / "proj")


def test_load_v2_enforces_c5_over_loaded_records(tmp_path: Path) -> None:
    _write_projection(tmp_path, [_record(repo_slug="grafana/grafana")])
    with pytest.raises(ValueError, match=REASON_CODE_C5_EXCLUDED_REPO):
        load_dataset_v2(tmp_path / "proj")


def test_load_v2_enforces_c8_with_exact_slug_opt_in(tmp_path: Path) -> None:
    record = _record(repo_slug="owner/gpl-repo")
    assert isinstance(record["lineage"], dict)
    record["lineage"]["license_decision"] = {
        "status": "rejected",
        "repo_slug": "owner/gpl-repo",
        "reason_code": REASON_CODE_C8_COPYLEFT_UNOPTED,
    }
    out = _write_projection(tmp_path, [record])
    with pytest.raises(ValueError, match=REASON_CODE_C8_COPYLEFT_UNOPTED):
        load_dataset_v2(out)
    # Opt-in via the function's new keyword admits the exact slug only.
    admitted = load_dataset_v2(out, allow_copyleft=frozenset({"owner/gpl-repo"}))
    assert [r["record_id"] for r in admitted] == ["rec-0001"]

    other = _record(repo_slug="owner/other-gpl-repo")
    assert isinstance(other["lineage"], dict)
    other["lineage"]["license_decision"] = {
        "status": "rejected",
        "repo_slug": "owner/other-gpl-repo",
        "reason_code": REASON_CODE_C8_COPYLEFT_UNOPTED,
    }
    _write_projection(tmp_path, [other])
    with pytest.raises(ValueError, match=REASON_CODE_C8_COPYLEFT_UNOPTED):
        load_dataset_v2(tmp_path / "proj", allow_copyleft=frozenset({"owner/gpl-repo"}))


def test_load_v2_admits_clean_records(tmp_path: Path) -> None:
    out = _write_projection(tmp_path, [_record(), _record(record_id="rec-0002")])
    records = load_dataset_v2(out)
    assert len(records) == 2
