"""Real projection loads require canonical record identity and C5 benchmark isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.training.admission import (
    REASON_CODE_C5_EXCLUDED_REPO,
)
from daydream.training.stacks import load_dataset_v2


def _record(**overrides: object) -> dict[str, object]:
    """A minimal current record carrying source repository identity."""
    record: dict[str, object] = {"schema_version": "3", "record_id": "rec-0001", "tier": "gold",
        "lineage": {"repo_slug": "owner/repo",
        },
    }
    record.update(overrides)
    # Convenience: a repo_slug override must propagate into the lineage's
    # identity, not land as a foreign top-level key.
    if "repo_slug" in overrides:
        lineage = cast(dict[str, object], record["lineage"])
        lineage["repo_slug"] = overrides["repo_slug"]
        # The override must not remain as a foreign top-level key.
        record.pop("repo_slug")
    return record


def _write_projection(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    out = tmp_path / "proj"
    out.mkdir(exist_ok=True)
    (out / "_SUCCESS").write_text("ok\n")
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    for name in ("validation.jsonl", "holdout.jsonl"):
        (out / name).write_text("", encoding="utf-8")
    return out


def _load_records(out: Path) -> list[dict[str, Any]]:
    lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


_MISSING = object()


def _edit(tmp_path: Path, dotted: str, value: object = _MISSING) -> None:
    out = tmp_path / "proj"
    records = _load_records(out)
    keys = dotted.split(".")
    for record in records:
        node: Any = record
        for key in keys[:-1]:
            node = node[key]
        if value is _MISSING:
            del node[keys[-1]]
        else:
            node[keys[-1]] = value
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

def test_load_v2_raises_on_stripped_repo_slug(tmp_path: Path) -> None:
    _write_projection(tmp_path, [_record()])
    _edit(tmp_path, "lineage.repo_slug")
    with pytest.raises(ValueError, match="repo_slug"):
        load_dataset_v2(tmp_path / "proj")

def test_load_v2_enforces_c5_over_loaded_records(tmp_path: Path) -> None:
    _write_projection(tmp_path, [_record(repo_slug="grafana/grafana")])
    with pytest.raises(ValueError, match=REASON_CODE_C5_EXCLUDED_REPO):
        load_dataset_v2(tmp_path / "proj")

@pytest.mark.parametrize("slug", ["GRAFANA/GRAFANA", "https://github.com/grafana/grafana.git",
                                  "git@github.com:grafana/grafana.git", " grafana/grafana.git "])
def test_loader_canonicalizes_before_benchmark_comparison(tmp_path: Path, slug: str) -> None:
    out = _write_projection(tmp_path, [_record(repo_slug=slug)])
    with pytest.raises(ValueError, match=REASON_CODE_C5_EXCLUDED_REPO):
        load_dataset_v2(out)


@pytest.mark.parametrize("slug", [None, "", 17, "owner/repo/extra"])
def test_loader_refuses_malformed_identity(tmp_path: Path, slug: object) -> None:
    out = _write_projection(tmp_path, [_record(repo_slug=slug)])
    with pytest.raises(ValueError, match="repo_slug"):
        load_dataset_v2(out)


def test_loader_admits_formerly_blocked_repository_without_decision(tmp_path: Path) -> None:
    out = _write_projection(tmp_path, [_record(repo_slug="gnu/coreutils")])
    assert [record["record_id"] for record in load_dataset_v2(out)] == ["rec-0001"]
