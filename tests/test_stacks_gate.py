"""Real projection loads require canonical record identity and C5 benchmark isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

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


def test_loader_canonicalizes_and_refuses_malformed_identity(tmp_path: Path) -> None:
    out = _write_projection(tmp_path, [_record(repo_slug="git@github.com:grafana/grafana.git")])
    with pytest.raises(ValueError, match=REASON_CODE_C5_EXCLUDED_REPO):
        load_dataset_v2(out)
    out = _write_projection(tmp_path, [_record(repo_slug="")])
    with pytest.raises(ValueError, match="repo_slug"):
        load_dataset_v2(out)
