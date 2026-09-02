"""Tests for the v2 directory loader wrapper (``load_v2_projection``).

Enters from the production entrypoint over real projection directories on the
real filesystem and asserts observable outcomes: per-split record access, a
deterministic directory-level digest, and a fail-closed split-drift gate that
refuses any record whose recorded ``lineage.split`` disagrees with the split
recomputed from its record id.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from daydream.training.corpus_v2.splits import assign_split
from daydream.training.stacks_v2 import V2Projection, load_v2_projection

SALT = "issue-1081-salt"
HOLDOUT_RATE = 0.2
VAL_RATE = 0.2


def _make_record(record_id: str, split: str) -> dict[str, Any]:
    """A minimal v2 record passing the existing identity/license gates."""
    return {
        "schema_version": "2",
        "record_id": record_id,
        "record_type": "outcome-finding",
        "tier": "gold",
        "lineage": {
            "repo_slug": "owner/repo",
            "split": split,
            "license_decision": {
                "status": "admitted",
                "repo_slug": "owner/repo",
                "reason_code": None,
            },
        },
    }


def _write_projection(tmp_path: Path, record_ids: list[str]) -> Path:
    """Build a real projection directory: records placed in the split file
    their record id deterministically assigns, plus lineage.json + _SUCCESS."""
    out = tmp_path / "proj"
    out.mkdir()
    by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "holdout": [],
    }
    for record_id in record_ids:
        split = assign_split(
            record_id, salt=SALT, holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE
        )
        by_split[split].append(_make_record(record_id, split))
    for split, records in by_split.items():
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
    (out / "lineage.json").write_text(
        json.dumps(
            {
                "schema_version": "corpus-v2",
                "salt": SALT,
                "holdout_rate": HOLDOUT_RATE,
                "val_rate": VAL_RATE,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return out


def test_load_v2_projection_returns_per_split_records_lineage_and_digests(
    tmp_path: Path,
) -> None:
    record_ids = [f"rec-{i:04d}" for i in range(30)]
    out = _write_projection(tmp_path, record_ids)

    proj = load_v2_projection(out)

    assert isinstance(proj, V2Projection)
    assert set(proj.by_split) == {"train", "validation", "holdout"}
    # Every record lands in the split its id deterministically assigns.
    for record in proj.records:
        expected = assign_split(
            str(record["record_id"]), salt=SALT, holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE
        )
        assert proj.by_split[expected] is not None
        assert record in proj.by_split[expected]
    assert len(proj.records) == len(record_ids)
    assert sum(len(v) for v in proj.by_split.values()) == len(record_ids)
    assert proj.lineage["salt"] == SALT
    assert set(proj.split_digests) == {
        "train.jsonl",
        "validation.jsonl",
        "holdout.jsonl",
    }
    for name, digest in proj.split_digests.items():
        import hashlib

        assert digest == hashlib.sha256((out / name).read_bytes()).hexdigest()


def test_load_v2_projection_digest_is_deterministic(tmp_path: Path) -> None:
    record_ids = [f"rec-{i:04d}" for i in range(12)]
    out = _write_projection(tmp_path, record_ids)

    first = load_v2_projection(out)
    second = load_v2_projection(out)

    assert first.digest == second.digest
    # An identical copy of the directory yields the same digest.
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    assert load_v2_projection(copy).digest == first.digest
    # And the digest is actually content-sensitive.
    (out / "train.jsonl").write_text(
        (out / "train.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert load_v2_projection(out).digest != first.digest


def test_load_v2_projection_raises_on_split_drift(tmp_path: Path) -> None:
    record_ids = [f"rec-{i:04d}" for i in range(20)]
    out = _write_projection(tmp_path, record_ids)

    # Tamper: change one record's recorded lineage.split without moving it
    # between files — the recompute gate must catch the drift fail-closed.
    lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    if not lines:  # train may be empty for tiny id sets; use whatever exists
        for name in ("validation.jsonl", "holdout.jsonl"):
            lines = (out / name).read_text(encoding="utf-8").splitlines()
            if lines:
                break
    records = [json.loads(line) for line in lines]
    target = records[0]
    tampered_split = next(
        s for s in ("train", "validation", "holdout")
        if s != str(target["lineage"]["split"])
    )
    target["lineage"]["split"] = tampered_split
    (out / "train.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="split drift"):
        load_v2_projection(out)


def test_load_v2_projection_missing_success_marker_fails_closed(
    tmp_path: Path,
) -> None:
    out = _write_projection(tmp_path, [f"rec-{i:04d}" for i in range(6)])
    (out / "_SUCCESS").unlink()
    with pytest.raises(ValueError, match="_SUCCESS"):
        load_v2_projection(out)


def test_load_v2_projection_non_v2_schema_version_fails_closed(
    tmp_path: Path,
) -> None:
    out = _write_projection(tmp_path, [f"rec-{i:04d}" for i in range(6)])
    records = [
        json.loads(line)
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if records:
        records[0]["schema_version"] = "1"
        (out / "train.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
    else:
        # Deterministic splits may leave train empty for this id set; tamper
        # whichever split file actually holds records.
        for name in ("validation.jsonl", "holdout.jsonl"):
            recs = [
                json.loads(line)
                for line in (out / name).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if recs:
                recs[0]["schema_version"] = "1"
                (out / name).write_text(
                    "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8"
                )
                break
    with pytest.raises(ValueError, match="schema_version"):
        load_v2_projection(out)
