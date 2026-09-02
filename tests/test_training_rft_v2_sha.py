"""Full-SHA validation on v2 task records before RFT rebuild (Req 7 / #714 rebase).

A v2 record's ``task_identity`` block is the frozen truth RFT replays against;
a truncated or malformed SHA must fail closed with a ``ValueError`` naming the
record id and field *before* any task reconstruction, never run against a
shortened identity. Mirrors the coordinator ``_rft_rows`` contract so Stage 2
and the replay agree on what a valid identity is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.training.rft import RftConfig, run_rft


def _record(rid: str = "r1", **overrides: object) -> dict[str, object]:
    rec: dict[str, object] = {
        "id": rid,
        "repo_slug": "owner/repo",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "diff": f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ {rid}\n",
        "findings": [
            {"id": f"{rid}-f1", "text": "Fix the off-by-one.", "grounded": True, "verdict": "consistent"},
        ],
        "format_valid": True,
        "length": 400,
    }
    rec.update(overrides)
    return rec


def _write_corpus(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "rft-inputs.jsonl"
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    return path


def _config(tmp_path: Path, records: list[dict[str, object]]) -> RftConfig:
    return RftConfig(
        inputs=_write_corpus(tmp_path, records),
        seed=7,
        rubric_version="2026.08.29-1",
        output_dir=tmp_path / "out",
    )


def test_valid_full_shas_build_the_task(tmp_path: Path) -> None:
    """A record with valid 40-hex base/head shas replays and writes winners."""
    cfg = _config(tmp_path, [_record("r1"), _record("r2")])
    result = run_rft(cfg)
    assert result.winners_path.is_file()
    assert result.inputs_sha256


def test_short_base_sha_fails_closed(tmp_path: Path) -> None:
    """A 6-char short sha is refused with ValueError before any rebuild."""
    cfg = _config(tmp_path, [_record("r1", base_sha="abc123")])
    with pytest.raises(ValueError, match=r"full sha|40"):
        run_rft(cfg)


def test_missing_head_sha_fails_before_task_work(tmp_path: Path) -> None:
    """A malformed identity (missing head_sha) raises before task/image work."""
    cfg = _config(tmp_path, [{"id": "r1", "base_sha": "a" * 40, "diff": "diff"}])
    with pytest.raises(ValueError, match="head_sha"):
        run_rft(cfg)


def test_non_hex_sha_fails_closed(tmp_path: Path) -> None:
    cfg = _config(tmp_path, [_record("r1", head_sha="z" * 40)])
    with pytest.raises(ValueError, match="full sha|40"):
        run_rft(cfg)


def test_missing_repo_slug_fails_closed(tmp_path: Path) -> None:
    cfg = _config(tmp_path, [_record("r1", repo_slug="")])
    with pytest.raises(ValueError, match="repo_slug"):
        run_rft(cfg)
