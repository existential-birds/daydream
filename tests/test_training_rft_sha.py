"""Reject malformed or truncated task SHAs before reconstruction, naming record and field. Match
coordinator validation so RFT replays only frozen full identities.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.training.rft import RftConfig, run_rft
from tests.test_training_rft import _record, _write_corpus


def _config(tmp_path: Path, records: list[dict[str, object]]) -> RftConfig:
    return RftConfig(
        inputs=_write_corpus(tmp_path, records), seed=7, rubric_version="2026.08.29-1", output_dir=tmp_path / "out",
    )


def test_valid_full_shas_build_the_task(tmp_path: Path) -> None:
    cfg = _config(tmp_path, [_record("r1"), _record("r2")])
    result = run_rft(cfg)
    assert result.winners_path.is_file()
    assert result.inputs_sha256

@pytest.mark.parametrize("record, match",
    [(_record("r1", base_sha="abc123"), r"full sha|40"),
        ({"id": "r1", "base_sha": "a" * 40, "diff": "diff"}, "head_sha"),
        (_record("r1", head_sha="z" * 40), r"full sha|40"), (_record("r1", repo_slug=""), "repo_slug"),
    ], ids=["short-base-sha", "missing-head-sha", "non-hex-sha", "missing-repo-slug"],
)
def test_malformed_identity_fails_closed(record: dict[str, object], match: str, tmp_path: Path) -> None:
    cfg = _config(tmp_path, [record])
    with pytest.raises(ValueError, match=match):
        run_rft(cfg)
