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


def test_malformed_identity_fails_closed(tmp_path: Path) -> None:
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"id": "r1", "base_sha": "a" * 40, "diff": "diff"}, "head_sha"),
        (_record("r1", head_sha="z" * 40), r"full sha|40"),
    )
    for index, (record, match) in enumerate(cases):
        root = tmp_path / str(index)
        root.mkdir()
        cfg = _config(root, [record])
        with pytest.raises(ValueError, match=match):
            run_rft(cfg)
