"""Output-share caps for corpus v2 (issue #1079, M1-M11)."""

from pathlib import Path
from typing import Any

import pytest


def _share_cfg(out_dir: Path, bundle_dir: Path, snapshot: Path, **kw: Any) -> Any:
    from tests.test_corpus_v2 import _cfg

    return _cfg(out_dir, bundle_dir, snapshot, **kw)


class TestShareCapValidation:
    def test_share_below_or_equal_zero_refuses(self, tmp_path: Path) -> None:
        from tests.test_corpus_v2 import _write_annotations_snapshot, _write_bundle

        bundle = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle)
        with pytest.raises(ValueError, match="max_stack_share"):
            _share_cfg(tmp_path / "out", bundle, snap, max_stack_share=0.0)

    def test_share_above_one_refuses(self, tmp_path: Path) -> None:
        from tests.test_corpus_v2 import _write_annotations_snapshot, _write_bundle

        bundle = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle)
        with pytest.raises(ValueError, match="max_repo_share"):
            _share_cfg(tmp_path / "out", bundle, snap, max_repo_share=1.5)

    def test_share_of_one_is_allowed(self, tmp_path: Path) -> None:
        from tests.test_corpus_v2 import _write_annotations_snapshot, _write_bundle

        bundle = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle)
        config = _share_cfg(tmp_path / "out", bundle, snap, max_profile_share=1.0)
        assert config.max_profile_share == 1.0

    def test_unset_shares_default_to_none(self, tmp_path: Path) -> None:
        from tests.test_corpus_v2 import _write_annotations_snapshot, _write_bundle

        bundle = _write_bundle(tmp_path)
        snap = _write_annotations_snapshot(bundle)
        config = _share_cfg(tmp_path / "out", bundle, snap)
        assert config.max_stack_share is None
        assert config.max_repo_share is None
        assert config.max_profile_share is None
