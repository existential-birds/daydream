"""The table sandbox must preserve the isolation formerly supplied per item."""

import os
from pathlib import Path

import pytest

from daydream.artifacts import ownership
from daydream.dataset import LocalRecordStore
from daydream.trajectory.context import _RECORDER_VAR
from tests.harness.table_cases import isolated_case
from tests.harness.trajectory import make_recorder


def test_table_cases_restore_host_state_and_start_with_fresh_roots(tmp_path: Path) -> None:
    original_environment = dict(os.environ)
    original_private = ownership._default_private_base()
    recorder = make_recorder(tmp_path)
    token = _RECORDER_VAR.set(recorder)
    try:
        with pytest.raises(RuntimeError, match="row failure"):
            with isolated_case(tmp_path, "0") as first:
                assert _RECORDER_VAR.get() is None
                assert ownership._default_private_base() / "runtime" == first.artifact_runtime_root
                assert LocalRecordStore(Path.home() / ".daydream/dataset").root == first.dataset_store
                first.monkeypatch.setenv("TABLE_CASE_CANARY", "first row")
                (first.tmp_path / "sentinel").write_text("first row")
                first.dataset_hub.fail_before = True
                _RECORDER_VAR.set(recorder)
                raise RuntimeError("row failure")

        assert _RECORDER_VAR.get() is recorder
        assert dict(os.environ) == original_environment
        assert ownership._default_private_base() == original_private
        with isolated_case(tmp_path, "1") as second:
            assert not (second.tmp_path / "sentinel").exists()
            assert not second.dataset_hub.fail_before
            assert second.dataset_store != first.dataset_store
            assert second.artifact_runtime_root != first.artifact_runtime_root
            assert Path(os.environ["DAYDREAM_ARCHIVE_DIR"]) == second.archive_dir
            assert Path(os.environ["GH_CONFIG_DIR"]).is_relative_to(second.tmp_path)
            config = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "plugins/installed_plugins.json"
            assert config.read_text() == '{"plugins": {}}'
        assert _RECORDER_VAR.get() is recorder
        assert dict(os.environ) == original_environment
    finally:
        _RECORDER_VAR.reset(token)
