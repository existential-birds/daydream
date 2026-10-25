"""Explicit per-row sandboxes for contract tables using pytest's native subtests.

Tables share their immutable setup. Mutating rows get the same private roots,
offline dataset boundary, patch rollback, and recorder reset as isolated items.
This helper does not collect tests or re-enter pytest's fixture machinery.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import pytest

from daydream.artifacts import ownership
from daydream.dataset import LocalRecordStore
from daydream.trajectory.context import _RECORDER_VAR
from daydream.trajectory.recorder import _ACTIVE_SIGNAL_RUNS
from tests.harness.dataset_hub import FakeDatasetHub


@dataclass(frozen=True)
class TableCase:
    tmp_path: Path
    monkeypatch: pytest.MonkeyPatch
    archive_dir: Path
    artifact_runtime_root: Path
    dataset_store: Path
    dataset_hub: FakeDatasetHub


@contextmanager
def isolated_case(root: Path, name: str) -> Iterator[TableCase]:
    """Create one fresh scenario; restore patches and host state even on failure.

    ``name`` is a table-local numeric index, not an input value or a filesystem
    path supplied by a scenario. Reusing an index fails instead of reusing bytes.
    """
    if not name.isdecimal():
        raise ValueError("table case names must be numeric indices")
    case_root = root / f"case-{name}"
    case_root.mkdir()
    tmp_path = case_root / "work"
    tmp_path.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()
    private = case_root / "private"
    dataset = case_root / "dataset"
    claude_config = case_root / "claude-config"
    (claude_config / "plugins").mkdir(parents=True)
    (claude_config / "plugins/installed_plugins.json").write_text('{"plugins": {}}')
    hub = FakeDatasetHub()
    token = _RECORDER_VAR.set(None)
    signals = list(_ACTIVE_SIGNAL_RUNS)
    _ACTIVE_SIGNAL_RUNS.clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(ownership, "_default_private_base", lambda: private)
            patch.setenv("DAYDREAM_ARCHIVE_DIR", str(archive))
            patch.setenv("CLAUDE_CONFIG_DIR", str(claude_config))
            patch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh-config"))
            patch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
            patch.delenv("HF_TOKEN", raising=False)
            patch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
            initialize = LocalRecordStore.__init__

            @wraps(initialize)
            def isolated_store(self: LocalRecordStore, root: Path, **kwargs: Any) -> None:
                if Path(root).expanduser().absolute() == (Path.home() / ".daydream" / "dataset").absolute():
                    root = dataset
                initialize(self, root, **kwargs)

            patch.setattr(LocalRecordStore, "__init__", isolated_store)
            yield TableCase(tmp_path, patch, archive, private / "runtime", dataset, hub)
    finally:
        _RECORDER_VAR.reset(token)
        _ACTIVE_SIGNAL_RUNS.clear()
        _ACTIVE_SIGNAL_RUNS.extend(signals)
