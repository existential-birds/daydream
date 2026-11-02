"""Real review lifecycle retains new JSONL evidence across offline HF publication."""

from pathlib import Path

import pytest

from daydream.run_config import RunConfig
from tests.harness.stub_backend import install_stub_backend, silence


@pytest.fixture
def config(multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> RunConfig:
    silence(monkeypatch)
    install_stub_backend(monkeypatch, multi_stack_target, pin_skill_availability=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    return RunConfig(target=str(multi_stack_target), output_mode="review", archive=False, run_eval=False,
        dataset_store_path=tmp_path / "records", trajectory_hub_repo="test-user/private-trajectories",
        shallow_fanout_threshold=0, cleanup=False)
