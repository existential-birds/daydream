from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import daydream
from daydream import runner
from daydream.benchmark.reviewbench.harbor_agent import DaydreamPublicReviewAgent
from daydream.benchmark.reviewbench.harbor_entrypoint import main
from daydream.trajectory import RUNS_DIRNAME
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git
from tests.harness.reviewbench import public_input


def _damage_public_input(environment: dict[str, str], fault: str) -> None:
    inputs = Path(environment['DAYDREAM_RB_INPUTS'])
    repository = Path(environment['DAYDREAM_RB_REPOSITORY'])
    if fault in {'schema', 'profile'} or fault.startswith('execution_'):
        profile_path = inputs / 'daydream-profile.json'
        profile = json.loads(profile_path.read_text())
        execution = profile['reviewer']['execution']
        if fault == 'schema':
            profile['schema'] = 1
        elif fault == 'profile':
            profile['reviewer']['model'] = 'unfrozen-model'
        elif fault == 'execution_missing':
            del profile['reviewer']['execution']
        elif fault == 'execution_bool':
            execution['stream_idle_timeout_s'] = True
        elif fault == 'execution_nan':
            execution['codex_terminal_drain_timeout_s'] = float('nan')
        else:
            execution['retry_policy']['attempts'] = 0.5
        if fault != 'profile':
            # Re-seal so only the semantic check, not the digest, can refuse it.
            policy = {key: value for key, value in profile.items() if key != 'profile_sha256'}
            profile['profile_sha256'] = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':'))
                                                      .encode()).hexdigest()
        profile_path.write_text(json.dumps(profile))
    elif fault == 'patch':
        (inputs / 'diff.patch').write_text('unrelated supplied patch')
    elif fault == 'refs':
        git(repository, 'update-ref', '-d', 'refs/reviewbench/base')
    elif fault == 'origin':
        git(repository, 'remote', 'add', 'origin', 'https://github.com/other/repo')
    elif fault == 'dirty':
        (repository / 'untracked.py').write_text('unfrozen source')
    elif fault == 'stale':
        output = Path(environment['DAYDREAM_RB_OUTPUT'])
        output.parent.mkdir()
        output.write_text('previous-evidence')
    else:
        (Path(environment['DAYDREAM_RB_CODEX_HOME']) / 'auth.json').unlink()


async def test_public_harbor_refuses_unbound_source_or_stale_artifacts_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh,
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('invalid public task reached generation transport')

    monkeypatch.setattr(runner, 'create_backend', forbidden)
    # Image setup probes the frozen task in the candidate image; a failed probe stops the trial.
    agent = DaydreamPublicReviewAgent(logs_dir=tmp_path / 'setup')
    assert agent.version() == daydream.__version__
    probes: list[str] = []

    class ProbeEnvironment:
        def __init__(self, return_code: int) -> None:
            self.return_code = return_code

        async def exec(self, command: str, **_kwargs: Any) -> SimpleNamespace:
            probes.append(command)
            return SimpleNamespace(return_code=self.return_code)

    await agent.setup(ProbeEnvironment(0))
    with pytest.raises(RuntimeError, match='image setup failed'):
        await agent.setup(ProbeEnvironment(1))
    assert all('load_public_task' in probe for probe in probes) and len(probes) == 2
    prior_archive = os.environ.get('DAYDREAM_ARCHIVE_DIR')
    for fault in ('schema', 'profile', 'patch', 'refs', 'origin', 'dirty', 'stale', 'auth',
                  'execution_missing', 'execution_bool', 'execution_nan', 'execution_retry'):
        (tmp_path / fault).mkdir()
        environment = public_input(tmp_path / fault)
        _damage_public_input(environment, fault)
        assert await main(environment) == 1, fault
        assert os.environ.get('DAYDREAM_ARCHIVE_DIR') == prior_archive
        output = Path(environment['DAYDREAM_RB_OUTPUT'])
        if fault == 'stale':
            assert output.read_text() == 'previous-evidence'
        else:
            assert not output.exists(), fault


async def test_native_storage_failure_retains_diagnostics_without_manufacturing_atif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, capsys: pytest.CaptureFixture[str],
    artifact_runtime_root: Path,
) -> None:
    environment = public_input(tmp_path)
    mkdir = Path.mkdir
    rejected: list[Path] = []

    def read_only_private_storage(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.parent.name == RUNS_DIRNAME and path.is_relative_to(artifact_runtime_root):
            rejected.append(path)
            raise OSError(errno.EROFS, 'private artifact storage is read-only')
        mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', read_only_private_storage)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail('artifact owner preflight failure reached model generation')

    monkeypatch.setattr(runner, 'create_backend', forbidden)
    assert await main(environment) == 1
    evidence = Path(environment['DAYDREAM_RB_EVIDENCE'])
    record = json.loads((evidence / 'execution.json').read_text())
    assert len(rejected) == 1
    assert record['status'] == 'failed' and record['terminal_result'] is None
    assert 'native reviewer raised OSError' in record['diagnostics']
    assert any('did not produce a finalized terminal result' in reason for reason in record['diagnostics'])
    assert any('native trajectory unavailable' in reason for reason in record['diagnostics'])
    assert not (evidence / 'trajectory.json').exists()
    assert not (evidence.parent / 'trajectory.json').exists()
    assert not Path(environment['DAYDREAM_RB_OUTPUT']).exists()
    assert 'FileNotFoundError' not in capsys.readouterr().err
