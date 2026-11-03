from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_auth
from tests.harness.reviewbench import native_auth


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, f'timed out waiting for {path.name}'
        time.sleep(0.01)


def test_subscription_only_rejects_unusable_auth_without_exposing_it(tmp_path: Path) -> None:
    for role in ('reviewer', 'judge'):
        for damage in ('api', 'partial', 'symlink'):
            canonical = tmp_path / f'{role}-{damage}'
            value = native_auth(canonical, 'codex' if role == 'reviewer' else 'judge')
            path = canonical / 'auth.json'
            if damage == 'api':
                if role == 'reviewer':
                    value['OPENAI_API_KEY'] = 'synthetic-secret'
                else:
                    value['openai-codex'] = {'type': 'api_key', 'key': 'synthetic-secret'}
                path.write_text(json.dumps(value))
            elif damage == 'partial':
                if role == 'reviewer':
                    value['tokens'].pop('refresh_token')
                else:
                    value['openai-codex'].pop('refresh')
                path.write_text(json.dumps(value))
            else:
                destination = canonical / 'actual-auth.json'
                path.rename(destination)
                path.symlink_to(destination)
            validate = (harbor_auth.validate_subscription_home if role == 'reviewer'
                        else harbor_auth.validate_judge_home)
            with pytest.raises(harbor_auth.SubscriptionAuthError) as caught:
                validate(canonical)
            assert 'synthetic-secret' not in str(caught.value)


def test_credential_stores_refuse_shared_cluttered_unsupported_or_unrelated_stores(tmp_path: Path) -> None:
    reviewer, judge = tmp_path / 'reviewer', tmp_path / 'judge'
    reviewer.mkdir()
    native_auth(judge, 'judge')
    native = {'nous': {'type': 'api_key', 'key': 'synthetic-nous-secret'},
              'other-provider': {'type': 'api_key', 'key': 'synthetic-other-secret'}}
    (reviewer / 'auth.json').write_text(json.dumps(native))
    originals = {path: path.read_bytes() for path in (reviewer / 'auth.json', judge / 'auth.json')}
    refusals: list[tuple[Path, Path, str, str]] = [
        (reviewer, judge, 'pi', 'synthetic'),            # unrelated provider credentials
        (judge, judge, 'pi', 'separate'),                # one store for both roles
        (reviewer, judge, 'claude', 'unsupported'),      # no admitted native capability
    ]
    for reviewer_home, judge_home, backend, expected in refusals:
        with pytest.raises(harbor_auth.SubscriptionAuthError) as caught:
            harbor_auth.validate_credential_stores(reviewer_home, judge_home, backend=backend, provider='nous')
        assert 'synthetic' not in str(caught.value)
        if expected != 'synthetic':
            assert expected in str(caught.value)
    (judge / 'config.toml').write_text('model = "unrelated"')
    with pytest.raises(harbor_auth.SubscriptionAuthError, match='mount only dedicated'):
        harbor_auth.validate_credential_stores(reviewer, judge, backend='pi', provider='nous')
    assert {path: path.read_bytes() for path in originals} == originals
    # A Codex reviewer whose refresh token is the judge's is one copied session, not two sign-ins.
    codex, codex_judge = tmp_path / 'codex-reviewer', tmp_path / 'codex-judge'
    native_auth(codex)
    judge_value = native_auth(codex_judge, 'judge')
    judge_value['openai-codex']['refresh'] = 'private-refresh-canary'
    (codex_judge / 'auth.json').write_text(json.dumps(judge_value))
    with pytest.raises(harbor_auth.SubscriptionAuthError, match='copied OAuth session') as copied:
        harbor_auth.validate_credential_stores(codex, codex_judge)
    assert 'private-refresh-canary' not in str(copied.value)


@pytest.mark.parametrize('role', ['reviewer', ])
def test_admission_tolerates_native_in_place_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str,
) -> None:
    home = tmp_path / 'credentials'
    auth = native_auth(home, 'codex' if role == 'reviewer' else 'judge')
    path = home / 'auth.json'
    path.write_text('{')
    read = Path.read_text

    def observed_read(current: Path, *args: Any, **kwargs: Any) -> str:
        content = read(current, *args, **kwargs)
        if current == path and content == '{':
            path.write_text(json.dumps(auth))
        return content

    monkeypatch.setattr(Path, 'read_text', observed_read)
    validate = harbor_auth.validate_subscription_home if role == 'reviewer' else harbor_auth.validate_judge_home
    validate(home)
    assert json.loads(read(path)) == auth


def test_judge_rotation_survives_fresh_invocations_and_cleanup(tmp_path: Path) -> None:
    """A cancelled judge retains its rotated credential, reaps its tools, and leaks no private home."""
    canonical = tmp_path / 'canonical'
    native_auth(canonical, 'judge')
    models = tmp_path / 'models.json'
    models.write_text('{}')
    script = tmp_path / 'judge.py'
    script.write_text('''import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1]); mode = sys.argv[2]
home = pathlib.Path(os.environ['PI_CODING_AGENT_DIR'])
path = pathlib.Path(os.environ['DAYDREAM_RB_PI_AUTH_PATH'])
assert not (home/'auth.json').exists()
assert (home/'models.json').read_text() == '{}'
assert not (home/'stale-config').exists()
assert not {'OPENAI_API_KEY','OPENAI_BASE_URL','CODEX_HOME'} & os.environ.keys()
auth = json.loads(path.read_text())
if mode == 'again':
 assert auth['openai-codex']['refresh'] == 'synthetic-rotated'
else:
 auth['openai-codex']['refresh'] = 'synthetic-rotated'
 path.write_text(json.dumps(auth))
(home/'stale-config').touch()
(root/('home-'+mode)).write_text(str(home))
(root/('pid-'+mode)).write_text(str(os.getpid()))
# Model a classifier tool that survives its leader and ignores cancellation.
subprocess.Popen([sys.executable, '-c',
 'import os,pathlib,signal,sys,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
 'pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(120)', str(root/('tool-'+mode))])
while not (root/('tool-'+mode)).exists():
 time.sleep(0.01)
(root/'ready').touch()
if mode == 'cancel':
 time.sleep(120)
sys.exit(int(mode) if mode.isdigit() else 0)
''')
    arguments = [sys.executable, str(Path(harbor_auth.__file__)), 'judge', '--canonical-home', str(canonical),
                 '--models', str(models), '--pi-bootstrap',
                 str(Path(harbor_auth.__file__).with_name('judge_auth.mjs')), '--',
                 sys.executable, str(script), str(tmp_path)]
    environment = dict(os.environ, OPENAI_API_KEY='forbidden', OPENAI_BASE_URL='https://invalid.example',
                       CODEX_HOME='forbidden')
    with subprocess.Popen([*arguments, 'cancel'], env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        _wait_for(tmp_path / 'ready', process)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 143, stdout + stderr
    assert 'synthetic-rotated' not in stdout + stderr
    tool_pid = (tmp_path / 'tool-cancel').read_text()
    state = subprocess.run(['ps', '-o', 'stat=', '-p', tool_pid], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith('Z'), 'judge tool survived private-state cleanup'
    home = Path((tmp_path / 'home-cancel').read_text())
    assert not home.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / 'pid-cancel').read_text()), 0)
    result = subprocess.run([*arguments, 'again'], env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not Path((tmp_path / 'home-again').read_text()).exists()
    assert {p.name for p in canonical.iterdir()} == {'auth.json'}
