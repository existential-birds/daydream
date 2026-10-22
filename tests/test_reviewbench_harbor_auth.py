"""Private role delivery and refresh ownership at real process/filesystem boundaries.

Existing native/backend replay tests own model lifecycle, not role-native
credential stores or the standalone verifier executable. Synthetic secrets never contact
an OAuth or inference endpoint.
"""
from __future__ import annotations

import base64
import hashlib
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
from daydream.benchmark.reviewbench.native import freeze_reviewer
from daydream.run_config import RunConfig
from tests.harness.reviewbench import supplied_source as supplied_source


def _token(expiry: int = 1) -> str:
    claims = base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
    return f"test.{claims}.test"


def _login(home: Path) -> dict[str, Any]:
    home.mkdir(parents=True, exist_ok=True)
    auth = {"auth_mode": "chatgpt", "OPENAI_API_KEY": None, "tokens": {
        "access_token": _token(), "refresh_token": "synthetic-refresh", "account_id": "test-account",
        "id_token": "synthetic-id",
    }, "last_refresh": "synthetic-date"}
    (home / "auth.json").write_text(json.dumps(auth))
    return auth


def _pi_login(home: Path) -> dict[str, Any]:
    home.mkdir(parents=True, exist_ok=True)
    auth = {"openai-codex": {"type": "oauth", "access": "synthetic-pi-access",
            "refresh": "synthetic-refresh", "accountId": "test-account", "expires": 1}}
    (home / "auth.json").write_text(json.dumps(auth))
    (home / "auth.json").chmod(0o600)
    return auth


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.parametrize("damage", ["missing", "api", "partial", "malformed", "symlink"])
@pytest.mark.parametrize("role", ["reviewer", "judge"])
def test_subscription_only_rejects_unusable_auth_without_exposing_it(
    tmp_path: Path, damage: str, role: str,
) -> None:
    """Native Pi admission must also reject API keys; Codex-only checks missed that route."""
    canonical = tmp_path / "canonical"
    value = _login(canonical) if role == "reviewer" else _pi_login(canonical)
    path = canonical / "auth.json"
    if damage == "missing":
        path.unlink()
    elif damage == "api":
        if role == "reviewer":
            value["OPENAI_API_KEY"] = "synthetic-secret"
        else:
            value["openai-codex"] = {"type": "api_key", "key": "synthetic-secret"}
        path.write_text(json.dumps(value))
    elif damage == "partial":
        if role == "reviewer":
            value["tokens"].pop("refresh_token")
        else:
            value["openai-codex"].pop("refresh")
        path.write_text(json.dumps(value))
    elif damage == "malformed":
        path.write_text("synthetic-secret:invalid-json")
    else:
        destination = canonical / "actual-auth.json"
        path.rename(destination)
        path.symlink_to(destination)
    validate = harbor_auth.validate_subscription_home if role == "reviewer" else harbor_auth.validate_judge_home
    with pytest.raises(harbor_auth.SubscriptionAuthError) as caught:
        validate(canonical)
    assert "synthetic-secret" not in str(caught.value)


@pytest.mark.parametrize('extra_provider', ['same-judge', 'other-provider'])
def test_native_pi_reviewer_store_refuses_unrelated_credentials(
    tmp_path: Path, extra_provider: str,
) -> None:
    """A separate directory cannot isolate judge credentials copied inside its file.

    Native Pi mounts the entire auth file. Existing disjoint-path and selected
    provider checks admit an additional judge login unless membership is closed.
    """
    reviewer, judge = tmp_path / 'reviewer', tmp_path / 'judge'
    reviewer.mkdir()
    judge_auth = _pi_login(judge)
    native = {'nous': {'type': 'api_key', 'key': 'synthetic-nous-secret'}}
    if extra_provider == 'same-judge':
        native.update(judge_auth)
    else:
        native['other-provider'] = {'type': 'api_key', 'key': 'synthetic-other-secret'}
    (reviewer / 'auth.json').write_text(json.dumps(native))
    originals = {path: path.read_bytes() for path in (reviewer / 'auth.json', judge / 'auth.json')}
    with pytest.raises(harbor_auth.SubscriptionAuthError) as caught:
        harbor_auth.validate_credential_stores(reviewer, judge, backend='pi', provider='nous')
    assert 'synthetic' not in str(caught.value)
    assert {path: path.read_bytes() for path in originals} == originals


@pytest.mark.parametrize('role', ['reviewer', 'judge'])
def test_admission_tolerates_native_in_place_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str,
) -> None:
    """Read-only snapshots must tolerate another runtime owner's truncate/write window.

    The SDK contention test owns locking, but host admission/evidence snapshots
    read independently. Release a real filesystem writer only after the first
    actual read observes its truncated file, so the regression cannot pass by
    scheduling the writer entirely before admission.
    """
    home = tmp_path / 'credentials'
    auth = _login(home) if role == 'reviewer' else _pi_login(home)
    path = home / 'auth.json'
    path.write_text('{')
    marker = tmp_path / 'read-observed'
    child = tmp_path / 'writer.py'
    child.write_text('''import json, pathlib, sys, time
path, marker = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
while not marker.exists():
 time.sleep(0.01)
path.write_text(sys.argv[3])
''')
    read = Path.read_text

    def observed_read(current: Path, *args: Any, **kwargs: Any) -> str:
        content = read(current, *args, **kwargs)
        if current == path and not marker.exists():
            assert content == '{'
            marker.touch()
        return content

    monkeypatch.setattr(Path, 'read_text', observed_read)
    with subprocess.Popen([sys.executable, str(child), str(path), str(marker), json.dumps(auth)]) as writer:
        try:
            validate = harbor_auth.validate_subscription_home if role == 'reviewer' else harbor_auth.validate_judge_home
            validate(home)
            writer.wait(timeout=5)
        finally:
            if writer.poll() is None:
                writer.kill()
    assert marker.exists() and writer.returncode == 0


def test_native_reviewer_processes_serialize_and_retain_rotation_after_death(tmp_path: Path) -> None:
    """Direct auth survives death without copyback; a second native invocation must still wait.

    The removed cross-runtime handoff tested conversion and recovery markers,
    which no longer exist. This retains the real independent flock contention.
    """
    canonical = tmp_path / "canonical"
    _login(canonical)
    child = tmp_path / "waiter.py"
    child.write_text('''import fcntl, json, os, pathlib, sys
from daydream.benchmark.reviewbench.harbor_auth import reviewer_credentials
root = pathlib.Path(sys.argv[1])
with (root/'canonical/.reviewer.lock').open('r+') as lock:
 try:
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
 except BlockingIOError:
  (root/'blocked').touch()
 else:
  raise AssertionError('reviewer did not block independent lock acquisition')
with reviewer_credentials(root/'canonical', root/'second') as home:
 data = json.loads((home/'auth.json').read_text())
 assert data['tokens']['refresh_token'] == 'synthetic-first-rotation'
 data['tokens']['refresh_token'] = 'synthetic-second-rotation'
 (home/'auth.json').write_text(json.dumps(data))
 os._exit(37)
''')
    with harbor_auth.reviewer_credentials(canonical, tmp_path / "first") as home:
        process = subprocess.Popen([sys.executable, str(child), str(tmp_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            _wait_for(tmp_path / 'blocked', process)
            data = json.loads((home / 'auth.json').read_text())
            data['tokens']['refresh_token'] = 'synthetic-first-rotation'
            (home / 'auth.json').write_text(json.dumps(data))
        except BaseException:
            process.kill()
            process.wait()
            raise
    stdout, stderr = process.communicate(timeout=15)
    assert process.returncode == 37, stdout + stderr
    with harbor_auth.reviewer_credentials(canonical, tmp_path / 'third') as home:
        assert json.loads((home / 'auth.json').read_text())['tokens']['refresh_token'] == 'synthetic-second-rotation'
    assert not (tmp_path / 'first/auth.json').exists()
    assert not (tmp_path / 'third/auth.json').exists()
    assert {p.name for p in canonical.iterdir()} == {'auth.json', '.reviewer.lock'}


@pytest.mark.parametrize('exit_code', [0, 29, 'cancel'])
def test_judge_rotation_survives_fresh_invocations_and_cleanup(tmp_path: Path, exit_code: int | str) -> None:
    """A failed/cancelled child must be joined before its fresh configuration is deleted.

    Lease tests missed direct Pi persistence and canonical auth identity across
    separate wrapper processes. No real OAuth requests or inference occur.
    """
    canonical = tmp_path / 'canonical'
    _pi_login(canonical)
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
    with subprocess.Popen([*arguments, str(exit_code)], env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True) as process:
        if exit_code == 'cancel':
            _wait_for(tmp_path / 'ready', process)
            process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == (143 if exit_code == 'cancel' else exit_code), stdout + stderr
    assert 'synthetic-rotated' not in stdout + stderr
    tool_pid = (tmp_path / ('tool-' + str(exit_code))).read_text()
    state = subprocess.run(['ps', '-o', 'stat=', '-p', tool_pid], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith('Z'), 'judge tool survived private-state cleanup'
    home = Path((tmp_path / ('home-' + str(exit_code))).read_text())
    assert not home.exists()
    if exit_code == 'cancel':
        with pytest.raises(ProcessLookupError):
            os.kill(int((tmp_path / 'pid-cancel').read_text()), 0)
    result = subprocess.run([*arguments, 'again'], env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not Path((tmp_path / 'home-again').read_text()).exists()
    assert {p.name for p in canonical.iterdir()} == {'auth.json'}


@pytest.fixture
def classifier_retention(tmp_path: Path) -> tuple[list[str], Path, Path, Path]:
    """A harmless child writes genuine-shaped Pi files through the public wrapper."""
    canonical = tmp_path / 'canonical'
    _pi_login(canonical)
    models = tmp_path / 'models.json'
    models.write_text('{}')
    sessions, evidence = tmp_path / 'judge/classifier-output/sessions', tmp_path / 'verifier/judge-sessions'
    script = tmp_path / 'classifier.py'
    script.write_text('''import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
home = pathlib.Path(os.environ['PI_CODING_AGENT_DIR'])
(root/'invoked').touch()
auth = json.loads((pathlib.Path(os.environ['DAYDREAM_RB_PI_AUTH_PATH'])).read_text())
original = auth['openai-codex']['refresh']
auth['openai-codex']['refresh'] = 'synthetic-new-judge-refresh'
(pathlib.Path(os.environ['DAYDREAM_RB_PI_AUTH_PATH'])).write_text(json.dumps(auth))
(root/'role-home').write_text(str(home))
sessions = root/'judge/classifier-output/sessions'
sessions.mkdir(parents=True, exist_ok=True)
payload = json.loads((root/'payload.json').read_text())
text = ''.join(json.dumps(entry)+'\\n' for entry in payload)
mode = sys.argv[3]
name = '2026-10-21T20-25-25-956Z_01a16023-debe-7edd-a755-3d10ced5c19f.jsonl'
path = sessions/name
if mode == 'symlink':
 path.symlink_to(pathlib.Path(os.environ['DAYDREAM_RB_PI_AUTH_PATH']))
elif mode == 'directory':
 path.mkdir()
elif mode == 'malformed':
 path.write_text('{partial session')
elif mode == 'unexpected':
 (sessions/'auth.json').write_text((pathlib.Path(os.environ['DAYDREAM_RB_PI_AUTH_PATH'])).read_text())
elif mode in ('initial-secret', 'refreshed-secret', 'tool-auth'):
 secret = original if mode == 'initial-secret' else auth['openai-codex']['refresh']
 if mode == 'tool-auth':
  secret = json.dumps({'openai-codex':{'type':'oauth','access':'opaque-intermediate-access',
   'refresh':'opaque-intermediate-refresh'},'other':{'access_token':'opaque-intermediate-token'}})
 payload.append({'type':'message','message':{'role':'toolResult','content':[{'type':'text','text':secret}]}})
 path.write_text(''.join(json.dumps(entry)+'\\n' for entry in payload))
else:
 path.write_text(text)
sys.exit(int(sys.argv[2]))
''')
    payload = [
        {'type': 'session', 'version': 3, 'id': 'synthetic-session', 'cwd': '/work/repo'},
        {'type': 'model_change', 'provider': 'openai-codex', 'modelId': 'gpt-6.1-sol'},
        {'type': 'thinking_level_change', 'thinkingLevel': 'medium'},
        {'type': 'message', 'message': {'role': 'user', 'content': 'Synthetic classifier context'}},
        {'type': 'message', 'message': {'role': 'assistant', 'provider': 'openai-codex',
            'model': 'gpt-6.1-sol', 'api': 'openai-codex-responses', 'stopReason': 'stop', 'timestamp': 123,
            'usage': {'input': 10, 'output': 4, 'cacheRead': 3, 'cost': {'total': 0.00012}},
            'content': [{'type': 'text', 'text': 'Synthetic classification'}]}},
    ]
    (tmp_path / 'payload.json').write_text(json.dumps(payload))
    arguments = ['judge', '--canonical-home', str(canonical), '--models', str(models),
                 '--classifier-sessions', str(sessions), '--judge-evidence', str(evidence), '--pi-bootstrap',
                 str(Path(harbor_auth.__file__).with_name('judge_auth.mjs')), '--',
                 sys.executable, str(script), str(tmp_path)]
    return arguments, canonical, sessions, evidence


@pytest.mark.parametrize('exit_code', [0, 29])
def test_judge_retains_exact_classifier_sessions_and_available_usage(
    classifier_retention: tuple[list[str], Path, Path, Path], exit_code: int,
) -> None:
    """Container cleanup must not erase classifier model/effort/token evidence, even on failure."""
    arguments, canonical, sessions, evidence = classifier_retention
    assert harbor_auth.main([*arguments, str(exit_code), 'valid']) == exit_code
    session = next(sessions.glob('*.jsonl'))
    assert (evidence / session.name).read_bytes() == session.read_bytes()
    index = json.loads((evidence / 'index.json').read_text())
    assert index['classifier_availability'] == 'retained'
    assert index['matcher_sessions'] == 'unavailable_upstream_in_memory'
    assert index['matcher_usage'] == 'not_retained_by_eval'
    assert index['monetary_usage_source'] == 'api-equivalent-estimate'
    assert index['subscription_cash_charges'] is None
    record = index['classifier_sessions'][0]
    assert record['sha256'] == hashlib.sha256(session.read_bytes()).hexdigest()
    assert set(path.name for path in evidence.iterdir()) == {session.name, 'index.json'}
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in evidence.iterdir())
    assert evidence.stat().st_mode & 0o777 == 0o700
    assert not Path((canonical.parent / 'role-home').read_text()).exists()
    assert json.loads((canonical / 'auth.json').read_text())['openai-codex']['refresh'] == 'synthetic-new-judge-refresh'
    assert all('synthetic-new-judge-refresh' not in path.read_text() for path in evidence.iterdir())


@pytest.mark.parametrize('fault', ['source-file', 'source-symlink', 'stale-source', 'stale-output', 'output-symlink'])
def test_judge_evidence_roots_refuse_stale_or_unsafe_paths_before_execution(
    classifier_retention: tuple[list[str], Path, Path, Path], fault: str,
) -> None:
    """An old session or symlink root must never be attributed to this judge attempt."""
    arguments, canonical, source, output = classifier_retention
    if fault.startswith('source') or fault == 'stale-source':
        source.parent.mkdir(parents=True)
        if fault == 'source-file':
            source.write_text('existing bytes')
        elif fault == 'source-symlink':
            source.symlink_to(canonical, target_is_directory=True)
        else:
            source.mkdir()
            (source / 'old-session.jsonl').write_text('old bytes')
    else:
        output.parent.mkdir()
        if fault == 'output-symlink':
            output.symlink_to(canonical, target_is_directory=True)
        else:
            output.mkdir()
            (output / 'index.json').write_text('old index')
    assert harbor_auth.main([*arguments, '0', 'valid']) == 1
    assert not (canonical.parent / 'invoked').exists()
    if fault == 'stale-output':
        assert (output / 'index.json').read_text() == 'old index'


@pytest.mark.parametrize('fault', ['symlink', 'directory', 'malformed', 'unexpected'])
def test_judge_refuses_unsafe_transcripts_without_publishing_credentials(
    classifier_retention: tuple[list[str], Path, Path, Path], fault: str,
) -> None:
    """Read tools/provider errors can expose auth; unsafe logs must fail closed before hashing/publication."""
    arguments, canonical, _source, output = classifier_retention
    assert harbor_auth.main([*arguments, '0', fault]) == 1
    assert (canonical.parent / 'invoked').exists()
    assert not output.exists()
    assert json.loads((canonical / 'auth.json').read_text())['openai-codex']['refresh'] == 'synthetic-new-judge-refresh'


def test_judge_session_publication_failure_preserves_partial_evidence_and_blocks_reuse(
    classifier_retention: tuple[list[str], Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure installing the final index must remain incomplete without overwriting a later attempt."""
    arguments, canonical, source, output = classifier_retention
    link = os.link

    def fail_index(source_path: Path, destination: Path, **kwargs: Any) -> None:
        if destination == output / 'index.json':
            raise OSError('synthetic index publication failure')
        link(source_path, destination, **kwargs)

    monkeypatch.setattr(os, 'link', fail_index)
    assert harbor_auth.main([*arguments, '0', 'valid']) == 1
    session = next(source.glob('*.jsonl'))
    assert (output / session.name).read_bytes() == session.read_bytes()
    assert not (output / 'index.json').exists()
    (canonical.parent / 'invoked').unlink()
    assert harbor_auth.main([*arguments, '0', 'valid']) == 1
    assert not (canonical.parent / 'invoked').exists()


@pytest.mark.parametrize('fault', ['initial-secret', 'refreshed-secret', 'tool-auth'])
def test_judge_sanitizes_auth_tool_results_before_retention_and_digest(
    classifier_retention: tuple[list[str], Path, Path, Path], fault: str,
) -> None:
    """Absolute-path reads may expose initial, rotated or intermediate OAuth stores."""
    arguments, _canonical, source, output = classifier_retention
    assert harbor_auth.main([*arguments, '0', fault]) == 0
    original = next(source.glob('*.jsonl'))
    retained = output / original.name
    text = retained.read_text()
    assert '[REDACTED_CREDENTIAL]' in text
    for credential in ('synthetic-refresh', 'synthetic-new-judge-refresh', 'opaque-intermediate-access',
                       'opaque-intermediate-refresh', 'opaque-intermediate-token'):
        assert credential not in text
    assert all(isinstance(json.loads(line), dict) for line in text.splitlines())
    index = json.loads((output / 'index.json').read_text())
    record = index['classifier_sessions'][0]
    assert record['format'] == 'sanitized-pi-jsonl' and record['redacted'] is True
    assert record['sha256'] == hashlib.sha256(retained.read_bytes()).hexdigest()
    assert record['sha256'] != hashlib.sha256(original.read_bytes()).hexdigest()
    assert 'source_sha256' not in record


def test_pinned_pi_factory_uses_shared_canonical_lock_and_persistent_rotation(
    tmp_path: Path, supplied_source: Path,
) -> None:
    """Real pinned Pi locks must contend across fresh agent dirs, not just point at one inode.

    Python filesystem replays cannot establish the SDK factory binding or Pi's
    sidecar identity. Replace only OAuth network refresh; retain its actual
    getApiKey, model discovery, storage and refresh locks.
    """
    import shutil

    canonical = tmp_path / 'canonical'
    _pi_login(canonical)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    scope = runtime / 'node_modules/@earendil-works'
    scope.mkdir(parents=True)
    sdk = supplied_source / 'node_modules/@earendil-works/pi-coding-agent'
    (scope / 'pi-coding-agent').symlink_to(sdk, target_is_directory=True)
    (scope / 'pi-ai').symlink_to(sdk / 'node_modules/@earendil-works/pi-ai', target_is_directory=True)
    binding = runtime / 'binding.mjs'
    shutil.copyfile(Path(harbor_auth.__file__).with_name('judge_auth.mjs'), binding)
    script = runtime / 'judge.mjs'
    script.write_text('''import {existsSync,writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {AuthStorage,ModelRegistry} from '@earendil-works/pi-coding-agent';
const require = createRequire(import.meta.resolve('@earendil-works/pi-coding-agent'));
import {registerOAuthProvider} from '@earendil-works/pi-ai/oauth';
globalThis.fetch = () => { throw Error('offline test attempted network'); };
const lockfile = require('proper-lockfile');
const root = process.argv[2], mode = process.argv[3];
const path = process.env.DAYDREAM_RB_PI_AUTH_PATH;
if(mode === 'second') {
 try { lockfile.lockSync(path, {realpath:false}); throw Error('missing contention'); }
 catch(error) { if(error.code !== 'ELOCKED') throw error; }
 writeFileSync(root+'/blocked', '');
}
const storage = AuthStorage.create();
// Production immediately discovers models after create(); no manual reload.
const registry = ModelRegistry.create(storage);
if(!registry.getAvailable().some(model => model.provider === 'openai-codex')) {
 throw Error('auth unavailable at startup');
}
registerOAuthProvider({id:'openai-codex',name:'synthetic',getApiKey: cred => cred.access,
 async refreshToken(cred) {
  if(mode !== 'first') throw Error('duplicate refresh');
  if(!existsSync(path+'.lock')) throw Error('missing canonical lock sidecar');
  writeFileSync(root+'/locked', '');
  while(!existsSync(root+'/release')) await new Promise(resolve => setTimeout(resolve, 10));
  return {...cred,access:'synthetic-pi-rotated-access',refresh:'synthetic-pi-rotated',expires:Date.now()+60000};
 }
});
if(await storage.getApiKey('openai-codex') !== 'synthetic-pi-rotated-access') throw Error('lost rotation');
if(storage.get('openai-codex').refresh !== 'synthetic-pi-rotated') throw Error('factory bypassed canonical path');
if(existsSync(process.env.PI_CODING_AGENT_DIR+'/auth.json')) throw Error('copied session');
''')
    environment = dict(os.environ, DAYDREAM_RB_PI_AUTH_PATH=str(canonical / 'auth.json'))
    models = runtime / 'models.json'
    models.write_text('{}')
    # Like pinned verify.ts, spawn process.execPath without inheriting execArgv.
    # The wrapper must install its binding through controlled NODE_OPTIONS.
    launcher = runtime / 'verify.mjs'
    launcher.write_text("import {spawnSync} from 'node:child_process';"
                        "const result=spawnSync(process.execPath,process.argv.slice(2),{stdio:'inherit'});"
                        "process.exit(result.status ?? 1);")
    command = [sys.executable, str(Path(harbor_auth.__file__)), 'judge', '--canonical-home', str(canonical),
               '--models', str(models), '--pi-bootstrap', str(binding), '--', 'node', str(launcher),
               str(script), str(tmp_path)]
    first = subprocess.Popen([*command, 'first'], env={**environment, 'PI_CODING_AGENT_DIR': str(tmp_path / 'pi1')},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    second = None
    try:
        _wait_for(tmp_path / 'locked', first)
        second = subprocess.Popen([*command, 'second'], env={**environment,
                                  'PI_CODING_AGENT_DIR': str(tmp_path / 'pi2')},
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _wait_for(tmp_path / 'blocked', second)
        # Exceed pinned Pi's ~180ms synchronous lock retry window. The
        # factory must still return authenticated storage after contention.
        time.sleep(0.5)
        assert second.poll() is None
        (tmp_path / 'release').touch()
        for process in (first, second):
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stdout + stderr
    finally:
        for pending in (first, second):
            if pending is not None and pending.poll() is None:
                pending.terminate()
                pending.communicate(timeout=15)
    assert not (canonical / 'auth.json.lock').exists()
    assert json.loads((canonical / 'auth.json').read_text())['openai-codex']['refresh'] == 'synthetic-pi-rotated'


def test_native_pi_reviewer_uses_its_own_canonical_auth_and_native_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The native reviewer SDK differs from the supplied judge's pinned SDK.

    Reusing its bootstrap against the bundled CLI leaves a different AuthStorage
    class unbound. Use the actual published unbundled native module and its real
    file locks, with synthetic credentials and no provider requests.
    """
    configured = os.environ.get('DAYDREAM_REVIEWBENCH_NATIVE_PI')
    if not configured:
        pytest.skip('set DAYDREAM_REVIEWBENCH_NATIVE_PI to the locked native Pi 1.3.0 package directory')
    package = Path(configured).resolve()
    assert json.loads((package / 'package.json').read_text())['version'] == '1.3.0'
    model_ids = ['deepseek/deepseek-v4.1-flash', 'deepseek/deepseek-v4.1-flash-1017']
    operator = tmp_path / 'operator'
    operator.mkdir()
    (operator / 'models.json').write_text(json.dumps({'providers': {'nous': {
        'baseUrl': 'https://inference-api.nousresearch.com/v1', 'apiKey': 'synthetic-ignored-fallback',
        'models': [{'id': name, 'name': name, 'api': 'openai-completions', 'reasoning': True,
                    'input': ['text'], 'contextWindow': 1000000, 'maxTokens': 8192,
                    'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}} for name in model_ids],
    }}}))
    monkeypatch.setenv('PI_CODING_AGENT_DIR', str(operator))
    monkeypatch.setenv('PI_PROVIDER', 'nous')
    frozen = freeze_reviewer(RunConfig(target=str(tmp_path), backend='pi', model=model_ids[0],
                                       parse_model=model_ids[1]))
    for name in ('pi1', 'pi2'):
        role = tmp_path / name
        role.mkdir()
        (role / 'models.json').write_text(json.dumps(frozen['pi_models']))
    canonical, judge = tmp_path / 'native-auth', tmp_path / 'judge-auth'
    canonical.mkdir()
    (canonical / 'auth.json').write_text(json.dumps({'nous': {'type': 'api_key', 'key': 'synthetic-native-key'}}))
    _pi_login(judge)
    judge_bytes = (judge / 'auth.json').read_bytes()
    harbor_auth.validate_credential_stores(canonical, judge, backend='pi', provider='nous')
    runtime = tmp_path / 'native-runtime'
    runtime.mkdir()
    (runtime / 'dist').symlink_to(package / 'dist', target_is_directory=True)
    binding = runtime / 'binding.mjs'
    binding.write_bytes(Path(harbor_auth.__file__).with_name('reviewer_auth.mjs').read_bytes())
    script = runtime / 'native.mjs'
    script.write_text('''import {existsSync,writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {AuthStorage,FileAuthStorageBackend} from './dist/core/auth-storage.js';
import {createAgentSessionServices} from './dist/core/agent-session-services.js';
const require=createRequire(import.meta.resolve('./dist/core/auth-storage.js'));
const lockfile=require('proper-lockfile');
globalThis.fetch=()=>{throw Error('native auth test attempted network');};
const root=process.argv[2],mode=process.argv[3],path=process.env.DAYDREAM_RB_PI_AUTH_PATH;
if(mode==='second') {
 try {lockfile.lockSync(path,{realpath:false});throw Error('canonical lock did not contend');}
 catch(error){if(error.code!=='ELOCKED')throw error;}
 writeFileSync(root+'/blocked','');
 while(!existsSync(root+'/release'))await new Promise(resolve=>setTimeout(resolve,10));
}
const services=await createAgentSessionServices({cwd:root,agentDir:process.env.PI_CODING_AGENT_DIR,
 resourceLoaderOptions:{noExtensions:true,noSkills:true,noPromptTemplates:true,noThemes:true}});
if(services.diagnostics.some(item=>item.type==='error'))throw Error('native session setup failed');
if(JSON.stringify(services.modelRuntime.getAllModels('nous').map(model=>model.id).sort())!==
 JSON.stringify(['deepseek/deepseek-v4.1-flash','deepseek/deepseek-v4.1-flash-1017']))
 throw Error('native phase model override unavailable');
const storage=services.modelRuntime.credentials.store;
if(storage.authPath!==path)throw Error('native factory lost canonical auth path');
try {AuthStorage.create(root+'/unadmitted-auth.json');throw Error('unadmitted native auth path accepted');}
catch(error){if(error.message!=='ReviewBench reviewer cannot select another credential store')throw error;}
if(mode==='first') {
 await new FileAuthStorageBackend(storage.authPath).withLockAsync(async current=>{
  writeFileSync(root+'/locked','');
  while(!existsSync(root+'/release'))await new Promise(resolve=>setTimeout(resolve,10));
  const value=JSON.parse(current);value.nous.key='synthetic-native-rotated';
  return {result:undefined,next:JSON.stringify(value)};
 });
} else {
 const value=await storage.read('nous');
 if(value.key!=='synthetic-native-rotated')throw Error('lost native credential rotation');
}
if(existsSync(process.env.PI_CODING_AGENT_DIR+'/auth.json'))throw Error('copied native credential');
''')
    command = ['node', '--import', str(binding), str(script), str(tmp_path)]
    environment = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'PI_OFFLINE': '1',
                   'DAYDREAM_RB_PI_AUTH_PATH': str(canonical / 'auth.json')}
    first = subprocess.Popen([*command, 'first'], env={**environment, 'PI_CODING_AGENT_DIR': str(tmp_path / 'pi1')},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    second = None
    try:
        _wait_for(tmp_path / 'locked', first)
        second = subprocess.Popen([*command, 'second'],
            env={**environment, 'PI_CODING_AGENT_DIR': str(tmp_path / 'pi2')},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _wait_for(tmp_path / 'blocked', second)
        (tmp_path / 'release').touch()
        for process in (first, second):
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stdout + stderr
    finally:
        for pending in (first, second):
            if pending is not None and pending.poll() is None:
                pending.terminate()
                pending.communicate(timeout=15)
    assert (judge / 'auth.json').read_bytes() == judge_bytes
    assert not (canonical / 'auth.json.lock').exists()
    assert json.loads((canonical / 'auth.json').read_text())['nous']['key'] == 'synthetic-native-rotated'
