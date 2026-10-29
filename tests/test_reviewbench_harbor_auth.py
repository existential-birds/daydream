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
from tests.harness.reviewbench import native_auth, supplied_source as supplied_source


def _wait_for(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None, process.communicate()
        assert time.monotonic() < deadline, f'timed out waiting for {path.name}'
        time.sleep(0.01)


def _run_contenders(command: list[str], environment: dict[str, str], root: Path) -> None:
    processes: list[subprocess.Popen[str]] = []
    try:
        for number, mode, marker in ((1, 'first', 'locked'), (2, 'second', 'blocked')):
            process = subprocess.Popen([*command, mode],
                env={**environment, 'PI_CODING_AGENT_DIR': str(root / f'pi{number}')},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(process)
            _wait_for(root / marker, process)
        # Exceed the supplied SDK's ~180ms synchronous lock retry window.
        time.sleep(0.5)
        assert processes[1].poll() is None
        (root / 'release').touch()
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=15)


@pytest.mark.parametrize("damage", ["missing", "api", "partial", "malformed", "symlink"])
@pytest.mark.parametrize("role", ["reviewer", "judge"])
def test_subscription_only_rejects_unusable_auth_without_exposing_it(
    tmp_path: Path, damage: str, role: str,
) -> None:
    canonical = tmp_path / "canonical"
    value = native_auth(canonical, "codex" if role == "reviewer" else "judge")
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
    reviewer, judge = tmp_path / 'reviewer', tmp_path / 'judge'
    reviewer.mkdir()
    judge_auth = native_auth(judge, 'judge')
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
    home = tmp_path / 'credentials'
    auth = native_auth(home, 'codex' if role == 'reviewer' else 'judge')
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


@pytest.mark.parametrize('legacy_lock', [False, True])
def test_native_reviewer_processes_overlap_and_retain_rotation_after_death(
    tmp_path: Path, legacy_lock: bool,
) -> None:
    canonical, judge = tmp_path / 'canonical', tmp_path / 'judge'
    initial = native_auth(canonical)['tokens']['refresh_token']
    native_auth(judge, 'judge')
    lock_path = canonical / '.reviewer.lock'
    if legacy_lock:
        lock_path.write_bytes(b'inert legacy lock')
    harbor_auth.validate_credential_stores(canonical, judge)
    child = tmp_path / 'reviewer.py'
    child.write_text('''import json, os, pathlib, sys, time
from daydream.benchmark.reviewbench.harbor_auth import reviewer_credentials
root, role, initial = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def wait_for(marker):
 deadline = time.monotonic() + 30
 while not marker.exists():
  assert time.monotonic() < deadline, 'reviewer rendezvous expired: ' + marker.name
  time.sleep(0.01)
with reviewer_credentials(root/'canonical', root/role) as home:
 path = home/'auth.json'
 assert path.is_symlink() and path.resolve() == root/'canonical/auth.json'
 assert home.stat().st_mode & 0o777 == 0o700
 (root/(role+'-entered')).touch()
 wait_for(root/(role+'-rotate'))
 data = json.loads(path.read_text())
 assert data['tokens']['refresh_token'] == (initial if role == 'first' else 'synthetic-first-rotation')
 data['tokens']['refresh_token'] = 'synthetic-'+role+'-rotation'
 path.write_text(json.dumps(data))
 (root/(role+'-rotated')).touch()
 wait_for(root/(role+'-release'))
 if role == 'second':
  os._exit(37)
assert not path.exists() and not path.is_symlink()
''')
    processes: list[subprocess.Popen[str]] = []
    try:
        # Neither review can rotate credentials or exit before BOTH have entered.
        for role in ('first', 'second'):
            process = subprocess.Popen([sys.executable, str(child), str(tmp_path), role, initial],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append(process)
            _wait_for(tmp_path / (role + '-entered'), process)
        assert all(process.poll() is None for process in processes)
        # Synthetic writes are ordered; this proves persistence, not OAuth refresh.
        for role, process in zip(('first', 'second'), processes, strict=True):
            (tmp_path / (role + '-rotate')).touch()
            _wait_for(tmp_path / (role + '-rotated'), process)
        assert all(process.poll() is None for process in processes)
        for role, process, exit_code in zip(('first', 'second'), processes, (0, 37), strict=True):
            (tmp_path / (role + '-release')).touch()
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == exit_code, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate(timeout=5)
    assert not (tmp_path / 'first/auth.json').is_symlink()
    # Abrupt death bypasses context cleanup; its private link retains canonical state.
    assert (tmp_path / 'second/auth.json').is_symlink()
    with pytest.raises(RuntimeError, match='review failed'):
        with harbor_auth.reviewer_credentials(canonical, tmp_path / 'third') as home:
            assert json.loads((home / 'auth.json').read_text())['tokens']['refresh_token'] == (
                'synthetic-second-rotation')
            raise RuntimeError('review failed')
    assert not (tmp_path / 'third/auth.json').exists()
    assert not (tmp_path / 'third/auth.json').is_symlink()
    harbor_auth.validate_credential_stores(canonical, judge)
    assert {p.name for p in canonical.iterdir()} == ({'auth.json', '.reviewer.lock'} if legacy_lock else {'auth.json'})
    if legacy_lock:
        assert lock_path.read_bytes() == b'inert legacy lock'


@pytest.mark.parametrize('exit_code', [0, 29, 'cancel'])
def test_judge_rotation_survives_fresh_invocations_and_cleanup(tmp_path: Path, exit_code: int | str) -> None:
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


@pytest.mark.parametrize('role', ['judge', 'reviewer'])
def test_native_sdk_factory_shares_canonical_lock_and_retains_rotation(
    tmp_path: Path, request: pytest.FixtureRequest, role: str,
) -> None:
    if role == 'judge':
        source: Path = request.getfixturevalue('supplied_source')
        package = source / 'node_modules/@earendil-works/pi-coding-agent'
    else:
        configured = os.environ.get('DAYDREAM_REVIEWBENCH_NATIVE_PI')
        if not configured:
            pytest.skip('set DAYDREAM_REVIEWBENCH_NATIVE_PI to the locked native Pi 1.4.0 package directory')
        package = Path(configured).resolve()
        assert json.loads((package / 'package.json').read_text())['version'] == '1.4.0'
    canonical, independent = tmp_path / 'canonical', tmp_path / 'independent-auth'
    native_auth(canonical, 'judge')
    native_auth(independent, 'judge')
    independent_bytes = (independent / 'auth.json').read_bytes()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    provider, field = ('openai-codex', 'refresh') if role == 'judge' else ('nous', 'key')
    if role == 'judge':
        scope = runtime / 'node_modules/@earendil-works'
        scope.mkdir(parents=True)
        (scope / 'pi-coding-agent').symlink_to(package, target_is_directory=True)
        (scope / 'pi-ai').symlink_to(package / 'node_modules/@earendil-works/pi-ai', target_is_directory=True)
        models: dict[str, Any] = {}
        module = '@earendil-works/pi-coding-agent'
        factory = '''const {ModelRegistry} = await import('@earendil-works/pi-coding-agent');
const {registerOAuthProvider} = await import('@earendil-works/pi-ai/oauth');
const storage = AuthStorage.create();
if(!ModelRegistry.create(storage).getAvailable().some(model => model.provider === 'openai-codex'))
 throw Error('auth unavailable at startup');
registerOAuthProvider({id:'openai-codex',name:'synthetic',getApiKey: cred => cred.access,
 async refreshToken(cred) {
  if(mode !== 'first') throw Error('duplicate refresh');
  if(!existsSync(path+'.lock')) throw Error('missing canonical lock sidecar');
  writeFileSync(root+'/locked', '');
  while(!existsSync(root+'/release')) await new Promise(resolve => setTimeout(resolve, 10));
  return {...cred,access:'synthetic-rotated',refresh:'synthetic-rotated',expires:Date.now()+60000};
 }
});
if(await storage.getApiKey('openai-codex') !== 'synthetic-rotated') throw Error('lost rotation');
'''
    else:
        (runtime / 'dist').symlink_to(package / 'dist', target_is_directory=True)
        (canonical / 'auth.json').write_text('{"nous":{"type":"api_key","key":"synthetic-native-key"}}')
        model_ids = ['deepseek/deepseek-v4.1-flash', 'deepseek/deepseek-v4.1-flash-1017']
        models = {'providers': {'nous': {'baseUrl': 'https://inference-api.nousresearch.com/v1',
            'models': [{'id': name, 'name': name, 'api': 'openai-completions', 'reasoning': True,
                'input': ['text'], 'contextWindow': 1000000, 'maxTokens': 8192,
                'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}} for name in model_ids]}}}
        module = './dist/core/auth-storage.js'
        factory = '''const {createAgentSessionServices} = await import('./dist/core/agent-session-services.js');
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
  const value=JSON.parse(current);value.nous.key='synthetic-rotated';
  return {result:undefined,next:JSON.stringify(value)};
 });
} else if((await storage.read('nous')).key!=='synthetic-rotated')throw Error('lost native rotation');
'''
    binding = runtime / 'binding.mjs'
    binding.write_bytes(Path(harbor_auth.__file__).with_name(role + '_auth.mjs').read_bytes())
    script = runtime / 'contender.mjs'
    script.write_text(f'''import {{existsSync,writeFileSync}} from 'node:fs';
import {{createRequire}} from 'node:module';
import {{AuthStorage,FileAuthStorageBackend}} from {module!r};
const require=createRequire(import.meta.resolve({module!r}));
const lockfile=require('proper-lockfile');
globalThis.fetch=()=>{{throw Error('offline auth test attempted network');}};
const root=process.argv[2],mode=process.argv[3],path=process.env.DAYDREAM_RB_PI_AUTH_PATH;
if(mode==='second') {{
 try {{lockfile.lockSync(path,{{realpath:false}});throw Error('missing canonical contention');}}
 catch(error){{if(error.code!=='ELOCKED')throw error;}}
 writeFileSync(root+'/blocked','');
 if({role!r}==='reviewer')
  while(!existsSync(root+'/release'))await new Promise(resolve=>setTimeout(resolve,10));
}}
''' + factory +
        "if(existsSync(process.env.PI_CODING_AGENT_DIR+'/auth.json'))throw Error('copied native credential');\n")
    models_path = runtime / 'models.json'
    models_path.write_text(json.dumps(models))
    for name in ('pi1', 'pi2'):
        home = tmp_path / name
        home.mkdir()
        (home / 'models.json').write_text(json.dumps(models))
    environment = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'PI_OFFLINE': '1',
                   'DAYDREAM_RB_PI_AUTH_PATH': str(canonical / 'auth.json')}
    command = ['node', '--import', str(binding), str(script), str(tmp_path)]
    if role == 'judge':
        launcher = runtime / 'verify.mjs'
        launcher.write_text("import {spawnSync} from 'node:child_process';"
            "const result=spawnSync(process.execPath,process.argv.slice(2),{stdio:'inherit'});"
            "process.exit(result.status ?? 1);")
        command = [sys.executable, str(Path(harbor_auth.__file__)), 'judge', '--canonical-home', str(canonical),
            '--models', str(models_path), '--pi-bootstrap', str(binding), '--', 'node', str(launcher),
            str(script), str(tmp_path)]
    _run_contenders(command, environment, tmp_path)
    assert (independent / 'auth.json').read_bytes() == independent_bytes
    assert not (canonical / 'auth.json.lock').exists()
    assert json.loads((canonical / 'auth.json').read_text())[provider][field] == 'synthetic-rotated'
