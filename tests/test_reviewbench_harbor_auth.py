"""Private role delivery and refresh ownership at real process/filesystem boundaries.

Existing native/backend replay tests own model lifecycle, not the new shared OAuth
lease or the standalone verifier executable. Synthetic secrets here never contact
an OAuth or inference endpoint.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_auth
from tests.harness.fake_gh import FakeGh


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


def test_refresh_returns_to_source_on_failed_judge_process(tmp_path: Path) -> None:
    """A quota/transport exit must not discard an already rotated refresh credential."""
    source, canonical = tmp_path / "source", tmp_path / "canonical"
    original = _login(source)
    models = tmp_path / "models.json"
    models.write_text('{"providers": {}}')
    result_path = tmp_path / "environment.json"
    script = tmp_path / "judge.py"
    script.write_text(
        "import json, os, pathlib, sys\n"
        "home = pathlib.Path(os.environ['PI_CODING_AGENT_DIR'])\n"
        "auth = json.loads((home / 'auth.json').read_text())\n"
        "assert auth['openai-codex']['type'] == 'oauth'\n"
        "assert (home / 'models.json').read_text() == '{\"providers\": {}}'\n"
        "auth['openai-codex']['refresh'] = 'synthetic-rotated'\n"
        "(home / 'auth.json').write_text(json.dumps(auth))\n"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps({\n"
        " 'api': 'OPENAI_API_KEY' in os.environ, 'url': 'OPENAI_BASE_URL' in os.environ,\n"
        " 'codex': 'CODEX_HOME' in os.environ, 'private_home': str(home),\n"
        " 'mode': (home / 'auth.json').stat().st_mode & 511}))\n"
        "sys.exit(29)\n"
    )
    with harbor_auth.subscription_session(source, canonical):
        environment = dict(os.environ, OPENAI_API_KEY="forbidden-api-key", OPENAI_BASE_URL="https://invalid.example")
        result = subprocess.run([
            sys.executable, str(Path(harbor_auth.__file__)), "judge", "--canonical-home", str(canonical),
            "--models", str(models), "--", sys.executable, str(script), str(result_path),
        ], env=environment, capture_output=True, text=True)
        assert result.returncode == 29, result.stderr
        observed = json.loads(result_path.read_text())
        assert observed == {"api": False, "url": False, "codex": False,
                            "private_home": observed["private_home"], "mode": 0o600}
        assert not Path(observed["private_home"]).exists()
        assert "synthetic-refresh" not in result.stdout + result.stderr
    updated = json.loads((source / "auth.json").read_text())
    assert updated["tokens"]["refresh_token"] == "synthetic-rotated"
    assert updated["tokens"]["id_token"] == original["tokens"]["id_token"]
    assert not (canonical / "auth.json").exists()


def test_cross_process_reviewer_and_judge_serialize_rotating_credentials(tmp_path: Path) -> None:
    """Concurrent trials must consume the latest canonical refresh after their lease."""
    canonical = tmp_path / "canonical"
    _login(canonical)
    child = tmp_path / "waiter.py"
    child.write_text(
        "import importlib.util, json, pathlib, sys\n"
        "spec = importlib.util.spec_from_file_location('auth', sys.argv[1])\n"
        "auth = importlib.util.module_from_spec(spec); spec.loader.exec_module(auth)\n"
        "root = pathlib.Path(sys.argv[2])\n"
        "(root / 'waiting').touch()\n"
        "with auth.subscription_lease(root / 'canonical', root / 'judge', role='judge') as home:\n"
        " data = json.loads((home / 'auth.json').read_text())['openai-codex']\n"
        " assert data['refresh'] == 'synthetic-reviewer-rotated'\n"
        " data['refresh'] = 'synthetic-judge-rotated'\n"
        " (home / 'auth.json').write_text(json.dumps({'openai-codex': data}))\n"
    )
    with harbor_auth.subscription_lease(canonical, tmp_path / "reviewer") as home:
        process = subprocess.Popen([sys.executable, str(child), str(Path(harbor_auth.__file__)), str(tmp_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            # A file handshake confirms the independent process entered its lease.
            import time
            deadline = time.monotonic() + 10
            while not (tmp_path / "waiting").exists():
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert process.poll() is None
            updated = json.loads((home / "auth.json").read_text())
            updated["tokens"]["refresh_token"] = "synthetic-reviewer-rotated"
            (home / "auth.json").write_text(json.dumps(updated))
        except BaseException:
            process.kill()
            process.wait()
            raise
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0, stdout + stderr
    assert json.loads((canonical / "auth.json").read_text())["tokens"]["refresh_token"] == "synthetic-judge-rotated"
    assert not (tmp_path / "reviewer" / "auth.json").exists()
    assert not (tmp_path / "judge" / "auth.json").exists()


@pytest.mark.parametrize("damage", ["missing", "api", "partial", "malformed", "symlink"])
def test_subscription_only_rejects_unusable_auth_without_exposing_it(tmp_path: Path, damage: str) -> None:
    """The new capability boundary must fail before any model/API fallback process."""
    canonical, private = tmp_path / "canonical", tmp_path / "private"
    value = _login(canonical)
    path = canonical / "auth.json"
    if damage == "missing":
        path.unlink()
    elif damage == "api":
        value["OPENAI_API_KEY"] = "synthetic-secret"
        path.write_text(json.dumps(value))
    elif damage == "partial":
        value["tokens"].pop("refresh_token")
        path.write_text(json.dumps(value))
    elif damage == "malformed":
        path.write_text("synthetic-secret:invalid-json")
    else:
        destination = canonical / "actual-auth.json"
        path.rename(destination)
        path.symlink_to(destination)
    with pytest.raises(harbor_auth.SubscriptionAuthError) as caught:
        with harbor_auth.subscription_lease(canonical, private):
            pytest.fail("invalid subscription auth reached reviewer")
    assert "synthetic-secret" not in str(caught.value)
    assert not (private / "auth.json").exists()


def test_external_source_rotation_preserves_both_stores(tmp_path: Path) -> None:
    """A noncooperating native client must not lose its newer login at job shutdown."""
    source, canonical = tmp_path / "source", tmp_path / "canonical"
    _login(source)
    with pytest.raises(harbor_auth.SubscriptionAuthError, match="changed outside"):
        with harbor_auth.subscription_session(source, canonical):
            outside = json.loads((source / "auth.json").read_text())
            outside["tokens"]["refresh_token"] = "synthetic-outside-rotation"
            (source / "auth.json").write_text(json.dumps(outside))
    assert json.loads((source / "auth.json").read_text())["tokens"]["refresh_token"] == "synthetic-outside-rotation"
    assert json.loads((canonical / "auth.json").read_text())["tokens"]["refresh_token"] == "synthetic-refresh"


def test_killed_role_blocks_replay_of_uncertain_refresh(tmp_path: Path) -> None:
    """Process death releases flock but cannot authorize reuse of pre-refresh auth."""
    canonical = tmp_path / "canonical"
    original = _login(canonical)
    script = tmp_path / "killed.py"
    script.write_text(
        "import importlib.util, json, os, pathlib, sys\n"
        "spec = importlib.util.spec_from_file_location('auth', sys.argv[1])\n"
        "auth = importlib.util.module_from_spec(spec); spec.loader.exec_module(auth)\n"
        "root = pathlib.Path(sys.argv[2])\n"
        "with auth.subscription_lease(root / 'canonical', root / 'private') as home:\n"
        " data = json.loads((home / 'auth.json').read_text())\n"
        " data['tokens']['refresh_token'] = 'synthetic-uncertain-rotation'\n"
        " (home / 'auth.json').write_text(json.dumps(data))\n"
        " os._exit(37)\n"
    )
    result = subprocess.run([sys.executable, str(script), str(Path(harbor_auth.__file__)), str(tmp_path)])
    assert result.returncode == 37
    with pytest.raises(harbor_auth.SubscriptionAuthError, match="credential recovery"):
        with harbor_auth.subscription_lease(canonical, tmp_path / "new-private"):
            pytest.fail("uncertain prior refresh reached another model invocation")
    assert json.loads((canonical / "auth.json").read_text()) == original
    shutil.rmtree(tmp_path / "private")  # Simulate disposable container filesystem removal.
    recovery = canonical / ".lease-auth.json"
    assert json.loads(recovery.read_text())["tokens"]["refresh_token"] == "synthetic-uncertain-rotation"
    assert recovery.stat().st_mode & 0o777 == 0o600
    assert json.loads((canonical / ".lease-state.json").read_text()) == {
        "role": "reviewer", "credential_file": ".lease-auth.json",
    }


@pytest.mark.parametrize('role', ['reviewer', 'judge'])
async def test_production_role_retains_rotated_credentials_after_writeback_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_gh: FakeGh, role: str,
) -> None:
    """Caller TemporaryDirectory cleanup must not erase the only rotated token.

    The earlier lease-only death test kept its caller home alive; it missed both
    production callers' automatic cleanup. Inject at the actual atomic replace
    after a real native-boundary invocation writes refreshed credentials.
    """
    from daydream import runner
    from daydream.benchmark.reviewbench.harbor_entrypoint import main as reviewer_main
    from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend
    from tests.test_reviewbench_harbor_agent import public_input

    environment = public_input(tmp_path)
    source, canonical = tmp_path / 'source', tmp_path / 'canonical'
    original = _login(source)
    environment['DAYDREAM_RB_CODEX_HOME'] = str(canonical)
    role_homes: list[Path] = []
    observed_home = tmp_path / 'observed-home'
    models = tmp_path / 'models.json'
    models.write_text('{}')
    script = tmp_path / 'rotate.py'
    script.write_text(
        'import json, os, pathlib, sys\n'
        'home = pathlib.Path(os.environ["PI_CODING_AGENT_DIR"])\n'
        'pathlib.Path(sys.argv[1]).write_text(str(home))\n'
        'auth = json.loads((home / "auth.json").read_text())\n'
        'auth["openai-codex"]["refresh"] = "synthetic-production-rotation"\n'
        '(home / "auth.json").write_text(json.dumps(auth))\n'
    )

    def create(name: str, **kwargs: Any) -> EmptyReviewBackend:
        assert name == 'codex'
        home = Path(kwargs['execution_input'].child_environment()['CODEX_HOME'])
        role_homes.append(home)
        auth = json.loads((home / 'auth.json').read_text())
        auth['tokens']['refresh_token'] = 'synthetic-production-rotation'
        (home / 'auth.json').write_text(json.dumps(auth))
        return EmptyReviewBackend(Path(environment['DAYDREAM_RB_REPOSITORY']),
                                  forbid_merge=False, forbid_supervise=False)

    monkeypatch.setattr(runner, 'create_backend', create)
    replace = os.replace

    def fail_canonical_replace(path: str | Path, target: str | Path) -> None:
        if target == canonical / 'auth.json' and (canonical / '.lease-state.json').exists():
            raise OSError('synthetic canonical replacement failure')
        return replace(path, target)

    monkeypatch.setattr(os, 'replace', fail_canonical_replace)
    with pytest.raises(harbor_auth.SubscriptionAuthError, match='prevents source credential reconciliation'):
        with harbor_auth.subscription_session(source, canonical):
            if role == 'reviewer':
                assert await reviewer_main(environment) == 1
            else:
                assert harbor_auth.main([
                    'judge', '--canonical-home', str(canonical), '--models', str(models), '--',
                    sys.executable, str(script), str(observed_home),
                ]) == 1
                role_homes.append(Path(observed_home.read_text()))
    assert role_homes and all(not home.exists() for home in role_homes)
    assert not fake_gh.calls("POST")
    assert json.loads((source / 'auth.json').read_text()) == original
    assert json.loads((canonical / 'auth.json').read_text()) == original
    recovery_path = canonical / '.lease-auth.json'
    recovery = json.loads(recovery_path.read_text())
    rotated = recovery['tokens']['refresh_token'] if role == 'reviewer' else recovery['openai-codex']['refresh']
    assert rotated == 'synthetic-production-rotation'
    assert recovery_path.stat().st_mode & 0o777 == 0o600
    assert set(path.name for path in canonical.iterdir()) == {
        'auth.json', '.lease.lock', '.lease-state.json', '.lease-auth.json',
    }


def test_competing_canonical_rotation_preserves_both_credentials(tmp_path: Path) -> None:
    """A noncooperating canonical writer must not be overwritten by lease cleanup."""
    canonical = tmp_path / 'canonical'
    original = _login(canonical)
    with pytest.raises(harbor_auth.SubscriptionAuthError, match='canonical auth changed outside'):
        with harbor_auth.subscription_lease(canonical, tmp_path / 'reviewer') as home:
            role = json.loads((home / 'auth.json').read_text())
            role['tokens']['refresh_token'] = 'synthetic-role-rotation'
            (home / 'auth.json').write_text(json.dumps(role))
            original['tokens']['refresh_token'] = 'synthetic-competing-rotation'
            (canonical / 'auth.json').write_text(json.dumps(original))
    assert json.loads((canonical / 'auth.json').read_text())['tokens']['refresh_token'] == (
        'synthetic-competing-rotation'
    )
    assert json.loads((canonical / '.lease-auth.json').read_text())['tokens']['refresh_token'] == (
        'synthetic-role-rotation'
    )


@pytest.mark.parametrize('recovery', [False, True])
def test_linux_container_root_returns_private_auth_to_host_owner(recovery: bool) -> None:
    """Root-written mode600 files otherwise strand a nonroot Linux host's login.

    Docker Desktop ownership translation can mask this regression. Exercise real
    Linux UID changes with the shipped standalone auth code, without credentials,
    image pulls, containers with network access, or model transports.
    """
    docker = shutil.which('docker')
    image = ('python:3.12.11-slim-bookworm@sha256:'
             '519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7')
    if docker is None:
        pytest.skip('Docker is unavailable for Linux ownership proof')
    if subprocess.run([docker, 'image', 'inspect', image], capture_output=True, timeout=15).returncode:
        pytest.skip('Pinned Python image is not cached; ownership proof never pulls images')
    script = '''
import base64, importlib.util, json, os, pathlib, sys, tempfile
spec = importlib.util.spec_from_file_location('auth', '/auth.py')
auth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(auth)
root = pathlib.Path(tempfile.mkdtemp())
root.chmod(0o700)
os.chown(root, 1000, 1000)
source, canonical, role = root/'source', root/'canonical', root/'role'
source.mkdir(mode=0o700)
os.chown(source, 1000, 1000)
claims = base64.urlsafe_b64encode(b'{"exp":1}').decode().rstrip('=')
original = {'auth_mode':'chatgpt','OPENAI_API_KEY':None,'tokens':{
 'access_token':f'test.{claims}.test','refresh_token':'synthetic-original',
 'account_id':'test-account','id_token':'synthetic-id'}}
(source/'auth.json').write_text(json.dumps(original))
(source/'auth.json').chmod(0o600)
os.chown(source/'auth.json', 1000, 1000)
os.setegid(1000)
os.seteuid(1000)
recovery = sys.argv[1] == 'True'
replace = os.replace
try:
 with auth.subscription_session(source, canonical):
  os.seteuid(0)
  def fail_replacement(path, target):
   if recovery and target == canonical/'auth.json' and (canonical/'.lease-state.json').exists():
    raise OSError('synthetic replacement fault')
   return replace(path, target)
  os.replace = fail_replacement
  try:
   with auth.subscription_lease(canonical, role) as home:
    updated = json.loads((home/'auth.json').read_text())
    updated['tokens']['refresh_token'] = 'synthetic-rotated'
    (home/'auth.json').write_text(json.dumps(updated))
  except OSError:
   assert recovery
  finally:
   os.replace = replace
   os.seteuid(1000)
  for file in canonical.iterdir():
   info = file.stat()
   assert (info.st_uid, info.st_gid, info.st_mode & 511) == (1000,1000,384), (file.name, info)
   file.read_text()
  with (canonical/'.lease.lock').open('r+'):
   pass
except auth.SubscriptionAuthError:
 assert recovery
else:
 assert not recovery
if recovery:
 assert json.loads((canonical/'.lease-auth.json').read_text())['tokens']['refresh_token'] == 'synthetic-rotated'
 assert json.loads((source/'auth.json').read_text()) == original
else:
 assert json.loads((source/'auth.json').read_text())['tokens']['refresh_token'] == 'synthetic-rotated'
 assert not (canonical/'auth.json').exists()
print(json.dumps({'host_uid':os.geteuid(),'private_files_readable':True,'recovery':recovery}))
'''
    result = subprocess.run([
        docker, 'run', '--rm', '--pull', 'never', '--network', 'none',
        '-v', f'{Path(harbor_auth.__file__).resolve()}:/auth.py:ro',
        '-v', f'{Path(harbor_auth.__file__).resolve().parents[2] / "json_utils.py"}:/daydream_json_utils.py:ro',
        image,
        'python', '-c', script, str(recovery),
    ], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {'host_uid': 1000, 'private_files_readable': True, 'recovery': recovery}
