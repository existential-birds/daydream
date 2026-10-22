"""Fail closed at the public image's no-model sandbox admission boundary."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from daydream.benchmark.reviewbench import harbor_sandbox_probe as probe
from daydream.benchmark.reviewbench.harbor_agent import DaydreamPublicReviewAgent


@pytest.mark.parametrize('outcome', ['unavailable', 'unverified', 'verified'])
def test_sandbox_probe_replays_cli_without_personal_auth_or_model_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    """A version check misses namespace and seccomp failures until after inference.

    Exercise actual subprocess invocation and fresh-home cleanup against a native
    CLI transport replay. This owns admission and credential isolation; actual
    native enforcement is separately demonstrated with the pinned Docker probe.
    """
    executable = tmp_path / 'bin' / 'codex'
    executable.parent.mkdir()
    record = tmp_path / 'cli.json'
    executable.write_text(
        f'#!{sys.executable}\n'
        'import json,os,pathlib,sys\n'
        'assert sys.argv[1:5]==["-c",\'sandbox_mode="read-only"\',"sandbox","--"]\n'
        'assert pathlib.Path(sys.argv[-1]).read_text()=="daydream sandbox preflight\\n"\n'
        'home=pathlib.Path(os.environ["CODEX_HOME"])\n'
        f'pathlib.Path({str(record)!r}).write_text(json.dumps({{'
        '"home":str(home),"auth_exists":(home/"auth.json").exists(),'
        '"has_api_key":"OPENAI_API_KEY" in os.environ}))\n'
        + ('print("bwrap: namespace unavailable",file=sys.stderr);sys.exit(1)\n' if outcome == 'unavailable' else
           'print(json.dumps({"source_read":True,"source_write_denied":' +
           ('False' if outcome == 'unverified' else 'True') +
           ',"internet_socket_denied":True,"effective_capabilities":0,"seccomp_filters":2}))\n')
    )
    executable.chmod(0o755)
    personal = tmp_path / 'personal-codex'
    personal.mkdir()
    (personal / 'auth.json').write_text('private operator credential canary')
    monkeypatch.setenv('PATH', str(executable.parent))
    monkeypatch.setenv('CODEX_HOME', str(personal))
    monkeypatch.setenv('OPENAI_API_KEY', 'forbidden API fallback canary')
    monkeypatch.setattr(platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'machine', lambda: 'aarch64')
    if outcome == 'verified':
        assert probe.run_probe('linux/arm64')['model_calls'] == 0
    else:
        with pytest.raises((RuntimeError, ValueError), match='unavailable|did not verify'):
            probe.run_probe('linux/arm64')
    actual = json.loads(record.read_text())
    assert not actual['auth_exists'] and not actual['has_api_key']
    assert Path(actual['home']) != personal and not Path(actual['home']).exists()
    assert (personal / 'auth.json').read_text() == 'private operator credential canary'


async def test_public_agent_setup_rejects_failed_native_sandbox_before_run(tmp_path: Path) -> None:
    """Harbor must receive a setup failure before it can admit reviewer inference."""
    class Environment:
        async def exec(self, command: str, **kwargs: Any) -> Any:
            if command == 'python -I -m daydream.benchmark.reviewbench.harbor_sandbox_probe':
                return SimpleNamespace(return_code=1, stdout='', stderr='native seccomp unavailable')
            return SimpleNamespace(return_code=0, stdout='', stderr='')

    agent = DaydreamPublicReviewAgent(logs_dir=tmp_path)
    with pytest.raises(RuntimeError, match='sandbox preflight failed before model execution'):
        await agent.setup(Environment())
    assert (tmp_path / 'sandbox-preflight.log').read_text() == 'native seccomp unavailable'


def test_shipped_seccomp_preserves_pinned_default_and_only_declared_bootstrap_additions() -> None:
    """A profile update must not silently replace Docker's default-deny policy.

    The provenance binds unchanged upstream rules and the exact appended delta;
    this is shipped isolation material independently of task/export bookkeeping.
    """
    root = Path(probe.__file__).parent
    policy_bytes = (root / 'harbor-seccomp.json').read_bytes()
    policy = json.loads(policy_bytes)
    provenance = json.loads((root / 'harbor-seccomp-provenance.json').read_text())
    assert hashlib.sha256(policy_bytes).hexdigest() == provenance['policy_sha256']
    additions = provenance['appended_rules']
    assert policy['syscalls'][-len(additions):] == additions
    policy['syscalls'] = policy['syscalls'][:-len(additions)]
    base = json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(base).hexdigest() == provenance['upstream']['canonical_sha256']
    assert policy['defaultAction'] == 'SCMP_ACT_ERRNO'
    assert {name for rule in additions for name in rule['names']} == {
        'clone', 'unshare', 'mount', 'pivot_root', 'umount2'}
    assert all(rule['action'] == 'SCMP_ACT_ALLOW' for rule in additions)
    assert all(rule.get('args') for rule in additions if rule['names'] != ['pivot_root'])
    # runc turns repeated argument indices into alternative rules, not an AND.
    for rule in additions:
        indices = [argument['index'] for argument in rule.get('args', [])]
        assert len(indices) == len(set(indices)), rule


def test_shipped_mount_filter_requires_bootstrap_flags_and_rejects_forbidden_bits(tmp_path: Path) -> None:
    """Exercise the shipped filter through Docker/runc, without auth or models.

    A positive native sandbox probe also passes a broadened outer mount rule.
    Distinct denied errno makes actual filter rejection observable independently
    of the kernel's capability/path rejection for an admitted mount syscall.
    """
    image = os.environ.get('DAYDREAM_REVIEWBENCH_REVIEWER_IMAGE')
    if not image:
        pytest.skip('set DAYDREAM_REVIEWBENCH_REVIEWER_IMAGE to a built public reviewer image')
    policy = json.loads(Path(probe.__file__).with_name('harbor-seccomp.json').read_bytes())
    policy['defaultErrnoRet'] = 99
    diagnostic = tmp_path / 'seccomp-denied-errno99.json'
    diagnostic.write_text(json.dumps(policy))
    script = '''import ctypes,json,pathlib
libc=ctypes.CDLL(None,use_errno=True)
libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_void_p]
libc.mount.restype=ctypes.c_int
rows={}
for name,flags in [('required',36898),('required_readonly',36899),('no_flags',0),
                   ('missing_remount',36866),('forbidden_synchronous',36914)]:
 ctypes.set_errno(0)
 assert libc.mount(b'none',b'/nonexistent-daydream-mount-target',b'tmpfs',flags,None)==-1
 rows[name]=ctypes.get_errno()
status=dict(line.split(':',1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert int(status['CapEff'],16)==0 and status['Seccomp'].strip()=='2'
print(json.dumps(rows))
'''
    result = subprocess.run([
        'docker', 'run', '--rm', '--network', 'none', '--cap-drop', 'ALL',
        '--security-opt', f'seccomp={diagnostic}', '--entrypoint', '/opt/daydream/bin/python', image,
        '-I', '-c', script,
    ], check=True, capture_output=True, text=True, timeout=45)
    observed = json.loads(result.stdout)
    assert set(observed) == {'required', 'required_readonly', 'no_flags', 'missing_remount', 'forbidden_synchronous'}
    assert observed['required'] != 99 and observed['required_readonly'] != 99, observed
    assert all(observed[name] == 99 for name in ('no_flags', 'missing_remount', 'forbidden_synchronous')), observed
