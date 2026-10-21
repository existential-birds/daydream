"""Fail closed at the public image's no-model sandbox admission boundary."""
from __future__ import annotations

import hashlib
import json
import platform
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
    assert hashlib.sha256((root / 'harbor-seccomp-LICENSE').read_bytes()).hexdigest() == (
        provenance['upstream']['license_sha256'])
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
