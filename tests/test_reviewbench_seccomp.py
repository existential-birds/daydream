from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

import httpx
import pytest

from daydream.benchmark.reviewbench import harbor_seccomp


@pytest.fixture
def moby_policy(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Replace only upstream HTTP; retain real seccomp generation and export."""
    upstream = b'{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[{"names":["read"],"action":"SCMP_ACT_ALLOW"}]}'
    monkeypatch.setattr(harbor_seccomp, 'MOBY_SHA256', hashlib.sha256(upstream).hexdigest())
    monkeypatch.setattr(httpx, 'get', lambda url, **kwargs: httpx.Response(
        200, content=upstream, request=httpx.Request('GET', url)))
    return upstream


def test_generated_seccomp_preserves_pinned_default_and_only_declared_bootstrap_additions(moby_policy: bytes) -> None:
    policy_bytes, provenance_bytes = harbor_seccomp.seccomp_profile()
    policy = json.loads(policy_bytes)
    provenance = json.loads(provenance_bytes)
    assert hashlib.sha256(policy_bytes).hexdigest() == provenance['policy_sha256']
    additions = provenance['appended_rules']
    assert policy['syscalls'][-len(additions):] == additions
    policy['syscalls'] = policy['syscalls'][:-len(additions)]
    assert policy == json.loads(moby_policy)
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


@pytest.mark.parametrize('status', [200, 404])
def test_seccomp_download_rejects_changed_or_unavailable_upstream(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setattr(httpx, 'get', lambda url, **kwargs: httpx.Response(
        status, content=b'{}', request=httpx.Request('GET', url)))
    with pytest.raises(ValueError if status == 200 else httpx.HTTPStatusError):
        harbor_seccomp.seccomp_profile()


def test_generated_mount_filter_requires_bootstrap_flags_and_rejects_forbidden_bits(tmp_path: Path) -> None:
    image = os.environ.get('DAYDREAM_REVIEWBENCH_REVIEWER_IMAGE')
    if not image:
        pytest.skip('set DAYDREAM_REVIEWBENCH_REVIEWER_IMAGE to a built public reviewer image')
    policy = json.loads(harbor_seccomp.seccomp_profile()[0])
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
    name = 'daydream-seccomp-' + uuid.uuid4().hex
    try:
        result = subprocess.run(['docker', 'run', '--name', name, '--network', 'none', '--cap-drop', 'ALL',
            '--security-opt', f'seccomp={diagnostic}', '--entrypoint', '/opt/daydream/bin/python',
            image, '-I', '-c', script], check=True, capture_output=True, timeout=225)
        observed = json.loads(result.stdout)
    finally:
        cleanup = subprocess.run(['docker', 'rm', '--force', name], capture_output=True, timeout=10)
        assert cleanup.returncode == 0 or b'No such container' in cleanup.stderr, cleanup.stderr
    assert set(observed) == {'required', 'required_readonly', 'no_flags', 'missing_remount', 'forbidden_synchronous'}
    assert observed['required'] != 99 and observed['required_readonly'] != 99, observed
    assert all(observed[name] == 99 for name in ('no_flags', 'missing_remount', 'forbidden_synchronous')), observed
