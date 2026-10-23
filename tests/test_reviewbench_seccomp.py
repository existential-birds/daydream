"""Generated reviewer seccomp policy preserves Docker defaults and bounded bootstrap rules."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
from pathlib import Path

import httpx
import pytest

from daydream.benchmark.reviewbench import harbor_seccomp
from tests.harness.reviewbench_docker import run_seccomp_probe


@pytest.fixture
def moby_policy(monkeypatch: pytest.MonkeyPatch) -> bytes:
    """Replace only upstream HTTP; preserve real policy generation and export."""
    upstream = b'{"defaultAction":"SCMP_ACT_ERRNO","syscalls":[{"names":["read"],"action":"SCMP_ACT_ALLOW"}]}'
    monkeypatch.setattr(harbor_seccomp, 'MOBY_SHA256', hashlib.sha256(upstream).hexdigest())
    monkeypatch.setattr(httpx, 'get', lambda url, **kwargs: httpx.Response(
        200, content=upstream, request=httpx.Request('GET', url)))
    return upstream


def test_generated_seccomp_preserves_pinned_default_and_only_declared_bootstrap_additions(moby_policy: bytes) -> None:
    """A profile update must not silently replace Docker's default-deny policy.

    The provenance binds unchanged upstream rules and the exact appended delta;
    this is runtime isolation material independently of task/export bookkeeping.
    """
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
    """Generation must fail rather than silently accept changed or missing rules.

    Shipped-file tests never exercised the new network trust boundary.
    """
    monkeypatch.setattr(httpx, 'get', lambda url, **kwargs: httpx.Response(
        status, content=b'{}', request=httpx.Request('GET', url)))
    with pytest.raises(ValueError if status == 200 else httpx.HTTPStatusError):
        harbor_seccomp.seccomp_profile()


async def test_generated_mount_filter_requires_bootstrap_flags_and_rejects_forbidden_bits(tmp_path: Path) -> None:
    """Exercise the generated filter through Docker/runc, without auth or models.

    Distinct denied errno makes actual filter rejection observable independently
    of the kernel's capability/path rejection for an admitted mount syscall.
    """
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
print('DAYDREAM_SECCOMP_READY',flush=True)
assert input()=='GO'
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
    observed = json.loads(await run_seccomp_probe(image, diagnostic, script))
    assert set(observed) == {'required', 'required_readonly', 'no_flags', 'missing_remount', 'forbidden_synchronous'}
    assert observed['required'] != 99 and observed['required_readonly'] != 99, observed
    assert all(observed[name] == 99 for name in ('no_flags', 'missing_remount', 'forbidden_synchronous')), observed


@pytest.mark.parametrize('behavior', ['delayed_start', 'startup_hang', 'filter_hang'])
async def test_seccomp_probe_separates_startup_and_execution_and_cleans_only_owned_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behavior: str,
) -> None:
    """Docker startup cannot consume the filter allowance or leak a container.

    Kernel errno assertions start only after Python runs; the original combined
    45s timeout cannot distinguish a created container from a hung probe. Preserve
    real pipes/processes/filesystem with a scripted external Docker protocol;
    the opt-in node separately verifies the actual image/filter through Docker.
    """
    binary = tmp_path / 'bin/docker'
    binary.parent.mkdir()
    state = tmp_path / 'state'
    state.mkdir()
    protected = state / 'other-agent'
    protected.write_text('preserve other container')
    binary.write_text(f'#!{sys.executable}\n' +
        'import json,os,sys,time\nfrom pathlib import Path\n' + f'root=Path({str(state)!r})\n' +
        'args=sys.argv[1:]\n'
        'if args[0]=="run":\n'
        ' name=args[args.index("--name")+1]\n'
        ' (root/"run.json").write_text(json.dumps({"args":args,"name":name,"pid":os.getpid()}))\n'
        ' (root/name).write_text("created")\n' +
        f' time.sleep({1.5 if behavior == "delayed_start" else 60 if behavior == "startup_hang" else 0})\n' +
        ' print("DAYDREAM_SECCOMP_READY",flush=True)\n'
        ' assert input()=="GO"\n'
        ' (root/"go").write_text("probe started")\n' +
        f' time.sleep({60 if behavior == "filter_hang" else 0})\n' +
        ' print("probe result",flush=True)\n'
        'elif args[0]=="rm":\n'
        ' assert args[:2]==["rm","--force"] and len(args)==3\n'
        ' name=args[2]\n'
        ' assert name.startswith("daydream-seccomp-")\n'
        ' (root/"removed.json").write_text(json.dumps(args))\n'
        ' (root/name).unlink(missing_ok=True)\n'
        'else: raise AssertionError(args)\n')
    binary.chmod(0o700)
    monkeypatch.setenv('PATH', str(binary.parent) + os.pathsep + os.environ['PATH'])
    profile = tmp_path / 'profile.json'
    profile.write_text('{}')
    try:
        if behavior == 'delayed_start':
            assert await run_seccomp_probe('synthetic-image', profile, '',
                startup_timeout=10, execution_timeout=1) == 'probe result\n'
        else:
            phase = 'startup' if behavior == 'startup_hang' else 'filter execution'
            with pytest.raises(AssertionError, match=phase):
                await run_seccomp_probe('synthetic-image', profile, '',
                    startup_timeout=1 if behavior == 'startup_hang' else 10, execution_timeout=1)
        started = json.loads((state / 'run.json').read_text())
        assert json.loads((state / 'removed.json').read_text()) == ['rm', '--force', started['name']]
        assert not (state / started['name']).exists()
        assert protected.read_text() == 'preserve other container'
        assert (state / 'go').exists() is (behavior != 'startup_hang')
        with pytest.raises(ProcessLookupError):
            os.kill(started['pid'], 0)
    finally:
        if (state / 'run.json').exists():
            try:
                os.kill(json.loads((state / 'run.json').read_text())['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
