from __future__ import annotations

import hashlib
import json

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
