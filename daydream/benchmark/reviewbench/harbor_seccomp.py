"""Generate the nested Codex sandbox policy from a verified Docker default."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

MOBY_REVISION = '4b8214fc4971e06172938a1f1554ccff6647ee80'
MOBY_URL = f'https://raw.githubusercontent.com/moby/profiles/{MOBY_REVISION}/seccomp/default.json'
MOBY_SHA256 = '6416b47770785a41ac59073cdc77d9fe98517df2799dc83ef207e622de3053f6'


def seccomp_profile() -> tuple[bytes, bytes]:
    """Fetch pinned upstream bytes, append bounded bootstrap rules, retain evidence.

    Only export needs network access. Exported datasets contain the complete
    policy and provenance, so execution never downloads or regenerates them.
    """
    response = httpx.get(MOBY_URL, timeout=30)
    response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != MOBY_SHA256:
        raise ValueError('Moby seccomp profile differs from its pinned digest')
    policy = json.loads(response.content)
    upstream_digest = hashlib.sha256(json.dumps(policy, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    additions: list[dict[str, Any]] = [
        {'names': [name], 'action': 'SCMP_ACT_ALLOW',
         'args': [{'index': index, 'value': value, 'op': 'SCMP_CMP_EQ'}]}
        for name, index, values in (
            ('clone', 0, (0x18020011, 0x58020011, 0x38020011, 0x78020011)),
            ('unshare', 0, (0x10000000,)),
            ('mount', 3, (0x8c000, 0x4c000, 6, 14, 10, 0xc0edd000, 0x9000, 0xd000)),
        )
        for value in values
    ]
    # One masked argument requires the remount flags and rejects forbidden bits.
    # Repeated argument indices are ORed by runc, so they cannot express this.
    additions.extend([
        {'names': ['mount'], 'action': 'SCMP_ACT_ALLOW',
         'args': [{'index': 3, 'value': 0xffffffffffdff3f2, 'valueTwo': 0x9022, 'op': 'SCMP_CMP_MASKED_EQ'}]},
        {'names': ['pivot_root'], 'action': 'SCMP_ACT_ALLOW'},
        {'names': ['umount2'], 'action': 'SCMP_ACT_ALLOW',
         'args': [{'index': 1, 'value': 2, 'op': 'SCMP_CMP_EQ'}]},
    ])
    policy['syscalls'].extend(additions)
    policy_bytes = (json.dumps(policy, indent=2) + '\n').encode()
    provenance = {
        'schema': 1,
        'upstream': {'repository': 'https://github.com/moby/profiles', 'revision': MOBY_REVISION,
                     'path': 'seccomp/default.json', 'url': MOBY_URL, 'sha256': MOBY_SHA256,
                     'canonical_sha256': upstream_digest, 'license': 'Apache-2.0'},
        'policy_sha256': hashlib.sha256(policy_bytes).hexdigest(),
        'modification_notice': 'Modified by Daydream: retain all pinned Moby default rules and append only '
                               'argument-bounded namespace/mount bootstrap allowances needed by native Codex '
                               'bubblewrap. No added container capabilities. Reviewer only; verifier retains '
                               'Docker default seccomp.',
        'appended_rules': additions,
        'validated_codex_cli': '0.163.0', 'validated_platforms': ['linux/arm64'],
    }
    return policy_bytes, (json.dumps(provenance, indent=2) + '\n').encode()
