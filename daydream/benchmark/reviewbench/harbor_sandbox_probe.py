"""No-model admission check for the native Codex read-only tool sandbox.

A version probe cannot detect denied nested namespaces or emulation that rejects
Codex's own seccomp filter. Use a fresh credential-free Codex home and a synthetic
source file, never an inference command, before admitting a benchmark reviewer.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROBE = '''import errno,json,pathlib,socket,sys
source=pathlib.Path(sys.argv[1])
assert source.read_text()=="daydream sandbox preflight\\n"
try:
 source.write_text("must not be writable")
 raise AssertionError("native read-only sandbox allowed a source write")
except OSError as error:
 assert error.errno in (errno.EROFS,errno.EACCES,errno.EPERM)
try:
 socket.socket(socket.AF_INET,socket.SOCK_STREAM)
 raise AssertionError("native sandbox allowed an internet socket")
except OSError as error:
 assert error.errno==errno.EPERM
lines=pathlib.Path("/proc/self/status").read_text().splitlines()
status={key:value.strip() for key,value in (line.split(":",1) for line in lines if ":" in line)}
assert int(status["CapEff"],16)==0
assert status["Seccomp"]=="2" and int(status["Seccomp_filters"])>=2
print(json.dumps({"source_read":True,"source_write_denied":True,"internet_socket_denied":True,"effective_capabilities":0,"seccomp_filters":int(status["Seccomp_filters"])}))
'''


def run_probe(expected_platform: str) -> dict[str, Any]:
    """Check the actual native sandbox without any subscription or API access."""
    architecture = {'aarch64': 'arm64', 'arm64': 'arm64', 'x86_64': 'amd64'}.get(platform.machine())
    if (platform.system() != 'Linux' or expected_platform not in {'linux/amd64', 'linux/arm64'}
            or expected_platform != 'linux/' + str(architecture)):
        raise ValueError('Reviewer platform differs from its frozen public profile')
    with tempfile.TemporaryDirectory(prefix='daydream-codex-sandbox-preflight-') as temporary:
        root = Path(temporary).resolve()
        home = root / 'home'
        home.mkdir()
        source = root / 'source.txt'
        source.write_text('daydream sandbox preflight\n')
        # No personal home, auth store, endpoint override, provider credential or
        # proxy settings can enter this command. `sandbox` does not call models.
        environment = {'PATH': os.environ.get('PATH', ''), 'LANG': 'C.UTF-8',
                       'HOME': str(home), 'CODEX_HOME': str(home), 'TMPDIR': str(root)}
        command = ['codex', '-c', 'sandbox_mode="read-only"', 'sandbox', '--',
                   sys.executable, '-c', _PROBE, str(source)]
        result = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True, timeout=45)
        if result.returncode != 0:
            raise RuntimeError('Native Codex read-only sandbox unavailable: ' + result.stderr[-8192:])
        checks = json.loads(result.stdout)
        if (not isinstance(checks, dict) or any(checks.get(key) is not True for key in
                ('source_read', 'source_write_denied', 'internet_socket_denied'))
                or checks.get('effective_capabilities') != 0
                or type(checks.get('seccomp_filters')) is not int or checks['seccomp_filters'] < 2):
            raise ValueError('Native Codex sandbox did not verify its restrictions')
        return {'platform': expected_platform, 'model_calls': 0, **checks}


def main() -> int:
    """Probe the image against its task's frozen platform before agent.run."""
    try:
        profile = json.loads(Path('/work/pr/daydream-profile.json').read_text())
        print(json.dumps(run_probe(profile['runtime']['platform']), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f'Public ReviewBench sandbox preflight failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
