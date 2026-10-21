"""Reproducible public reviewer image inputs from an actual installed wheel."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any

NODE_IMAGE = 'node:24.18.1-bookworm-slim@sha256:235600a8101ab264e117b1768e925532262668dc9b581ef1dd7d96ced463b8e7'
UV_IMAGE = 'ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc'
HARBOR_REVISION = '6ce6abee0b9fa94a50c88f0557d030a419998d78'


def reviewer_build(wheel: Path) -> tuple[str, dict[str, bytes], dict[str, Any]]:
    """Freeze wheel and resolved dependency bytes; no source-checkout install."""
    if not wheel.is_file() or wheel.suffix != '.whl':
        raise ValueError('An actual Daydream wheel is required')
    if not re.fullmatch(r'daydream-[A-Za-z0-9_.-]+\.whl', wheel.name):
        raise ValueError('Unsafe Daydream wheel filename')
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1 or '\nName: daydream\n' not in archive.read(metadata[0]).decode():
            raise ValueError('Reviewer wheel must contain the Daydream distribution')
        required = 'daydream/benchmark/reviewbench/harbor_entrypoint.py'
        if required not in archive.namelist():
            raise ValueError('Reviewer wheel lacks the public Harbor runtime')
        package = Path(__file__).resolve().parents[2]
        installed_names = {
            'daydream/' + path.relative_to(package).as_posix()
            for path in package.rglob('*') if path.is_file()
            and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}
            and path.name != '.DS_Store'
        }
        wheel_names = {name for name in archive.namelist() if name.startswith('daydream/') and not name.endswith('/')}
        if wheel_names != installed_names:
            raise ValueError('Reviewer wheel package membership differs from the executing candidate')
        for name in archive.namelist():
            if name.startswith('daydream/') and not name.endswith('/'):
                installed = package / name.removeprefix('daydream/')
                if not installed.is_file() or installed.read_bytes() != archive.read(name):
                    raise ValueError('Export must execute the same candidate package as its reviewer wheel')
    requirements = Path(__file__).with_name('harbor-runtime.lock').read_bytes()
    assets = {wheel.name: wheel.read_bytes(), 'runtime.lock': requirements}
    # The lock contains the public host runtime too. Installing it in the reviewer
    # keeps the exact same package graph, but exposes no verifier assets/goldens.
    suffix = (
        f'\nCOPY --from={UV_IMAGE} /uv /usr/local/bin/uv\n'
        'ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy\n'
        'RUN uv python install 3.14.3 && uv venv --python 3.14.3 /opt/daydream\n'
        'COPY runtime.lock /opt/runtime.lock\n'
        f'COPY {wheel.name} /opt/{wheel.name}\n'
        f'RUN uv pip install --python /opt/daydream/bin/python -r /opt/runtime.lock /opt/{wheel.name}\n'
        'RUN npm install --global --ignore-scripts @openai/codex@0.163.0\n'
        'ENV PATH="/opt/daydream/bin:${PATH}"\n'
        'RUN python --version && codex --version\n'
    )
    identity = {'wheel_name': wheel.name, 'wheel_sha256': hashlib.sha256(assets[wheel.name]).hexdigest(),
                'runtime_lock_sha256': hashlib.sha256(requirements).hexdigest(),
                'codex_cli': '0.163.0', 'python': '3.14.3', 'uv_image': UV_IMAGE,
                'platform': 'linux/amd64', 'harbor_revision': HARBOR_REVISION}
    return suffix, assets, identity
