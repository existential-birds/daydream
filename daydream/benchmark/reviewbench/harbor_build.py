"""Reproducible public reviewer image inputs from an actual installed wheel."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Any

from daydream.benchmark.reviewbench.candidate import candidate_package_files

NODE_IMAGE = 'node:24.18.1-bookworm-slim@sha256:235600a8101ab264e117b1768e925532262668dc9b581ef1dd7d96ced463b8e7'
UV_IMAGE = 'ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc'
HARBOR_REVISION = '6ce6abee0b9fa94a50c88f0557d030a419998d78'
REVIEWBENCH_PLATFORMS = ('linux/amd64', 'linux/arm64')


def reviewer_build(wheel: Path, *, platform: str = 'linux/amd64') -> tuple[str, dict[str, bytes], dict[str, Any]]:
    """Freeze wheel and resolved dependency bytes; no source-checkout install."""
    if platform not in REVIEWBENCH_PLATFORMS:
        raise ValueError('Public ReviewBench requires an explicit supported Linux platform')
    if not wheel.is_file() or wheel.suffix != '.whl':
        raise ValueError('An actual Daydream wheel is required')
    if not re.fullmatch(r'daydream-[A-Za-z0-9_.-]+\.whl', wheel.name):
        raise ValueError('Unsafe Daydream wheel filename')
    wheel_bytes = wheel.read_bytes()
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1 or '\nName: daydream\n' not in archive.read(metadata[0]).decode():
            raise ValueError('Reviewer wheel must contain the Daydream distribution')
        required = 'daydream/benchmark/reviewbench/harbor_entrypoint.py'
        if required not in archive.namelist():
            raise ValueError('Reviewer wheel lacks the public Harbor runtime')
        installed = candidate_package_files()
        wheel_names = {name for name in archive.namelist() if not name.endswith('/') and (
            name.startswith(('daydream/', 'daydream_ext/')) or name == 'daydream_ext.py'
        )}
        if wheel_names != installed.keys():
            raise ValueError('Reviewer wheel package membership differs from the executing candidate')
        for name, path in installed.items():
            if path.read_bytes() != archive.read(name):
                raise ValueError('Export must execute the same candidate package as its reviewer wheel')
    requirements = Path(__file__).with_name('reviewer-runtime.lock').read_bytes()
    host_requirements = Path(__file__).with_name('harbor-runtime.lock').read_bytes()
    assets = {wheel.name: wheel_bytes, 'reviewer-runtime.lock': requirements}
    # Only the host installs Harbor/ReviewBench. Their Git source distributions
    # include reference answers, including in uv's retained source checkout.
    # Install the hash-locked core graph without caches, then only our own wheel.
    suffix = (
        f'\nCOPY --from={UV_IMAGE} /uv /usr/local/bin/uv\n'
        'ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_NO_CACHE=1\n'
        'RUN uv python install 3.14.3 && uv venv --python 3.14.3 /opt/daydream\n'
        'COPY reviewer-runtime.lock /opt/reviewer-runtime.lock\n'
        f'COPY {wheel.name} /opt/{wheel.name}\n'
        'RUN uv pip install --no-cache --require-hashes --python /opt/daydream/bin/python '
        '-r /opt/reviewer-runtime.lock\n'
        f'RUN uv pip install --no-cache --no-deps --python /opt/daydream/bin/python /opt/{wheel.name} '
        '&& uv pip check --python /opt/daydream/bin/python\n'
        'RUN npm install --global --ignore-scripts @openai/codex@0.163.0\n'
        'ENV PATH="/opt/daydream/bin:${PATH}"\n'
        'RUN python --version && codex --version && python -c '
        "\"from importlib.util import find_spec; "
        "assert find_spec('harbor') is None; assert find_spec('reviewbench_harbor') is None; "
        "from daydream.benchmark.reviewbench.harbor_entrypoint import main\" "
        '&& test ! -d /root/.cache/uv/git-v0\n'
    )
    identity = {'wheel_name': wheel.name, 'wheel_sha256': hashlib.sha256(assets[wheel.name]).hexdigest(),
                'reviewer_runtime_lock_sha256': hashlib.sha256(requirements).hexdigest(),
                'host_runtime_lock_sha256': hashlib.sha256(host_requirements).hexdigest(),
                'codex_cli': '0.163.0', 'python': '3.14.3', 'uv_image': UV_IMAGE,
                'platform': platform, 'harbor_revision': HARBOR_REVISION}
    return suffix, assets, identity
