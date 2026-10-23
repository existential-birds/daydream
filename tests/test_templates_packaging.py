"""Workflow templates and native tools must ship in built wheels.

The packaged single-file variant must agree with the setup secret/variable names.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from daydream import config
from daydream.benchmark.reviewbench import harbor_build

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPECTED_TEMPLATES = {"daydream-review.yml", "daydream-command.yml", "daydream-post.yml"}
# Optional manual-copy variant; NOT part of the default `daydream setup` deposit.
_SINGLE_TEMPLATE_PATH = "daydream/templates/workflows/single/daydream.yml"


def test_templates_and_native_tools_present_in_built_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build a real wheel to check packaged paths and canonical workflow names."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["uv", "build", "--wheel", "--out-dir", tmp], cwd=_REPO_ROOT, check=True, capture_output=True,)
        wheels = list(Path(tmp).glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
        expected_paths = {f"daydream/templates/workflows/{name}" for name in _EXPECTED_TEMPLATES}
        expected_paths.add(_SINGLE_TEMPLATE_PATH)  # optional variant ships too
        expected_paths.add('daydream/backends/pi_structured_output.ts')
        expected_paths.add('daydream/backends/pi_host_tools.ts')
        expected_paths.add('daydream/backends/pi_diagnostics.ts')
        with zipfile.ZipFile(wheels[0]) as zf:
            names_in_wheel = set(zf.namelist())
            # Seccomp assets are generated at export, not shipped as copied JSON.
            assert not any(name.endswith(('harbor-seccomp.json', 'harbor-seccomp-provenance.json'))
                           for name in names_in_wheel)
            installed = Path(tmp) / 'installed'
            zf.extractall(installed)
            single = zf.read(_SINGLE_TEMPLATE_PATH).decode("utf-8")
            for secret in config.SETUP_SECRET_NAMES:
                assert secret in single
            assert config.BOT_HANDLE_VAR in single
        assert expected_paths <= names_in_wheel, (f"Built wheel is missing packaged runtime resources. "
            f"Found resources: {sorted(names_in_wheel.intersection(expected_paths))!r}. "
            f"Expected paths: {sorted(expected_paths)!r}. "
            f"Check [tool.hatch.build.targets.wheel] include in pyproject.toml."
        )
        # Execute installed resources outside the source checkout.
        script = """from itertools import product
from pathlib import Path
import sys
from daydream.benchmark.reviewbench.harbor_build import reviewer_build
from daydream.benchmark.reviewbench import harbor_auth

for backend, platform in product(('codex', 'pi'), ('linux/amd64', 'linux/arm64')):
    suffix, assets, identity = reviewer_build(Path(sys.argv[1]), backend=backend, platform=platform)
    assert identity['platform'] == platform
    assert identity[backend + '_cli'] == ('1.3.0' if backend == 'pi' else '0.163.0')
    assert 'reviewer-runtime.lock' in assets and 'runtime.lock' not in assets
    assert identity['reviewer_runtime_lock_sha256'] != identity['host_runtime_lock_sha256']
    if backend == 'pi':
        names = ('package.json', 'package-lock.json', 'daydream-auth.mjs')
        assert all('pi-reviewer/' + name in assets for name in names)
        assert 'npm ci' in suffix and 'codex_cli' not in identity
"""
        installed_environment = {**os.environ, 'PYTHONPATH': str(installed)}
        subprocess.run([sys.executable, '-c', script, str(wheels[0])], cwd=tmp,
                       env=installed_environment, check=True, capture_output=True, text=True)
        help_result = subprocess.run([sys.executable, '-m', 'daydream', 'benchmark', 'reviewbench',
                                      'harbor-export', '--help'], cwd=tmp,
                                     env=installed_environment,
                                     check=True, capture_output=True, text=True)
        assert all(flag in help_result.stdout for flag in ('--wheel', '--source', '--platform'))
        # Concurrent filename replacement must preserve the admitted wheel bytes.
        wheel = wheels[0]
        admitted = wheel.read_bytes()
        replacement = Path(tmp) / 'replacement.whl'
        replacement_bytes = b'competing unvalidated wheel bytes'
        replacement.write_bytes(replacement_bytes)
        runtime_lock = Path(harbor_build.__file__).with_name('reviewer-runtime.lock')
        read_bytes = Path.read_bytes

        def replace_after_validation(path: Path) -> bytes:
            if path == runtime_lock:
                os.replace(replacement, wheel)
            return read_bytes(path)

        monkeypatch.setattr(Path, 'read_bytes', replace_after_validation)
        _, assets, identity = harbor_build.reviewer_build(wheel)
        assert wheel.read_bytes() == replacement_bytes
        assert identity['wheel_sha256'] == hashlib.sha256(admitted).hexdigest()
        assert assets[wheel.name] == admitted


def test_public_reviewer_image_excludes_supplied_verifier_material() -> None:
    image = os.environ.get('DAYDREAM_REVIEWBENCH_REVIEWER_IMAGE')
    if not image:
        pytest.skip('set DAYDREAM_REVIEWBENCH_REVIEWER_IMAGE to the final built public reviewer image')
    script = '''import importlib.util
from pathlib import Path

for module in ("harbor", "reviewbench_harbor"):
    assert importlib.util.find_spec(module) is None, f"verifier module visible: {module}"
cache = Path("/root/.cache/uv")
assert not cache.exists() or not any(cache.iterdir()), "reviewer retained uv cache"
for root in (Path("/root"), Path("/opt"), Path("/tmp")):
    for pattern in ("corpus/manifest.json", "corpus/test/findings.json", "scripts/eval/matcher.ts",
                    "scripts/classifier/classify.ts", "golden/*.json"):
        assert not list(root.rglob(pattern)), f"verifier material visible under {root}: {pattern}"
from daydream.benchmark.reviewbench.harbor_entrypoint import main
'''
    subprocess.run([
        'docker', 'run', '--rm', '--network', 'none', '--entrypoint', '/opt/daydream/bin/python', image, '-c', script,
    ], check=True, capture_output=True, text=True, timeout=180)
