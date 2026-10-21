"""Workflow templates and native tools must ship in built wheels.

The packaged single-file variant must agree with the setup secret/variable names.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from daydream import config

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXPECTED_TEMPLATES = {"daydream-review.yml", "daydream-command.yml", "daydream-post.yml"}
# Optional manual-copy variant; NOT part of the default `daydream setup` deposit.
_SINGLE_TEMPLATE_PATH = "daydream/templates/workflows/single/daydream.yml"


def test_templates_and_native_tools_present_in_built_wheel() -> None:
    """Build a real wheel to check packaged paths and canonical workflow names."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["uv", "build", "--wheel", "--out-dir", tmp], cwd=_REPO_ROOT, check=True, capture_output=True,)
        wheels = list(Path(tmp).glob("*.whl"))
        assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
        expected_paths = {f"daydream/templates/workflows/{name}" for name in _EXPECTED_TEMPLATES}
        expected_paths.add(_SINGLE_TEMPLATE_PATH)  # optional variant ships too
        expected_paths.add('daydream/backends/pi_structured_output.ts')
        expected_paths.update({
            'daydream/benchmark/reviewbench/harbor-runtime.in',
            'daydream/benchmark/reviewbench/harbor-runtime.lock',
            'daydream/benchmark/reviewbench/harbor_entrypoint.py',
            'daydream/benchmark/reviewbench/harbor_auth.py',
            'daydream/benchmark/reviewbench/harbor_sandbox_probe.py',
            'daydream/benchmark/reviewbench/harbor-seccomp.json',
            'daydream/benchmark/reviewbench/harbor-seccomp-provenance.json',
            'daydream/benchmark/reviewbench/harbor-seccomp-LICENSE',
        })
        expected_paths.add('daydream/backends/pi_host_tools.ts')
        expected_paths.add('daydream/backends/pi_diagnostics.ts')
        with zipfile.ZipFile(wheels[0]) as zf:
            names_in_wheel = set(zf.namelist())
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
        # Regression: source-tree resource reads pass while installed execution
        # outside a checkout cannot find its runtime lock or public entrypoint.
        script = (
            'import json,sys; from pathlib import Path; '
            'from daydream.benchmark.reviewbench.harbor_build import reviewer_build; '
            'suffix,assets,identity=reviewer_build(Path(sys.argv[1])); '
            '_,_,arm_identity=reviewer_build(Path(sys.argv[1]),platform="linux/arm64"); '
            'print(json.dumps({"identity":identity,"arm_identity":arm_identity,"assets":list(assets),"suffix":suffix}))'
        )
        result = subprocess.run([sys.executable, '-c', script, str(wheels[0])], cwd=tmp,
                                env={**os.environ, 'PYTHONPATH': str(installed)},
                                check=True, capture_output=True, text=True)
        build = json.loads(result.stdout)
        assert build['identity']['platform'] == 'linux/amd64'
        assert build['arm_identity']['platform'] == 'linux/arm64'
        assert build['identity']['codex_cli'] == '0.163.0'
        assert 'runtime.lock' in build['assets']
        help_result = subprocess.run([sys.executable, '-m', 'daydream', 'benchmark', 'reviewbench',
                                      'harbor-export', '--help'], cwd=tmp,
                                     env={**os.environ, 'PYTHONPATH': str(installed)},
                                     check=True, capture_output=True, text=True)
        assert '--wheel' in help_result.stdout
        assert '--source' in help_result.stdout
        assert '--platform' in help_result.stdout
