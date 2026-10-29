#!/usr/bin/env bash
# Install the shipped wheel and locks without the checkout's environment.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

uv build --wheel --out-dir "$work_dir" "$repo_root"
wheel=("$work_dir"/*.whl)
python="$(uv python find 3.14)"
"$python" - "${wheel[0]}" "$work_dir" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

with ZipFile(sys.argv[1]) as archive:
    for resource, name in (
        ('daydream/benchmark/reviewbench/harbor-runtime.lock', 'host.lock'),
        ('daydream/benchmark/harbor/runtime-requirements.lock', 'reviewer.lock'),
    ):
        Path(sys.argv[2], name).write_bytes(archive.read(resource))
PY

cd "$work_dir"
uv venv --python "$python" host
uv pip install --python host/bin/python -r host.lock "${wheel[0]}"
uv pip check --python host/bin/python
host/bin/python -I -c 'from daydream.benchmark.reviewbench.harbor_cli import verify_runtime; verify_runtime()'

uv venv --python "$python" reviewer
uv pip install --python reviewer/bin/python --require-hashes -r reviewer.lock
uv pip install --python reviewer/bin/python --no-deps "${wheel[0]}"
uv pip check --python reviewer/bin/python
reviewer/bin/python -I -c 'from daydream.benchmark.reviewbench.harbor_entrypoint import main'
