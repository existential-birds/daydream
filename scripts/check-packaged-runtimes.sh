#!/usr/bin/env bash
# Install the shipped wheel and locks without the checkout's environment.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

uv build --wheel --out-dir "$work_dir" "$repo_root"
wheel=("$work_dir"/*.whl)
python="$(uv python find 3.14)"
"$python" - "${wheel[0]}" "$work_dir" "$repo_root" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

with ZipFile(sys.argv[1]) as archive:
    shipped = set(archive.namelist())
    source = Path(sys.argv[3])
    resources = {path.relative_to(source).as_posix() for pattern in (
        'daydream/templates/workflows/**/*.yml', 'daydream/benchmark/harbor/templates/**/*',
        'daydream/backends/pi_*.ts',
    ) for path in source.glob(pattern) if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.')}
    if missing := sorted(resources - shipped):
        sys.exit(f'wheel is missing packaged resources: {missing}')
    Path(sys.argv[2], 'reviewer.lock').write_bytes(archive.read('daydream/benchmark/harbor/runtime-requirements.lock'))
PY

cd "$work_dir"
uv venv --python "$python" reviewer
uv pip install --python reviewer/bin/python --require-hashes -r reviewer.lock
uv pip install --python reviewer/bin/python --no-deps "${wheel[0]}"
uv pip check --python reviewer/bin/python
reviewer/bin/python -I -c 'from daydream.benchmark.harbor.entrypoint import main'
