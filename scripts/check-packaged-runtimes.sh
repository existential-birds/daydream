#!/usr/bin/env bash
# Install the shipped wheel with its locked dependencies, without the checkout's environment.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

uv build --wheel --out-dir "$work_dir" "$repo_root"
wheel=("$work_dir"/*.whl)
python="$(uv python find 3.14)"
"$python" - "${wheel[0]}" "$repo_root" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

with ZipFile(sys.argv[1]) as archive:
    shipped = set(archive.namelist())
    source = Path(sys.argv[2])
    resources = {path.relative_to(source).as_posix() for pattern in (
        'daydream/templates/workflows/**/*.yml', 'daydream/backends/pi_*.ts',
    ) for path in source.glob(pattern) if path.is_file() and '__pycache__' not in path.parts and not path.name.startswith('.')}
    if missing := sorted(resources - shipped):
        sys.exit(f'wheel is missing packaged resources: {missing}')
PY

uv export --project "$repo_root" --frozen --no-dev --no-emit-project --format requirements-txt \
    --output-file "$work_dir/runtime.lock" > /dev/null
cd "$work_dir"
uv venv --python "$python" runtime
uv pip install --python runtime/bin/python --require-hashes -r runtime.lock
uv pip install --python runtime/bin/python --no-deps "${wheel[0]}"
uv pip check --python runtime/bin/python
runtime/bin/python -I -c 'from daydream.cli import main'
