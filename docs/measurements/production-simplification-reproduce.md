# Reproducing PR #1444 measurements

Use Python 3.12 and `uvx`. Run from the repository root. This materializes the
exact owned Python inventory at a Git revision, including runtime templates and
all added production modules. The baseline revision is fixed; use the PR's latest
commit for the second invocation. Measurements of a dirty checkout are provisional.

```bash
uv run --no-project --python 3.12 python - 58cc88e4d7091b27afe1b2a6eec486504e255dfd /tmp/daydream-baseline <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

revision, destination = sys.argv[1:]
snapshot = Path(destination)
snapshot.mkdir(parents=True, exist_ok=False)
inventory = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", revision], text=True
).splitlines()
paths = sorted(
    path for path in inventory
    if path.endswith(".py")
    and path.startswith(("daydream/", "scripts/", "rl/daydream_review/"))
    and not path.startswith(("daydream/atif/", "rl/daydream_review/tests/"))
)
physical = 0
for path in paths:
    content = subprocess.check_output(["git", "show", f"{revision}:{path}"])
    output = snapshot / path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    physical += len(content.decode("utf-8").splitlines())
inventory_bytes = ("\n".join(paths) + "\n").encode("utf-8")
print(json.dumps({
    "revision": revision,
    "files": len(paths),
    "physical_loc": physical,
    "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
}, indent=2))
PY
uvx --python 3.12 scb-check==0.2.0 check /tmp/daydream-baseline --report --include-all > /tmp/daydream-baseline-report.json
```

Repeat the Python block with the PR commit and a fresh destination such as
`/tmp/daydream-after`, then run the same pinned `scb-check` command on that directory.
The report's `total_loc` is source LOC, `files_scanned` must match the inventory,
and `clone_loc`, `high_cc_functions`, and `high_cog_functions` are reported unchanged
by this methodology. `scb-check` can return 1 when it reports complexity findings;
that does not invalidate an otherwise complete JSON report. A crash or incomplete
report is a failed measurement.

The physical reduction is `100 * (113364 - physical_loc) / 113364`; the source
reduction is `100 * (85527 - total_loc) / 85527`. Both must reach 10%. The integer
ceilings are 102,027 physical lines and 76,974 source lines. Comments, docstrings,
blank lines, documentation, and tests cannot satisfy the source target. Inventory
changes must be inspected, and all added production Python must remain included.
