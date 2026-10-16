# Deterministic quality analysis (#1402)

Measured on macOS ARM64 with CPython 3.14.3, against an immutable `git archive`
of `724e771791cdd775854f54be72016a4af4cce89a`. The same 759 eligible Python
files parsed successfully for both implementations. Inputs never included the
implementation changes or their new tests. No coverage or profiler was enabled
for the elapsed/CPU comparison. Each scenario had one warm-up and five serial
samples, with no concurrent heavy workload. Raw samples, canonical hashes and
separate profile counts are in [quality-analysis.json](quality-analysis.json).

| Input | Baseline median elapsed / CPU | Optimized median elapsed / CPU |
| --- | --- | --- |
| Whole workspace | 8.510787 / 8.510031 s | 5.750909 / 5.750059 s |
| Candidate `daydream/eval/quality.py`, all peers indexed | 1.515045 / 1.514903 s | 1.512165 / 1.512031 s |
| Explicit empty candidates | 0.000162 / 0.000163 s | 0.000170 / 0.000171 s |
| One eligible file plus excluded dependency tree | 0.079612 / 0.079528 s | 0.000290 / 0.000290 s |

Whole-workspace elapsed/CPU work decreased 32.4%. Candidate time is essentially
unchanged: parsing every peer and clone indexing still dominate that mode. Empty
scope remains an enumeration-free shortcut; these tiny timing differences are
noise. The dependency fixture has 1,000 `node_modules/package-N/nested`
directories containing five Python files each, plus an eligible three-line
function. All four canonical JSON outputs match exactly, including ordering,
scope counts and calibration. No timing fields were removed from quality output.

Separate cProfile runs observed 1,518 → 759 `read_bytes` calls and
21,625,837 → 4,528,694 `_iter_function` generator calls/resumptions over the
repository. The 28,904 per-assignment `_count_later_references` calls disappeared.
The excluded fixture's native `scandir` calls fell from 4,004 to one.
Clone analysis was profiled before considering further changes: its 760
`_repeated_block_rows` calls used 3.455 s cumulative baseline and 3.473 s after.
Its implementation remains unchanged. Profiled totals (20.932 / 13.668 s) are
diagnostics, not the speedup estimate; nested cumulative times overlap.

## Reproduction

Export the same input once, and save the old module outside either input tree:

```sh
mkdir -p /tmp/quality-input
git archive 724e771791cdd775854f54be72016a4af4cce89a | tar -x -C /tmp/quality-input
git show 724e771791cdd775854f54be72016a4af4cce89a:daydream/eval/quality.py > /tmp/quality-before.py
```

From the implementation checkout, run the following with `uv run python` after
`uv lock --check`. Use the exported input for both modules; repeat with
`candidate_paths={"daydream/eval/quality.py"}` and `candidate_paths=set()`, and
with the dependency fixture described above. Run profiles separately with
`cProfile.Profile().runcall(invoke)` and inspect `pstats.Stats`.

```python
import importlib.util
import json
import statistics
import time
from pathlib import Path

outputs = []
for module_path in ("/tmp/quality-before.py", "daydream/eval/quality.py"):
    spec = importlib.util.spec_from_file_location("measured_quality", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    invoke = lambda: module.analyze_quality(Path("/tmp/quality-input/.daydream"))
    canonical = json.dumps(invoke(), sort_keys=True, separators=(",", ":"))
    samples = []
    for _ in range(5):
        cpu, elapsed = time.process_time(), time.perf_counter()
        result = invoke()
        samples.append((time.perf_counter() - elapsed, time.process_time() - cpu))
        assert json.dumps(result, sort_keys=True, separators=(",", ":")) == canonical
    print(samples, tuple(statistics.median(row[i] for row in samples) for i in (0, 1)))
    outputs.append(canonical)
assert outputs[0] == outputs[1]
```

Finalization trace scopes are documented in [observability](../observability.md#trajectory-timing).
They measure the actual worker-thread tail separately from model phases while
evaluation and manifest lifecycle timing continue to use frozen trajectory bytes.
