# Reproducing the recorded CI study

The accompanying `compact-evidence.json` declares its publication state. A draft
is incomplete and supports no median acceptance claim. Final mode requires nine
successful, independently verified runs, three per configuration, complete
runner/activity evidence and original artifact ZIPs. Earlier 97a9e0fb validation
and b5642969 failed-study samples are excluded from these nine. Local operation
profiles ran on b5642969; their CI phase evidence came from 97a9e0fb. Relevant
production blob equality with the measured head is recorded separately. Local
profile timings are never acceptance benchmarks.

## Timing and original inputs

At the measured source SHA, use the existing helper:

```bash
python3 scripts/ci_benchmark.py report \
  --manifest PATH_TO_PACKAGE/launch-manifest.json \
  --output PATH_TO_NEW_LOCAL_OUTPUT
```

This re-fetches the exact run identities and rejects changed attempts, source,
workflow or event. Original run/job snapshots are also retained under `raw/` and
hashed in the evidence. The gate is `check.completed_at - run.created_at`.
Runner minutes sum every non-skipped job's start/end interval. Initial queue
starts at run creation; plan-to-shard delay includes scheduling because the API
has no separate ready timestamp. Baseline `Run tests` includes script tests.
Sharded setup is the interval before `Run pytest shard`; artifact/combine/final
gate overhead remains in the full gate metric. Timestamp resolution is seconds.
For each group use the median of its three successful gate times, then reduction
`100 * (1 - median_treatment / median_baseline)`. Observe runner minutes and
max/mean shard imbalance alongside wall time.

Each artifact has an immutable ID, stable API/archive and web URLs, archive SHA256,
size, expiry, capture time and extracted file digests in compact evidence. Fetch
metadata by ID and check its repository/run/head against the recorded attempt's
receipts. Download by immutable ID, rather than substituting a later artifact
with the same name. In final packages `original-artifacts/ID.zip` contains the
original download; its archive digest and every extracted file digest are checked
by export. GitHub redirects to short-lived signed download URLs: retain the API
endpoint, not a signed redirect URL. Artifacts/logs expire; stable links and hashes
cannot reconstruct expired bytes. The final local ZIP preserves the original
artifacts and raw snapshots, but durable storage of that ZIP is the recipient's
responsibility. No external publication is performed.
([GitHub artifact API](https://docs.github.com/en/rest/actions/artifacts))

## Rebuild the inputs from GitHub while retained

Reviewers without the accompanying local ZIP can reconstruct the inputs from
the committed compact evidence and immutable artifact IDs. From this repository
with the measured commit available locally and authenticated `gh`, run the
following with the path to compact evidence and a **new** output directory.
This creates a portable package skeleton; it does not dispatch or rerun CI.
Then execute the independent check below against that directory. If an artifact
has expired or its attempt has been replaced, stop and obtain the retained
original archive rather than substituting another run or artifact.

```bash
python3 - PATH_TO_COMPACT_EVIDENCE.json PATH_TO_NEW_DIRECTORY <<'PY'
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

source, target = Path(sys.argv[1]), Path(sys.argv[2])
e = json.loads(source.read_text())
target.mkdir(parents=True, exist_ok=False)
(target / 'compact-evidence.json').write_bytes(source.read_bytes())
def sha(data):
    return hashlib.sha256(data).hexdigest()
def gh(endpoint):
    return subprocess.check_output(['gh', 'api', endpoint], timeout=60)
def need(value, message):
    if not value:
        raise ValueError(message)
for pin in e['source_pins']:
    data = subprocess.check_output(['git', 'show', f"{e['head']}:{pin['path']}"], timeout=60)
    need(sha(data) == pin['sha256'], 'Pinned source differs')
    p = target / 'measured-source' / pin['path']
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
for r in e['runs']:
    need(r['state'] == 'verified', 'Only complete verified samples can be reconstructed')
    run = json.loads(gh(f"repos/{e['repository']}/actions/runs/{r['id']}"))
    need((run['id'], run['head_sha'], run['run_attempt'], run['event'], run['path'], run['status'], run['conclusion']) ==
         (r['id'], e['head'], r['attempt'], 'workflow_dispatch', '.github/workflows/ci.yml', 'completed', 'success'), 'Run identity/status changed')
    need(run['repository']['full_name'] == e['repository'], 'Wrong repository')
    for a in r['artifacts']:
        endpoint = f"repos/{e['repository']}/actions/artifacts/{a['id']}"
        meta = json.loads(gh(endpoint))
        need(not meta['expired'] and meta['name'] == a['name'] and meta['digest'] == 'sha256:' + a['archive_sha256'], 'Artifact expired/changed')
        need((meta['workflow_run']['id'], meta['workflow_run']['head_sha']) == (r['id'], e['head']), 'Artifact source changed')
        data = gh(endpoint + '/zip')
        need(sha(data) == a['archive_sha256'], 'Archive digest differs')
        p = target / a.get('local_archive', f"original-artifacts/{a['id']}.zip")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        directory = target / 'raw' / 'verification' / 'downloads' / str(r['id']) / a['name']
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                name = PurePosixPath(info.filename)
                need(not name.is_absolute() and '..' not in name.parts, 'Unsafe archive path')
                p = directory / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(archive.read(info))
print('Recorded source and original artifacts reconstructed; now run the identity/floor check.')
PY
```

## Independent exact identity and coverage check

The following standard-library recipe checks the package without the private
collector/verifier or the original `/tmp` directory. Supply the package directory
as the first argument. A draft checks available successful samples and explicitly
reports incompleteness; it never prints final success. Hashes use the original
node strings sorted by Python Unicode ordering, encoded with
`json.dumps(..., ensure_ascii=False)` using default separators, UTF-8 and no final
newline. There is no Unicode/path normalization. Duplicate checks precede hashes.

JUnit mapping splits only the address before the first parameter `[` and retains
the entire parameter suffix, including embedded `::`, dots, slashes, brackets,
quotes and Rust snippets. Ambiguous `(classname, name)` mappings are rejected.
A skip is a setup or call phase with outcome `skipped`, compared with JUnit's
`skipped` child. All exact report/skip identities must match, not only totals.

```bash
python3 - PATH_TO_PACKAGE <<'PY'
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
e = json.loads((root / 'compact-evidence.json').read_text())
def need(value, message):
    if not value:
        raise ValueError(message)
def sha(data):
    return hashlib.sha256(data).hexdigest()
def nodes_hash(nodes):
    return sha(json.dumps(sorted(nodes), ensure_ascii=False).encode('utf-8'))
def junit_key(node):
    address, bracket, parameters = node.partition('[')
    parts = address.split('::')
    parts[0] = parts[0].removesuffix('.py').replace('/', '.')
    parts[-1] += bracket + parameters
    return '.'.join(parts[:-1]), parts[-1]
for pin in e['source_pins']:
    need(sha((root / 'measured-source' / pin['path']).read_bytes()) == pin['sha256'], 'Pinned source differs')
config = tomllib.loads((root / 'measured-source' / 'pyproject.toml').read_text())
need(config['tool']['coverage']['report']['fail_under'] == e['coverage_floor'], 'Configured floor differs')
canonical = expected_skips = None
available = [r for r in e['runs'] if r['state'] == 'verified']
for r in available:
    directory = root / 'raw' / 'verification' / 'downloads' / str(r['id'])
    for a in r['artifacts']:
        for f in a['files']:
            need(sha((directory / a['name'] / f['path']).read_bytes()) == f['sha256'], 'File digest differs')
        if a.get('local_archive'):
            need(sha((root / a['local_archive']).read_bytes()) == a['archive_sha256'], 'ZIP digest differs')
    xml = ET.parse(directory / 'coverage-report' / 'coverage.xml').getroot()
    counts = {k: int(xml.attrib[k]) for k in ('lines-covered', 'lines-valid', 'branches-covered', 'branches-valid')}
    need(0 <= counts['lines-covered'] <= counts['lines-valid'] and counts['lines-valid'] > 0, 'Invalid line counters')
    need(0 <= counts['branches-covered'] <= counts['branches-valid'] and counts['branches-valid'] > 0, 'Invalid branch counters')
    pct = 100 * (counts['lines-covered'] + counts['branches-covered']) / (counts['lines-valid'] + counts['branches-valid'])
    need(pct >= e['coverage_floor'], 'Below unchanged coverage floor')
    need(all(r['verification']['coverage'][k] == v for k, v in counts.items()), 'Coverage receipt differs')
    if r['jobs'] == 1:
        continue
    full = None
    selected = []
    skips = set()
    for i in range(r['jobs']):
        d = directory / f'shard-{i}'
        m = json.loads((d / 'manifest.json').read_text())
        need(m['schema'] == 1 and m['head'] == e['head'] and m['tree'] == e['tree'], 'Wrong source identity')
        need(str(m['run_id']) == str(r['id']) and str(m['run_attempt']) == str(r['attempt']), 'Wrong workflow attempt')
        need((m['index'], m['count'], m['workers']) == (i, r['jobs'], 4), 'Wrong configuration')
        need(m['exitstatus'] == 0 and not m['errors'], 'Failed execution')
        c = m['collection']
        need(c and c == sorted(set(c)) and (full is None or c == full), 'Collection mismatch')
        full = c
        need(m['selected'] == c[i::r['jobs']] and set(m['reports']) == set(m['selected']), 'Selection/report mismatch')
        selected.extend(m['selected'])
        need(sha((d / 'coverage').read_bytes()) == m['coverage_sha256'], 'Raw coverage differs')
        for node, phases in m['reports'].items():
            need({'setup', 'teardown'} <= set(phases) <= {'setup', 'call', 'teardown'}, 'Incomplete phases')
            need(phases['setup']['outcome'] != 'passed' or 'call' in phases, 'Missing call')
            need(all(p['outcome'] in {'passed', 'skipped'} and math.isfinite(p['seconds']) and p['seconds'] >= 0 for p in phases.values()), 'Bad outcome/duration')
            if any(phases.get(p, {}).get('outcome') == 'skipped' for p in ('setup', 'call')):
                skips.add(node)
    need(len(selected) == len(set(selected)) and set(selected) == set(full), 'Not a disjoint complete union')
    need(canonical is None or full == canonical, 'Treatment collection differs')
    need(expected_skips is None or skips == expected_skips, 'Treatment skips differ')
    canonical, expected_skips = full, skips
    need(nodes_hash(full) == r['verification']['collection_sha256'], 'Collection hash differs')
    need(nodes_hash(skips) == r['verification']['skip_sha256'], 'Skip hash differs')
need(canonical is not None, 'No treatment collection available')
lookup = {}
for node in canonical:
    key = junit_key(node)
    need(key not in lookup, 'Ambiguous JUnit mapping')
    lookup[key] = node
for r in available:
    if r['jobs'] != 1:
        continue
    path = root / 'raw' / 'verification' / 'downloads' / str(r['id']) / 'coverage-report' / 'baseline-tests.xml'
    mapped, skips = [], set()
    for case in ET.parse(path).getroot().findall('.//testcase'):
        key = (case.attrib['classname'], case.attrib['name'])
        need(key in lookup and case.find('failure') is None and case.find('error') is None, 'Unexpected/failed baseline case')
        node = lookup[key]
        mapped.append(node)
        if case.find('skipped') is not None:
            skips.add(node)
    need(len(mapped) == len(set(mapped)) and set(mapped) == set(canonical), 'Baseline identity differs')
    need(skips == expected_skips, 'Baseline skips differ')
    need(nodes_hash(mapped) == r['verification']['collection_sha256'] and nodes_hash(skips) == r['verification']['skip_sha256'], 'Baseline hashes differ')
complete = (len(available) == 9 and Counter(r['jobs'] for r in available) == Counter({1: 3, 2: 3, 4: 3}) and e['all_nine_verified'])
need(e['publication_state'] != 'FINAL' or complete, 'Final package incomplete')
print(json.dumps({'identity_and_floor_verified_runs': len(available), 'nodes': len(canonical), 'skips': len(expected_skips), 'all_nine_complete': complete, 'state': e['publication_state']}))
PY
```

The coverage floor comes from the pinned `pyproject.toml`; combined branch
coverage is `(covered lines + covered branches) / (valid lines + valid branches)`.
Neither XML `line-rate` alone nor equality of raw XML bytes establishes it.
Record denominators and investigate differences without discarding useful test
coverage or silently replacing a sample. Complete job success, runner class,
source/dependency pins and repository activity remain separate provenance checks
in compact evidence and original snapshots.
