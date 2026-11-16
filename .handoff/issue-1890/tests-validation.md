# #1890 — tests / fixtures / validation lane (read-only investigation)

Repo `/home/user/daydream`, HEAD `1756782b` ("test(review-completion): route finalization-boundary gh calls through FakeGh (#1801) (#1906)").
No tracked file was edited (`git status --short` empty, `git diff --check` clean after all work). Written only:
gitignored `.venv/` (via `make install`), gitignored `tests/**/__pycache__/` (pytest), and files under
`.scratch/issue-1890/` (gitignored by `.gitignore:72 .scratch/`).

## TL;DR

* Bug **reproduced on current code**, both at unit level (`_supporting_facts`) and end to end through the named
  dirty fixing journey (scratch copy). A supporting new-side ref on a `tracked_worktree_edits` revision is measured
  against HEAD's blob: wrong digest when line counts match, refused when the cited line exists only uncommitted.
* A concrete extension of `test_runner_rejects_missing_canonical_before_fix_or_test` (sketch below) **fails on current
  code at the new assertion and nowhere earlier**, with `[] == [('base.txt','new',{2..2},<reviewed sha256>)]`.
  With a scratch-only simulated repair, the whole extended test passes (later no-execution/verdict assertions intact).
* It needs one small harness extension: `StubBackend._scripted_review` can only cite assigned `source_options`
  today; add an optional per-claim `"supporting"` list appended to `evidence_refs`.
* No existing test pins HEAD-reads *for a tracked-worktree revision*. The one HEAD-read pin,
  `tests/deep_orchestrator/test_review_native_sources.py::test_source_selectors_construct_supporting_evidence_from_the_pinned_git_blob`,
  uses a revision **without** the marker and is the negative control: it stays green under a fix conditional on
  `tracked_worktree_edits`, and fails under an over-broad "read the worktree whenever no `source_id`" fix (verified).
* Test count stays net-neutral (extend the existing journey; no new test function).

## 1. The named test and how it drives the run

File: `tests/test_missing_test_command_acceptance.py` (179 lines). Test: `:59-121`.

**Fixtures / arrangement**
* `linked_worktree` (`tests/conftest.py:119-135`): main repo with `base.txt = "base\n"` committed on `main`
  (`tests/conftest.py:62-68`, `_make_repo_with_main`); branch `feature` adds `services/taste/{parser,lexer,token,ast}.go`;
  a linked worktree checks out `feature`. The test uses the linked worktree as `target` (`:71`).
* `_arrange` (`:39-48`): commits `api.py = "VALUE = 2\n"` and `requirements.txt` on `feature`; builds
  `StubBackend(target)` with `review_claims = [{"title": "Sample issue", "file": "api.py", "line": 1}]`,
  `fix_edit_line` sentinel, and monkeypatches `daydream.runner.create_backend` to return that one instance
  (same object across both loop iterations).
* `.daydream.toml` `test_command = '""'` (empty argv, `:73`) → real TOML loader → readiness rejection
  `empty-argv`/`config`. A fake `pytest` on `PATH` appends to a counter file if ever executed (`:75-79`).
  `owner-untracked.txt` is an untracked file (`:74`).
* `make_config(..., output_mode="loop", assume="yes", test_command=None, file_config=load_file_config(target), trajectory_path=...)` (`:81-82`; fixture `tests/conftest.py:267-281`).

**Driving the run.** Real `daydream.runner.run(config)` (`:90`); only the external backend is replaced. The
StubBackend is *not* a canned-output fake for review: when `host_tools` is passed (`tests/harness/stub_backend.py:166-178`)
it answers each committed-review assignment through the **real host binding** —
`_scripted_review` (`:107-154`) parses the assignment, calls the real `review_source` host tool to fetch targets
(`:134`), builds claims whose `evidence_refs` are assigned selectors `{source_id, basis, source_range}` taken from the
target's `source_options` (`:139-151`), then `host_tools.admit("terminal","structured_output",value,terminal=True)` and
`host_tools.submit(...)` (`:173-174`). Admission goes through the production path
`StageInputFactory.admit_submission` → `_SubmissionSources` (`daydream/review_stage_inputs.py:172-188`) →
`Review._refs` → `sources.canonicalize` (`daydream/committed_review.py:253-257`, `:269-271`) →
`canonicalize_source_selectors` → `_supporting_facts` (`daydream/review_source_refs.py:226-229`). Challenge returns
`review_verdict` ("confirmed") for every claim (`stub_backend.py:128-132`).

**Dirtying `base.txt`.** Loop `for dirty in (False, True)` (`:86`); in the second iteration
`(target / "base.txt").write_text("owner dirty bytes\n")` (`:88`) then `before = _state(target)` (`:89`). In that run
the orchestrator's fixing path takes the live-checkout diff `git_ops.diff(work.repo, work.base_branch, ...)` and sets
`tracked_worktree_edits = git_ops.has_worktree_edits(work.repo)` (`daydream/deep/orchestrator.py:310-314`), passed into
`AnalyzedRevision(..., source_id=None, tracked_worktree_edits=True)` (`orchestrator.py:455-458`). `source_id` is `None`
because `work.captured_source is None` on this path (`orchestrator.py:267`). The reviewed diff has 7 changed files
(4 `.go`, `api.py`, `requirements.txt`, dirty `base.txt`).

**Current assertions.**
* each iteration: `rc != 0` (`:91`); dirty notice printed iff dirty (`:92`); `review-coverage.json`
  `(schema_version, analyzed_revision.tracked_worktree_edits) == (3, True)` dirty / `(1, None)` clean (`:93-95`).
* after loop: owner HEAD/index/tracked+untracked bytes unchanged (`:96`); fake pytest never ran (`:97`);
  `backend.test_suite_calls == 0`, no fix prompts, no heal prompts (`:98-100`); no `test`/`fix` phases in trajectory
  (`:101-104`); `test-verdict.json` is `no-execution`, `empty-argv`/`config`, diagnostics mention `--test-command` and
  `[tool.daydream]`, no repairs (`:105-113`); persisted recipe unresolved with candidate `("pytest",)` (`:114-119`);
  no repair job/checkpoint (`:120-121`).
* Nothing today inspects evidence refs or source digests.

**Observable output for the new assertion.** `.daydream/deep/committed-review.json` (written at
`daydream/deep/review_terminal.py:226`, schema 13 for this run) carries `claims[*].payload.evidence_refs` and
`findings[*].payload.evidence_refs` with the canonical refs, including supporting refs
(`artifact_sha256`, `revision`, `path`, `side`, `source_range`, `basis`, `role`). `merged-items.json` does **not**
carry evidence refs (checked). `deep/` is `rmtree`'d per run (`orchestrator.py:364-367`), so after the loop the file
is the dirty run's.

### Probe of the journey on current code (scratch copy, StubBackend subclass)

`.scratch/issue-1890/test_probe_journey.py`, command:
`PYTHONPATH=/home/user/daydream uv run pytest -p tests.conftest .scratch/issue-1890/test_probe_journey.py -q -s -p no:cacheprovider`
→ exit 0, `2 passed in 1.59s` (probe prints, does not assert). Log: `.scratch/issue-1890/probe-journey.log`.

| variant | dirty `base.txt` | supporting selector | current outcome |
|---|---|---|---|
| same-lines (exact existing dirty bytes) | `owner dirty bytes\n` | new, lines 1..1 | **admitted, mis-measured**: `artifact_sha256 = f34848ca…` = sha256(`base\n`, HEAD) instead of `7148b101…` = sha256(reviewed); `revision` = head_sha; claim and finding present; `test-verdict.json` written |
| added-line | `base\nowner dirty bytes\n` | new, lines 2..2 | **refused**: `HostToolRejected: evidence_refs supporting path.source_range: outside the file's lines … (1 claim: 'base.txt' (new) 1..1)` → "Review failed: no confirmed findings", `claims=0`, `findings=0`, **no `test-verdict.json`** (run stops before readiness) |

Both runs: coverage `(3, True)`, committed result `schema_version` 13.

### Recommended extension (sketch) — validated against current code

Harness change (`tests/harness/stub_backend.py`): document an optional per-claim `"supporting"` list in the
`review_claims` comment (`:31-35`) and append it in `_scripted_review` (`:151`):

```python
"evidence_refs": [{key: option[key] for key in ("source_id", "basis", "source_range")},
                  *claim.get("supporting", ())]}})
```

Test change (`tests/test_missing_test_command_acceptance.py`), `import hashlib`, then:

```python
    for dirty in (False, True):
        if dirty:
            # Line 2 exists only in the tracked uncommitted edit; HEAD's base.txt has one line.
            (target / "base.txt").write_text("base\nowner dirty bytes\n")
            backend.review_claims = [{"title": "Sample issue", "file": "api.py", "line": 1, "supporting": [
                {"path": "base.txt", "side": "new", "source_range": {"start": 2, "end": 2},
                 "basis": "source_inspection"}]}]
            before = _state(target)
        ...existing per-iteration asserts...
    # Before the test-verdict reads: on the defect the review fails and writes no verdict.
    committed = json.loads((deep / "committed-review.json").read_text())
    reviewed = hashlib.sha256((target / "base.txt").read_bytes()).hexdigest()
    assert [(ref["path"], ref["side"], ref["source_range"], ref["artifact_sha256"])
            for finding in committed["findings"] for ref in finding["payload"]["evidence_refs"]
            if ref["role"] == "supporting"] == [("base.txt", "new", {"start": 2, "end": 2}, reviewed)], (
        "a supporting reference into an uncommitted line resolves against the reviewed bytes")
```

Notes on the sketch:
* Set the selector **only in the dirty iteration**; in the clean iteration line 2 does not exist and the clean
  review would be refused (a different, unintended path).
* Write the full claim literal (as above): `review_claims` is `list[dict[str, Any]] | None`, so
  `backend.review_claims[0]` fails mypy without narrowing.
* Placing the assertion after the loop and **before** `test-verdict.json` is read gives the intended-reason failure;
  otherwise pre-fix fails with `FileNotFoundError` on the verdict.
* The added-line variant catches both failure modes (line-count authority and digest). The same-lines variant
  (keep `owner dirty bytes\n`, range 1..1) catches only the digest, with a cleaner admitted-but-wrong diff; either
  fails pre-fix.
* `revision` deliberately not asserted: `validate_committed_result` already forces
  `ref.revision == source_revision(analyzed_revision, side)` (`daydream/committed_review_model.py:256-263`), and the
  value is the implementer's design choice (today head_sha for a marker-only revision).
* The journey cannot tell a *live* worktree read from a *reviewed-snapshot* read: nothing changes `base.txt` between
  planning and submission. If the repair snapshots the reviewed bytes, a stronger proof is to mutate `base.txt` inside
  the scripted review (before submit) and still expect the reviewed digest; that could stay in this same test.

Pre-fix control, actually run: scratch copy of the whole test with this extension (subclass in place of the
harness edit), `.scratch/issue-1890/test_extended_sketch.py`:
`PYTHONPATH=/home/user/daydream uv run pytest -p tests.conftest ".scratch/issue-1890/test_extended_sketch.py::test_runner_rejects_missing_canonical_before_fix_or_test" -q -p no:cacheprovider`
→ **exit 1, `1 failed in 1.94s`**, at the new assertion:

```
E       AssertionError: a supporting reference into an uncommitted line resolves against the reviewed bytes
E       assert [] == [('base.txt',...6a03b140ec8')]
E         Right contains one more item: ('base.txt', 'new', {'start': 2, 'end': 2}, 'dd2c27244f8658daf866e7fc1b9ce38a76f19e28e9098978a40fb6a03b140ec8')
```

All earlier assertions (rc, notice, coverage (3, True)) passed. Log: `.scratch/issue-1890/extended-sketch.log`.

Scratch-only consistency check (NOT proof of a fix): the same file plus a monkeypatched `_source_contents` that reads
`root/path` when `captured is None and side == "new" and revision.get("tracked_worktree_edits")`
(`.scratch/issue-1890/test_extended_sketch_simulated_fix.py`) → **exit 0, `1 passed in 1.98s`**. The later
no-execution verdict/recipe/no-fix assertions stay consistent once the reference is admitted. The real pre/post
proof must use the real repair with the same harness and inputs (test-audit "Bug regressions").

## 2. Existing tests touching source refs / captured source / `source_id` / `tracked_worktree_edits`

Grep (`tests/`, excluding the harness CLI): `review_source_refs`, `canonicalize_source_selectors`,
`validate_source_refs`, `_supporting_facts`, `_SubmissionSources`, `source_revision`, `SourceSelectorRejected`,
`source_inspection`, `tracked_worktree_edits`, `captured_source`, `source_id`.

| Test | What it pins | Effect of #1890 |
|---|---|---|
| `tests/deep_orchestrator/test_review_native_sources.py:58` `test_source_selectors_construct_supporting_evidence_from_the_pinned_git_blob` | **HEAD-read pin**: revision `{head_sha, merge_base_sha}` (no marker); worktree `support.py` rewritten to `MUTABLE_WORKTREE_CONTENT` "must never become the canonical supporting digest or LF range authority" (`:80-100`); `_SubmissionSources` reuse/revocation (`:103-109`); `validate_source_refs` admission rows (`:115-121`); refusal messages (`:125-151`); storage-loss `HostToolFatal` (`:155-175`). | **No change needed if the repair is conditional on `tracked_worktree_edits`.** It is the negative control. Verified by `.scratch/issue-1890/test_guard_overbroad.py` (imports and calls this test under a monkeypatch): conditional simulated repair → pass; unconditional worktree read → **fail** (`SourceSelectorRejected … 'support.py' (new).source_range: allowed 1..1`). Possible extension site for a unit-level row (add the marker to `revision` and expect worktree digest), but the issue mandates the journey, and adding both would be overlap. |
| `tests/test_committed_review.py:712-722` (inside `test_committed_review_completes_through_native_host_tools`-adjacent block ~`:690-730`) | `tracked_worktree_edits=True` → result version 13 / terminal 3; version/marker binding. No supporting refs. | Unaffected. |
| `tests/test_committed_review.py:359-362`, `:1186-1210`, `:1880-1990` | Supporting-selector refusal mechanics (`:(glob)*.py`, out-of-range, missing blob, grouping/echo bounds) on clean committed inputs. | Unaffected (no marker). |
| `tests/test_review_result.py:63-84` | Terminal result rejects `source_id` + `tracked_worktree_edits` together. | Unaffected. |
| `tests/test_findings.py:185-198` | Commit-bound findings artifact refuses `source_id` / `tracked_worktree_edits`. | Unaffected. |
| `tests/test_captured_source.py` (13 tests) | Native captured-source mechanics (`CapturedSource`, `source_id`). | Unaffected (captured path already reads captured bytes; `_source_contents` `:72-75`). |
| `tests/test_host_tools.py` | Only imports `SOURCE_SELECTORS_SCHEMA`. | Unaffected. |
| `tests/test_missing_test_command_acceptance.py:59` | Only tracked-worktree fixing journey in the suite (only `owner dirty bytes` / dirty-notice / `tracked_worktree_edits` test in `tests/`). | The owner test to extend. |

No test needs updating for HEAD-read behavior on a marked revision (none exists). Other `review_claims` users
(`tests/test_integration.py:120`, `test_archive_data_capture.py:102`, `test_cli_main_integration.py:94`,
`test_trajectory_phase_events.py:155`, `test_extension_seam_integration.py:58`,
`deep_orchestrator/test_review_completion.py:177,438`, `deep_orchestrator/test_findings_artifacts.py:91`,
`deep_orchestrator/test_repair_job_acceptance.py:45`, `test_runner.py:552,714`, `test_cli_github_app.py:212`) never
pass a `"supporting"` key, so the optional harness key is backwards compatible. `tests/harness/test_stub_backend.py`
is an empty module (docstring only).

Production callers of the supporting reads (for the implementer's scope check):
`review_source_refs.py:122` (`_SubmissionSources._supporting`), `:229`, `:421`; `committed_review.py:256` (`sources.canonicalize`);
`committed_review_model.py:265` calls `validate_source_refs` **without** `source_root`, so persisted-result validation
never re-reads supporting blobs (only checks `revision`, `:256-263`); `review_result.py:280-284` likewise checks only
`revision`. Reviewer instructions still say supporting selectors resolve "against captured Git blobs"
(`committed_review.py:935-940` (text at `:938`)).

## 3. Environment and runs (actually executed)

| Command | Result | Duration |
|---|---|---|
| `uv lock --check` | exit 0, `Resolved 149 packages in 15ms` | 1.7 s wall |
| `make install` (`uv sync --all-extras`) | exit 0; `Using CPython 3.14.6`, `Creating virtual environment at: .venv`. Only noise: `warning: The UV_NATIVE_TLS environment variable is deprecated …` (environment, harmless; appears on every `uv` call). Log `.scratch/issue-1890/make-install.log` | 4.3 s |
| `uv run pytest "tests/test_missing_test_command_acceptance.py::test_runner_rejects_missing_canonical_before_fix_or_test" -q` | **1 passed in 4.00s** (call 2.51 s, setup 0.27 s; 0.92 s call when warm in the batch below) | 6.2 s wall |
| `uv run pytest tests/test_missing_test_command_acceptance.py tests/deep_orchestrator/test_review_native_sources.py tests/test_committed_review.py tests/test_review_result.py tests/test_findings.py tests/test_captured_source.py tests/test_host_tools.py -q` | **122 passed, 2 warnings in 71.89s** (serial). Slowest: `test_committed_review_completes_through_native_host_tools[claude]` 15.48 s, `[codex]` 14.46 s, `test_bounded_navigation_reaches_late_bytes_and_retains_partial_work` 10.41 s. Log `.scratch/issue-1890/related-tests.log` | 1m13 s wall |
| `uv run pytest <25 files referencing StubBackend/review_claims> -q -n logical` (4 CPUs; list in log) | **118 passed, 9 warnings in 19.86s** — baseline for the harness change. Log `.scratch/issue-1890/stub-consumers.log` | 20.5 s wall |
| `make lint` | exit 0, `All checks passed!` | 0.1 s |
| `make deadcode` | exit 0 (vulture clean) | 3.2 s |
| `make typecheck` | exit 0, `Success: no issues found in 581 source files` | 17.3 s |
| `make check-naming` | exit 0 | <0.1 s |
| `uv run python .scratch/issue-1890/probe_supporting_facts.py` | see below | <2 s |
| journey probe / extended sketch / simulated fix / guard check | see §1 and §2 | ~2 s each |

Unit probe output (`.scratch/issue-1890/probe-supporting-facts.log`), temp repo: `base.txt="base\n"` on main,
feature commit `api.py`, then tracked uncommitted `base.txt = "base\nowner dirty bytes\n"`, revision with
`tracked_worktree_edits: True`, no captured source:

```
has_worktree_edits: True
reviewed diff includes base.txt: True
_supporting_facts(new, base.txt): _SupportingFacts(digest='f34848ca92665c342abd5816c9e3eda0e82180671195362bcd0080544a3bc2ac', lines=1, utf8=True)
  sha256(HEAD bytes)    = f34848ca92665c342abd5816c9e3eda0e82180671195362bcd0080544a3bc2ac
  sha256(reviewed bytes)= dd2c27244f8658daf866e7fc1b9ce38a76f19e28e9098978a40fb6a03b140ec8
  RESOLVES AGAINST: HEAD (bug)
canonicalize (line 2): REFUSED -> evidence_refs[1] supporting path 'base.txt' (new).source_range: allowed 1..1 with start <= end; choose a contained range or remove this optional supporting selector.
```

Not run: `make check`, `make test` (coverage), `make package-check`, `make actionlint`, `make test-scripts`.

## 4. Verified check commands (from `Makefile`, `pyproject.toml`, `CONTRIBUTING.md:71-109`)

Focused (implementation loop):
* `uv lock --check` (`make lockcheck`) — run before any `uv run`/sync (test-audit Validation step 1).
* `uv run pytest tests/test_missing_test_command_acceptance.py tests/deep_orchestrator/test_review_native_sources.py -q`
  (owner + negative control; ~5 s).
* Wider owner/sibling: add `tests/test_committed_review.py tests/test_review_result.py tests/test_findings.py tests/test_captured_source.py tests/test_host_tools.py` (~72 s serial; add `-n logical` to parallelize).
* Harness consumers (StubBackend change): the 25-file list in `.scratch/issue-1890/stub-consumers.log`, `-n logical` (~20 s).
* `make lint` = `uv run ruff check daydream tests scripts/ci_*.py scripts/measure_finalization_memory.py`
* `make typecheck` = `uv run mypy daydream tests scripts/ci_*.py scripts/measure_finalization_memory.py`
* `make deadcode` = `uv run vulture daydream tests`
* `git diff --check`
* Pre-fix control: run the extended test against unrepaired production code in an isolated checkout (test-audit
  "Use isolated checkouts for pre-fix controls"); expected failure is the `[] == [(... reviewed digest)]` above.

Gate: `make check` = `lockcheck install static test coverage-report`; `static` runs
`lint deadcode typecheck package-check actionlint check-naming test-scripts` in parallel; `test` =
`uv run pytest -n logical --cov --cov-branch --cov-report=term-missing --cov-report=xml`. Budget < 2 min
(AGENTS.md "Testing standard"). `make actionlint` may report a Docker-unavailable skip; report it as a skip.

Coverage ratchet (`docs/coverage.md`): targeted pytest runs have no floor; `fail_under` in `pyproject.toml`
`[tool.coverage.report]` is enforced only by `make test`/`make check`. Never lower the floor; raise it only by
measuring the full suite on current main, rounding down to a whole percent, recording SHA/date/environment/raw
percentage in the commit. CI aggregates eight shards and fails on missing/corrupt shard evidence.

## 5. Test-rule fit (AGENTS.md / test-audit authoring gate)

1. Contract: a supporting new-side reference on a reviewed diff that includes tracked uncommitted edits is measured
   against the reviewed bytes (digest and line authority), recorded in the committed result's findings.
2. Credible regression: `_source_contents` falls back to `frozen_source(root, head_sha, path)` for marker-only
   revisions (`review_source_refs.py:76-78`), shown failing both ways above.
3. Owner test: the only tracked-worktree fixing journey; the issue names it; extending it keeps test count neutral and
   adds ~0 s (sketch run 1.94 s total vs 4.00 s first cold run of the original).
4. No production hook needed: the harness drives the real host binding; the new `"supporting"` key is test-support only.
5. Negative control already exists at the owner unit boundary (`test_review_native_sources.py:58`) and needs no edit
   if the repair is conditional on the marker.

## Scratch artifacts

`.scratch/issue-1890/`: `probe_supporting_facts.py`, `test_probe_journey.py`, `test_extended_sketch.py`,
`test_extended_sketch_simulated_fix.py`, `test_guard_overbroad.py`, and logs (`make-*.log`, `probe-*.log`,
`related-tests.log`, `stub-consumers.log`, `extended-sketch*.log`).
