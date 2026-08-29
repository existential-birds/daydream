# Private PR Benchmark Runbook (Harbor)

This runbook is the operator's guide to Daydream's **private PR benchmark** — a
GitHub-only, crash-consistent workspace where the repository's own PR-review
cases are imported, curated, built into a Harbor dataset, and scored. It is the
single authoritative reference for the shipped `daydream benchmark` surface.

The workflow is split into seven sections: prerequisites/privacy → initialize/
import/validate → build/run → inspect/objective/aggregate → candidate-profile
trust → upgrade path → failure/cleanup.

The shipped `daydream benchmark` surface is: `daydream benchmark init`,
`daydream benchmark status`, `daydream benchmark validate`,
`daydream benchmark build-harbor`, `daydream benchmark upgrade`,
`daydream benchmark import-prs`, `daydream benchmark curate`,
`daydream benchmark calibrate-judge`, `daydream benchmark run`,
`daydream benchmark clean`, `daydream benchmark objective`, and
`daydream benchmark aggregate`.

## 1. Prerequisites and privacy boundary

Before anything else, confirm the environment is exactly what the benchmark
expects — and understand the data/egress boundary that keeps private source
material in your control.

### Prerequisites

- **Harbor** `0.22` in range `[0.22, 0.23)` installed in the **same Python
  environment** as Daydream (`uv sync` installs it alongside the `daydream`
  console script).
- **Docker Desktop for macOS** in its standard, local configuration. The first
  build/run performs a live `nftables` capability preflight against the local
  engine — the run itself validates the Docker + Harbor integration, there is no
  fake end-to-end fixture.
- **Explicitly not supported** and refused/unsupported by the runbook: Docker
  Desktop *Business* air-gapped containers, OrbStack, any proxy or custom
  network bridge, Linux/cloud Docker hosts, or a public-network fallback. The
  workspace targets a locked-down local engine and these setups will not behave
  correctly.

### Reviewer / judge boundary

- **Pi** is the reviewer backend. **OpenRouter** is the provider for **both**
  the reviewer and the judge.
- Both runtime phases are restricted to the single egress host `openrouter.ai`
  (see `--reviewer-host` / `--judge-host` below).
- On `init`, the repository host allowlists (`--reviewer-host`, `--judge-host`)
  must be **nonempty** — the workspace refuses a review/judge edge with no
  allowlisted host. Host values are normalized (lowercase, scheme/credential/
  port/path stripped).

### Data and egress inventory

- The base container runs in a **`no-network`** environment; no traffic leaves
  the job except what an explicit allowlist opens.
- The **reviewer allowlist** admits `openrouter.ai` for the reviewer; a
  **separate verifier allowlist** admits only the verifier's own required egress
  — the two edges are scoped independently.
- Each imported case is **frozen** into a deterministic, offline-replayable
  source snapshot before anything reviews it.
- Gold reviews are **hidden** from the reviewer; a run passes the **Oracle**
  self-match/reward gate before its findings are trusted (see
  [section 3](#3-build-and-run)).
- **No-publication boundary:** the runbook never publishes private profile
  content, source, gold, candidate findings, credentials, or judge reasoning.

### Credentials

- A **minimal GitHub token** (no unnecessary scopes) is read from the
  environment for `import-prs`; API keys for the reviewer/judge OpenRouter route
  are read from the environment as well.
- Credentials are **never written into the workspace or the compiled task** —
  they stay in the environment that launched the command.

### Privacy rule

Every example in this runbook uses placeholders only: `OWNER/REPO`, `<40-hex>`,
`<run-id>`, `<case-id>`. There are no real repository names, PR numbers,
credentials, source paths, gold text, candidate findings, or judge reasoning
anywhere in the examples. If you see a real value in a command you are about to
run, you have copied it from a place that should never have contained it.

## 2. Initialize / import / validate

All commands operate on a workspace directory (below, `~/bench-owner-repo`).

### `init` — create the private workspace

```bash
daydream benchmark init ~/bench-owner-repo --repo OWNER/REPO \
  --reviewer-host openrouter.ai --judge-host openrouter.ai
```

`init` refuses a nonempty target directory and persists an immutable
forge-identity block rather than a bare `OWNER/REPO`. Both `--reviewer-host` and
`--judge-host` are repeatable; each must be nonempty. The created workspace is
`0700`-rooted with `imports/ cases/ snapshots/ transactions/ runtime/ cache/
harbor/` and a self-ignoring `.gitignore`.

### `status` — read-only derived state

```bash
daydream benchmark status ~/bench-owner-repo
```

Reports the derived workspace state (`empty` / `collecting` / `ready` / …),
whether the repository identity is **unresolved**, the PR ledger, and per-indexed
case snapshot state (`ready` / `unreplayable` / `imported`) with the frozen head
prefix each case was caught at. Safe to run concurrently with other read-only
commands.

### `validate` — 0/2/1 exit codes

```bash
daydream benchmark validate ~/bench-owner-repo
```

Returns a numeric **exit** code: `0` ready, `2` structurally valid but
incomplete (for example an unresolved repository identity on a fresh workspace),
`1` corrupt (invalid/missing `benchmark.yaml`, an orphan or missing indexed file,
or a checksum-mismatched ready-snapshot bundle).

### `import-prs` — explicit private evidence

```bash
daydream benchmark import-prs ~/bench-owner-repo --pr 123
daydream benchmark import-prs ~/bench-owner-repo --pr-file prs.txt
daydream benchmark import-prs ~/bench-owner-repo --head PR=<40-hex>
daydream benchmark import-prs ~/bench-owner-repo --pr 456 --head PR=<40-hex> --refresh
```

- `--pr` accepts a PR number (or a GitHub pull URL, `https://github.com/OWNER/REPO/pull/N`);
  repeatable.
- `--pr-file` accepts a file listing PR numbers/URLs, one per line; repeatable.
- `--head PR=<40-hex>` ties an explicit head SHA to its PR (a bare `<40-hex>` is
  accepted for back-compat and treated as the sole requested PR).
- `--refresh` re-fetches already-imported PRs, marking stale cases without
  overwriting curation.

Each requested head is frozen into a `base → head` snapshot bundle under
`snapshots/`; the case is written with a `ready|unreplayable` snapshot instead of
`imported`. A classified git failure yields a schema-valid `unreplayable` case,
never a silent failure. A six-step preflight (binaries, `gh` auth, authenticated
user, repository identity + read access, credentialed `git ls-remote`, summary
print) runs before any fetch; the first successful repository resolution fills
`repository_id`/`visibility` atomically and immutably. A failed fetch leaves no
import file and marks the PR `fetch_failed` in the resumable ledger. The command
never selects gold and never filters bot authors — bot classification is retained
as metadata only.

### `curate` — golden review

```bash
daydream benchmark curate ~/bench-owner-repo --case <case-id>
daydream benchmark curate ~/bench-owner-repo --case <case-id> --apply-gold gold.yaml
```

`curate` edits a case's golden review; `--apply-gold` applies a reviewed gold YAML
draft (deriving all forbidden fields, never flipping a case to `ready` by itself).

All workspace writes are atomic and journaled (`prepared | committing |
complete`) under the workspace lock, so a crash mid-mutation restores either the
whole before- or after-state — never a checksum-drifted partial.

### Authoring anchors and exact acceptance

Exact single-keystroke acceptance of an inline review candidate is judged
solely from a strict, versioned **authoring anchor** on the evidence record —
the authoring commit/path/line range as it existed when the comment was
written — never from GitHub's re-anchored `commit_id`/`path`/`start_line`
fields.

- Anchors are derived **once, during case materialization**, from the
  authenticated mirror, immediately before candidate projection, and only for
  root inline comments (replies are evidence, never candidates). An
  `imported`-status snapshot — no freeze, no mirror — and any pre-anchor
  import leave the anchor unset, and exact acceptance fails closed.
- Derivation is fail-closed. An anchor is `derived` (carrying the authoring
  commit, the mirror-traced authoring path — the observed path when it exists
  in the authoring tree, else its unique rename between the authoring commit
  and the mapped head — and the authoring line range) or carries exactly one
  fixed closed status with all data unset: `history-unavailable` (no authoring
  commit or a failed rename trace), `path-unavailable` (no unique authoring
  path), or `range-unavailable` (no authoring line range). A guessed or
  ambiguous path never becomes an anchor.
- Inline exact acceptance requires a `derived` anchor whose `commit_id` equals
  the snapshot's selected original head SHA, a usable authoring location, a
  projectable title, and a record that is neither `outdated` nor `dismissed`.
  Everything else is edit-required with a fixed not-exact reason — including
  `re-anchored`, when the anchor derives on a commit other than the selected
  head. Review bodies are file-agnostic and keep their single submission
  `commit_id` gate.
- `curate` shows the authoring commit prefix (`auth:`) beside the observed
  re-anchored `commit_id` (`commit:`) and the fixed not-exact reason whenever
  exact acceptance is unavailable, so you can see at a glance whether a record
  projects exactly or needs manual edit. `import-prs --refresh` recomputes
  anchors only for records that are missing one or genuinely changed;
  curation state is preserved.

## 3. Build and run

### `build-harbor` — package the workspace for Harbor

```bash
daydream benchmark build-harbor ~/bench-owner-repo --daydream-wheel dist/daydream.whl
```

Packages a validated workspace for Harbor `0.22`. `--daydream-wheel` names the
wheel for this Daydream version; the emitted `harbor/benchmark.lock.json`
`daydream` block records its version and SHA-256.

### `upgrade` — repair legacy case documents

```bash
daydream benchmark upgrade ~/bench-owner-repo --dry-run
daydream benchmark upgrade ~/bench-owner-repo
```

Deterministically upgrades pre-provenance-split case documents (`finding_id` +
`schema_version`). `--dry-run` reports without writing. Full detail in
[section 6](#6-upgrade-path).

### `run` — supervised Harbor run

```bash
daydream benchmark run ~/bench-owner-repo --yes
daydream benchmark run ~/bench-owner-repo --oracle --yes
```

`run <dir>` supervises a Harbor run behind the **Oracle self-match / reward**
gate; `--oracle` runs the Oracle self-match pass. `--yes` confirms the paid run
without prompting.

The run itself validates the Docker and Harbor integration — it exercises the
real engine, not a fake end-to-end fixture.

### `calibrate-judge` — optional diagnostic

```bash
daydream benchmark calibrate-judge ~/bench-owner-repo --yes
```

`calibrate-judge` is an optional, **diagnostic**-only pass that measures the
configured judge's agreement with an **unverified** fixture. It is not an
authorization, correctness, or validation check, and it carries no operational
authority.

### `clean` — disposable artifacts

```bash
daydream benchmark clean ~/bench-owner-repo --derived
daydream benchmark clean ~/bench-owner-repo --all --yes
```

`clean <dir>` removes ledger-derived disposable artifacts. `--cache` removes the
disposable clone + build stage under `cache/`; `--jobs` removes ledgered Harbor
job dirs and their recorded Docker images; `--trajectories` removes contained
`agent/trajectory.json` files in ledgered job dirs; `--derived` is the union of
`--cache --jobs --trajectories` (preserving curated source/gold); `--all` deletes
every deletable artifact including curated source/gold (needs `--yes`).

### Paid maintainer gates

`calibrate-judge` and `run --oracle` are **paid maintainer gates** — they spend
on hosted model calls (72-call calibration, the Oracle self-match pass). State it
plainly: **these paid maintainer gates are never executed in CI.**

### OpenRouter data handling

Live private-source runs that route through **OpenRouter** send private source
data to a third-party provider. Before running such a run you must explicitly
accept that provider's **data-handling and retention policy** — especially for
its free endpoints, which may retain or further process inputs. Never put API
keys or source findings in docs or receipts. This `data handling` contract is
part of operating the benchmark at all; it is separate from the diagnostic
vs. Oracle distinction above, and it never becomes a skip-able step.

### Diagnostics vs the Oracle gate

Keep three concepts distinct:

1. **Optional unverified judge-agreement diagnostics** — `calibrate-judge`
   reports how closely the judge agrees with the unverified fixture. It is
   optional, diagnostic-only, and never a pass/fail prerequisite; no calibration
   fixture or receipt gates any run.
2. **The Oracle self-match / reward gate.** The supervised run uses `--oracle`
   and gates on the reviewer achieving the Oracle's **self-match / reward** result
   — a verified, operational state, not a calibration fixture.
3. **The normal benchmark-after-Oracle gate.** Once an Oracle receipt exists, a
   normal (non-`--oracle`) run is gated on that receipt being current for the
   workspace's compiled lock state (an allowlist change invalidates it).

## 4. Inspect / objective / aggregate

`objective` and `aggregate` are **read-only** commands. They resolve and project
**completed Harbor results** that already exist in the ledger. They **do not run
Harbor**, they **do not call a judge**, and they never generate candidates,
select an objective, or implement hill-climbing.

### `objective` — one exact run as JSON

```bash
daydream benchmark objective ~/bench-owner-repo --run-id <run-id> --json -
```

Resolves an exact ledgered run by explicit `<run-id>`, requiring a terminal
`complete` state, and writes strict machine-readable JSON. `--json PATH|-` writes
the JSON to a path or `-` for stdout (`--json` omitted prints a human summary).

The output is opaque and privacy-safe: only opaque run/benchmark ids and counts
pass through. No repository slug, PR number, source path, gold/candidate text,
judge reasoning, or source code is ever emitted.

```json
{
  "run_id": "run-8f3a1c2e",
  "mode": "benchmark",
  "schema_version": 1,
  "identity": {
    "objective_schema_version": 1,
    "profile_schema_version": 1,
    "profile_name": "openrouter-pi-next",
    "profile_digest": "sha256:<40-hex>",
    "daydream_version": "0.12.0",
    "daydream_wheel_sha256": "sha256:<40-hex>",
    "compiled_lock_sha256": "sha256:<40-hex>",
    "harbor_version": "0.22",
    "reviewer_backend": "pi",
    "reviewer_model": "z-ai/glm-5.2",
    "reviewer_base_url": "",
    "reviewer_effort": null,
    "judge_provider": "openrouter",
    "judge_model": "openrouter/auto",
    "judge_host": "openrouter.ai",
    "verifier_template_sha256": "sha256:<40-hex>",
    "threshold": 0.7,
    "attempts": 1
  },
  "objective": {
    "tp": 18,
    "fp": 7,
    "fn": 4,
    "precision": 0.72,
    "recall": 0.818,
    "f1": 0.767,
    "clean_task_count": 2,
    "clean_pass_count": 1,
    "clean_accuracy": 0.5,
    "task_count": 8,
    "scored_task_count": 8,
    "candidate_count": 25,
    "gold_count": 22,
    "infra_error_task_count": 0,
    "verifier_error_task_count": 0,
    "malformed_task_count": 0,
    "failed_task_count": 0,
    "comparison_eligible": true,
    "mean_task_score": 0.75,
    "tokens": 81234.5,
    "cost": 0.62
  }
}
```

The top-level keys are exactly `run_id`, `mode`, `schema_version`, `identity`,
`objective`. `schema_version` is `1`; `mode` is `oracle` or `benchmark`. The
`objective` dict carries the count-derived micro-metrics plus counts
(`comparison_eligible`, `task_count`, `candidate_count`, `gold_count`), the
reported location/severity axes (below), with optional `tokens`/`cost`.

### Reported location and severity axes

Each scored task's per-trial reward JSON (verifier template version 4) carries
the content reward plus two **reported** diagnostic axes over the matched
(tp) pairs — location and severity agreement. A per-task reward row looks
like:

```json
{
  "tp": 5,
  "fp": 1,
  "fn": 2,
  "reward": 0.77,
  "location_exact": 3,
  "location_near": 1,
  "location_file": 1,
  "location_miss": 0,
  "location_credit": 0.8,
  "location_present": 1,
  "severity_exact": 4,
  "severity_within_1": 1,
  "severity_mean_distance": 0.2,
  "severity_credit": 0.9,
  "severity_pairs": 5,
  "severity_present": 1,
  "verifier_error": 0
}
```

**The axes are reported-only.** They are computed over matched pairs and
reported for diagnostics; they never gate tp/fp/fn, the reward, or the
self-match tiebreak. A pair with no location (locationless findings) or no
severity contributes to no count, mean, or credit — it is never imputed as a
zero (axis-presence doctrine); `location_present`/`severity_present` are 0/1
flags reporting whether at least one pair scored that axis. Location tiers are
`exact` (same file, distance 0 — either candidate endpoint lies inside or on
the gold's inclusive range, so an overlapping range that is not identical
still scores exact), `near` (within `LOCATION_TOLERANCE` = 3 lines),
`file` (same file, different location), and `miss` (different file); severity
is scored as exact, within-1 severity step, and mean ordinal distance with
credit 1.0/0.5/0.0 for distance 0/1/2+.

`aggregate` pools the axes across the suite over axis-present tasks only,
alongside the pooled TP/FP/FN micro metrics:

```json
{
  "location_exact": 11,
  "location_exact_rate": 0.65,
  "location_near_rate": 0.18,
  "location_miss_rate": 0.0,
  "location_pairs_scored": 17,
  "severity_pairs_scored": 17,
  "severity_within_1": 16,
  "severity_credit": 0.88
}
```

Axis rates use the axis pair count as the denominator and are 0.0 when no
pairs scored that axis — an absent axis is missing signal, not a perfect one.

### `aggregate` — pool a suite manifest

```bash
daydream benchmark aggregate suite.json --json -
```

`aggregate <manifest> [--json PATH|-]` pools a suite manifest of exact,
already-completed runs into one compatible objective JSON. A suite *manifest* is
a small JSON file naming the workspaces/runs to pool:

```json
{
  "schema_version": 1,
  "entries": [
    { "workspace": "/workspaces/repo-alpha", "run_id": "run-11aa22bb" },
    { "workspace": "/workspaces/repo-beta",  "run_id": "run-33cc44dd" }
  ]
}
```

Each entry is `{workspace, run_id}`. The suite validates the manifest, resolves
every entry fail-closed (any missing, incomplete, malformed, comparison-ineligible,
or duplicated entry fails the whole command — never a silently-subsetted pool),
and requires the full compatibility identity to match across every entry.

The suite **pools** TP/FP/FN across repositories and derives **micro**
precision/recall/F1 from the flattened per-task rows. Per-repository F1 is
**never averaged** — the pooled counts are the only number that means anything
across repositories. The aggregate *output* is keyed by `experiment_id`,
`profile_digest`, `identity`, and `objective` (the `entries`/`schema_version`
belong to the manifest *input*, not the output). A stable `experiment_id` is a
SHA-256 over the canonicalized manifest plus the shared identity.

Both commands write through an atomic write; on an expected `ObjectiveError` they
print to stderr and exit `1` without touching an existing output file.

## 5. Candidate-profile trust boundary

A benchmark accepts only an **explicit validated candidate profile** from the
trusted **control plane** — or the **packaged default**. Harbor ignores ambient
user environment profiles and every target-repository `.daydream.toml` /
profile; target source, gold, and verifier content cannot mutate the candidate.

Attribution records four fields for every emitted objective: the `schema version`
(`objective_schema_version` / `profile_schema_version`), the `profile name`, the
`source kind` (control plane vs packaged default), and the canonical `digest`
(`profile_digest`, plus the wheel/compiled-lock digests in the identity block).
The profile is model/backend/effort/schemas/privacy-controls/verifier-threshold/
matching/gold/scoring-independent — everything that is *not* the candidate
profile stays outside it.

Profiles are a **seam for a future optimizer**, not an implemented hill-climber.
The runbook never publishes private profile content, and a digest is a
commitment to content — it does **not** reveal the content it fingerprints.

## 6. Upgrade path

Pre-provenance-split workspaces need a one-time repair before they can build and
run. `daydream benchmark upgrade <dir>` (and `--dry-run`) is that repair path:

- It backfills `requested_base_sha` from the recorded `original_base_sha` on
  every `ready`/`imported` snapshot — **v1 and already-v2 alike**.
- It re-derives the v1→v2 **case-scoped `finding_id`** for the same snapshots.
- **Authored content is left untouched** — curation, gold, and any author edits
  are preserved byte-for-byte.
- `unreplayable` cases are left byte-unchanged (they have no replayable base to
  provenance-split).
- A second run is a **no-op** (idempotent); `--dry-run` reports the upgrade that
  *would* be written without writing.
- An **un-upgraded** such workspace fails `CaseDocument` validation and reports
  as **corrupt**.

## 7. Failure and cleanup states

### Failure / recovery states

- **Unreplayable case.** A case whose head could not be frozen into a replayable
  snapshot. It stays `unreplayable` (valid, never silently accepted); `upgrade`
  leaves it byte-unchanged and `curate` cannot flip it to `ready`.
- **Stale case.** An already-imported case whose snapshot is stale; `--refresh`
  re-fetches and marks it without overwriting curation.
- **Cleanup-pending.** A run that finished but whose disposable artifacts have
  not yet been collected; `objective` refuses a non-`complete` (cleanup-pending)
  run, and `clean --jobs`/`--derived` collects the artifacts.
- **Failed verifier.** A run whose verifier errored on one or more trials;
  `comparison_eligible` becomes `false` and a suite refuses to pool it.
- **Interrupted run.** A crash mid-run restores either the whole before- or
  after-state via the atomic journal (`prepared | committing | complete`); the
  ledger stays consistent and the run can be resumed or cleaned.

### Cleanup behavior

| `clean` scope | Removes |
|---|---|
| `--cache` | disposable clone + build stage under `cache/` |
| `--jobs` | ledgered Harbor job dirs + recorded Docker images |
| `--trajectories` | contained `agent/trajectory.json` files in ledgered job dirs |
| `--derived` | union of `--cache --jobs --trajectories` (preserves curated source/gold) |
| `--all` | every deletable artifact including curated source/gold (needs `--yes`) |

Run these scopes to reclaim space or to invalidate a stale compiled lock
deliberately; the Oracle/receipt gate then requires a fresh compile, which is the
intended recovery path for an allowlist or profile change.