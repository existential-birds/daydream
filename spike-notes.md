# Spike Notes — Per-Phase Terminal Signal Coverage (issue #762)

Re-verified against HEAD (`d31cb99d`). Conclusion: the spec's Key Decision 2 survives
intact — existing phase-event + deep-artifact signals cover every per-phase terminal
state. No spec revision needed.

## Confirmed signals

1. **Merge ran/failed** — via `deep/per-stack-failures.json["__merge__"]`:
   - `MERGE_FAILURE_KEY = "__merge__"` (orchestrator.py:1952).
   - Written **only** on the merge-failure consolidation path (`_salvage_merge_failure`,
     orchestrator.py:2160).
   - `merged-items.json` is written on BOTH the merge-failure path (`_salvage_merge_failure`,
     ~2197) and success paths (`_step_single_stack_merge`, `phase_cross_stack_merge`).
   - **Load-bearing:** merge `succeeded` ⇔ `merged-items.json` present AND `"__merge__"`
     absent from `per-stack-failures.json`. Presence of `merged-items.json` alone is NOT
     sufficient (it is also written on the failure path).
   - `DaydreamPhase` has NO `MERGE` member; `phase_scope(DaydreamPhase...)` call sites cover
     EXPLORATION/INTENT/ALTERNATIVES/DEEP(stage=…)/PARSE/VERIFY/FIX/TEST — no merge. So merge
     terminal state must come from deep artifacts, not phase events.

2. **Fix via `fix-failures.json`:**
   - `fix_failures_p` written only when `all_non_success` (dropped/reverted groups) is
     non-empty; unlinked when clean (orchestrator.py:3248-3262).
   - Present ⇔ fix phase had dropped/reverted groups (partial); absent ⇔ clean or never ran.

3. **Test via `test-verdict.json`:**
   - `test_verdict_path(dd).write_text(...)` occurs BEFORE the failure early-return
     (orchestrator.py:3445), so `{"passed": bool, "retries": int, "ignored": bool}` is written
     for BOTH outcomes.
   - Present ⇔ test phase ran (both outcomes); absent ⇔ test never ran (merge-failure-no-test,
     review-only, improve).

4. **Cancelled via `archive_status == "partial"` && no fix_failures:**
   - `Trajectory.write_partial()` → `on_write(self, "partial")` (trajectory.py:2042), signal flush = cancelled.
   - `_write` → `on_write(self, "complete")` (trajectory.py:1986).
   - `_archive_run_inner` coerces `complete → partial` when `fix_failures` present (archive/__init__.py:152-156).
   - **Confirmed:** `cancelled` ⇔ `archive_status == "partial"` AND no `fix_failures`; `partial` ⇔ `fix_failures` present.

## Flow awareness
`_flow_fix_test_steps(recorder.run_flow, config.flow_name)` (manifest.py:75) returns
`(runs_fix, runs_test)` by runtime mode/registry: `TTT` → (False, False); `PR` → (True, False)
(fix only); NORMAL/DEEP/IMPROVE/CUSTOM classified by registered pipeline. A phase the flow
does not run is `absent` and neutral; only a phase the flow DOES run that is absent forces
`partial` (run stopped early).

These four confirmed signals + the flow-awareness requirement drive `derive_phase_states` /
`derive_pipeline_status` in Task 3 and the `_archive_run_inner` wiring in Task 4.

---

# Issue #779 Task 0 spike — CONFIRMED (schema compatible; loader API recovered)

Date: 2026-08-23
Branch: `eb/daydream/issue-779`
Base: `45c753c`

## Result: CONFIRMED — proceed with implementation

The plan-s8 field set and both job configs validate against the REAL Harbor 0.21 models.
The only discrepancy was the plan's example loader import (`harbor.task.Task`), which does not
exist in Harbor 0.21; the real API is `harbor.models.task.Task`. This is exactly the
spike-recoverable loader-API adjustment the plan's Task 13 note anticipated ("adjust the
imports to the real Harbor 0.21 API the spike confirms"). NOT a schema divergence — do NOT
route back to spec/plan.

## Lock export (confirmed)

```bash
uv export --frozen --no-dev --no-emit-project --format requirements-txt
```
- 397 lines containing `--hash`
- 0 lines beginning `daydream==`
- `httpx==0.28.1`

## OCI base-image digest (resolved)

`python:3.12-slim` → `sha256:876416ecde9aca2bcc90e1fb0c7a9500bbf749f5788b70f82d4c5a5c2357f8b4`
(record as `ENV_BASE_IMAGE` / `VERIFIER_BASE_IMAGE`).

## Harbor 0.21 schema-1.4 acceptance — CONFIRMED (task.toml)

The exact plan-s8 task.toml validates via the real model:
```python
from harbor.models.task.config import TaskConfig
cfg = TaskConfig.model_validate_toml(open("/tmp/s8_task.toml").read())
```
- `schema_version = "1.4"` ✓
- `[metadata]` is a free-form `dict[str, Any]` → accepts `benchmark_case_key` + `source_kind` ✓
- `[agent]` `timeout_sec`, `network_mode="allowlist"`, `allowed_hosts` ✓ (AgentConfig(PhaseNetworkPolicyConfig))
- `[environment]` `network_mode="no-network"`, `workdir`, `build_timeout_sec`, `cpus`, `memory_mb`, `storage_mb` ✓ (EnvironmentConfig(BaselineNetworkPolicyConfig))
- `[verifier]` `timeout_sec`, `environment_mode="separate"` ✓ (VerifierEnvironmentMode.SEPARATE)
- `[verifier.environment]` allowlist/`api.anthropic.com`/build_timeout/cpus=1/memory=2048/storage=4096 ✓

## Harbor 0.21 job-config acceptance — CONFIRMED (harbor-job.yaml / harbor-oracle.yaml)

Both plan-s8 configs validate via the real model:
```python
import yaml
from harbor.models.job.config import JobConfig
cfg = JobConfig.model_validate(yaml.safe_load(text))
```
- `jobs_dir="jobs"`, `n_attempts=1`, `n_concurrent_trials=4` ✓
- `environment: {type: docker, delete: true}` ✓ (EnvironmentType.DOCKER, delete default True)
- `agents: [{import_path: daydream.benchmark.harbor.agent:DaydreamReviewAgent, env: {DAYDREAM_REVIEW_*}}]` ✓ (trial AgentConfig.import_path/.env)
- oracle `agents: [{name: oracle}]` ✓ (AgentConfig.name)
- `datasets: [{path: .}]` ✓ (DatasetConfig.path)
- `metrics: [{type: uv-script, kwargs: {script_path: metric.py}}]` ✓ (MetricConfig.type=uv-script)
- `verifier: {env: {DAYDREAM_JUDGE_*}}` ✓ (VerifierConfig.env)

## RECOVERED Harbor 0.21 API (use these in Task 13/15, NOT the plan's `harbor.task.Task` / `harbor.config.load_config`)

- **Task model:** `from harbor.models.task import Task`. Constructor takes the task **DIRECTORY** (not a task.toml file path): `Task(str(case_dir))`. It reads `task.toml` from the dir via `TaskPaths.config_path` and `TaskConfig.model_validate_toml`. Note: the default constructor runs `_validate_tests` unless `disable_verification=True`; for the construct-only `validate --compiled` proof, pass `disable_verification=True` (or ensure the case `tests/` layout satisfies Harbor's test validation).
- **Job config:** `from harbor.models.job.config import JobConfig`; parse with `JobConfig.model_validate(yaml.safe_load(text))`.
- `MetricType` includes `uv-script`; `EnvironmentType` includes `docker` — both confirmed.

## Verifier Dockerfile / judge-flow

Not completed in the initial spike run (it stopped early on the loader-path false alarm).
The digest-pinned, entrypoint-free verifier `templates/tests/Dockerfile` is rewritten in
Task 7 and its build + judge flow are proven in Task 15's audit execution tests — those cover
the remaining spike step 4 verification.
