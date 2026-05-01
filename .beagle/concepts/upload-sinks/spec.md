# Spec: Cloud Upload Sinks for Daydream Run Archives

**Slug:** `upload-sinks`
**Status:** Draft

## Core Value

Daydream gains a pluggable upload pipeline so a run's full archive bundle can be shipped to a configured remote sink — disabled by default, with one reference adapter, ready for an org to flip on once they decide where their trajectories should go.

## Problem Statement

Today every daydream run produces a complete artifact bundle (ATIF v1.6 trajectory, review output, manifest, diff, optional deep artifacts) at `~/.daydream/archive/runs/{session_id}/`. These bundles never leave the laptop or CI runner that produced them. Adopters who want to centralize agent runs across their team — for ML research, fine-tune dataset construction, or eventual debug/observability work — have no in-product way to do it. They have to roll their own out-of-band sync.

The proximate user is whoever stands up the centralized sink. The downstream user — the one whose work the sink serves — is the ML researcher pulling trajectories into a notebook for analysis or fine-tune dataset curation.

Why now: ATIF v1.6 trajectories and the local archive landed recently. The data is finally structured enough to be useful; it just has nowhere to go. We want the upload plumbing in place *before* there's a hosted sink so adopters can configure-and-go without waiting on us, and so the next sink (a strictly-OSI option) is a copy-paste-and-modify rather than a refactor.

## Requirements

### Must have

- **Pluggable `Sink` adapter contract.** Each implementation registers a name, accepts a full archive bundle (a read-only view over the archive run directory plus the parsed manifest), returns a structured push result.
- **Sink contract includes a `healthcheck`** that callers (especially `daydream sync`) invoke before walking the archive, so misconfiguration fails fast.
- **Phoenix reference adapter** that uploads the ATIF trajectory via Arize's `arize-phoenix-client.upload_atif_trajectories_as_spans` (idempotent via SHA-256 of session_id, ATIF v1.0–v1.6 supported).
- **No-op test stub adapter** that records pushes in memory, used in the test suite and available as a `--dry-run`-style sink.
- **Auto-push trigger on run completion.** When a sink is configured, `archive_run` invokes it after the local archive is written. Upload failures are non-fatal warnings — they never affect the primary run.
- **`daydream sync` command** that walks `~/.daydream/archive/`, filters by manifest fields (at minimum `--since=…` and `--repo=…`), and pushes any run not yet confirmed-uploaded by the configured sink. Recovers from outages, ephemeral CI failures, sink switching, and onboarding pre-existing archive contents.
- **Per-sink upload status persisted in the existing SQLite index**, keyed by `(session_id, sink_name)`. No double-pushes; failed pushes can be retried; remote IDs are recorded for downstream tracing.
- **Disabled by default.** No sink configured ⇒ zero behavior change vs. today. No new env vars read, no config file required, no network calls.
- **Configuration via environment variables.** Sink selection and connection settings (e.g., `DAYDREAM_SINK=phoenix`, `DAYDREAM_PHOENIX_URL`, `DAYDREAM_PHOENIX_API_KEY`) are read at startup. No config file at v1.
- **Secrets never written to the archive, manifest, or upload payload.** Existing `Redactor` in `daydream/trajectory.py` continues to run at trajectory-write time; sink configuration secrets live only in adopter env.
- **Test coverage** that exercises (a) the no-op stub end-to-end through `archive_run`, (b) an in-memory Phoenix mock proving the trajectory makes the round trip, (c) a "sink raises" path that proves the run still completes and the archive is intact, (d) `daydream sync` skipping already-uploaded runs based on the SQLite state, (e) `daydream sync` resuming after a partial run.

### Should have

- **`daydream sync --dry-run`** that lists what would be pushed without sending.
- **`daydream sync --sink=<name>`** so adopters with multiple configured sinks can target a specific one.
- **`daydream sync --rm-after-upload`** that removes a bundle's on-disk contents (trajectory.json, review-output.md, deep/, diff.patch) only after the SQLite index records a confirmed successful upload to the configured sink. The SQLite manifest row is preserved so the system retains a record that the run existed and was uploaded. Intended for laptops where archive disk pressure becomes a concern; explicit opt-in per invocation.
- **End-of-run UI surfacing** for auto-push failures (warning line in the existing summary panel) and a one-line summary at the end of `daydream sync` (`X uploaded, Y skipped, Z failed`).

### Out of scope (with reasons)

- **Standing up Phoenix (or any other sink) operationally.** Reason: explicit user scope — this milestone prepares daydream for upload, it does not deploy infrastructure.
- **S3 / object-store bundle adapter.** Reason: not needed to validate the abstraction; explicitly deferred. See Future Considerations.
- **Filesystem-mirror adapter.** Same reason.
- **Langfuse / OpenInference / OTel adapter (the strictly-OSI sink).** Reason: the adapter pattern must make this trivially addable, but shipping it now would conflate "build the abstraction" with "build a non-trivial mapper." See Future Considerations.
- **A debug UI for browsing uploaded runs.** Reason: AI-engineer audience is post-MVP; if the Phoenix adapter is in use, Phoenix already provides one.
- **Leadership dashboards / metrics aggregation.** Reason: leadership audience is post-MVP; same Phoenix free-ride applies.
- **Multi-tenant isolation, per-org auth, hosted SaaS offering.** Reason: daydream is OSS and self-hosted; each adopter runs a single-tenant deployment.
- **Re-shaping ATIF on upload (transformation pipelines, schema mappers).** Reason: ATIF interop with the Harbor ecosystem is a stated foundational constraint; sinks consume the bundle as-is.
- **Additional PII / code redaction beyond what the trajectory recorder already does.** Reason: existing `Redactor` already scrubs secrets; adopters trust their own infrastructure.
- **Outcome-label round-trip / re-push when local labels change.** Reason: useful, but a Phase 2 problem once labels are actually being added at scale. See Future Considerations.

## Constraints

- **ATIF v1.6 fidelity preserved.** Sinks consume the canonical bundle without transformation. Reason: ATIF interop is a foundational project constraint.
- **Local archive is the source of truth by default.** Uploads are derivative. The single explicit exception is `daydream sync --rm-after-upload`, which trades local durability for disk reclamation and only after the SQLite index records confirmed remote upload. Reason: the durability assumption is what makes "auto + sync" possible without re-engineering retry logic per sink; the rm flag is explicit opt-in for adopters who accept the tradeoff.
- **No new runtime dependency unless a configured sink needs it.** Phoenix adapter's dependency tree must be opt-in (e.g., an extras install or a soft import that errors only when the Phoenix sink is selected). Reason: adopters who never enable Phoenix shouldn't pay for its install footprint.
- **No `harbor` runtime dependency.** Reason: pre-existing project constraint; ATIF code stays vendored under `daydream/atif/`.
- **Auto-push must not block or fail the run path.** Reason: pre-existing project pattern — `archive_run`'s body is already exception-wrapped to never affect the primary run; sinks inherit that contract.
- **Upload state lives in the existing SQLite index.** No new database, no new state file. Reason: duplicate state stores invite divergence; the existing index already supports cross-project querying and the filters `daydream sync` needs.
- **Disabled by default.** No env var, no config, no upload. Reason: zero-impact for adopters who haven't opted in is the OSS social contract.
- **No `Sink` construction or sink-specific code in `runner.py`, `phases.py`, or `ui.py`.** All sink wiring lives in a new module under `daydream/`, echoing the `Backend` protocol pattern. Reason: matches the existing module-bloat ban on `phases.py`/`ui.py` for ATIF construction.

## Key Decisions

1. **Phoenix is the v1 reference adapter — not a deployment commitment.**
   *Considered:* Langfuse first (most popular OSS, MIT). Rejected — no native ATIF means a mapper is required, which would conflate "build the abstraction" with "build a mapper" and obscure whether the abstraction is honest.
   *Considered:* Opik first (Apache 2.0, has Harbor integration). Rejected — the Opik/Harbor integration is "wrap-and-trace at execution time," not "ingest pre-recorded ATIF bundles." Wrong shape for daydream's archive-first model.
   *Chosen:* Phoenix because it is the only OSS sink that natively eats ATIF v1.0–v1.6 (matches our pinned version exactly), with idempotent re-uploads via session_id-derived span IDs. Validates the abstraction with the smallest possible adapter and zero mapping logic.

2. **Auto-push + `daydream sync` for backfill — not auto-only, not manual-only.**
   *Considered:* Auto only. Rejected — CI runs that die mid-push lose data permanently; adopters can't repoint at a new sink without out-of-band work.
   *Considered:* Manual only. Rejected — defeats "flip the switch and forget."
   *Chosen:* Auto on `archive_run` completion + `daydream sync` for backfill, retry, sink switching, and onboarding pre-existing archives. The local archive is already durable and acts as the upload queue.

3. **Full bundle is the unit of upload, not just the trajectory.**
   *Considered:* Trajectory only. Rejected — ML researchers will eventually want manifest + review-output + diff together; changing the unit later is a breaking interface change.
   *Chosen:* Sinks receive a `Bundle` view over the archive directory (paths + parsed manifest); each sink decides what to push. Phoenix adapter pushes only the trajectory; future S3/Langfuse adapters will push more.

4. **Per-sink upload status in the existing SQLite index.**
   *Considered:* Rely on remote-side idempotency. Rejected — Phoenix has it, future sinks may not, and `daydream sync --since=…` filtering needs the index anyway.
   *Considered:* A new state file per sink. Rejected — the SQLite index already exists and indexed lookups are exactly what `daydream sync` needs.
   *Chosen:* A new table in the existing SQLite index keyed by `(session_id, sink_name)`, recording status, remote_id, uploaded_at, and last error.

5. **Adapter is opt-in via configuration; no default sink ships enabled.**
   *Considered:* A "noop" sink as the default. Rejected — every adopter would have to actively turn it off; surprises in OSS land badly.
   *Chosen:* No sink configured ⇒ no upload behavior whatsoever, exactly the same code path as today.

6. **`Sink` is a Protocol, not a base class.**
   *Considered:* Abstract base class with shared retry/timeout logic. Rejected — `Backend` in this codebase is already a Protocol; consistency wins, and any shared retry/timeout logic that emerges should live in the dispatcher, not a base class.
   *Chosen:* Structural typing matching the existing `Backend` pattern. Factory function `create_sink(name, config)` parallels `create_backend(name, model)`.

7. **Configuration via environment variables only at v1.**
   *Considered:* Config file (`~/.daydream/config.toml`). Rejected for now — env vars match "flip a switch" simplicity, work cleanly in CI, and don't introduce a new file format to maintain. Config file is in Future Considerations if env-var surface grows past comfort.
   *Chosen:* Env-only.

## Reference Points

- **ATIF v1.6 spec (Harbor RFC 0001):** https://github.com/laude-institute/harbor/blob/main/docs/rfcs/0001-trajectory-format.md
- **Phoenix ATIF Trajectory Upload (April 2026 release notes):** https://arize.com/docs/phoenix/release-notes/04-2026/04-03-2026-atif-trajectory-upload
- **Phoenix self-hosting docs:** https://arize.com/docs/phoenix/self-hosting
- **Existing local archive implementation:** `daydream/archive/__init__.py` (`archive_run`, `_archive_run_inner`)
- **Existing manifest schema:** `daydream/archive/manifest.py` (`Manifest`, `MANIFEST_SCHEMA_VERSION`)
- **Existing SQLite index:** `daydream/archive/index.py` (`upsert_run`)
- **Existing trajectory recorder + redactor:** `daydream/trajectory.py` (`TrajectoryRecorder`, `Redactor`)
- **Architectural analog for the `Sink` protocol:** `daydream/backends/__init__.py` — the `Backend` protocol + `create_backend(name, model)` factory is exactly the shape this should mirror.

## Open Questions

None — all clarifications resolved during brainstorming.

## Future Considerations

- **Config file (`~/.daydream/config.toml` or similar)** if env-var surface grows past comfort or adopters request it.
- **S3-compatible bundle adapter.** Bucket-keyed `{repo}/{date}/{session_id}/`, full bundle as discrete files. The "raw object-store lake" leg of the larger picture.
- **Langfuse adapter (the strictly-OSI sink).** Requires an ATIF → OTel/OpenInference span mapper; that mapper is itself a reusable artifact for any OTel-shaped sink.
- **Opik adapter** if there's adopter demand. Apache 2.0; would need a different integration shape than the existing Opik-Harbor wrapper.
- **`daydream-lake` notebook helper library** for ML researchers working against a bundle adapter — `read_parquet` / `read_json` over the archive layout, lazy session loading, dataset filters.
- **Normalizer / indexer service** that on upload extracts metrics + run metadata into a columnar store (DuckDB / Parquet / ClickHouse). Only worth building once we know which queries actually matter.
- **AI-engineer debug UI** for browsing uploaded runs (the second audience). Free if the sink is Phoenix; a real product surface otherwise.
- **Leadership dashboards** for cost/findings/fix-rate trends (the third audience). Same Phoenix free-ride applies.
- **Outcome-label round-trip / re-push.** When local `Manifest.outcome_labels` change, `daydream sync` should patch the remote. Needs an "update existing run" semantic that v1 sinks may not support uniformly.
- **Per-sink redaction policy.** Public-research sink redacts code; private internal sink doesn't. Today's redaction is global at trajectory-write time.
- **Hosted SaaS offering of any kind.** Out of scope for the OSS distribution; if anyone wants to operate this as a service, the adapter pattern allows it without changes to daydream itself.
