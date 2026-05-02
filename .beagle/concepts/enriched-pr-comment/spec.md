---
name: enriched-pr-comment
description: Replace daydream's single-line "Mode" PR summary comment with a structured rollup + per-phase ATIF metrics breakdown so ML researchers, product engineers, and engineering leadership can judge the value of a review at a glance.
---

# Enriched PR Comment Summary

## Core Value

Daydream's GitHub PR summary comment shows enough detail at a glance — model, cost, tokens, depth, version — for an ML researcher, product engineer, or engineering leader to judge the value of a review without leaving the PR page.

## Problem Statement

Daydream posts a summary comment to GitHub PRs in three flows (`--comment`, `daydream feedback <pr#>`, and the default deep-mode fix loop). Today the comment's `<details>ℹ️ Review info</details>` block contains a single line: `- **Mode:** {mode_label}`. That fact is too thin to answer the questions the actual readers ask.

**Who reads these comments and what they want to know:**

- **ML/AI researchers** — was prompt caching effective? How many turns? What model? What did this trajectory cost in compute terms?
- **Product engineers** — should I trust this review? Was the model serious about it (lots of tool use, real depth) or shallow?
- **Engineering leadership** — is daydream paying for itself? What does each review cost?

None of those questions are answerable from "Mode: trust-the-technology" alone. All of the data needed to answer them is already captured by ATIF v1.6 trajectories — the project just isn't using it on the way out.

## Requirements

### Must have

- **M1.** The PR summary comment includes a visible rollup with: Mode, Model (or "mixed — see breakdown"), Cost, Tokens (input / cached / output + cache hit %), Steps, Tool calls.
- **M2.** The PR summary comment includes a per-phase breakdown inside a collapsed `<details><summary>Per-phase breakdown</summary>` block, presented as a markdown table with columns: Phase, Model, Steps, Tools, Input, Cached, Output, Cost.
- **M3.** The PR summary comment includes a small-text footer naming the daydream version that produced the comment.
- **M4.** All three summary-posting flows (`--comment`, `daydream feedback <pr#>`, and the default deep-mode fix loop) produce comments with the identical field set and layout.
- **M5.** Cost is computed from a fixed price table covering: `gpt-5.5`, `gpt-5.5-pro`, `gpt-5-codex`, `gpt-5.3-codex`. Anthropic-backed runs use cost values already supplied by the Claude SDK; no synthesis needed.
- **M6.** When any phase ran on a model not in the price table, the rollup Cost cell renders `—` and a footnote names the unknown model. The per-phase row's Cost cell also renders `—`.
- **M7.** When phases used different models, the rollup Model cell renders `mixed — see breakdown`.
- **M8.** The Codex backend surfaces `cached_input_tokens` on its metrics events (currently always `None` — known bug).
- **M9.** When trajectory data is missing or unparseable, the comment degrades to today's single `Mode:` line plus a footer note "*run details unavailable*". The comment must still post.
- **M10.** Numbers ≥1,000 render with thousand separators; cost values <$0.01 render as `<$0.01`; cache hit ratio omitted when input tokens = 0.

### Should have

- **S1.** The existing `test_build_payload_shape` test (in `tests/test_pr_review.py`) is extended to assert the new field set is present in the rendered comment body.
- **S2.** New tests cover the renderer's edge cases: unknown model, missing trajectory, mixed-model run, single-phase run, deep-mode multi-trajectory run.
- **S3.** All currently-passing tests continue to pass post-change.

### Out of scope

- **User-overridable price tables.** A static, code-reviewed table is enough for now. *Why:* user-overridable prices add config surface and a "whose prices are right?" support burden that doesn't pay back at the current scale. *Tracked:* [#61](https://github.com/existential-birds/daydream/issues/61).
- **Wall-clock latency per phase.** *Why:* not in current ATIF rollup; would require new instrumentation. *Tracked:* [#58](https://github.com/existential-birds/daydream/issues/58).
- **Cost-per-finding metric.** *Why:* couples the comment renderer to review output schemas. *Tracked:* [#59](https://github.com/existential-birds/daydream/issues/59).
- **Pricing API integration / live price lookup.** *Why:* adds an external dependency and auth surface for a number that changes ~quarterly.
- **Inline review comments.** *Why:* this project is the summary comment only; inline comments stay as they are.
- **A `--version` CLI flag.** *Why:* orthogonal feature; the version is read internally for the footer regardless.

## Constraints

- **C1.** Comment generator runs after the daydream phases finish. Trajectory data is the source of truth — the comment renderer reads finalized data, not live in-memory recorder state. *Rationale:* uniform behavior across `--comment`, `daydream feedback <pr#>`, and the default deep-mode fix loop. Deep mode produces sibling fork trajectory files; the renderer aggregates across all of them without changing the deep orchestrator.
- **C2.** ATIF v1.6 schema is fixed for this project. No new ATIF fields. *Rationale:* schema version is pinned per the upstream daydream/ATIF milestone.
- **C3.** No external network calls during comment rendering (no live price API, no version API). *Rationale:* PR comment posting must be reliable and deterministic.
- **C4.** Comment must remain valid GitHub-flavored markdown and respect the existing `gh api ... pulls/{n}/reviews` posting mechanism. *Rationale:* don't break the integration that already works.
- **C5.** Daydream's project decision **D-16 ("no synthesis of cost from token prices")** is repealed by this project. *Rationale:* without synthesis, Codex runs can never show cost, and "judge value of review" fails for anyone using Codex.

## Key Decisions

- **K1.** **Both rollup and per-phase breakdown.** A single rollup line buries per-phase detail; a table-only layout makes leadership skim past the cost. Keeping both, with the table collapsed by default, serves all three audiences.
- **K2.** **Repeal D-16; allow cost synthesis from a hardcoded price table.** Considered alternatives: (a) keep D-16, render cost only for Claude — leadership can't see Codex cost; (b) live pricing API — extra failure surface and auth. Hardcoded table wins on simplicity and reliability.
- **K3.** **Price-table coverage: GPT-5.5 family + the codex-tuned models** (`gpt-5.5`, `gpt-5.5-pro`, `gpt-5-codex`, `gpt-5.3-codex`). Considered alternatives: GPT-5.5 family only (misses the actual Codex CLI defaults); broader 5.x table (more maintenance churn).
- **K4.** **Bundle the codex-cached-tokens fix into this project.** The CLAUDE.md note saying "no `cached_tokens` for codex" is stale — Codex emits `cached_input_tokens` on `turn.completed`; daydream just isn't extracting it. Fixing it inside this project means the cache-hit-ratio metric works for both backends from day one.
- **K5.** **Unknown models render `—` + footnote naming the model**, rather than guessing or hiding the row. Considered alternatives: hide unknown rows (silently misleads reader); render token-only "best-effort cost" (lies about precision). Honest sentinel + named footnote wins.
- **K6.** **Daydream version goes in a small footer line, not the rollup.** Considered alternatives: rollup line (steals attention from the metrics that drive the value judgment); inside the collapsed block (hidden when readers want provenance). Footer is the conventional place.
- **K7.** **Field labels: "Steps" (not "Turns") and "Tool calls".** ATIF's native vocabulary is "step." ML researchers will read it precisely; product engineers will infer "model invocations" from context. "Turns" is ambiguous (turn = exchange or just model output?).
- **K8.** **Comment degrades, never blocks.** If trajectory parsing fails, the comment still posts with today's behavior plus a small "run details unavailable" note. *Rationale:* a missing trajectory must never cost a user their review feedback.

## Reference Points

- **Current rendering site:** `daydream/pr_review.py:737-742` — same `<details>ℹ️ Review info</details>` shell, more content. The shell stays.
- **Posting mechanism:** `gh api /repos/{owner}/{repo}/pulls/{pr_number}/reviews --method POST --input <payload>` (`daydream/git_ops.py:761-826`). Unchanged.
- **Data source:** ATIF v1.6 `FinalMetrics` rollups + step/tool-call iteration over `Trajectory.steps`. Multiple Invocations within a single Trajectory are summed; sibling fork trajectory files (deep mode) are summed alongside the parent.
- **Codex protocol fact:** `TokenUsage.cached_input_tokens` is emitted by `codex-rs/protocol/src/protocol.rs:2042` on `turn.completed`. Daydream's `daydream/backends/codex.py:297-325` currently hardcodes `cached_tokens=None` and needs a small fix.
- **OpenAI pricing snapshot, May 2026 (per 1M tokens):** `gpt-5.5` $5.00 in / $0.50 cached / $30.00 out; `gpt-5.5-pro` $30.00 in / $180.00 out; `gpt-5-codex` $1.25 in / $10.00 out; `gpt-5.3-codex` $1.75 in / $14.00 out. (Cached-input prices for the codex-tuned and pro variants to be confirmed at build time against [OpenAI's pricing page](https://openai.com/api/pricing/) — values that aren't published get the input-token price as a conservative upper bound.)
- **Comment shape (rollup):**
  ```markdown
  - **Mode:** trust-the-technology
  - **Model:** gpt-5.5
  - **Cost:** $0.42
  - **Tokens:** 33,600 in (22,600 cached, 67% hit) → 6,900 out
  - **Steps / tool calls:** 23 / 47
  ```
- **Comment shape (collapsed table):**
  ```markdown
  | Phase            | Model    | Steps | Tools | Input  | Cached | Output | Cost   |
  |------------------|----------|-------|-------|--------|--------|--------|--------|
  | Review           | gpt-5.5  | 8     | 19    | 12,400 | 8,200  | 1,800  | $0.13  |
  | Parse feedback   | gpt-5.5  | 1     | 0     | 800    | 600    | 200    | $0.01  |
  | Fix              | gpt-5.5  | 12    | 25    | 18,000 | 12,000 | 4,500  | $0.22  |
  | Test & heal      | gpt-5.5  | 2     | 3     | 2,400  | 1,800  | 400    | $0.02  |
  ```
- **Footer:** `<sub>Generated by daydream v0.14.0</sub>`

## Open Questions

- **OQ1.** Cached-input price for `gpt-5.5-pro`, `gpt-5-codex`, and `gpt-5.3-codex` — search results don't surface these consistently. Confirm against [OpenAI's pricing page](https://openai.com/api/pricing/) at build time. Fallback if unpublished: use input-token price (slight overcount, transparent).
- **OQ2.** GPT-5.5's >272K-context multiplier (2× input, 1.5× output) — model only as a flat per-token price, or apply the multiplier when input tokens cross 272,000? The latter requires tracking per-call input size, not just session totals. Default proposal: flat price for v1, note if real runs commonly exceed 272K input tokens.
- **OQ3.** Phase name canonical strings — orchestrators emit phase identifiers, but the human-readable labels in the table ("Test & heal" vs "test_and_heal") need a single mapping. Is the existing phase name set in `daydream/phases.py` already display-quality, or does it need a label dict?
- **OQ4.** What's the maintenance trigger for refreshing the price table? Manual ad-hoc update on each daydream release? A pre-release checklist item? A scheduled review? Not blocking v1 but worth deciding before merge.

## Future Considerations

The deferred items below are tracked as GitHub issues:

- Wall-clock latency per phase ("how long did this take?") — most useful "value" signal not currently captured in ATIF rollups. → [#58](https://github.com/existential-birds/daydream/issues/58)
- Cost-per-finding ratio — couples to review output structure but answers "are we paying $1 for one finding or thirty?" → [#59](https://github.com/existential-birds/daydream/issues/59)
- A `daydream summarize <traj-dir>` subcommand that reuses the same renderer to produce comments from archived trajectories (useful for offline retrospectives, replay). → [#60](https://github.com/existential-birds/daydream/issues/60)
- User-overridable price table (e.g. `~/.daydream/prices.toml`) — defer until enterprise / cost-conscious users actually ask. → [#61](https://github.com/existential-birds/daydream/issues/61)
- Per-stack rollups inside the per-phase table when in deep mode (group rows by stack, with subtotals). → [#62](https://github.com/existential-birds/daydream/issues/62)
- Reasoning-token column for models with explicit reasoning output (GPT-5.x reasoning, extended-thinking Claude). Currently rolled into output tokens; pulling it out makes "how much was the model thinking vs producing visible output" visible. → [#63](https://github.com/existential-birds/daydream/issues/63)
- Project-level / cross-run dashboards aggregating these metrics over time.
