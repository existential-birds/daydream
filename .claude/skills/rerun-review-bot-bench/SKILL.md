---
name: rerun-review-bot-bench
description: >-
  Benchmark daydream against a GitHub review bot's PR history — harvest the
  bot's reviews into a corpus, replay daydream at each bot snapshot, and score
  the overlap with the in-process judge. Use when asked to "re-run the
  review-bot benchmark", "rerun the GLM pilot", "replay daydream against
  coderabbit/greptile", or to benchmark daydream vs a SaaS review bot on a
  repo's real PRs. Encodes the verified traps (pi --session-id version floor,
  z.ai overload, disjoint pi token counters) so the re-run doesn't waste hours.
---

# Benchmark daydream against a review bot

This runs entirely through the production harness — `daydream bench`. There are
no standalone scripts (the `bench/review-bot-compare/` pilot was absorbed into
`daydream/benchmark/`).

Two commands:

1. `daydream bench harvest` — pull the bot's review history into a **harvested
   corpus** (`<DIR>/results/benchmark_data.json` + `<DIR>/harvest/pr-<N>.json` +
   `<DIR>/index.json`). The bot's standalone inline comments become the golden
   set; its own review is injected as a `tool=<bot stem>` review entry.
2. `daydream bench --harvest-dir <DIR>` — acquire each PR at the **bot's
   snapshot** (head = the bot review's `commit_id`, base = `merge-base(
   origin/<base_ref>, head)`, i.e. GitHub's 3-dot compare base), run daydream,
   inject the review, and score.

Worked example below: the **shelfspace / coderabbit** GLM (pi) pilot.
Parameterize `--repo`, `--bot`, and the reviewer flags for any repo/bot.

Deep mode is ~10–20 min/PR on GLM. Budget accordingly.

## Steps

### 1. Preflight pi (do this FIRST — skips traps 1 and 2 below)

```bash
# (a) version floor: daydream's pi backend passes `--session-id <uuid>`,
#     which only exists in pi >= 0.80.2. pi 0.74.2 rejects it and every run
#     aborts in ~2s. Both checks must pass:
pi --version                                  # must be >= 0.80.2
pi --help | grep -- --session-id              # must show: --session-id <id>

# If either fails, upgrade (the `pi` on PATH is the HOMEBREW npm global,
# NOT nvm's — use the homebrew npm explicitly):
/opt/homebrew/bin/npm install -g @earendil-works/pi-coding-agent@latest

# (b) one-shot smoke — catches z.ai overload AND a broken provider:
pi -p --no-tools --provider zai --model glm-5.2 'reply with {"ok":true}'
```

- One-shot 429s/errors → **z.ai is overloaded. STOP** and retry later; deep
  runs will all fail.
- One-shot is clean but a real run still prints `Unknown option: --session-id`
  → that's the **version trap (1a)**, not overload. Re-run the version check.

### 2. Harvest the bot's review history

```bash
# Confirm the real bot handle first:
gh api repos/OWNER/REPO/pulls/N/comments --jq '.[].user.login' | sort -u

daydream bench harvest \
    --repo shelfspace-app/shelfspace-mono \
    --bot "coderabbitai[bot]" \
    --out ./out/shelfspace-coderabbit \
    --limit 300 --state all
```

Harvest is pure `gh api` — no checkout, no model calls. Re-harvest only when
the PR set has changed.

### 3. Run the benchmark

A harvested corpus **must** score via `--judge-route anthropic-direct`: the
`martian` route shells `python -m code_review_benchmark.step*` with the corpus
root as cwd, and that package only exists inside the withmartian checkout. The
CLI rejects the combination.

```bash
export ANTHROPIC_API_KEY=...   # the judge
daydream bench \
    --harvest-dir ./out/shelfspace-coderabbit \
    --reviewer-backend pi --reviewer-model glm-5.2 --reviewer-provider zai \
    --tool-label daydream-glm \
    --judge-route anthropic-direct --model anthropic/claude-opus-4-5-20251101
#   --reviewer NAME               instead, to expand a [tool.daydream.bench.reviewers.NAME] preset
#   --limit N / --only <pr-url>   to narrow the PR set
#   --force                       to re-review PRs already injected
#   --no-score                    to review only, score later (scoring is ON by default)
#   --trials N                    repeat the sweep and get a distribution
```

`--model` is the **judge** model; the reviewer under test is configured with
the `--reviewer-*` flags.

Run it with `run_in_background: true` and tail the output.

Per-PR failures are isolated — one PR's `GitError` or `DaydreamRunError` never
aborts the sweep. Transient backend overload and tail-end stream drops are
handled inside `run_daydream_review` (see trap 3).

While it runs, judge live progress ONLY via these signals:

```bash
# daydream parent + a pi child means it's actively working:
ps aux | grep -E "[d]aydream|[p]i --mode json"
# fresh pi session writes in the last 3 min = forward progress:
find ~/.pi/agent/sessions -name '*.jsonl' -newermt '-3 minutes'
```

### 4. Read the report

```bash
cat ./out/shelfspace-coderabbit/.daydream-bench/report-daydream-glm.json
```

The JSON report carries per-PR `elapsed_s`, tokens, `cost_usd` +
`cost_source` ("measured" | "synthesized" | "unknown"), and tp/fp/fn/precision/
recall leaves, plus an `aggregate` block. GLM reports `cost_usd=0`, so cost is
synthesized from tokens via `daydream/pricing.py` and labeled `synthesized` —
not faked. Add a GLM price card through `$DAYDREAM_PRICES_FILE`.

`bench/benchmark-report/build.py` renders the HTML view over run dirs.

## Gotchas / traps

1. **pi version floor `>= 0.80.2`.** `--session-id <uuid>` is rejected by
   0.74.2 (`Error: Unknown option: --session-id`) → every run aborts in ~2s
   with a "Backend Execution Error". Preflight per step 1a.

2. **z.ai overload.** The one-shot smoke (step 1b) 429ing means deep runs will
   all fail — stop and retry later, don't burn the batch.

3. **Transient failure and tail-end stream drops are already handled.**
   `daydream/benchmark/daydream_run.py` retries overload/rate-limit signatures
   twice with exponential backoff, and treats a non-zero exit whose
   `merged-items.json` + trajectory `final_metrics` are complete on disk as a
   success (the review finished; only the closing socket died). Don't add a
   retry wrapper around the CLI.

4. **App-identity hard-abort is already handled** — `run_daydream_review`
   strips `DAYDREAM_APP_ID` / `DAYDREAM_APP_PRIVATE_KEY` from the child env so
   the review doesn't hard-abort under a GitHub App identity. Don't re-add them.

5. **`[bot]` suffix mismatch is handled** — GitHub REST keeps `coderabbitai[bot]`,
   GraphQL drops it to `coderabbitai`. `bot_login_matches` matches both; harvest
   with the REST form (`--bot "coderabbitai[bot]"`).

6. **pi token counters are disjoint** — `prompt`=input, `cached`=cacheRead (a
   *separate* cache-read bucket). They bill separately; don't sum them as one
   input total. `report.synthesize_cost` branches on backend for exactly this.

7. **Re-runs skip already-injected PRs.** The corpus-level skip is the resume
   mechanism; pass `--force` for a genuinely clean re-review. There are no
   stale per-PR artifact files to hand-clear anymore.

8. **A giant-diff PR can kill the pi process.** shelfspace **PR #1292** is a
   ~17k-line diff that can OOM/terminate pi ("terminated"). Per-PR isolation
   records the failure and moves on; retry it alone with `--only <pr-url>`.

9. **No merge-base means the PR fails loudly.** A snapshot whose base ref has
   no merge-base with the bot's commit raises `GitError` rather than silently
   reviewing a different diff than the bot saw. Fix the `base_ref` in
   `index.json`; don't work around it.

10. **Never `--no-verify`, never bypass a failing gate.** If something fails,
    fix the root cause (usually the pi version or z.ai availability above).
