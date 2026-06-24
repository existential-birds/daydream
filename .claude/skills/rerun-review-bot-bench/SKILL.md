---
name: rerun-review-bot-bench
description: >-
  Re-run the review-bot-compare benchmark pilot — replay daydream against a
  GitHub review bot's PR history and score the overlap. Use when asked to
  "re-run the review-bot benchmark", "rerun the GLM pilot", "replay daydream
  against coderabbit/greptile", "redo the review-bot-compare run", or to
  benchmark daydream vs a SaaS review bot on a repo's real PRs. Encodes the
  verified traps (stale artifacts, pi --session-id version floor, z.ai
  overload, no --provider on replay.py) so the re-run doesn't waste hours.
---

# Re-run the review-bot-compare benchmark

Harness lives at `bench/review-bot-compare/` (run all `python3` commands from
there). Pipeline: `harvest.py` → `replay.py` → `judge.py` → `compare.py`.
Artifacts land in `out/<owner__repo>/{findings,traj,logs,replay,judge,worktrees}/pr-<N>.json`
plus `report.md` + `comparison.csv`. The `<owner__repo>` slug replaces `/` with
a double underscore (e.g. `shelfspace-app/shelfspace-mono` → `shelfspace-app__shelfspace-mono`).

Worked example below: the **shelfspace / coderabbit** GLM (pi) pilot.
Parameterize `--repo`, `--bot`, `--source`, and the PR set for any repo/bot.

Deep mode (no `--shallow`) is ~10–20 min/PR on GLM. Budget accordingly; use
`--shallow` only for a fast single-stack smoke.

## Steps

### 1. Preflight pi (do this FIRST — skips the #1 and #2 traps below)

```bash
# (a) version floor: daydream's pi backend passes `--session-id <uuid>`,
#     which only exists in pi >= 0.80.2. pi 0.74.2 rejects it and every run
#     aborts in ~2s. Both checks must pass:
pi --version                                  # must be >= 0.80.2
pi --help | grep -- --session-id              # must show: --session-id <id>  Use exact project session ID, creating it if missing

# If either fails, upgrade (the `pi` on PATH is the HOMEBREW npm global,
# NOT nvm's — use the homebrew npm explicitly):
/opt/homebrew/bin/npm install -g @earendil-works/pi-coding-agent@latest
# Reference source if you need to inspect: /Users/ka/github/reference_agents/pi-mono

# (b) one-shot smoke — catches z.ai overload AND a broken provider:
pi -p --no-tools --provider zai --model glm-5.2 'reply with {"ok":true}'
```

- One-shot 429s/errors → **z.ai is overloaded. STOP** and retry later; deep
  replays will all fail.
- One-shot is clean but a real replay still prints `Unknown option:
  --session-id` → that's the **version trap (1a)**, not overload. Re-run the
  version check.

### 2. Clear stale per-PR artifacts for the target PRs (before any clean re-run)

`replay/` and `judge/` records are **NOT** auto-cleared, so a fast-abort run
can leave them showing the PREVIOUS run's numbers (real case: "findings=6/28"
reported for PRs that actually aborted in 2s — old codex-era findings on disk).
Clear them for every PR you're about to re-run:

```bash
REPO_SLUG=shelfspace-app__shelfspace-mono
for N in 1290 1291 1292 1293 1295; do
  rm -f out/$REPO_SLUG/findings/pr-$N.json \
        out/$REPO_SLUG/traj/pr-$N.json \
        out/$REPO_SLUG/judge/pr-$N.json \
        out/$REPO_SLUG/replay/pr-$N.json
done
# Do NOT delete out/$REPO_SLUG/logs/ — keep crash logs. (Just know logs LAG;
# never diagnose an in-progress run from them — see Gotchas.)
```

### 3. (If re-harvesting) refresh the bot's review history — usually skip

Harvest is pure `gh api`, no checkout. Only needed if PRs changed since last
harvest:

```bash
python3 harvest.py --repo shelfspace-app/shelfspace-mono --bot "coderabbitai[bot]" --out ./out --limit 300
# Confirm the real bot handle first:
#   gh api repos/OWNER/REPO/pulls/N/comments --jq '.[].user.login' | sort -u
```

### 4. Replay daydream at each snapshot (run in BACKGROUND, then monitor live)

`replay.py` has **NO `--provider`/`--model` flags** — `--backend pi` alone
gives GLM (its pi backend defaults to `--provider zai` + `glm-5.2`). Passing
`--provider` to replay.py errors.

```bash
python3 replay.py --repo shelfspace-app/shelfspace-mono \
    --source /Users/ka/github/shelfspace-app/shelfspace-mono \
    --in ./out --backend pi --pr 1295 --pr 1293 --pr 1292 --pr 1291 --pr 1290 \
    --timeout 2700 --retries 3 --backoff 30
#   --limit N  instead of repeated --pr to take the first N harvested PRs
#   --shallow  for a fast single-stack smoke (skip for the real deep review)
#   --retry-failed  to re-run only PRs whose replay record is not 'ok'
```

Run this with `run_in_background: true` and tail its output file. The replay's
own **stdout result line** (`-> ok | findings=N | cost=$...`) is the
authoritative per-PR completion signal — NOT the on-disk logs/findings.

While it runs, judge live progress ONLY via these signals:

```bash
# daydream parent + a pi child means it's actively working:
ps aux | grep -E "[d]aydream|[p]i --mode json"
# fresh pi session writes in the last 3 min = forward progress:
find ~/.pi/agent/sessions -name '*.jsonl' -newermt '-3 minutes'
```

### 5. LLM judge (semantic overlap — one light call per PR)

`judge.py` DOES take `--provider`/`--model` (unlike replay.py):

```bash
python3 judge.py --repo shelfspace-app/shelfspace-mono --in ./out \
    --backend pi --provider zai --model glm-5.2 --retries 3
#   --pr N  to judge specific PRs
```

### 6. Compare + read the report

```bash
python3 compare.py --repo shelfspace-app/shelfspace-mono --bot-name coderabbit \
    --in ./out --min-conf 0.5
cat out/shelfspace-app__shelfspace-mono/report.md
```

GLM reports `cost_usd=0`, so `compare.py` synthesizes $ from tokens via the
`glm-5.2` price card (`--price-model glm-5.2`, the default known card). Cost is
labeled synthetic in the report — not faked.

## Gotchas / traps

1. **Stale artifacts are the #1 time-waster — NEVER diagnose an in-progress
   run from `logs/`, `findings/`, or `traj/`.** `replay.py` writes
   `logs/pr-N.log` and `findings/pr-N.json` only AS/AFTER a PR's daydream run
   finishes (findings only on a *successful* exit), and the `replay/pr-N.json`
   record only after `replay_one` returns. During a re-run, those files still
   hold the PREVIOUS run's output and will show you old errors. A PR's
   artifacts are trustworthy only once its `replay/pr-N.json` is rewritten OR
   the replay stdout prints its result line. Trust live signals (step 4),
   not on-disk per-PR files.

2. **Clear stale artifacts before a clean re-run** (step 2). `replay.py` does
   **NOT** pre-clear anything — line 111 only `mkdir`s parent dirs. `findings/`
   and `traj/` are overwritten by daydream *only on a successful exit*; `logs/`
   only after the run returns; `replay/` + `judge/` records are never
   auto-cleared. So a fast-abort run leaves the PREVIOUS run's `findings`,
   `traj`, `replay`, and `judge` files intact → stale numbers (the real
   "findings=6/28" misreport). Manually `rm` them for every PR you re-run.

3. **pi version floor `>= 0.80.2`.** `--session-id <uuid>` is rejected by
   0.74.2 (`Error: Unknown option: --session-id`) → every run aborts in ~2s
   with a "Backend Execution Error". Preflight per step 1a.

4. **z.ai overload.** The one-shot smoke (step 1b) 429ing means deep replays
   will all fail — stop and retry later, don't burn the batch.

5. **`replay.py` takes no `--provider`/`--model`.** `--backend pi` is enough
   for GLM. (`judge.py` is the script that takes those flags.)

6. **App-identity hard-abort is already handled** — `replay.py` strips
   `DAYDREAM_APP_ID` / `DAYDREAM_APP_PRIVATE_KEY` from the child env so
   `--review` doesn't hard-abort under a GitHub App identity. Don't re-add them.

7. **`[bot]` suffix mismatch is handled** — GitHub REST keeps `coderabbitai[bot]`,
   GraphQL drops it to `coderabbitai`. The harness matches both; harvest with
   the REST form (`--bot "coderabbitai[bot]"`).

8. **pi token counters are disjoint** — `prompt`=input, `cached`=cacheRead (a
   *separate* cache-read bucket). They bill separately; don't sum them as
   one input total.

9. **A giant-diff PR can kill the pi process.** shelfspace **PR #1292** is a
   ~17k-line diff that can OOM/terminate pi ("terminated"). The harness
   isolates per-PR failures — accept 4/5 and move on, or retry just that one:
   `python3 replay.py ... --backend pi --pr 1292 --retries 3`.

10. **Never `--no-verify`, never bypass a failing gate.** If something fails,
    fix the root cause (usually the pi version or z.ai availability above).
