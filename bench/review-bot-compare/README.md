# review-bot-compare

Benchmark **daydream** against any GitHub review bot (CodeRabbit, Greptile, …)
on a repo's real PR history, by replaying daydream at the *exact snapshot* the
bot reviewed.

Three stages, each a standalone script (stdlib + `gh`/`git` CLIs only):

```text
harvest.py   bot's historic reviews  → out/<owner__repo>/pr-<N>.json + index.json
replay.py    daydream @ same snapshot → out/<owner__repo>/findings|traj|replay/
judge.py     LLM semantic matching    → out/<owner__repo>/judge/pr-<N>.json   (optional)
compare.py   align + score            → out/<owner__repo>/report.md + comparison.csv
```

## Why "same snapshot" matters

A review bot reviews a PR at a specific commit. The GitHub review object carries
`commit_id`. `replay.py` reconstructs the bot's exact view of the diff:

- `head` = the bot review's `commit_id`
- `base` = `git merge-base origin/<base_ref> <head>` (GitHub's 3-dot compare base)

It checks `head` out in a **detached worktree** and runs daydream in-place with
`--base <merge-base>`, so daydream reviews byte-for-byte the same diff the bot saw
— not today's HEAD.

## Usage

```bash
# 1. Harvest (no checkout needed — pure gh api)
python3 harvest.py --repo OWNER/REPO --bot "coderabbitai[bot]" --out ./out --limit 300

# 2. Replay daydream at each snapshot (needs a local clone with the remote)
python3 replay.py --repo OWNER/REPO --source /path/to/clone \
    --in ./out --backend codex --limit 5          # pilot: first 5 PRs
#   add --shallow for a fast single-stack pass; drop it for the real deep review
#   --pr 1292 --pr 1290  to target specific PRs

# 3. (optional but recommended) LLM judge for semantic overlap — one light call per PR
python3 judge.py --repo OWNER/REPO --in ./out --backend pi --provider zai --model glm-5.2

# 4. Compare (auto-uses judge matches when present, else deterministic lower bound)
python3 compare.py --repo OWNER/REPO --bot-name coderabbit --in ./out --min-conf 0.5
```

## Why the LLM judge

Two tools phrase the same issue differently and anchor it to different lines, so
the deterministic file+line+wording matcher reports ~0 overlap even when they
agree. `judge.py` makes ONE light LLM call per PR (no tools, no repo checkout —
works on subscription backends that rate-limit deep reviews) to decide which
findings describe the same underlying issue. `compare.py` picks up the judge
artifact automatically. Example: on a pilot PR the deterministic matcher found 0
overlap; the judge correctly matched 2 issues both tools flagged (an `os.Exit`
in a deps-init path, and DB errors masked as HTTP 200s).

## Generalization

- `--bot` is any `user.login`. The harness matches tolerant of GitHub's
  REST-vs-GraphQL `[bot]`-suffix mismatch (`coderabbitai[bot]` ≡ `coderabbitai`).
  Confirm the real handle first: `gh api repos/OWNER/REPO/pulls/N/comments
  --jq '.[].user.login' | sort -u`.
- `--repo` is any `owner/repo`.
- `--backend` is daydream's backend (`codex`, `claude`, `pi`). Codex/Pi allow
  subscription auth for unattended batches; Claude needs an `ANTHROPIC_API_KEY`
  (subscription keys are disallowed for automation).

## What's measured

| Axis | daydream | bot |
|---|---|---|
| Issues raised | findings artifact | inline comments |
| Overlap / unique | deterministic file+line+token match (a **lower bound**) | |
| Acted-upon precision | findings hitting a **resolved** bot thread | resolved-thread rate |
| Cost | measured (`trajectory.final_metrics.total_cost_usd` + tokens) | **not observable** (SaaS) — use amortized list price |
| Latency | wall-clock per PR | review timestamp − PR open |

## Known limitations

- **Overlap is a lower bound.** Deterministic matching misses semantically-equal
  findings phrased differently or snapped to different lines. Layer an LLM judge
  on top for the headline overlap number; the resolved-thread axis is unaffected.
- **Bot cost is amortized, daydream's is measured.** Stated explicitly in the
  report; not faked.
- **Bot summary bodies aren't split into discrete findings** — only inline
  comments are counted as bot findings (add a per-bot normalizer to change this).
- Codex cost fields may be partial depending on backend parity; tokens are always
  populated and are the more honest cross-backend comparison.
