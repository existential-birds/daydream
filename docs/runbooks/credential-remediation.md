# Credential Remediation Runbook (issue #981, M22)

When a Git credential is discovered in an archived bundle — an uploaded Hub
dataset, a local archive, or a quarantined derivative — this runbook walks the
remediation from scoping through verification. It exists because the code
deliberately stops at the safe boundary: the sanitizer, scanner, and inventory
tooling are agent-runnable, but **revocation, rotation, and Hub history
rewrite are destructive operations that require a human and are never
automated.**

Audience: the daydream operator (a human with provider dashboards open and the
archive checkout in their own terminal).

---

## 1. Scope the incident

Run the value-free inventory over the archive:

```bash
python -c "
from pathlib import Path
from daydream.archive.sanitize import report_inventory
report_inventory(Path('/path/to/archive_dir'))
"
```

`report_inventory()` classifies each bundle's `git.remote_url` via the
normalizer and prints **counts by category only** — session counts, never a
URL fragment or a matched credential value. Categories mirror
`daydream.archive.git_safe.classify_remote_url`: `userinfo` (covers both
`user:token`/PAT-shaped userinfo and `x-access-token` token-only `user@`
forms), `query` (credential-like query parameters), and `clean` (benign URLs),
plus `unparseable` for manifests it could not read.

Record the output. It tells you how many manifests/bundles are affected per
category, which determines how wide the revocation net in step 3 must be.

If the incident involves bundles already uploaded to the Hub, also list the
affected dataset revisions (upload timestamps vs. affected session dates)
before touching anything.

## 2. Identify affected credentials

Work from `sanitized/audit.jsonl` records under `<archive_dir>/sanitized/`.
Each record carries, per processed bundle:

- `source` — the original run directory path
- `session_id` — the archived session
- `derivative_digest` — content digest of the released derivative
- `status` — `"sanitized"` or `"quarantined"`
- `completed_at` — timestamp

**Never print a credential value.** Do not grep for the token itself, do not
paste matched regions into tickets, logs, or agent tooling. If the operator
must inspect a value (e.g. to match a token prefix against a provider
dashboard), they do it **in their own terminal, from the source-of-record**
(provider settings page, secret store, CI config) — outside agent tooling.

Map affected sessions to the repos they touched:

1. For each affected `session_id`, read `manifest.json` in the source run
   directory and note the repo identity (owner/repo) — the *identity*, not the
   raw URL.
2. The inventory's single `userinfo` category covers both `x-access-token`
   token-only `user@` forms (GitHub App installation tokens) and `user:token`
   or PAT-shaped userinfo (personal access tokens), so it cannot tell the two
   apart at category granularity. Distinguish them from the source
   `manifest.json` `git.remote_url` records instead, not the inventory
   category counts.
3. The result is a table of (credential type, repo(s), session ids, date
   range) — safe to share, contains no secret values.

## 3. Verify expiry / revoke or rotate

> **Gate: revocation and rotation require human approval. Never automated.
> No code path in daydream revokes, rotates, or expires a credential.**

Per provider:

- **GitHub PAT (classic / fine-grained):** in GitHub → Settings → Developer
  settings, check each candidate token's last-used date and expiry against the
  affected date range. Revoke tokens that overlap, or rotate (issue a
  replacement, update the secret store, then revoke the old one). Prefer
  revocation over rotation when the token's scope is uncertain.
- **GitHub App installation tokens (`x-access-token`):** these are short-lived
  by construction, but if the *installation's* credential material (the App
  private key) could have leaked alongside, rotate the App private key from
  the App settings page. Check the App's installation audit log for anomalous
  repo access within the affected window.
- **Other providers:** apply the same rule — verify scope and expiry from the
  provider's own dashboard, then revoke first, rotate second.

After revocation, confirm in the provider's audit log that the credential no
longer authenticates.

## 4. Remediate Hub history

Choose exactly one option, in escalating order of destructiveness:

- **(a) Leave quarantined bundles unreleased (non-destructive default).**
  Bundles that failed the fail-closed scan live under
  `<archive_dir>/quarantine/<session_id>/` and are never released. Doing
  nothing is a valid, safe outcome for anything not yet uploaded.
- **(b) Delete specific Hub uploads.** Delete the affected dataset revision(s)
  from the HuggingFace dataset repo (revisions uploaded before revocation, or
  re-upload sanitized derivatives after revocation).
- **(c) Full history rewrite** of the dataset repo — only when the credential
  shipped in many revisions and (b) is impractical.

> **Gate for (b) and (c):** all of the following are required before acting:
> 1. Explicit operator approval, recorded in writing (who, when, what scope).
> 2. An **executed revocation first** (step 3 is complete) — deleting history
>    is useless if the credential still works.
> 3. A written record of exactly what was deleted (dataset repo, revision
>    SHAs or date range, operator, timestamp), kept with the incident record.

## 5. Non-destructive vs destructive operations

| Operation | Destructive? | Who runs it | Gate |
|---|---|---|---|
| `report_inventory()` scoping | No (read-only, value-free) | Agent or human | — |
| Reading `sanitized/audit.jsonl` | No | Agent or human | Never print values |
| `sanitize_archive()` / `sanitize_bundle()` | No (produces derivatives; bronze sources never modified) | Agent or human | Fail-closed scan; failures quarantine |
| Quarantine **release** (moving a derivative out of `quarantine/` after review) | No | Agent or human | Must pass `scan_run_dir()` clean first |
| Credential revocation / rotation | **Destructive** | **Human only** | Human approval; never automated |
| Hub revision deletion (4b) | **Destructive** | **Human only** | Approval + executed revocation + written deletion record |
| Hub history rewrite (4c) | **Destructive** | **Human only** | Approval + executed revocation + written deletion record |

## 6. Verify

Post-remediation, confirm the incident is closed:

1. **Re-run the inventory:** `report_inventory()` again. Every previously
   affected category must now read zero; only clean categories (and
   `unparseable`, if pre-existing) remain.
2. **Re-scan:** run the fail-closed scanner (`daydream.archive.scan.scan_run_dir`)
   over sanitized derivatives and any bundle that will egress. It must report
   clean.
3. **Revocation holds:** attempt authentication with a revoked token from the
   operator's own terminal and confirm it fails.
4. **Going forward:** the fail-closed scan (upload preflight in
   `daydream/archive/hub.py`, the `--dump-artifacts` copy gate, and the
   sanitizer release gate) blocks any bundle that is not provably clean from
   every egress path. Future incidents should be caught at that gate, before
   upload.

## 7. How the harvest clone step authenticates

`corpus harvest` clones repos from manifests that have no local checkout into
its clone cache. It never clones the archived raw URL: the URL is rewritten via
`normalize_remote_url` (see `daydream.archive.git_safe`) into a credential-free
HTTPS identity like `https://github.com/owner/repo`, and only allowlisted git
hosts are accepted. All credential material is therefore absent from the URL
by construction.

Authentication for private repos is out-of-band and operator-supplied:

- Set the `DAYDREAM_GIT_TOKEN` environment variable (e.g. a GitHub PAT with
  read access to the target repos) when running `corpus harvest`. It is
  injected into the git command via `-c http.extraHeader` at the argv layer;
  the URL string never contains the token.
- Without the variable, the clone falls back to the ambient git credential
  helper. `GIT_TERMINAL_PROMPT=0` fails closed on auth errors instead of
  prompting.
- A clone or fetch failure is warning-only at harvest time (the affected run
  is skipped, never aborted) and never survives into the archive.

Treat `DAYDREAM_GIT_TOKEN` as a credential in its own right: if it is
suspected leaked, add it to the step-3 revocation net. It is an input to the
clone step, not archived data, so it never appears in uploaded bundles or in
`report_inventory()` output.
