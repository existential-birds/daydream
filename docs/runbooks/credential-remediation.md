# Credential Remediation Runbook (issue #981, M22)

When a Git credential is discovered in an archived bundle — an uploaded Hub
dataset, a local archive, or a quarantined derivative — this runbook walks the
remediation from scoping through verification. It exists because the code
deliberately stops at the safe boundary: the sanitizer and scanner
are agent-runnable, but **revocation, rotation, and Hub history
rewrite are destructive operations that require a human and are never
automated.**

Ordinary local archives and `--dump-artifacts DIR` preserve assembled evidence;
diagnostic dumps always copy the exact assembled bytes, including
credential-shaped strings and binary files, without scanning or sanitization.
Byte preservation does not undo upstream trajectory redaction. For shared
diagnostics, use standalone `sanitize_bundle()` to create
separate derivatives and review their reports. Hydration
sanitizes incoming sources and verifies derivatives; its final curated payload,
including supporting ledgers, is scanned before upload and verified afterward.

Direct run-bundle uploads always scan and refuse blocking credential findings,
scanner exceptions, any `scan_error` finding regardless of severity, and
incomplete results. Advisory-only findings are allowed with value-free warnings;
matched values and exception payloads are never printed. Upload failure
preserves local evidence. The scanner covers limited credential patterns; a
passing scan cannot guarantee that a bundle contains no sensitive information.

Audience: the daydream operator (a human with provider dashboards open and the
archive checkout in their own terminal).

---

## 1. Scope the incident

Record affected bundle paths, session dates, repository identities, and scan
categories without copying credential values into the incident record. Use the
single-bundle sanitizer to create and scan a separate derivative:

```bash
uv run python - <<'PYTHON'
from pathlib import Path
from daydream.archive.scan import scan_run_dir
from daydream.archive.sanitize import sanitize_bundle

archive_dir = Path('/path/to/archive_dir')
run_dir = archive_dir / 'runs/SESSION'
scan_result = scan_run_dir(run_dir)
result = sanitize_bundle(run_dir, archive_dir)
print(result.session_id, result.status, result.derivative_digest, scan_result.summary())
PYTHON
```

The source remains untouched. Released derivatives live under `sanitized/`;
blocked derivatives remain under `quarantine/`. Review the audit record and
value-free scan diagnostics for each affected bundle. There is no bulk sanitizer,
resume ledger, or inventory API.

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
2. Distinguish GitHub App installation tokens from personal access tokens
   using the provider settings and original credential source. A scan category
   alone cannot identify the credential type.
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
| Reading `sanitized/audit.jsonl` | No | Agent or human | Never print values |
| `sanitize_bundle()` | No (produces derivatives; bronze sources never modified) | Agent or human | Fail-closed scan; blocking findings quarantine, advisory findings are reported and released |
| Quarantine **release** (moving a derivative out of `quarantine/` after review) | No | Agent or human | Must pass `scan_run_dir()` with no blocking finding first (see §6.2); advisory findings do not hold a release |
| Credential revocation / rotation | **Destructive** | **Human only** | Human approval; never automated |
| Hub revision deletion (4b) | **Destructive** | **Human only** | Approval + executed revocation + written deletion record |
| Hub history rewrite (4c) | **Destructive** | **Human only** | Approval + executed revocation + written deletion record |

## 6. Verify

Post-remediation, confirm the incident is closed:

1. **Check the incident record:** account for every affected bundle and uploaded
   revision. Original local evidence remains unchanged; verify the derivatives
   intended for sharing and the remediated remote revisions.
2. **Re-scan:** run the fail-closed scanner (`daydream.archive.scan.scan_run_dir`)
   over sanitized derivatives and any bundle that will egress. It must report
   **no blocking findings** (`ScanResult.blocking` empty). Direct upload also
   requires a complete result without any `scan_error` finding. The scanner
   reports two tiers:
   - **Blocking** — high-confidence credential formats: API-key prefixes
     (`api_key`), PEM key material (`pem_key`), JWTs (`jwt`), literal
     `user:pass@` userinfo and token-only userinfo (`url_credential`),
     credential-bearing query parameters (`query_credential`), and any
     `scan_error` (a scan that could not complete never reads clean). A
     credential finding refuses sanitized publication and direct uploads.
     Scanner errors always refuse direct upload, including an advisory-severity
     `scan_error`, as do scanner exceptions and incomplete results.
   - **Advisory** — shapes the scanner cannot attribute to a credential value:
     a secret-*named* variable whose value is not secret-shaped (`env_var`,
     e.g. `SORT_KEY = "created_at"`), and a userinfo template whose parts are
     entirely `{placeholder}` interpolation. These are reported to the operator
     but no longer refuse egress.

   So a complete scan with only advisory findings and no scan errors will **not**
   report `clean` yet can still allow egress. That is deliberate: a rule that
   cannot identify a credential value must not gate irreversible publication.
   Read the advisory list anyway —
   it names the path, location, and category (never a value), and it is where a
   credential in an unrecognized format would show up. If an advisory finding
   looks like a real credential, treat it as an incident and go back to step 1
   rather than releasing.
3. **Revocation holds:** attempt authentication with a revoked token from the
   operator's own terminal and confirm it fails.
4. **Going forward:** direct uploads, standalone sanitizer release, curated
   payload publication and clean-room verification retain blocking credential
   checks. Adjudication publication independently
   checks metadata and SQLite payloads. Local copying does
   not impose a publication scanner. Review content before sharing; credentials
   outside the scanner's recognized patterns can pass undetected.

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
clone step, not archived data, so it must never appear in uploaded bundles
or incident diagnostics.
