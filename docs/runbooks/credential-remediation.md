# Credential remediation

When credentials appear in captured evidence, record the affected run IDs,
observation IDs, repository identities, publication commits, and exposure dates.
Keep secret values out of incident reports and logs. Credential revocation and
rotation, and any remote dataset history rewrite, require the operator's explicit
decision; Daydream does not perform those actions automatically.

Capture and local diagnostic archives/dumps retain assembled evidence, including
credential-shaped strings and binary files. Existing upstream trajectory redaction
still applies. Review local evidence before sharing it. Diagnostic byte preservation
is independent of dataset publication.

The canonical JSONL publisher scans complete records and refuses blocking secret
findings with value-free diagnostics. Failed publication leaves the original local
records and retry queue available. A passing scan covers only recognized patterns;
operators remain responsible for deciding what evidence to share.

Use `daydream corpus dataset status --store RECORD_STORE --trajectory-hub-repo OWNER/REPO`
to inspect queued/published/failed counts. For previously published evidence, use
`dataset download` at an exact commit to identify affected records without changing
remote history. Immutable records must never be edited in place to erase evidence
or defeat integrity checks.

The operator should revoke exposed credentials with their provider, rotate dependent
secrets, and verify the old credentials no longer authorize access. Check dependent
jobs without printing secret values. If remote copies exist, decide separately how
to restrict access or remediate history. Merely removing a current file does not
remove earlier commits or caches.

For private repository harvesting, `DAYDREAM_GIT_TOKEN` travels through Git's child
process configuration, never a URL or command argument. GitHub license acquisition
uses `GITHUB_TOKEN` from the launching environment and pins the request to an exact
repository commit. Tokens must not become run or observation evidence.
