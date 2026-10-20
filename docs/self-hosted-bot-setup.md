# Self-hosted bot setup

Read before installing or changing the bot setup flow. With authenticated `gh`
and `ANTHROPIC_API_KEY` in the launching environment:

```bash
daydream setup /path/to/repo --repo OWNER/REPO
daydream setup /path/to/repo --repo OWNER/REPO --verify
```

Setup registers an App through a localhost manifest callback, deposits secrets,
and opens a workflow PR. A human must click **Install** to grant repository
access. The bot runs on the repository's own Actions and credentials.
`--verify` is read-only; it checks secret names, not their values. App installation
and permission checks need local `DAYDREAM_APP_ID` and raw-PEM
`DAYDREAM_APP_PRIVATE_KEY`; without them those checks are reported as skipped.

For manual installation, create an App with webhooks disabled; grant pull requests
and issues write, contents/metadata read, and Actions write. Download its PEM,
install it on the target, then add Actions secrets `DAYDREAM_APP_ID`,
`DAYDREAM_APP_PRIVATE_KEY`, and `ANTHROPIC_API_KEY`. Set `DAYDREAM_BOT_HANDLE`
without `@`. Organization-scoped secrets may be shared across selected repos.

Copy the three [packaged workflows](../daydream/templates/workflows/README.md)
and follow their pinning and approval contract. The manual
[single-workflow variant](../daydream/templates/workflows/single/README.md)
does not need Actions write; setup always emits the three-file variant.

A trusted OWNER/MEMBER/COLLABORATOR comments `@<bot> review`,
`@<bot> add sequence diagram`, or `@<bot> add flowchart`. Approval binds the exact
head SHA; a moved head needs a new approval. Parse only fixed case-sensitive tokens;
comment text must never enter a shell or model prompt. Out-of-scope issue filing
requires explicit repository opt-in (`scope_issue_filing`); preserve its default-off
policy and the existing fix-scope checks.
