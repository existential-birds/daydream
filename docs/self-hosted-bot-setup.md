# Self-hosted Daydream review bot — setup guide

This is the canonical, ordered guide for installing the Daydream review bot in
your own repository, running under **your** GitHub App. The maintainer hosts
nothing: the bot runs entirely in your repository's GitHub Actions, on your
credentials.

There are two equivalent paths and they install **exactly the same** workflow
files and **the same** secret/variable names — they cannot drift, because both
read the templates that ship inside the `daydream` package.

- **CLI (recommended, terminal):** one command automates every step GitHub
  permits — see [The one-command path](#the-one-command-path-daydream-setup).
- **Browser-only (no terminal):** follow the numbered steps below.

Either way, when you are done you can confirm the install with:

```bash
daydream setup /path/to/repo --repo OWNER/REPO --verify
```

`--verify` is read-only. It checks that the App is installed, all three secrets
and the bot-handle variable are present, the App's permissions are sufficient,
and the three workflow files exist — and it exits non-zero, naming each missing
piece, if anything is wrong. It works for installs done by **either** path.

---

## What a correct install contains

| Item | Name | Where |
|---|---|---|
| Secret | `DAYDREAM_APP_ID` | Actions secrets |
| Secret | `DAYDREAM_APP_PRIVATE_KEY` | Actions secrets |
| Secret | `ANTHROPIC_API_KEY` | Actions secrets |
| Variable | `DAYDREAM_BOT_HANDLE` | Actions variables |
| Workflow | `daydream-review.yml` | `.github/workflows/` |
| Workflow | `daydream-command.yml` | `.github/workflows/` |
| Workflow | `daydream-post.yml` | `.github/workflows/` |

These names are fixed by the workflows themselves. Do not rename them.

---

## The one-command path (`daydream setup`)

If you have a terminal with `gh` (GitHub CLI) authenticated:

```bash
daydream setup /path/to/repo --repo OWNER/REPO        # repository scope
daydream setup /path/to/repo --org   ORG-NAME         # organization scope
```

This registers a new GitHub App via GitHub's App-from-manifest flow with a
**localhost** callback (no maintainer server), captures the App ID and PEM,
deposits the three secrets and the `DAYDREAM_BOT_HANDLE` variable, and opens a
**pull request** adding the three workflow files. The bot goes live when you
merge that PR.

Two steps remain manual no matter the path, because GitHub forces them:

- **Clicking "Install"** on the freshly-created App — GitHub never lets an App
  grant itself access to your repositories. `daydream setup` opens the install
  page and waits for you.
- (Browser path only) **downloading the PEM** — see the limits note below.

Supply your Anthropic key via the `ANTHROPIC_API_KEY` environment variable, or
the command will prompt for it.

---

## The browser-only path

No terminal required. Each step is a GitHub web page.

### 1. Register the GitHub App

Go to **Settings → Developer settings → GitHub Apps → New GitHub App** (for an
organization, use the organization's Settings → Developer settings instead).

Fill in:

- **GitHub App name:** anything, e.g. `my-daydream-review`. This becomes the
  bot's identity; comments will be posted by `<name>[bot]`.
- **Homepage URL:** any URL (your repo is fine).
- **Webhook:** uncheck **Active** — the bot is driven by Actions, not webhooks.
- **Repository permissions** — set exactly:
  - **Pull requests: Read and write** — posting and minimizing review comments.
  - **Contents: Read-only** — required by the posting token's least-privilege trio.
  - **Metadata: Read-only** — implicit baseline.
  - **Actions: Read and write** — the command workflow mints a dispatch token
    with `actions: write`; a `workflow_dispatch` made with the built-in
    `GITHUB_TOKEN` never fires the downstream `workflow_run` trigger, so the
    post workflow would otherwise never run on the `@<bot> review` path.
- **Where can this GitHub App be installed?** — **Only on this account**.

Click **Create GitHub App**.

### 2. Note the App ID and download the PEM

On the App's settings page:

- Copy the **App ID** (a number near the top). This is your `DAYDREAM_APP_ID`.
- Under **Private keys**, click **Generate a private key**. GitHub **downloads**
  a `.pem` file. The full contents of that PEM file (including the
  `-----BEGIN ... PRIVATE KEY-----` / `-----END ... PRIVATE KEY-----` lines)
  are your `DAYDREAM_APP_PRIVATE_KEY`.

  > **Accepted limit:** the PEM **must be downloaded and pasted by hand**, even
  > on the CLI path. Auto-retrieving it would require a maintainer-hosted
  > redirect server, which is deliberately ruled out (the maintainer hosts
  > nothing). This manual PEM download is the irreducible floor of the
  > browser-only path.

### 3. Install the App on your repository (or org)

On the App's settings page, open **Install App** and install it on the target
repository (or, for an organization App, on the org / selected repos).

> **Accepted limit:** GitHub requires a human to click **Install** — an App can
> never grant itself access. This click is unavoidable on every path.

### 4. Add the three secrets

In the **target repository**, go to **Settings → Secrets and variables →
Actions → Secrets → New repository secret** and add:

- `DAYDREAM_APP_ID` — the App's numeric ID from step 2.
- `DAYDREAM_APP_PRIVATE_KEY` — the PEM contents from step 2, pasted verbatim.
- `ANTHROPIC_API_KEY` — your Anthropic API key (used only by the unprivileged
  review job).

> **Org scope:** if you used an organization App, add these as **organization**
> secrets instead (Org Settings → Secrets and variables → Actions). Org-scoped
> secrets reduce the whole install to **one setup per org** — every repo the App
> is installed on shares them.

### 5. Add the bot-handle variable

In the same area, switch to the **Variables** tab → **New repository variable**:

- `DAYDREAM_BOT_HANDLE` — the mention handle the command workflow matches,
  **without** the `@` (e.g. `daydream-review`, so `@daydream-review review`
  triggers it).
- _Optional:_ `DAYDREAM_AUTO_REVIEW` — set to `false` to disable auto-review on
  PR open; the `@<bot> review` command keeps working.

### 6. Add the three workflow files

Copy these three files into the target repository's `.github/workflows/`
directory (via the GitHub web editor's **Add file → Create new file**, or any
means you prefer):

- `daydream-review.yml`
- `daydream-command.yml`
- `daydream-post.yml`

The exact, current contents ship inside the `daydream` package under
`daydream/templates/workflows/`; the CLI path copies the same files. Their
roles, trigger matrix, and security model are documented in that directory's
[`README.md`](../daydream/templates/workflows/README.md).

### 7. Confirm

Open a pull request in the repo. With auto-review on, the bot reviews it and
posts inline comments as `<your-app>[bot]`; otherwise comment `@<bot> review`.
If you have a terminal available, run the `--verify` command shown at the top of
this guide to audit the install component-by-component.

---

## Accepted honest limits

This install is as automated as GitHub allows. Two steps are irreducibly manual
on every path, and one more on the browser path:

- **Clicking "Install"** (all paths) — GitHub requires a human to grant an App
  access to repositories.
- **Downloading and pasting the PEM** (browser path) — auto-retrieval needs a
  redirect server, which is ruled out.
- Using **organization-scoped** secrets reduces the work to **one setup per
  org** rather than per repository.

There is intentionally **no** maintainer-hosted template repository and no
hosted setup page: the in-package workflow templates and the localhost callback
make a hosted surface unnecessary, and a hosted surface would be an ongoing
maintenance and liability burden the whole project is designed to avoid.
