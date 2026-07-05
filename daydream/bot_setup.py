"""Self-hosted review-bot setup orchestrator.

Takes an operator from nothing to a live, self-hosted review bot. The first
leg is the App-from-manifest localhost handshake: a one-shot ``http.server``
serves an auto-submitting HTML form that POSTs a GitHub App *manifest* to
GitHub's app-creation page, GitHub redirects the operator's browser back to a
``/callback`` route with a temporary conversion ``code``, and that code is
exchanged for the freshly created App's credentials and slug.

The live browser/manifest leg is isolated behind :class:`_ManifestListener` so
the code-exchange behavior is testable without real GitHub: drive
:meth:`_ManifestListener._handle_code` directly.
"""

from __future__ import annotations

import getpass
import html
import json
import os
import shutil
import sys
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from daydream import config, git_ops
from daydream.agent import console
from daydream.git_ops import GitError
from daydream.github_app import (
    APP_ID_ENV,
    APP_PRIVATE_KEY_ENV,
    AppCredentials,
    GitHubAppError,
    _scoped_gh_token,
    exchange_manifest_code,
    get_app_metadata,
    mint_jwt,
    resolve_credentials,
)
from daydream.templates import workflow_template_files
from daydream.ui import print_error, print_info, print_success, print_warning

_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
_SETUP_BRANCH = "daydream/setup-bot"

# Events the #147 workflow templates consume (daydream-review.yml →
# pull_request, daydream-command.yml → issue_comment, daydream-post.yml →
# workflow_run). Declared in the manifest so the App is subscribed to exactly
# what the shipped workflows listen for.
_MANIFEST_EVENTS = ("pull_request", "issue_comment", "workflow_run")

_APP_NAME_DEFAULT = "Daydream Review Bot"
_GITHUB_NEW_APP_URL = "https://github.com/settings/apps/new"
_GITHUB_NEW_APP_ORG_URL = "https://github.com/organizations/{org}/settings/apps/new"


def _manifest_payload(*, redirect_url: str, org: str | None) -> dict[str, object]:
    """Build the GitHub App manifest JSON for the from-manifest flow.

    Args:
        redirect_url: The ``http://localhost:<port>/callback`` URL GitHub
            redirects to with the conversion code.
        org: Organization login when registering an org-owned App, else None.

    Returns:
        The manifest dict: name, the homepage/redirect URLs, the
        :data:`config.APP_PERMISSIONS` default permissions, and the
        :data:`_MANIFEST_EVENTS` the shipped workflows consume.
    """
    return {
        "name": _APP_NAME_DEFAULT,
        "url": "https://github.com/anthropics/daydream",
        "redirect_url": redirect_url,
        "public": False,
        "default_permissions": dict(config.APP_PERMISSIONS),
        "default_events": list(_MANIFEST_EVENTS),
    }


def _manifest_form_html(*, action_url: str, manifest: dict[str, object]) -> str:
    """Render an auto-submitting HTML form POSTing the manifest to GitHub.

    Returns:
        A self-contained HTML page that submits on load.
    """
    manifest_json = html.escape(json.dumps(manifest), quote=True)
    return (
        "<!DOCTYPE html><html><head><title>Daydream setup</title></head>"
        "<body onload='document.forms[0].submit()'>"
        "<p>Redirecting to GitHub to create your review-bot App&hellip;</p>"
        f"<form action='{html.escape(action_url, quote=True)}' method='post'>"
        f"<input type='hidden' name='manifest' value='{manifest_json}'>"
        "<noscript><button type='submit'>Continue to GitHub</button></noscript>"
        "</form></body></html>"
    )


class _ManifestListener:
    """One-shot localhost listener for the App-from-manifest handshake.

    Serves ``/`` (the auto-submitting manifest form) and ``/callback`` (where
    GitHub redirects with the conversion ``code``). :meth:`serve` binds an
    ephemeral port, opens the browser at ``/``, and blocks until the callback
    arrives, then returns the exchanged credentials. :meth:`_handle_code` is
    the testable seam: it validates the code and performs the exchange,
    isolated from the blocking serve loop.

    Attributes:
        repo_dir: Working directory threaded to ``exchange_manifest_code``.
        org: Organization login when org-scoped, else None.
    """

    def __init__(self, *, repo_dir: Path, org: str | None) -> None:
        self.repo_dir = repo_dir
        self.org = org
        self._result: tuple[AppCredentials, str] | None = None
        self._error: GitHubAppError | None = None
        self._port: int = 0
        self._done = threading.Event()

    def _action_url(self) -> str:
        """GitHub's app-creation URL — org variant when an org is set."""
        if self.org:
            return _GITHUB_NEW_APP_ORG_URL.format(org=self.org)
        return _GITHUB_NEW_APP_URL

    def _handle_code(self, code: str | None) -> tuple[AppCredentials, str]:
        """Validate the callback code and exchange it for App credentials.

        Args:
            code: The ``code`` query param from GitHub's callback redirect.

        Returns:
            The ``(credentials, slug)`` tuple from
            :func:`exchange_manifest_code`.

        Raises:
            GitHubAppError: If the callback carried no code (the operator
                declined the App creation); never proceeds with empty creds.
        """
        if not code:
            raise GitHubAppError("App registration was cancelled")
        return exchange_manifest_code(self.repo_dir, code)

    def serve(self) -> tuple[AppCredentials, str]:
        """Bind a localhost port, open the browser, and block on the callback.

        Raises:
            GitHubAppError: If the operator declined or the exchange failed.
        """
        listener = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # noqa: A003 - silence stdlib access log
                return

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._serve_form()
                elif parsed.path == "/callback":
                    self._serve_callback(parsed.query)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serve_form(self) -> None:
                redirect_url = f"http://localhost:{listener._port}/callback"
                manifest = _manifest_payload(redirect_url=redirect_url, org=listener.org)
                body = _manifest_form_html(action_url=listener._action_url(), manifest=manifest).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def _serve_callback(self, query: str) -> None:
                code = parse_qs(query).get("code", [None])[0]
                try:
                    listener._result = listener._handle_code(code)
                except GitHubAppError as exc:
                    listener._error = exc
                message = (
                    "Daydream: App created. You can close this tab and return to the terminal."
                    if listener._error is None
                    else "Daydream: App registration was cancelled. Return to the terminal."
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(f"<!DOCTYPE html><html><body><p>{html.escape(message)}</p></body></html>".encode())
                listener._done.set()

        server = HTTPServer(("localhost", 0), _Handler)
        port = server.socket.getsockname()[1]
        self._port = port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            webbrowser.open(f"http://localhost:{port}/")
            self._done.wait(timeout=300)  # 5-minute bound; avoids indefinite hang if browser flow is abandoned
        finally:
            server.shutdown()
            thread.join(timeout=5)

        if self._error is not None:
            raise self._error
        if self._result is None:
            raise GitHubAppError("App registration was cancelled")
        return self._result


def register_app_via_manifest(repo_dir: Path, *, org: str | None = None) -> tuple[AppCredentials, str]:
    """Register a GitHub App via the from-manifest localhost browser flow.

    Binds a localhost ``http.server`` on an ephemeral port, opens the
    operator's browser to an auto-submitting form that POSTs the App manifest
    to GitHub's app-creation page (the org variant when *org* is set), blocks
    until GitHub redirects back to ``/callback?code=...``, and exchanges that
    code for the new App's credentials.

    Args:
        repo_dir: Working directory for the ``gh`` subprocess used by the
            manifest-code exchange.
        org: Organization login to register an org-owned App, else None for a
            personal-account App.

    Returns:
        A ``(credentials, slug)`` tuple: the new App's id/PEM and its slug.

    Raises:
        GitHubAppError: If the operator declined (callback without a code) or
            the manifest-code exchange failed.
    """
    return _ManifestListener(repo_dir=repo_dir, org=org).serve()


@dataclass(frozen=True)
class Scope:
    """Target scope for deposited Actions secrets/variables.

    Exactly one of *repo* or *org* must be set; the orchestrator threads it to
    the ``gh`` ``--repo``/``--org`` flags. Validated at construction so an
    ambiguous/empty scope can never reach a ``gh`` call.

    Attributes:
        repo: ``owner/repo`` slug for a repository-scoped deposit, else None.
        org: Organization login for an org-scoped deposit, else None.
    """

    repo: str | None = None
    org: str | None = None

    def __post_init__(self) -> None:
        if bool(self.repo) == bool(self.org):
            raise ValueError("Scope requires exactly one of repo='owner/repo' or org='name'")

    def _secret_kwargs(self) -> dict[str, str]:
        """Map the scope to the ``gh_secret_set``/``gh_variable_set`` keyword."""
        if self.repo:
            return {"repo_slug": self.repo}
        return {"org": self.org} if self.org else {}


def deposit_secrets(
    repo_dir: Path,
    creds: AppCredentials,
    *,
    anthropic_key: str,
    bot_handle: str,
    scope: Scope,
) -> None:
    """Deposit the App credentials and bot handle as Actions secrets/variables.

    Sets the three :data:`config.SETUP_SECRET_NAMES` secrets
    (``DAYDREAM_APP_ID`` = the numeric App id, ``DAYDREAM_APP_PRIVATE_KEY`` =
    the PEM, ``ANTHROPIC_API_KEY`` = the operator key) via
    :func:`git_ops.gh_secret_set` (value piped on stdin, never argv), and the
    :data:`config.BOT_HANDLE_VAR` Actions variable via
    :func:`git_ops.gh_variable_set`, threading *scope* to ``--repo``/``--org``.

    Idempotent: ``gh`` overwrites an existing secret/variable, so a re-run is
    safe. Pre-existing secrets are listed first and logged by name only —
    secret *values* are never logged. The PEM is passed on stdin so it cannot
    leak into process listings.

    Args:
        repo_dir: Working directory (ambient ``gh`` auth context).
        creds: The App credentials (``app_id``, ``private_key``).
        anthropic_key: The operator's ``ANTHROPIC_API_KEY`` value.
        bot_handle: The bot's login handle for ``DAYDREAM_BOT_HANDLE``.
        scope: Repo- or org-scoped target (exactly one).

    Raises:
        GitHubAppError: If any ``gh`` set call fails — named so no partial,
            silently-incomplete deposit is reported as success.
    """
    scope_kwargs = scope._secret_kwargs()
    secret_values = {
        "DAYDREAM_APP_ID": str(creds.app_id),
        "DAYDREAM_APP_PRIVATE_KEY": creds.private_key,
        "ANTHROPIC_API_KEY": anthropic_key,
    }

    try:
        existing = set(git_ops.gh_secret_list(repo_dir, **scope_kwargs))
    except GitError as exc:
        raise GitHubAppError(f"Could not list existing secrets: {exc}") from exc

    already = [name for name in config.SETUP_SECRET_NAMES if name in existing]
    if already:
        print_info(console, f"Overwriting existing secrets: {', '.join(already)}")

    for name in config.SETUP_SECRET_NAMES:
        try:
            git_ops.gh_secret_set(repo_dir, name, secret_values[name], **scope_kwargs)
        except GitError as exc:
            raise GitHubAppError(f"Failed to set secret {name}: {exc}") from exc

    try:
        git_ops.gh_variable_set(repo_dir, config.BOT_HANDLE_VAR, bot_handle, **scope_kwargs)
    except GitError as exc:
        raise GitHubAppError(f"Failed to set variable {config.BOT_HANDLE_VAR}: {exc}") from exc


# Returned by :func:`land_workflows` when all three workflow files already exist
# verbatim, so no branch/PR was opened. Deliberately not a URL (does not start
# with ``http``) so the caller can distinguish a no-op from a freshly opened PR.
WORKFLOWS_ALREADY_INSTALLED = "already-installed"

_WORKFLOWS_DIR = ".github/workflows"

_PR_TITLE = "Add Daydream review-bot workflows"
_PR_BODY = (
    "Adds the Daydream self-hosted review-bot GitHub Actions workflows.\n\n"
    "These workflows run code review in your own Actions runners under your "
    "GitHub App identity. Review the setup guide before merging:\n\n"
    "- Setup guide: `docs/self-hosted-bot-setup.md`\n"
    "- Security model: see the *Security model* section of that guide.\n\n"
    "Merging this PR makes the bot live on this repository."
)


def land_workflows(repo_dir: Path, *, branch: str) -> str:
    """Copy the packaged workflow templates and open a reviewable PR.

    Copies each :func:`daydream.templates.workflow_template_files` into
    ``<repo>/.github/workflows/``, then creates *branch*, commits **only** the
    three workflow files, pushes the branch, and opens a pull request against
    the repository's default branch. Never commits to or pushes the default
    branch — the workflows always land via a reviewable PR.

    Idempotent: a template whose target file already exists with identical
    content is skipped. When all three already match, no branch or PR is
    created and the :data:`WORKFLOWS_ALREADY_INSTALLED` sentinel is returned
    (distinguishable from a PR URL).

    Args:
        repo_dir: Repository working directory (ambient ``gh``/``git`` context).
        branch: The branch name to create the workflow files on.

    Returns:
        The opened PR's URL, or :data:`WORKFLOWS_ALREADY_INSTALLED` when every
        workflow file already exists verbatim.

    Raises:
        GitError: If a git/``gh`` operation (branch, commit, push, PR) fails.
        BranchNotFoundError: If the default branch cannot be resolved.
    """
    workflows_dir = repo_dir / _WORKFLOWS_DIR
    workflows_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for template in workflow_template_files():
        target = workflows_dir / template.name
        content = template.read_text()
        if target.exists() and target.read_text() == content:
            continue
        target.write_text(content)
        copied.append(Path(_WORKFLOWS_DIR) / template.name)

    if not copied:
        print_info(console, "Workflow files already present; skipping branch/PR.")
        return WORKFLOWS_ALREADY_INSTALLED

    base = git_ops.default_branch(repo_dir)
    if git_ops.branch_exists(repo_dir, branch):
        git_ops.checkout_branch(repo_dir, branch)
    else:
        git_ops.create_branch(repo_dir, branch)
    git_ops.commit_paths(repo_dir, copied, _PR_TITLE)
    git_ops.push_branch(repo_dir, branch)
    existing_prs = git_ops.gh_pr_list_for_branch(repo_dir, branch)
    if existing_prs:
        return existing_prs[0]["url"]
    return git_ops.gh_pr_create(repo_dir, head=branch, base=base, title=_PR_TITLE, body=_PR_BODY)


@dataclass(frozen=True)
class Check:
    """One verify-doctor check outcome.

    Attributes:
        name: Short check identifier (e.g. ``"secrets"``).
        passed: True when the check passed (a skipped optional check is
            reported as passed with an explanatory *detail*).
        detail: Human-readable result. On failure it names the exact missing
            secret/variable/file/permission **and** the remediation.
        required: When False, the check is informational and does not affect
            :attr:`VerifyResult.ok` (used for checks skipped because no local
            App credentials are available to read App-level state).
    """

    name: str
    passed: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class VerifyResult:
    """Aggregate outcome of :func:`run_verify`.

    Attributes:
        checks: The ordered per-check results.
        ok: True iff every *required* check passed.
    """

    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        """True when every required check passed (skipped optionals ignored)."""
        return all(c.passed for c in self.checks if c.required)


def _owner_of(scope: Scope) -> str:
    """Resolve the target owner login from a repo/org scope."""
    if scope.org:
        return scope.org
    # scope.repo is "owner/repo" (Scope validates exactly one is set).
    parsed = git_ops.split_owner_repo(scope.repo or "")
    return parsed[0] if parsed is not None else ""


def _bot_handle_for(slug: str | None, scope: Scope) -> str:
    """Return the bot handle to deposit: the App slug when known, else the scope owner."""
    return slug if slug else _owner_of(scope)


def _installed_owner_logins(repo_dir: Path, jwt_token: str) -> set[str | None]:
    """Return the set of account logins from ``GET /app/installations``.

    Callers are responsible for catching :class:`GitError` and converting it
    to their own error type or return value as appropriate.
    """
    bearer = {"Authorization": f"Bearer {jwt_token}"}
    with _scoped_gh_token(jwt_token):
        installations = git_ops.gh_api(repo_dir, "/app/installations", headers=bearer, idempotent=True)
    return {
        (inst.get("account") or {}).get("login")
        for inst in (installations if isinstance(installations, list) else [])
    }


def _check_app_installed(repo_dir: Path, scope: Scope, creds: AppCredentials | None) -> Check:
    """Check (1): the App is installed on the target owner.

    Reads ``GET /app/installations`` under an App JWT and confirms the target
    owner appears. Skipped (reported as a passing, non-required note) when no
    local App credentials are available to authenticate the App-level read.
    """
    if creds is None:
        return Check(
            name="app_installed",
            passed=True,
            detail=(
                "Skipped: no local App credentials "
                f"({APP_ID_ENV}/{APP_PRIVATE_KEY_ENV}) to read installations. "
                "Export them to verify the App is installed on the target."
            ),
            required=False,
        )
    owner = _owner_of(scope)
    jwt_token = mint_jwt(creds.app_id, creds.private_key)
    try:
        logins = _installed_owner_logins(repo_dir, jwt_token)
    except GitError as exc:
        return Check(
            name="app_installed",
            passed=False,
            detail=f"Could not read App installations: {exc}. Confirm the App credentials are valid.",
        )
    if owner in logins:
        return Check(name="app_installed", passed=True, detail=f"App is installed on '{owner}'.")
    return Check(
        name="app_installed",
        passed=False,
        detail=(
            f"App is not installed on '{owner}'. Install it via "
            f"https://github.com/settings/installations (or the org's settings)."
        ),
    )


def _check_secrets_and_var(repo_dir: Path, scope: Scope) -> Check:
    """Check (2): all required secrets + the bot-handle variable are present."""
    scope_kwargs = scope._secret_kwargs()
    try:
        secrets = set(git_ops.gh_secret_list(repo_dir, **scope_kwargs))
        variables = set(git_ops.gh_variable_list(repo_dir, **scope_kwargs))
    except GitError as exc:
        return Check(
            name="secrets",
            passed=False,
            detail=f"Could not list secrets/variables: {exc}. Confirm gh is authenticated for the target.",
        )

    missing_secrets = [name for name in config.SETUP_SECRET_NAMES if name not in secrets]
    var_missing = config.BOT_HANDLE_VAR not in variables
    if not missing_secrets and not var_missing:
        return Check(
            name="secrets",
            passed=True,
            detail=f"All {len(config.SETUP_SECRET_NAMES)} secrets and {config.BOT_HANDLE_VAR} are set.",
        )

    parts: list[str] = []
    if missing_secrets:
        parts.append(
            "Missing secret(s): "
            + ", ".join(missing_secrets)
            + f" — set via `gh secret set <NAME> {_scope_flag(scope)}`."
        )
    if var_missing:
        parts.append(
            f"Missing variable {config.BOT_HANDLE_VAR} — "
            f"set via `gh variable set {config.BOT_HANDLE_VAR} {_scope_flag(scope)}`."
        )
    return Check(name="secrets", passed=False, detail=" ".join(parts))


def _check_permissions(repo_dir: Path, creds: AppCredentials | None) -> Check:
    """Check (3): App permissions are a superset of the required set."""
    if creds is None:
        return Check(
            name="permissions",
            passed=True,
            detail=(
                "Skipped: no local App credentials "
                f"({APP_ID_ENV}/{APP_PRIVATE_KEY_ENV}) to read App permissions."
            ),
            required=False,
        )
    try:
        meta = get_app_metadata(repo_dir, creds.app_id, creds.private_key)
    except GitHubAppError as exc:
        return Check(
            name="permissions",
            passed=False,
            detail=f"Could not read App permissions: {exc}. Confirm the App credentials are valid.",
        )
    granted = meta.get("permissions") or {}
    _LEVELS = {"none": 0, "read": 1, "write": 2, "admin": 3}
    missing = [
        name
        for name in config.APP_PERMISSIONS
        if _LEVELS.get(granted.get(name, "none"), 0)
        < _LEVELS.get(config.APP_PERMISSIONS[name], 0)
    ]
    if not missing:
        return Check(name="permissions", passed=True, detail="App grants all required permissions.")
    detail = ", ".join(f"{name}={config.APP_PERMISSIONS[name]}" for name in missing)
    return Check(
        name="permissions",
        passed=False,
        detail=(
            f"App is missing required permission(s): {detail}. "
            "Update the App's permissions in its GitHub settings and re-accept the install."
        ),
    )


def _check_workflows(repo_dir: Path) -> Check:
    """Check (4): the three workflow files exist on the default branch."""
    try:
        base = git_ops.default_branch(repo_dir)
    except git_ops.BranchNotFoundError as exc:
        return Check(
            name="workflows",
            passed=False,
            detail=f"Could not resolve the default branch: {exc}.",
        )

    missing: list[str] = []
    for template in workflow_template_files():
        path = f"{_WORKFLOWS_DIR}/{template.name}"
        try:
            git_ops.show(repo_dir, f"origin/{base}", path)
        except GitError:
            missing.append(path)
    if not missing:
        return Check(name="workflows", passed=True, detail=f"All workflow files present on '{base}'.")
    return Check(
        name="workflows",
        passed=False,
        detail=(
            f"Missing workflow file(s) on '{base}': "
            + ", ".join(missing)
            + " — run `daydream setup` (or merge the setup PR) to land them."
        ),
    )


def _scope_flag(scope: Scope) -> str:
    """Render the ``gh`` scope flag for a remediation hint."""
    if scope.org:
        return f"--org {scope.org}"
    return f"--repo {scope.repo}"


def run_verify(repo_dir: Path, *, scope: Scope) -> VerifyResult:
    """Run the read-only setup doctor against the target scope.

    Runs four checks and aggregates them into a :class:`VerifyResult`:

    1. **App installed** — the App appears in ``GET /app/installations`` for the
       target owner (skipped, non-required, when no local App credentials are
       available to authenticate the read).
    2. **Secrets & variable** — every :data:`config.SETUP_SECRET_NAMES` secret
       and the :data:`config.BOT_HANDLE_VAR` variable exist at the scope.
    3. **Permissions** — the App grants a superset of
       :data:`config.APP_PERMISSIONS` (skipped, non-required, without creds).
    4. **Workflows** — the three packaged workflow files exist on the default
       branch.

    This is strictly read-only: it never sets a secret, variable, or file. Each
    failed required check's ``detail`` names the exact missing element and the
    remediation; :attr:`VerifyResult.ok` is True iff every required check passed.
    """
    creds = resolve_credentials()
    checks = (
        _check_app_installed(repo_dir, scope, creds),
        _check_secrets_and_var(repo_dir, scope),
        _check_permissions(repo_dir, creds),
        _check_workflows(repo_dir),
    )
    return VerifyResult(checks=checks)


def print_verify_result(result: VerifyResult) -> None:
    """Render a :class:`VerifyResult` to the console (one line per check).

    Passed checks print as a success line; failed required checks print as an
    error naming the missing element and remediation; skipped optional checks
    print as a dim/info note. The CLI handler maps :attr:`VerifyResult.ok` to
    the process exit code separately.
    """
    for check in result.checks:
        if check.passed and check.required:
            print_success(console, f"[{check.name}] {check.detail}")
        elif check.passed:
            print_info(console, f"[{check.name}] {check.detail}")
        else:
            print_error(console, f"[{check.name}] check failed", check.detail)
    if result.ok:
        print_success(console, "All required checks passed; the bot is configured.")
    else:
        print_warning(console, "Setup is incomplete; address the failed checks above.")


def _confirm_installation(repo_dir: Path, scope: Scope, creds: AppCredentials) -> bool:
    """Return True once the App is installed on the target owner.

    Re-checks ``GET /app/installations`` under the App JWT. If the owner is not
    yet present, prints the install URL, blocks on the manual **Install** click
    via the :func:`_wait_for_install_click` seam, then re-checks once more.

    The installations read is attempted first, so an install that already
    happened (e.g. an org-wide App) is auto-satisfied without prompting — this
    is what lets the real-path test drive the full-auto flow without blocking
    on stdin.

    Returns:
        True if the owner appears in the installations after at most one wait.
    """
    owner = _owner_of(scope)
    if _owner_installed(repo_dir, owner, creds):
        return True

    install_url = (
        f"https://github.com/organizations/{scope.org}/settings/installations"
        if scope.org
        else "https://github.com/settings/installations"
    )
    print_info(
        console,
        f"Install the new App on '{owner}' to finish: {install_url}",
    )
    _wait_for_install_click()
    return _owner_installed(repo_dir, owner, creds)


def _owner_installed(repo_dir: Path, owner: str, creds: AppCredentials) -> bool:
    """True when *owner* appears in ``GET /app/installations`` under the App JWT.

    Raises:
        GitHubAppError: If the installations read fails (so a transport error is
            never silently read as "not installed").
    """
    jwt_token = mint_jwt(creds.app_id, creds.private_key)
    try:
        logins = _installed_owner_logins(repo_dir, jwt_token)
    except GitError as exc:
        raise GitHubAppError(f"Could not read App installations: {exc}") from exc
    return owner in logins


def _wait_for_install_click() -> None:
    """Block until the operator confirms the manual GitHub App Install click.

    Isolated as a module-level seam so the real-path test can drive
    :func:`run_setup` without blocking on real stdin (the happy path
    auto-satisfies the installations re-check before this is reached).
    """
    input("Press Enter once you have installed the App on the target...")


def _prompt_for_anthropic_key() -> str | None:
    """Prompt the operator for their ``ANTHROPIC_API_KEY`` when it is not preset.

    The key is bound for deposit as an Actions secret, so asking for it inline
    keeps the "one command" promise instead of stranding a new operator on a
    pre-flight error. Input is read with :func:`getpass.getpass` so the secret
    is never echoed to the terminal or shell history.

    Isolated as a module-level seam so the real-path test can drive
    :func:`run_setup` without blocking on real stdin.

    Returns:
        The entered key (stripped, non-empty), or ``None`` when stdin is not a
        TTY (unattended/CI run) or the operator cancels — so the caller falls
        back to a clean pre-flight error rather than blocking or guessing.
    """
    if not sys.stdin.isatty():
        return None
    print_info(
        console,
        f"{_ANTHROPIC_KEY_ENV} is not set. Enter it now — it will be stored as an Actions secret.",
    )
    try:
        entered = getpass.getpass(f"{_ANTHROPIC_KEY_ENV} (input hidden): ")
    except (EOFError, KeyboardInterrupt):
        return None
    return entered.strip() or None


def run_setup(
    target_dir: Path,
    *,
    scope: Scope,
    force: bool,
    anthropic_key: str | None,
) -> int:
    """Take an operator from nothing to a landed self-hosted review bot.

    Orchestrates the full-auto path:

    1. **Pre-flight** — ``gh`` is on ``PATH`` and the ``ANTHROPIC_API_KEY`` is
       resolvable (explicit arg, environment, or an interactive hidden prompt;
       a non-interactive stdin without the key fails cleanly here).
    2. **Register** — unless the credentials are already deposited (idempotency)
       and *force* is not set, register the App via
       :func:`register_app_via_manifest` (the localhost browser handshake).
    3. **Install** — confirm the App is installed on the target owner, blocking
       on the manual Install click when it is not yet present.
    4. **Deposit** — :func:`deposit_secrets` writes the three secrets + handle.
    5. **Land** — :func:`land_workflows` opens a reviewable PR with the three
       workflow files (never pushing to the default branch).

    Args:
        target_dir: Repository working directory.
        scope: Repo- or org-scoped deposit target (exactly one).
        force: Re-register the App even if credentials are already deposited.
        anthropic_key: The ``ANTHROPIC_API_KEY`` value, or None to read it from
            the environment.

    Returns:
        ``0`` on success (including a no-op when workflows already exist); ``1``
        on a recoverable failure surfaced to the operator.

    Raises:
        GitHubAppError: Propagated from registration/deposit on hard failure.
        GitError: Propagated from the git/``gh`` landing operations.
    """
    if shutil.which("gh") is None:
        print_error(
            console,
            "gh not found",
            "The GitHub CLI (`gh`) must be installed and authenticated. See https://cli.github.com/.",
        )
        return 1

    resolved_key = anthropic_key or os.environ.get(_ANTHROPIC_KEY_ENV)
    if not resolved_key:
        resolved_key = _prompt_for_anthropic_key()
    if not resolved_key:
        print_error(
            console,
            "ANTHROPIC_API_KEY missing",
            f"Set {_ANTHROPIC_KEY_ENV} in the environment (or pass it) before running setup, "
            "or run interactively to be prompted for it.",
        )
        return 1

    already = set(git_ops.gh_secret_list(target_dir, **scope._secret_kwargs()))
    creds_present = all(name in already for name in config.SETUP_SECRET_NAMES)

    if creds_present and not force:
        print_info(
            console,
            "App credentials already deposited; skipping registration (use --force to re-register).",
        )
        creds = resolve_credentials()
        if creds is None:
            print_error(
                console,
                "Cannot reuse credentials",
                (
                    f"Secrets exist at the target but {APP_ID_ENV}/{APP_PRIVATE_KEY_ENV} are not "
                    "set locally to confirm the install. Re-run with --force, or export them."
                ),
            )
            return 1
        try:
            slug = get_app_metadata(target_dir, creds.app_id, creds.private_key).get("slug") or None
        except GitHubAppError:
            slug = None
    else:
        creds, slug = register_app_via_manifest(target_dir, org=scope.org)
        print_success(console, f"Registered GitHub App '{slug}'.")

    if not _confirm_installation(target_dir, scope, creds):
        print_error(
            console,
            "App not installed",
            f"The App is still not installed on '{_owner_of(scope)}'. Install it, then re-run setup.",
        )
        return 1

    bot_handle = _bot_handle_for(slug, scope)
    deposit_secrets(
        target_dir,
        creds,
        anthropic_key=resolved_key,
        bot_handle=bot_handle,
        scope=scope,
    )
    print_success(console, "Deposited App credentials and bot handle as Actions secrets/variables.")

    pr_url = land_workflows(target_dir, branch=_SETUP_BRANCH)
    if pr_url == WORKFLOWS_ALREADY_INSTALLED:
        print_info(console, "Workflow files already present on this repository; nothing to land.")
    else:
        print_success(console, f"Opened workflow PR: {pr_url}")
        print_info(console, "Merge the PR to go live — the bot reviews PRs once it lands on the default branch.")
    return 0
