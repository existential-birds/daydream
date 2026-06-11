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

import html
import json
import threading
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from daydream import config, git_ops
from daydream.agent import console
from daydream.git_ops import GitError
from daydream.github_app import AppCredentials, GitHubAppError, exchange_manifest_code
from daydream.ui import print_info

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

    Args:
        action_url: GitHub's app-creation URL (org variant when org-scoped).
        manifest: The manifest payload to POST as the ``manifest`` field.

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

        Returns:
            The exchanged ``(credentials, slug)`` tuple.

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
            self._done.wait()
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
