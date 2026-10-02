"""Local browser handshake for registering a GitHub App from its manifest."""

from __future__ import annotations

import html
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from daydream import config, git_ops
from daydream.github_app import AppCredentials, GitHubAppError, exchange_manifest_code

# Events consumed by the approval-gated workflows: trusted review commands and
# completed review runs. Credential-bearing review workflows are not subscribed
# to pull_request events.
_MANIFEST_EVENTS = ("issue_comment", "workflow_run")

_APP_NAME_DEFAULT = "Daydream Review Bot"
_GITHUB_NEW_APP_URL = "https://github.com/settings/apps/new"
_GITHUB_NEW_APP_ORG_URL = "https://github.com/organizations/{org}/settings/apps/new"


def _manifest_payload(*, redirect_url: str) -> dict[str, object]:
    """Build the private App manifest with the packaged permissions and events."""
    return {
        "name": _APP_NAME_DEFAULT,
        "url": "https://github.com/anthropics/daydream",
        "redirect_url": redirect_url,
        "public": False,
        "default_permissions": dict(config.APP_PERMISSIONS),
        "default_events": list(_MANIFEST_EVENTS),
    }


def _manifest_form_html(*, action_url: str, manifest: dict[str, object]) -> str:
    """Render a self-submitting manifest form, escaping the action and JSON."""
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
    """One-shot localhost form/callback server with a five-minute browser timeout."""

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
        """Exchange a nonempty callback code; a missing code means registration was cancelled."""
        if not code:
            raise GitHubAppError("App registration was cancelled")
        return exchange_manifest_code(self.repo_dir, code, auth=git_ops.INHERIT_GITHUB_AUTH)

    def serve(self) -> tuple[AppCredentials, str]:
        """Serve the localhost browser callback; cancellation or exchange failure raises GitHubAppError."""
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
                manifest = _manifest_payload(redirect_url=redirect_url)
                body = _manifest_form_html(action_url=listener._action_url(), manifest=manifest).encode("utf-8")
                self._respond_html(body)

            def _respond_html(self, body: bytes) -> None:
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
                self._respond_html(
                    f"<!DOCTYPE html><html><body><p>{html.escape(message)}</p></body></html>".encode()
                )
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
    """Register an App through a localhost browser handshake and return credentials/slug.

    GitHubAppError reports cancellation or code-exchange failure. ``org`` selects
    an organization-owned App; omission creates a personal-account App.
    """
    return _ManifestListener(repo_dir=repo_dir, org=org).serve()
