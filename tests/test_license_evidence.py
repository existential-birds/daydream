"""License acquisition pins the Git commit and confines credentials to HTTP headers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from email.message import Message
from typing import Any

import pytest

from daydream.training.license_evidence import _GITHUB_API, GithubLicenseResolver, LicenseEvidenceError


def test_github_resolver_reads_token_from_env_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use GitHub's real response shapes: branch head SHA and a distinct license blob SHA."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secrettokenvalue")
    resolver = GithubLicenseResolver()

    captured: dict[str, Any] = {}
    commit = "b" * 40
    responses: dict[str, dict[str, Any]] = {
        f"{_GITHUB_API}/repos/acme/widget": {"full_name": "acme/widget", "default_branch": "main"},
        f"{_GITHUB_API}/repos/acme/widget/branches/main": {"name": "main", "commit": {"sha": commit}},
        f"{_GITHUB_API}/repos/acme/widget/license?ref={commit}": {
            "name": "LICENSE", "sha": "c" * 40, "license": {"spdx_id": "MIT"},
        },
    }

    def fake_urlopen(req: Any, **kwargs: Any) -> Any:
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())

        class R:
            def __enter__(self) -> Any:
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(responses[req.full_url]).encode()

        return R()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    evidence = resolver.resolve("acme/widget", None)
    assert evidence is not None and evidence.spdx_id == "MIT"
    assert "ghp_secrettokenvalue" not in captured["url"]
    assert captured["headers"]["Authorization"] == "Bearer ghp_secrettokenvalue"
    assert captured["url"] == f"{_GITHUB_API}/repos/acme/widget/license?ref={commit}"
    assert evidence.source == f"github:acme/widget@{'b' * 40}"


@pytest.mark.parametrize("status", [404, 401, 403, 500])
def test_only_confirmed_license_absence_returns_none(
    monkeypatch: pytest.MonkeyPatch, status: int,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_private_failure_payload")
    calls: list[str] = []

    def failed_http(request: Any, **kwargs: Any) -> Any:
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, status, "SECRET response body", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", failed_http)
    resolver = GithubLicenseResolver()
    if status == 404:
        assert resolver.resolve("acme/widget", "b" * 40) is None
    else:
        with pytest.raises(LicenseEvidenceError, match="^license_source_request_failed$") as failure:
            resolver.resolve("acme/widget", "b" * 40)
        assert failure.value.__cause__ is None
    assert calls == [f"{_GITHUB_API}/repos/acme/widget/license?ref={'b' * 40}"]


@pytest.mark.parametrize("failure", ["missing_token", "moving_revision", "malformed_revision"])
def test_invalid_acquisition_never_reaches_github(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    if failure == "missing_token":
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_offline_fixture")

    def forbidden_http(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("License resolution must reject unpinned or unauthorized acquisition before HTTP")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden_http)
    revision = {"missing_token": "b" * 40, "moving_revision": "main", "malformed_revision": "b" * 39}[failure]
    expected = "GITHUB_TOKEN is not set" if failure == "missing_token" else "^invalid_repository_revision$"
    with pytest.raises(LicenseEvidenceError, match=expected):
        GithubLicenseResolver().resolve("acme/widget", revision)


def test_rate_limited_license_requests_retry_the_same_pin_with_bounded_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_offline_fixture")
    calls: list[str] = []
    delays: list[float] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def read(self) -> bytes:
            return b'{"license":{"spdx_id":"MIT"},"sha":"license-blob"}'

    def retrying_http(request: Any, **kwargs: Any) -> Any:
        calls.append(request.full_url)
        if len(calls) <= 2:
            headers = Message()
            headers["X-RateLimit-Remaining"] = "0"
            headers["Retry-After"] = "99999" if len(calls) == 1 else "not-a-number"
            raise urllib.error.HTTPError(request.full_url, 403, "SECRET response body", headers, None)
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", retrying_http)
    monkeypatch.setattr("daydream.training.license_evidence.time.sleep", delays.append)
    evidence = GithubLicenseResolver().resolve("acme/widget", "B" * 40)
    assert evidence is not None and evidence.repo_commit == "b" * 40
    assert delays == [120.0, 4.0]
    assert calls == [f"{_GITHUB_API}/repos/acme/widget/license?ref={'b' * 40}"] * 3
