"""The optional HF SDK is mocked only at the external network boundary."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RemoteEntryNotFoundError

REPO = "test-user/private-trajectories"
HEAD = "a" * 40
NEXT = "b" * 40


class FakeApi:
    def __init__(self) -> None:
        self.info = SimpleNamespace(private=True, sha=HEAD)
        self.error: Exception | None = None
        self.result = SimpleNamespace(oid=NEXT)
        self.download: Path | None = None
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _check(self, method: str, kwargs: dict[str, Any]) -> None:
        self.calls.append((method, kwargs))
        if self.error is not None:
            raise self.error

    def repo_info(self, **kwargs: Any) -> SimpleNamespace:
        self._check("repo_info", kwargs)
        return self.info

    def create_commit(self, **kwargs: Any) -> SimpleNamespace:
        self._check("create_commit", kwargs)
        return self.result

    def hf_hub_download(self, **kwargs: Any) -> str:
        self._check("download", kwargs)
        assert self.download is not None
        return str(self.download)


def install_sdk(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, FakeApi]:
    from huggingface_hub import errors

    from daydream import hub as boundary

    api = FakeApi()
    sdk = SimpleNamespace(HfApi=lambda **kwargs: api, CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(boundary, "_load_hf", lambda: (sdk, errors))
    return boundary.HfDatasetHub(), api


def http_error(status: int, text: str = "SECRET payload failure") -> HfHubHTTPError:
    request = httpx.Request("POST", "https://huggingface.co/api/datasets/private/commit/main")
    return HfHubHTTPError(text, response=httpx.Response(status, request=request))


def test_private_revision_is_a_private_dataset_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    client, api = install_sdk(monkeypatch)
    assert client.private_revision(REPO) == HEAD
    assert api.calls == [("repo_info", {"repo_id": REPO, "repo_type": "dataset", "revision": "main"})]


@pytest.mark.parametrize("visibility", [False, None, "private", 1])
def test_public_or_unconfirmed_destinations_are_rejected(
    monkeypatch: pytest.MonkeyPatch, visibility: object,
) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    api.info.private = visibility
    with pytest.raises(HubError) as failure:
        client.private_revision(REPO)
    assert failure.value.code == "public_destination"
    assert "SECRET" not in str(failure.value)


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, None])
def test_invalid_repository_pins_are_rejected(monkeypatch: pytest.MonkeyPatch, revision: object) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    api.info.sha = revision
    with pytest.raises(HubError) as failure:
        client.private_revision(REPO)
    assert failure.value.code == "invalid_revision"


def test_commit_installs_complete_bytes_in_one_guarded_dataset_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    client, api = install_sdk(monkeypatch)
    files: Mapping[str, bytes] = {"runs/one.jsonl": b'{"run":1}\n', "manifest.json": b'{"count":1}\n'}
    assert client.commit(REPO, files, HEAD) == NEXT
    call = api.calls[0][1]
    assert call["repo_id"] == REPO
    assert call["repo_type"] == "dataset"
    assert call["parent_commit"] == HEAD
    assert call["revision"] == "main"
    assert call["create_pr"] is False
    assert call["run_as_future"] is False
    assert {op.path_in_repo: op.path_or_fileobj for op in call["operations"]} == files
    assert len(api.calls) == 1


@pytest.mark.parametrize("status", [401, 403, 404, 409, 500, 503])
def test_network_failures_never_expose_payloads_or_claim_conflicts(
    monkeypatch: pytest.MonkeyPatch, status: int,
) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    api.error = http_error(status)
    actions: list[Callable[[], Any]] = [
        lambda: client.private_revision(REPO), lambda: client.read_file(REPO, "manifest.json", HEAD),
        lambda: client.commit(REPO, {"manifest.json": b"{}\n"}, HEAD),
    ]
    for action in actions:
        with pytest.raises(HubError) as failure:
            action()
        assert failure.value.code == "network_failed"
        assert str(failure.value) == "network_failed"
        assert failure.value.__cause__ is None


def test_only_http_precondition_failed_is_a_commit_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.hub import HubConflict

    client, api = install_sdk(monkeypatch)
    api.error = http_error(412)
    with pytest.raises(HubConflict) as failure:
        client.commit(REPO, {"manifest.json": b"{}\n"}, HEAD)
    assert "SECRET" not in str(failure.value)
    assert failure.value.__cause__ is None


def test_reads_exact_revision_and_resolves_normal_hf_cache_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    client, api = install_sdk(monkeypatch)
    blob = tmp_path / "blob"
    blob.write_bytes(b'{"complete":true}\n')
    snapshot = tmp_path / "snapshot"
    snapshot.symlink_to(blob)
    api.download = snapshot
    assert client.read_file(REPO, "runs/one.jsonl", HEAD) == blob.read_bytes()
    assert api.calls == [("download", {"repo_id": REPO, "filename": "runs/one.jsonl",
                                     "repo_type": "dataset", "revision": HEAD})]


def test_only_missing_remote_entry_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    client, api = install_sdk(monkeypatch)
    request = httpx.Request("GET", "https://huggingface.co/private/resolve/file")
    api.error = RemoteEntryNotFoundError("SECRET absent", response=httpx.Response(404, request=request))
    assert client.read_file(REPO, "manifest.json", HEAD) is None


def test_local_cache_miss_never_looks_like_an_absent_remote_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    api.error = LocalEntryNotFoundError("SECRET offline cache miss")
    with pytest.raises(HubError, match="^network_failed$"):
        client.read_file(REPO, "manifest.json", HEAD)


@pytest.mark.parametrize("operation", ["read", "commit"])
@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40])
def test_mutable_or_noncanonical_pins_never_reach_the_network(
    monkeypatch: pytest.MonkeyPatch, operation: str, revision: str,
) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    with pytest.raises(HubError, match="^invalid_revision$"):
        if operation == "read":
            client.read_file(REPO, "manifest.json", revision)
        else:
            client.commit(REPO, {"manifest.json": b"{}\n"}, revision)
    assert api.calls == []


def test_invalid_commit_response_never_acknowledges_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    api.result.oid = "SECRET not a commit"
    with pytest.raises(HubError, match="^invalid_revision$"):
        client.commit(REPO, {"manifest.json": b"{}\n"}, HEAD)


def test_constructor_412_is_not_a_remote_commit_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)

    def reject_operation(**kwargs: Any) -> None:
        raise http_error(412)

    monkeypatch.setattr(client._hf, "CommitOperationAdd", reject_operation)
    with pytest.raises(HubError, match="^network_failed$"):
        client.commit(REPO, {"manifest.json": b"{}\n"}, HEAD)
    assert api.calls == []


@pytest.mark.parametrize("kind", ["missing", "directory", "dangling_symlink"])
def test_unreadable_or_nonregular_cache_results_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str,
) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    source = tmp_path / "source"
    if kind == "directory":
        source.mkdir()
    elif kind == "dangling_symlink":
        source.symlink_to(tmp_path / "absent")
    api.download = source
    with pytest.raises(HubError, match="^network_failed$"):
        client.read_file(REPO, "manifest.json", HEAD)


def test_missing_extra_reports_a_sanitized_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.hub import HfDatasetHub, HubError

    real_import = builtins.__import__

    def without_hf(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "huggingface_hub":
            raise ImportError("SECRET missing package detail")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_hf)
    with pytest.raises(HubError, match="^missing_hub_dependency$"):
        HfDatasetHub()
