"""The optional HF SDK is mocked only at the external network boundary."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from huggingface_hub.errors import HfHubHTTPError, RemoteEntryNotFoundError

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
    request = httpx2.Request("POST", "https://huggingface.co/api/datasets/private/commit/main")
    return HfHubHTTPError(text, response=httpx2.Response(status, request=request))


def test_private_revision_is_a_private_dataset_pin_and_bad_pins_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    assert client.private_revision(REPO) == HEAD
    assert api.calls == [("repo_info", {"repo_id": REPO, "repo_type": "dataset", "revision": "main"})]
    for visibility in (False, 1):
        client, api = install_sdk(monkeypatch)
        api.info.private = visibility
        with pytest.raises(HubError) as failure:
            client.private_revision(REPO)
        assert failure.value.code == "public_destination"
        assert "SECRET" not in str(failure.value)
    client, api = install_sdk(monkeypatch)
    api.info.sha = "A" * 40
    with pytest.raises(HubError) as failure:
        client.private_revision(REPO)
    assert failure.value.code == "invalid_revision"
    # Mutable or noncanonical pins never reach the network.
    client, api = install_sdk(monkeypatch)
    with pytest.raises(HubError, match="^invalid_revision$"):
        client.read_file(REPO, "manifest.json", "A" * 40)
    with pytest.raises(HubError, match="^invalid_revision$"):
        client.commit(REPO, {"manifest.json": b"{}\n"}, "main")
    assert api.calls == []


def test_commit_installs_complete_bytes_in_one_guarded_dataset_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.hub import HubConflict, HubError

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
    # Only a remote HTTP precondition failure is a commit conflict.
    api.error = http_error(412)
    with pytest.raises(HubConflict) as conflict:
        client.commit(REPO, {"manifest.json": b"{}\n"}, HEAD)
    assert "SECRET" not in str(conflict.value)
    assert conflict.value.__cause__ is None
    # Network failures never expose payloads or claim conflicts.
    api.error = http_error(409)
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
    # A local constructor 412 is not a remote commit conflict.
    client, api = install_sdk(monkeypatch)

    def reject_operation(**kwargs: Any) -> None:
        raise http_error(412)

    monkeypatch.setattr(client._hf, "CommitOperationAdd", reject_operation)
    with pytest.raises(HubError, match="^network_failed$"):
        client.commit(REPO, {"manifest.json": b"{}\n"}, HEAD)
    assert api.calls == []


def test_reads_exact_revision_and_only_regular_cache_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from daydream.hub import HubError

    client, api = install_sdk(monkeypatch)
    blob = tmp_path / "blob"
    blob.write_bytes(b'{"complete":true}\n')
    snapshot = tmp_path / "snapshot"
    snapshot.symlink_to(blob)
    api.download = snapshot
    assert client.read_file(REPO, "runs/one.jsonl", HEAD) == blob.read_bytes()
    assert api.calls == [("download", {"repo_id": REPO, "filename": "runs/one.jsonl",
                                     "repo_type": "dataset", "revision": HEAD})]
    directory, dangling = tmp_path / "directory", tmp_path / "dangling"
    directory.mkdir()
    dangling.symlink_to(tmp_path / "absent")
    for source in (directory, dangling):
        api.download = source
        with pytest.raises(HubError, match="^network_failed$"):
            client.read_file(REPO, "manifest.json", HEAD)
    request = httpx2.Request("GET", "https://huggingface.co/private/resolve/file")
    api.error = RemoteEntryNotFoundError("SECRET absent", response=httpx2.Response(404, request=request))
    assert client.read_file(REPO, "manifest.json", HEAD) is None


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
