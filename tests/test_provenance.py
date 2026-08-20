import pytest

from daydream.archive import provenance


def test_version_is_package_version():
    import daydream
    p = provenance.capture_executable_provenance()
    assert p.version == daydream.__version__


def test_container_digest_from_optin_env(monkeypatch):
    monkeypatch.setenv("DAYDREAM_IMAGE_DIGEST", "sha256:abc123")
    assert provenance.capture_executable_provenance().container_digest == "sha256:abc123"


def test_container_digest_unknown_when_unset(monkeypatch):
    monkeypatch.delenv("DAYDREAM_IMAGE_DIGEST", raising=False)
    assert provenance.capture_executable_provenance().container_digest == "unknown"


def test_commit_and_dirty_resolve_or_unknown():
    p = provenance.capture_executable_provenance()
    # The package dir is a git checkout in CI/dev; when git state is
    # unresolvable the field must be the explicit string "unknown", never
    # None and never a target-repo sha.
    assert p.commit in {"unknown"} or (p.commit and len(p.commit) == 40)
    assert p.dirty in (True, False, "unknown")


def test_install_source_is_known_or_unknown():
    p = provenance.capture_executable_provenance()
    assert p.install_source in {"editable", "git", "package", "unknown"}


def test_to_dict_never_omits_unknown():
    p = provenance.capture_executable_provenance()
    d = p.to_dict()
    assert set(d) == {"version", "install_source", "commit", "dirty", "container_digest"}