"""Backend protocol and factory contracts; event dataclass fields/defaults live in test_backends_events.py."""
from pathlib import Path

import pytest

from daydream.backends import (
    AuditIsolationError,
    create_backend,
)


def test_create_backend_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid")


@pytest.mark.parametrize("name", ["codex", "pi", "osprey"])
def test_create_backend_refuses_unsupported_audit_backend(name: str, tmp_path: Path) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    with pytest.raises(AuditIsolationError) as exc_info:
        create_backend(name, audit_root=root)
    assert exc_info.value.backend_name == name
    assert exc_info.value.reason == "unsupported_backend"

def test_create_backend_unknown_name_stays_a_value_error_with_audit_root(tmp_path: Path) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid", audit_root=root)
