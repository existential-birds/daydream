"""Backend protocol and factory contracts; event dataclass fields/defaults live in test_backends_events.py."""
from pathlib import Path

import pytest

from daydream.backends import (
    AuditIsolationError,
    create_backend,
)
from tests.harness.table_cases import isolated_case


def test_create_backend_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid")


def test_create_backend_refuses_unsupported_audit_backend(
    tmp_path: Path,
    subtests: pytest.Subtests,
) -> None:
    table_root = tmp_path
    row_index = 0
    cases: list[str] = ["codex", "pi", "osprey"]
    for name in cases:
        row_index += 1
        with subtests.test(name=name):
            with isolated_case(table_root, str(row_index)) as case:
                tmp_path = case.tmp_path
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
