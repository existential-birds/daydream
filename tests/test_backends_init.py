"""Backend protocol and factory contracts; event dataclass fields/defaults live in test_backends_events.py."""
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.backends import (
    AUDIT_ROOT_ISOLATION,
    AuditIsolationError,
    ClaudeBackend,
    ContinuationToken,
    ResultEvent,
    create_backend,
)
from daydream.backends.codex import CodexBackend
from daydream.backends.pi import PiBackend
from daydream.config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL
from tests.harness.claude_sdk import (
    MockAssistantMessage,
    MockResultMessage,
    MockTextBlock,
    patch_claude_sdk,
    scripted_client,
)


def test_continuation_token_fields() -> None:
    token = ContinuationToken(backend="codex", data={"thread_id": "abc"})
    assert token.backend == "codex"
    assert token.data == {"thread_id": "abc"}


@pytest.mark.parametrize(
    ("name", "cls", "model", "expected"),
    [
        ("claude", ClaudeBackend, None, DEFAULT_CLAUDE_MODEL),
        ("claude", ClaudeBackend, "sonnet", "sonnet"),
        ("codex", CodexBackend, None, DEFAULT_CODEX_MODEL),
        ("codex", CodexBackend, "o3-pro", "o3-pro"),
    ],
    ids=["claude-default", "claude-custom", "codex-default", "codex-custom"],
)
def test_create_backend_model_resolution(
    name: str, cls: type, model: str | None, expected: str
) -> None:
    backend = create_backend(name) if model is None else create_backend(name, model=model)
    assert isinstance(backend, cls)
    assert backend.model == expected

def test_create_backend_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend("invalid")

def test_pi_backend_concise_fix_prompts_true() -> None:
    backend = PiBackend(model="glm-5.2")
    assert backend.concise_fix_prompts is True

async def test_create_backend_claude_execute_accepts_agents_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    patch_claude_sdk(
        monkeypatch,
        scripted_client(
            [MockAssistantMessage(content=[MockTextBlock(text="OK")]), MockResultMessage()], captured=captured,
        ),
    )
    backend = create_backend("claude", model="test")
    events = [event async for event in backend.execute(Path("/tmp"), "test", agents=None)]
    assert any(isinstance(event, ResultEvent) for event in events)
    assert getattr(captured["options"], "agents", None) is None

def test_create_backend_forwards_reasoning_effort_to_every_driver() -> None:
    for name in ("claude", "codex", "pi"):
        backend: Any = create_backend(name, reasoning_effort="max")
        assert backend.reasoning_effort == "max", name

def test_create_backend_without_reasoning_effort_leaves_it_unset() -> None:
    for name in ("claude", "codex", "pi"):
        assert cast(Any, create_backend(name)).reasoning_effort is None, name

def test_create_backend_binds_claude_to_exact_audit_root(tmp_path: Path) -> None:
    root = tmp_path / "audit root"
    root.mkdir()
    outward = frozenset({root / "outside-link"})
    backend = create_backend("claude", model="test", audit_root=root, audit_outward_symlinks=outward)
    assert isinstance(backend, ClaudeBackend)
    assert backend.audit_root_isolation == AUDIT_ROOT_ISOLATION
    assert backend.audit_root == root.resolve(strict=True)
    assert backend.audit_outward_symlinks == outward

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

@pytest.mark.parametrize("name", ["claude", "codex", "pi", "osprey"])
def test_create_backend_without_audit_root_keeps_ordinary_backends(name: str) -> None:
    backend = create_backend(name, model="test")
    assert getattr(backend, "audit_root_isolation", None) is None
