"""Optional hosted Sprites adapter tests (live cases SKIPPED).

The Sprites adapter is a *reference hosted-executor integration* kept strictly
adapter-scoped: no Sprite name, SDK, or API type appears in the common models
of ``daydream.executors.contract`` (verified here). Live Sprite execution is a
separately credentialed, opt-in staging gate — every live-feeling case below
is skipped unless ``DAYDREAM_SPRITES_STAGING=1`` is set AND a connection is
provided, so the hermetic gate never touches a Sprite.

Because the adapter refuses to run without live staging, the hermetic cases
here assert adapter structure/safety (kind, capability admission, vendor-name
isolation, quarantined behaviour), not live lifecycle. Live lifecycle is
deferred to staging per contract.
"""

from __future__ import annotations

import pytest

from daydream.executors.contract import REQUIRED_CAPABILITIES, ExecutorError
from daydream.executors.sprites import SpritesExecutor, sprite_staging_enabled


def test_sprites_kind_and_capabilities_declared() -> None:
    adapter = SpritesExecutor(connection=None)
    assert adapter.kind == "sprites"
    assert adapter.adapter_version == 1
    # The adapter must not weaken the required capability set.
    assert REQUIRED_CAPABILITIES.issubset(adapter.capabilities)


def test_sprites_live_gate_defaults_off() -> None:
    assert sprite_staging_enabled() is False


@pytest.mark.asyncio
async def test_sprites_hermetic_call_refuses_without_staging() -> None:
    adapter = SpritesExecutor(connection=None)
    # Hermetic path: never touches a Sprite, refuses loudly instead.
    from daydream.executors.contract import ExecutionRef

    ref = ExecutionRef(executor_kind="sprites", adapter_version=1, opaque_handle="x", attempt_id="a")
    with pytest.raises(ExecutorError, match="staging"):
        await adapter.inspect(ref)
    with pytest.raises(ExecutorError, match="staging"):
        await adapter.start(
            __import__("daydream.executors.contract", fromlist=["ExecutorJob"]).ExecutorJob(attempt_id="a")
        )


_SPRITES_STAGING = __import__("os").environ.get("DAYDREAM_SPRITES_STAGING") == "1"


@pytest.mark.skipif(not _SPRITES_STAGING, reason="live Sprites staging gate")
@pytest.mark.asyncio
async def test_sprites_live_lifecycle_requires_explicit_connection() -> None:
    """Live staging only; never runs in the hermetic gate.

    With live staging on and a nominal connection, ``start`` must still refuse
    to fabricate an execution until a real Sprite client is wired up.
    """
    adapter = SpritesExecutor(connection=object())
    assert adapter.live is True  # staging is opted in, so the live flag is set
    from daydream.executors.contract import ExecutorJob

    with pytest.raises(ExecutorError):
        await adapter.start(ExecutorJob(attempt_id="live"))


def test_common_contract_has_no_sprite_names() -> None:
    """Vendor-name isolation: 'sprite' must not appear in common model field names."""
    import dataclasses

    from daydream.executors import contract

    for cls in (contract.ExecutionRef, contract.ExecutionSnapshot, contract.ArtifactEnvelope):
        for field in dataclasses.fields(cls):
            assert "sprite" not in field.name.lower(), f"common model leaked Sprite infra field '{field.name}'"
    # And the common contract module itself must not import the Sprite adapter.
    assert not hasattr(contract, "SpritesExecutor")
