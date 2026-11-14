"""Behavioral contracts for shared backend scripts and their real Backend.execute signatures.

Retain turn selection/repetition, streamed failures and closure, argument recording,
because consumer assertions depend on this harness. Host lifecycle coverage lives with its consumers.
"""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any

from daydream.backends import Backend

# The shared ``tests/harness`` backend fakes, by (label, module, class). Every
# fake the suite constructs directly or installs through ``create_backend`` is
# listed here so the protocol-parity check below covers the whole harness.
_SHARED_HARNESS_BACKENDS = (
    ("ScriptedBackend", "tests.harness.backend", "ScriptedBackend"),
    ("StubBackend", "tests.harness.stub_backend", "StubBackend"),
    # Most ImproveStubBackend specializations inherit execute through the
    # pre hook. Rendezvous wraps the stream lifetime for cancellation,
    # so its separate protocol declaration also needs signature coverage.
    ("ImproveStubBackend", "tests.harness.improve_backend", "ImproveStubBackend"),
    ("RendezvousImproveBackend", "tests.harness.improve_backend", "RendezvousImproveBackend"),
)


def test_shared_harness_backends_satisfy_the_backend_protocol_signature() -> None:
    """A harness fake that drops or renames a protocol parameter silently under-exercises it."""
    def _params(func: Any) -> dict[str, tuple[Any, Any]]:
        return {
            name: (param.kind, param.default)
            for name, param in inspect.signature(func).parameters.items()
            if name != "self"
        }

    expected = _params(Backend.execute)
    for label, module_name, class_name in _SHARED_HARNESS_BACKENDS:
        fake = getattr(import_module(module_name), class_name)
        declared = _params(fake.execute)
        # A fake that declares host tools takes the two keywords run_agent passes with a binding.
        capability = {"review_instructions", "host_tools"} if getattr(fake, "supports_host_tools", False) else set()
        missing = [name for name in (*expected, *capability) if name not in declared]
        extra = [name for name in declared if name not in expected and name not in capability]
        drift = [name for name in expected if name in declared and declared[name] != expected[name]]
        order = [name for name in declared if name in expected]
        assert order == [name for name in expected if name in declared], (
            f"{label}.execute declares the protocol parameters out of order"
        )
        assert not (missing or extra or drift), (
            f"{label}.execute drifted from daydream.backends.Backend.execute: "
            f"missing={missing} extra={extra} kind/default drift={drift}"
        )


