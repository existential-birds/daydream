"""Round-2 deep-review regressions for daydream issue #1445.

Pins the two dispositions this round landed by hand:

* ``test_strict_gate_has_a_single_spelling``   -- merged finding item:1 (LOW),
  which the fix agent left unresolved: the four out-of-module consumers still
  reached ``json_utils.validates_schema`` through the private
  ``daydream.agent._validates_schema`` alias whose own comment claimed every
  in-repo consumer used the public helper.
* ``test_pi_backend_honors_structured_output_opt_out`` -- the python stack's
  finding, which the cross-stack arbiter never merged: ``PiBackend`` applied
  schema-aware selection unconditionally, so the
  ``validate_structured_output=False`` exemption documented in ``agent.py``'s
  host fallback did not hold on the pi path.
"""

import json
import pathlib

import pytest

import daydream.agent as agent_module
import daydream.backends.pi as pi_module
import daydream.deep.adjudication_steps as adjudication_steps_module
import daydream.deep.review_steps as review_steps_module
import daydream.json_utils as json_utils
import daydream.phases.adjudication as adjudication_module
import daydream.phases.review as review_module

_STRICT_GATE_CONSUMERS = (
    agent_module,
    review_module,
    adjudication_module,
    review_steps_module,
    adjudication_steps_module,
    pi_module,
)


def test_strict_gate_has_a_single_spelling() -> None:
    """item:1 -- every consumer imports the public helper; no private alias remains."""
    assert not hasattr(agent_module, "_validates_schema"), (
        "the private alias is a second spelling of the strict gate; consumers must "
        "import daydream.json_utils.validates_schema directly"
    )
    for module in _STRICT_GATE_CONSUMERS:
        assert module.__file__ is not None, f"{module.__name__} has no source file"
        source = pathlib.Path(module.__file__).read_text()
        assert "_validates_schema" not in source, f"{module.__name__} still references the removed private alias"
        if module is not agent_module:
            assert "validates_schema" in source, f"{module.__name__} calls the gate without importing the public helper"
    # The gate itself is untouched by the migration: still one implementation.
    assert json_utils.validates_schema({"a": 1}, {"type": "object"})


@pytest.mark.asyncio
async def test_pi_backend_honors_structured_output_opt_out() -> None:
    """The pi path keeps largest-span extraction for an opted-out caller."""
    from daydream.backends import ResultEvent
    from daydream.backends.pi import PiBackend
    from daydream.phases.schemas import PER_STACK_RECORD_SCHEMA
    from tests.harness.pi_replay import make_mock_process_from_fixture
    from tests.harness.process_replay import replay_process

    # The fixture's assistant text opens with an incidental array and closes with
    # the authoritative {"issues": []}. Schema-aware selection returns the object;
    # largest-span extraction returns the array.
    fixture = "issue1445_empty_results.jsonl"

    backend = PiBackend(model="glm-5.2")
    events, _ = await replay_process(
        backend,
        make_mock_process_from_fixture(fixture),
        pathlib.Path("/tmp"),
        "Parse",
        output_schema=PER_STACK_RECORD_SCHEMA,
        finalization=True,  # The excluded legacy path retains schema-aware prose selection.
    )
    validated = [e for e in events if isinstance(e, ResultEvent)]
    assert len(validated) == 1
    assert validated[0].structured_output == {"issues": []}

    opted_out = PiBackend(model="glm-5.2")
    events, _ = await replay_process(
        opted_out,
        make_mock_process_from_fixture(fixture),
        pathlib.Path("/tmp"),
        "Parse",
        output_schema=PER_STACK_RECORD_SCHEMA,
        validate_structured_output=False,
    )
    unvalidated = [e for e in events if isinstance(e, ResultEvent)]
    assert len(unvalidated) == 1
    # Largest-span extraction: the array, NOT the trailing schema-valid object.
    assert unvalidated[0].structured_output == ["module", "moduleVersion", "surface"]


def test_agent_threads_the_opt_out_to_capable_backends() -> None:
    """agent.py forwards the caller's opt-out only to backends that advertise it."""
    source = pathlib.Path(agent_module.__file__).read_text()
    assert 'getattr(\n                        backend, "supports_structured_output_opt_out", False' in source or (
        'backend, "supports_structured_output_opt_out", False' in source
    ), "agent.py must gate the opt-out kwarg on the backend capability"
    assert 'execute_kwargs["validate_structured_output"] = False' in source
    assert pi_module.PiBackend.supports_structured_output_opt_out is True


def test_selected_type_is_not_public_surface() -> None:
    """item:2 (auto-resolved) -- the unread field stays off the result dataclass."""
    assert not hasattr(json_utils.SchemaAwareSelection, "selected_type"), (
        "selected_type has no consumer in daydream/ or tests/; it must not ship as "
        "public surface on the selection record"
    )
    selection = json_utils.extract_json_by_schema(
        json.dumps({"issues": []}),
        schema={"type": "object"},
        accept=json_utils.validates_schema,
    )
    assert selection.value == {"issues": []}
