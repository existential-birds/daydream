"""Real protobuf identity, flags, trace state, and dropped-count fidelity.

First prove the stock encoder’s flag/state losses, then require repair while
preserving bounded-container order and independent limit counters."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.util import BoundedList
from opentelemetry.trace import Link, SpanContext, TraceFlags, TraceState

from daydream.observability.config import ObservabilityConfig
from daydream.observability.exporters import honeyhive_exporter, langsmith_exporter, otlp_exporter
from tests.harness.otlp import attributes, otlp_collector

_FLAGS_HAS_IS_REMOTE = 256
_FLAGS_IS_REMOTE = 512
_FLAGS_SAMPLED = 1

_TRACE_ID = 0x11111111111111111111111111111111
_SPAN_ID = 0x3333333333333333
_PARENT_SPAN_ID = 0x2222222222222222
_LINK_TRACE_ID = 0x44444444444444444444444444444444
_LINK_SPAN_ID = 0x5555555555555555

_VENDORS = ("honeyhive", "langsmith", "otlp")


def _ctx(trace_id: int, span_id: int, *, is_remote: bool = False, sampled: bool = True,
    state_entries: list[tuple[str, str]] | None = None,
) -> SpanContext:
    return SpanContext(
        trace_id=trace_id, span_id=span_id, is_remote=is_remote, trace_flags=TraceFlags(0x01 if sampled else 0x00),
        trace_state=TraceState(entries=state_entries) if state_entries else None,
    )


def _span(*, trace_id: int = _TRACE_ID, span_id: int = _SPAN_ID, parent: SpanContext | None = None, name: str = "span",
    sampled: bool = True, links: list[Link] | None = None, events: list[Event] | None = None, attributes: Any = None,
) -> ReadableSpan:
    return ReadableSpan(name=name, context=_ctx(trace_id, span_id, sampled=sampled), parent=parent,
        resource=Resource({"service.name": "daydream-test"}), attributes=attributes or {}, events=list(events or []),
        links=list(links or []),
    )


def _hex_id(value: str) -> str:
    return base64.b64decode(value).hex()


def _wire_spans(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [span
        for resource in request["body"]["resourceSpans"]
        for scope in resource["scopeSpans"]
        for span in scope["spans"]
    ]


def _vendor_wire(monkeypatch: pytest.MonkeyPatch, vendor: str, spans: list[ReadableSpan]) -> list[dict[str, Any]]:
    """Export through a production destination factory and return wire spans."""
    with otlp_collector() as receiver:
        if vendor == "honeyhive":
            monkeypatch.setenv("HH_API_URL", receiver.base_url)
            monkeypatch.setenv("HH_API_KEY", "opaque-key")
            exporter = honeyhive_exporter(ObservabilityConfig())
        elif vendor == "langsmith":
            monkeypatch.setenv("LANGSMITH_ENDPOINT", receiver.base_url)
            monkeypatch.setenv("LANGSMITH_API_KEY", "opaque-key")
            exporter = langsmith_exporter(ObservabilityConfig())
        else:
            monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", receiver.base_url + "/v1/traces")
            exporter = otlp_exporter(ObservabilityConfig())
        try:
            exporter.export(spans)
            assert exporter.force_flush()
        finally:
            exporter.shutdown()
        assert len(receiver.requests) == 1
        return _wire_spans(receiver.requests[0])


def test_stock_encoder_drops_low_trace_flags_and_link_state() -> None:
    """Documents the pinned SDK's fidelity gap the Task 4A repair exists for."""
    link = Link(_ctx(_LINK_TRACE_ID, _LINK_SPAN_ID, is_remote=True, sampled=True, state_entries=[("vendor", "1")]))
    span = _span(parent=_ctx(_TRACE_ID, _PARENT_SPAN_ID, is_remote=True, sampled=False), links=[link])
    wire = encode_spans([span]).resource_spans[0].scope_spans[0].spans[0]
    assert wire.flags & _FLAGS_SAMPLED == 0  # sampled bit lost on the span
    assert wire.links[0].flags & _FLAGS_SAMPLED == 0  # and on the link
    assert wire.links[0].trace_state == ""  # link trace state lost entirely


def test_wire_preserves_identity_order_and_fields_for_all_destinations(monkeypatch: pytest.MonkeyPatch,) -> None:
    """The real exporter preserves one heterogeneous span batch under simultaneous limit pressure.

    Separate scalar exports could miss position-based repair and interference between dropped
    containers. Literal per-span/link expectations also protect repeated identities with distinct flags.
    """
    attrs = BoundedAttributes(
        maxlen=3,
        attributes={
            "discarded": 0,
            "daydream.span.kind": "attempt",
            "daydream.billing.owner": "structural_attempt",
            "gen_ai.usage.input_tokens": 100,
        },
    )
    events: BoundedList[Event] = BoundedList(2)
    events.extend(
        (
            Event("discarded"),
            Event("retained", attributes=BoundedAttributes(maxlen=1, attributes={"discarded": 0, "retained": 1})),
            Event("last"),
        )
    )
    links: BoundedList[Link] = BoundedList(4)
    links.append(Link(_ctx(_LINK_TRACE_ID, _LINK_SPAN_ID + 1)))
    for remote_flag, sampled_flag, state in (
        (False, False, "a"),
        (False, True, "b"),
        (True, False, "c"),
        (True, True, "d"),
    ):
        links.append(
            Link(
                _ctx(
                    _LINK_TRACE_ID,
                    _LINK_SPAN_ID,
                    is_remote=remote_flag,
                    sampled=sampled_flag,
                    state_entries=[(state, "1")],
                ),
                attributes=BoundedAttributes(maxlen=1, attributes={"discarded": 0, "retained": 1}),
            )
        )
    root = ReadableSpan(
        name="root",
        context=_ctx(_TRACE_ID, _SPAN_ID),
        parent=None,
        resource=Resource({"service.name": "daydream-test"}),
        attributes=attrs,
        events=events,
        links=links,
    )
    local = _span(name="local-child", span_id=0x4444444444444444, parent=root.context, sampled=False)
    remote_parent = _ctx(_TRACE_ID, _PARENT_SPAN_ID, is_remote=True, sampled=False)
    remote = _span(name="remote-child", span_id=0x6666666666666666, parent=remote_parent, sampled=False)
    sampled = _span(name="sampled-remote", span_id=0x7777777777777777, parent=remote_parent)
    # The destination receives one actual span population, including observed-zero and missing
    # usage on structural scopes. Native billing aliases must follow the resolved owner per span.
    mapping: list[ReadableSpan] = []
    zero_usage = {
        "gen_ai.usage.input_tokens": 0,
        "gen_ai.usage.output_tokens": 0,
        "gen_ai.usage.cost": 0.0,
        "gen_ai.usage.cache_read.input_tokens": 0,
        "gen_ai.usage.cache_creation.input_tokens": 0,
        "gen_ai.usage.reasoning.output_tokens": 0,
    }
    identity = {
        "daydream.run.id": "run-bind",
        "daydream.flow": "review",
        "gen_ai.conversation.id": "native-conversation",
    }
    for session_present in (False, True):
        for usage_known in (False, True):
            for kind in ("run", "phase", "agent", "tool", "attempt"):
                scope_attributes: dict[str, Any] = {**identity, "daydream.span.kind": kind}
                if session_present:
                    scope_attributes["traceloop.association.properties.session_id"] = "session-bind"
                if usage_known:
                    scope_attributes.update(zero_usage)
                if kind == "agent":
                    scope_attributes.update(
                        {
                            "gen_ai.agent.name": "review",
                            "daydream.agent.name": "review",
                            "daydream.agent.role": "root" if session_present else "subagent",
                        }
                    )
                if kind == "attempt":
                    scope_attributes.update(
                        {"daydream.billing.owner": "structural_attempt", "daydream.agent.role": "root"}
                    )
                mapping.append(
                    _span(
                        name=f"{kind}-{session_present}-{usage_known}",
                        span_id=_SPAN_ID + 0x100 + len(mapping),
                        parent=root.context,
                        attributes=scope_attributes,
                    )
                )
    mapping.extend(
        (
            _span(
                name="generation-attempt",
                span_id=_SPAN_ID + 0x200,
                attributes={
                    **identity,
                    "daydream.span.kind": "attempt",
                    "daydream.billing.owner": "generation_children",
                    "gen_ai.usage.input_tokens": 100,
                },
            ),
            _span(
                name="billed-generation",
                span_id=_SPAN_ID + 0x201,
                attributes={
                    **identity,
                    "daydream.span.kind": "generation",
                    "daydream.generation.billed": True,
                    "gen_ai.usage.input_tokens": 60,
                    "gen_ai.usage.output_tokens": 10,
                    "gen_ai.usage.cost": 0.05,
                },
            ),
            _span(
                name="unbilled-generation",
                span_id=_SPAN_ID + 0x202,
                attributes={
                    **identity,
                    "daydream.span.kind": "generation",
                    "daydream.generation.billed": False,
                    "gen_ai.usage.input_tokens": 60,
                },
            ),
            _span(
                name="owner-none",
                span_id=_SPAN_ID + 0x203,
                attributes={
                    **identity,
                    "daydream.span.kind": "attempt",
                    "daydream.billing.owner": "none",
                    "gen_ai.usage.input_tokens": 100,
                },
            ),
            _span(
                name="pressure-two",
                span_id=_SPAN_ID + 0x204,
                attributes=BoundedAttributes(
                    maxlen=4, attributes={"a": "1", "b": "2", "c": "3", "d": "4", "e": "5", "f": "6"}
                ),
            ),
        )
    )
    originals = (root, local, remote, sampled, *mapping)
    before = [dict(span.attributes or {}) for span in originals]
    for vendor in _VENDORS:
        batch = _vendor_wire(monkeypatch, vendor, list(originals))
        assert len(batch) == 29
        wires, mapped = batch[:4], batch[4:]
        assert [wire["name"] for wire in wires] == ["root", "local-child", "remote-child", "sampled-remote"]
        assert [_hex_id(wire["spanId"]) for wire in wires] == [
            "3333333333333333",
            "4444444444444444",
            "6666666666666666",
            "7777777777777777",
        ]
        assert all(_hex_id(wire["traceId"]) == "11111111111111111111111111111111" for wire in wires)
        assert "parentSpanId" not in wires[0]
        assert [_hex_id(wire["parentSpanId"]) for wire in wires[1:]] == [
            "3333333333333333",
            "2222222222222222",
            "2222222222222222",
        ]
        assert [wire["flags"] for wire in wires] == [257, 256, 768, 769]
        assert all(wire["kind"] == "SPAN_KIND_INTERNAL" for wire in wires)
        wire = wires[0]
        assert wire["droppedAttributesCount"] == wire["droppedEventsCount"] == wire["droppedLinksCount"] == 1
        assert [event["name"] for event in wire["events"]] == ["retained", "last"]
        assert wire["events"][0]["droppedAttributesCount"] == 1
        assert [link["flags"] for link in wire["links"]] == [256, 257, 768, 769]
        assert [link["traceState"] for link in wire["links"]] == ["a=1", "b=1", "c=1", "d=1"]
        assert all(_hex_id(link["spanId"]) == "5555555555555555" for link in wire["links"])
        assert all(link["droppedAttributesCount"] == 1 for link in wire["links"])
        assert all(attributes(link) == {"retained": 1} for link in wire["links"])
        native = attributes(wire)
        assert native["daydream.span.kind"] == "attempt"
        assert native["gen_ai.usage.input_tokens"] == 100
        if vendor == "honeyhive":
            assert native["honeyhive_event_type"] == "chain"
        elif vendor == "langsmith":
            assert native["langsmith.span.kind"] == "chain"
        for child in wires[1:]:
            assert child.get("droppedAttributesCount", 0) == child.get("droppedEventsCount", 0) == 0
            assert child.get("droppedLinksCount", 0) == 0
        assert [wire["name"] for wire in mapped] == [span.name for span in mapping]
        for original, wire in zip(mapping, mapped, strict=True):
            assert original.context is not None
            assert _hex_id(wire["spanId"]) == f"{original.context.span_id:016x}"
            native = attributes(wire)
            assert all(native[key] == value for key, value in (original.attributes or {}).items())
        for original, wire in zip(mapping[:20], mapped[:20], strict=True):
            source = dict(original.attributes or {})
            native = attributes(wire)
            kind_name = str(source["daydream.span.kind"])
            billed = kind_name == "attempt" and "gen_ai.usage.input_tokens" in source
            if vendor == "honeyhive":
                assert (
                    native["honeyhive_event_type"]
                    == {"run": "chain", "phase": "chain", "agent": "chain", "attempt": "chain", "tool": "tool"}[
                        kind_name
                    ]
                )
                assert native["honeyhive.session_id"] == source.get(
                    "traceloop.association.properties.session_id", "run-bind"
                )
                assert (
                    native["honeyhive.session_name"] == "daydream.review"
                    and native["honeyhive.session_auto_create"] is True
                )
                assert {key: value for key, value in native.items() if key.startswith("honeyhive_metadata.")} == (
                    {
                        "honeyhive_metadata.prompt_tokens": 0,
                        "honeyhive_metadata.completion_tokens": 0,
                        "honeyhive_metadata.cost": 0.0,
                        "honeyhive_metadata.cache_read_input_tokens": 0,
                        "honeyhive_metadata.cache_write_input_tokens": 0,
                        "honeyhive_metadata.reasoning_tokens": 0,
                    }
                    if billed
                    else {}
                )
                if kind_name == "agent":
                    assert native["gen_ai.agent.name"] == "review"
                    assert not any(key.startswith(("_", "honeyhive_metadata.agent_")) for key in native)
            elif vendor == "langsmith":
                assert (
                    native["langsmith.span.kind"]
                    == {"run": "chain", "phase": "chain", "agent": "chain", "attempt": "chain", "tool": "tool"}[
                        kind_name
                    ]
                )
                assert ("langsmith.usage_metadata" in native) == billed
                if billed:
                    assert json.loads(native["langsmith.usage_metadata"]) == {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_cost": 0.0,
                        "input_token_details": {"cache_read": 0, "cache_creation": 0},
                        "output_token_details": {"reasoning": 0},
                    }
                assert native.get("langsmith.metadata.ls_agent_type") == (
                    source["daydream.agent.role"] if kind_name == "agent" else None
                )
        special = {wire["name"]: attributes(wire) for wire in mapped[20:]}
        assert mapped[-1]["droppedAttributesCount"] == 2
        if vendor in {"honeyhive", "langsmith"}:
            usage_key = "honeyhive_metadata.prompt_tokens" if vendor == "honeyhive" else "langsmith.usage_metadata"
            assert all(
                usage_key not in special[name] for name in ("generation-attempt", "unbilled-generation", "owner-none")
            )
            if vendor == "honeyhive":
                assert special["billed-generation"]["honeyhive_event_type"] == "model"
                assert special["billed-generation"]["honeyhive_metadata.prompt_tokens"] == 60
                assert special["billed-generation"]["honeyhive_metadata.cost"] == 0.05
                assert special["generation-attempt"]["honeyhive_event_type"] == "chain"
            else:
                assert special["billed-generation"]["langsmith.span.kind"] == "llm"
                assert special["unbilled-generation"]["langsmith.span.kind"] == "llm"
                assert json.loads(special["billed-generation"]["langsmith.usage_metadata"]) == {
                    "input_tokens": 60,
                    "output_tokens": 10,
                    "total_cost": 0.05,
                }
                assert (
                    special["generation-attempt"]["langsmith.span.kind"]
                    == special["owner-none"]["langsmith.span.kind"]
                    == "chain"
                )
        assert [dict(span.attributes or {}) for span in originals] == before
        assert attrs.dropped == events.dropped == links.dropped == 1
