"""Real-wire OTLP fidelity, limit and encoder-repair tests (P18 Task 4A Step 1).

Exercises the destination exporters against real ``ReadableSpan`` instances with
bounded containers and real protobuf encoding: identity/order on the wire,
span/link flags, link trace state, and exact dropped counters under independent
limit pressure. The stock encoder's proved fidelity losses (low trace flags,
link trace state) are asserted first, then required to be repaired in transit.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.attributes import BoundedAttributes
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
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


@pytest.fixture(autouse=True)
def _isolate_trace_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    import os

    for key in os.environ:
        if key.startswith(("OTEL_", "LANGSMITH_", "HH_", "_OTEL_")):
            monkeypatch.delenv(key)
    yield


def _ctx(
    trace_id: int,
    span_id: int,
    *,
    is_remote: bool = False,
    sampled: bool = True,
    state_entries: list[tuple[str, str]] | None = None,
) -> SpanContext:
    return SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=is_remote,
        trace_flags=TraceFlags(0x01 if sampled else 0x00),
        trace_state=TraceState(entries=state_entries) if state_entries else None,
    )


def _span(
    *,
    trace_id: int = _TRACE_ID,
    span_id: int = _SPAN_ID,
    parent: SpanContext | None = None,
    name: str = "span",
    links: list[Link] | None = None,
    events: list[Event] | None = None,
    attributes: Any = None,
) -> ReadableSpan:
    return ReadableSpan(
        name=name,
        context=_ctx(trace_id, span_id),
        parent=parent,
        resource=Resource({"service.name": "daydream-test"}),
        attributes=attributes or {},
        events=list(events or []),
        links=list(links or []),
    )


def _hex_id(value: str) -> str:
    return base64.b64decode(value).hex()


def _wire_spans(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        span
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
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        exporter.export(spans)
        exporter.shutdown()
        provider.shutdown()
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


def test_wire_repairs_span_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    remote_parent = _ctx(_TRACE_ID, _PARENT_SPAN_ID, is_remote=True, sampled=False)
    cases = [
        # Remote parent with an unsampled span context: masks only.
        (
            ReadableSpan(
                name="span",
                context=_ctx(_TRACE_ID, _SPAN_ID, sampled=False),
                parent=remote_parent,
                resource=Resource({"service.name": "daydream-test"}),
                attributes={},
                events=[],
                links=[],
            ),
            _FLAGS_HAS_IS_REMOTE | _FLAGS_IS_REMOTE,
        ),
        # Local root, unsampled span context.
        (
            ReadableSpan(
                name="span",
                context=_ctx(_TRACE_ID, 0x4444444444444444, sampled=False),
                parent=None,
                resource=Resource({"service.name": "daydream-test"}),
                attributes={},
                events=[],
                links=[],
            ),
            _FLAGS_HAS_IS_REMOTE,
        ),
        # Local root, sampled span context: the low sampled bit is restored.
        (_span(span_id=0x5555555555555555, parent=None), _FLAGS_HAS_IS_REMOTE | _FLAGS_SAMPLED),
        # Remote parent, sampled span context.
        (
            _span(parent=_ctx(_TRACE_ID, _PARENT_SPAN_ID, is_remote=True, sampled=True)),
            _FLAGS_HAS_IS_REMOTE | _FLAGS_IS_REMOTE | _FLAGS_SAMPLED,
        ),
    ]
    for span, expected in cases:
        for vendor in _VENDORS:
            wire = _vendor_wire(monkeypatch, vendor, [span])[0]
            assert wire.get("flags", 0) == expected, vendor


@pytest.mark.parametrize("remote_sampled", [(False, False), (False, True), (True, False), (True, True)])
def test_wire_repairs_link_flags_and_trace_state(
    monkeypatch: pytest.MonkeyPatch, remote_sampled: tuple[bool, bool]
) -> None:
    remote, sampled = remote_sampled
    link = Link(_ctx(_LINK_TRACE_ID, _LINK_SPAN_ID, is_remote=remote, sampled=sampled, state_entries=[("vendor", "1")]))
    span = _span(links=[link])
    expected = _FLAGS_HAS_IS_REMOTE | (_FLAGS_IS_REMOTE if remote else 0) | (_FLAGS_SAMPLED if sampled else 0)
    for vendor in _VENDORS:
        wire = _vendor_wire(monkeypatch, vendor, [span])[0]
        assert wire["links"][0].get("flags", 0) == expected, vendor
        assert wire["links"][0]["traceState"] == "vendor=1", vendor


def test_wire_keeps_duplicate_position_links_with_distinct_flags_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    links = [
        Link(_ctx(_LINK_TRACE_ID, _LINK_SPAN_ID, is_remote=False, sampled=True, state_entries=[("a", "1")])),
        Link(_ctx(_LINK_TRACE_ID, _LINK_SPAN_ID, is_remote=True, sampled=False, state_entries=[("b", "2")])),
    ]
    span = _span(links=links)
    for vendor in _VENDORS:
        wire = _vendor_wire(monkeypatch, vendor, [span])[0]
        assert len(wire["links"]) == 2, vendor
        assert wire["links"][0].get("flags", 0) == _FLAGS_HAS_IS_REMOTE | _FLAGS_SAMPLED, vendor
        assert wire["links"][0]["traceState"] == "a=1", vendor
        assert wire["links"][1].get("flags", 0) == _FLAGS_HAS_IS_REMOTE | _FLAGS_IS_REMOTE, vendor
        assert wire["links"][1]["traceState"] == "b=2", vendor


def test_wire_preserves_identity_order_and_fields_for_all_destinations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _span(name="root")
    child = _span(name="child", span_id=0x4444444444444444, parent=root.context)
    for vendor in _VENDORS:
        wires = _vendor_wire(monkeypatch, vendor, [root, child])
        assert [wire["name"] for wire in wires] == ["root", "child"], vendor
        assert _hex_id(wires[0]["traceId"]) == f"{_TRACE_ID:032x}", vendor
        assert _hex_id(wires[0]["spanId"]) == f"{_SPAN_ID:016x}", vendor
        assert "parentSpanId" not in wires[0], vendor  # empty root parent
        assert _hex_id(wires[1]["spanId"]) == f"{0x4444444444444444:016x}", vendor
        assert _hex_id(wires[1]["parentSpanId"]) == f"{_SPAN_ID:016x}", vendor
        assert wires[0]["kind"] == "SPAN_KIND_INTERNAL", vendor
        assert wires[0].get("droppedAttributesCount", 0) == 0, vendor


@pytest.mark.parametrize(
    "pressure",
    ["attributes", "events", "links", "event_attributes", "link_attributes"],
)
def test_wire_preserves_exact_dropped_counters_under_limit_pressure(
    monkeypatch: pytest.MonkeyPatch,
    pressure: str,
) -> None:
    """Original bounded dropped counters stay exact after vendor additions."""
    span: ReadableSpan
    if pressure == "attributes":
        attrs = BoundedAttributes(maxlen=2, attributes={"a": 1, "b": 2, "c": 3})
        span = _span(attributes=attrs)
    elif pressure == "events":
        events: BoundedList[Event] = BoundedList(2)
        for index in range(3):
            try:
                events.append(Event(f"e{index}"))
            except Exception:
                pass
        span = ReadableSpan(
            name="span",
            context=_ctx(_TRACE_ID, _SPAN_ID),
            parent=None,
            resource=Resource({"service.name": "daydream-test"}),
            attributes={},
            events=events,
            links=BoundedList(0),
        )
    elif pressure == "links":
        links: BoundedList[Link] = BoundedList(2)
        for index in range(3):
            try:
                links.append(Link(_ctx(_LINK_TRACE_ID, _LINK_SPAN_ID + index)))
            except Exception:
                pass
        span = ReadableSpan(
            name="span",
            context=_ctx(_TRACE_ID, _SPAN_ID),
            parent=None,
            resource=Resource({"service.name": "daydream-test"}),
            attributes={},
            events=BoundedList(0),
            links=links,
        )
    elif pressure == "event_attributes":
        event = Event("e", attributes=BoundedAttributes(maxlen=1, attributes={"x": 1, "y": 2}))
        span = _span(events=[event])
    else:
        link = Link(
            _ctx(_LINK_TRACE_ID, _LINK_SPAN_ID),
            attributes=BoundedAttributes(maxlen=1, attributes={"x": 1, "y": 2}),
        )
        span = _span(links=[link])
    for vendor in _VENDORS:
        wire = _vendor_wire(monkeypatch, vendor, [span])[0]
        if pressure == "attributes":
            assert wire["droppedAttributesCount"] == 1, vendor
            assert wire.get("droppedEventsCount", 0) == 0 and wire.get("droppedLinksCount", 0) == 0, vendor
        elif pressure == "events":
            assert wire["droppedEventsCount"] == 1, vendor
            assert len(wire["events"]) == 2, vendor  # the two surviving events
        elif pressure == "links":
            assert wire["droppedLinksCount"] == 1, vendor
            assert len(wire["links"]) == 2, vendor
        elif pressure == "event_attributes":
            assert wire["events"][0]["droppedAttributesCount"] == 1, vendor
        else:
            assert wire["links"][0]["droppedAttributesCount"] == 1, vendor


def test_vendor_wire_attributes_survive_additions_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Vendor additions land in wire attributes without erasing originals."""
    with otlp_collector() as receiver:
        monkeypatch.setenv("HH_API_URL", receiver.base_url)
        monkeypatch.setenv("HH_API_KEY", "opaque-key")
        exporter = honeyhive_exporter(ObservabilityConfig())
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with provider.get_tracer("daydream-test").start_as_current_span(
            "attempt",
            attributes={
                "daydream.span.kind": "attempt",
                "daydream.billing.owner": "structural_attempt",
                "gen_ai.usage.input_tokens": 100,
            },
        ):
            pass
        provider.shutdown()
    wire = _wire_spans(receiver.requests[0])[0]
    keys = [attr["key"] for attr in wire["attributes"]]
    assert "daydream.span.kind" in keys
    assert "honeyhive_event_type" in keys
    native = attributes(receiver.spans[0])
    assert native["honeyhive_event_type"] == "chain"
