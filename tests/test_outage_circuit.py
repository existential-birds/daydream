"""Drive the pure outage state machine with explicit clock values.

Real-path coordination lives in test_agent_budget.py."""

from __future__ import annotations

from daydream.outage_circuit import OutageCircuit


def test_the_circuit_opens_at_the_threshold_and_admits_exactly_one_probe() -> None:
    circuit = OutageCircuit(failure_threshold=3, probe_interval_s=30.0)
    assert circuit.record_failure(0.0) is False
    assert circuit.record_failure(1.0) is False
    circuit.record_success()
    for instant in (2.0, 3.0, 4.0):
        assert circuit.admit_retry(instant).allowed is True  # closed: ladders run
        # Only the third consecutive failure crosses the threshold and opens.
        assert circuit.record_failure(instant) is (instant == 4.0)
    assert circuit.state() == "open"
    assert circuit.admit_retry(5.0).allowed is False  # open, before the interval
    assert circuit.admit_retry(34.0).allowed is True  # half-open: one probe
    assert circuit.admit_retry(34.0).allowed is False  # the probe is already out
    # The failed probe re-opens the circuit for a full interval.
    assert circuit.record_failure(35.0) is True
    assert circuit.state() == "open" and circuit.admit_retry(40.0).allowed is False
    assert circuit.admit_retry(66.0).allowed is True  # next interval, next probe
    circuit.record_success()
    assert circuit.state() == "closed"
