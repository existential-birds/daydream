"""Unit tests for the run-scoped outage circuit state machine.

The circuit is pure and caller-clocked: every observation takes an explicit
``now``, so these tests drive it with raw floats and never touch a real clock.
The real-path coordination test lives in ``tests/test_agent_budget.py``.
"""

from __future__ import annotations

from daydream.outage_circuit import OutageCircuit


def test_the_circuit_opens_at_the_threshold_and_admits_exactly_one_probe() -> None:
    circuit = OutageCircuit(failure_threshold=3, probe_interval_s=30.0)

    for instant in (0.0, 1.0, 2.0):
        assert circuit.admit_retry(instant).allowed is True  # closed: ladders run
        # Only the third consecutive failure crosses the threshold and opens.
        assert circuit.record_failure(instant) is (instant == 2.0)

    assert circuit.state() == "open"
    assert circuit.admit_retry(5.0).allowed is False  # open, before the interval
    assert circuit.admit_retry(32.0).allowed is True  # half-open: one probe
    assert circuit.admit_retry(32.0).allowed is False  # the probe is already out
    # The failed probe re-opens the circuit for a full interval.
    assert circuit.record_failure(33.0) is True
    assert circuit.state() == "open" and circuit.admit_retry(40.0).allowed is False
    assert circuit.admit_retry(64.0).allowed is True  # next interval, next probe
    circuit.record_success()
    assert circuit.state() == "closed"


def test_state_is_a_pure_read_that_never_transitions() -> None:
    circuit = OutageCircuit(failure_threshold=1, probe_interval_s=30.0)
    # The very first failure crosses the threshold, so it opens the circuit.
    assert circuit.record_failure(0.0) is True

    # Reading well past the probe interval must not open the half-open door.
    assert circuit.state() == "open"
    assert circuit.state() == "open"
    # The first admission is what transitions, and it grants exactly one probe.
    assert circuit.admit_retry(1_000.0).allowed is True
    assert circuit.state() == "half_open"
    assert circuit.admit_retry(1_000.0).allowed is False


def test_a_success_resets_the_consecutive_failure_count() -> None:
    circuit = OutageCircuit(failure_threshold=3, probe_interval_s=30.0)
    # Two sub-threshold failures, so neither one opens the circuit.
    assert circuit.record_failure(0.0) is False
    assert circuit.record_failure(1.0) is False
    circuit.record_success()
    # The success reset the count, so this is again a sub-threshold failure.
    assert circuit.record_failure(2.0) is False

    # Two failures before the success were forgotten, so this is only the first
    # consecutive failure and the circuit stays closed.
    assert circuit.state() == "closed"
    assert circuit.admit_retry(2.0).allowed is True
