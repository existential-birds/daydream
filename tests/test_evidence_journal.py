"""Durable local evidence: lossless OTLP encoding, verified seals, and truthful incompleteness."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from daydream.trajectory.journal import (
    SEAL_EVENT,
    JournalWriter,
    any_value,
    from_any_value,
    read_journal,
)


def _writer(tmp_path: Path) -> JournalWriter:
    return JournalWriter(tmp_path / "evidence.otlp.jsonl", resource={"service.name": "daydream"}, version="test")


def test_any_value_round_trips_json_shapes_without_loss_or_truncation() -> None:
    """Large bodies, lone surrogates, null, booleans, non-finite doubles and ordered keys survive exactly."""
    value: dict[str, Any] = {"text": "caf\udce9 " + "x" * (8 * 1024 * 1024), "none": None, "flag": True,
                             "count": 1, "ratio": 0.5, "nested": [{"b": 2, "a": [False, None]}], "empty": {}}
    encoded = json.loads(json.dumps(any_value(value), ensure_ascii=True))
    assert from_any_value(encoded) == value and list(from_any_value(encoded)) == list(value)
    assert math.isnan(from_any_value(any_value(math.nan))) and from_any_value(any_value(-math.inf)) == -math.inf
    assert from_any_value(any_value(True)) is True and from_any_value(any_value(1)) == 1


def test_only_a_verified_seal_makes_a_document_complete(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    for index in range(3):
        assert writer.append("daydream.test", {"index": index}, {"gen_ai.usage.input_tokens": index})
    assert writer.seal({"status": "complete"})
    raw = writer.path.read_bytes()
    contents = read_journal(raw)
    assert contents.complete and [record.body for record in contents.records[:3]] == [{"index": i} for i in range(3)]
    assert contents.seal is not None and contents.seal.body["record_count"] == 3
    assert contents.records[1].attributes == {"gen_ai.usage.input_tokens": 1}
    assert all(json.loads(line)["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]["eventName"]
               for line in raw.splitlines())

    lines = raw.splitlines(keepends=True)
    truncated = b"".join(lines[:2]) + lines[2][:-10]
    tampered = raw.replace(b'"intValue":"1"', b'"intValue":"7"', 1)
    after_seal = raw + lines[0].replace(b'"intValue":"0"}', b'"intValue":"4"}', 1)
    unsealed = b"".join(lines[:3])
    for damaged, problem in ((truncated, "truncated"), (tampered, "seal does not match"),
                             (after_seal, "sequence is broken"), (unsealed, "not sealed")):
        contents = read_journal(damaged)
        assert not contents.complete and problem in str(contents.problem)
        assert [record.seq for record in contents.records if record.event_name != SEAL_EVENT][:2] == [0, 1]


@pytest.mark.parametrize("fault", ["write", "encoding"])
def test_write_and_encoding_faults_stop_capture_before_the_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    """After the first fault no record or seal is accepted, and the accepted prefix stays readable."""
    writer = _writer(tmp_path)
    assert writer.append("daydream.test", {"index": 0})
    accepted = writer.reference()
    if fault == "write":
        def denied(_fd: int, _data: Any) -> int:
            raise OSError("disk full")
        monkeypatch.setattr(os, "write", denied)
        assert not writer.append("daydream.test", {"index": 1})
        monkeypatch.undo()
    else:
        assert not writer.append("daydream.test", {"too_large": 1 << 64})
    assert writer.failure == ("OSError" if fault == "write" else "EvidenceEncodingError")
    assert not writer.append("daydream.test", {"index": 2}) and not writer.seal({})
    assert writer.reference() == accepted
    contents = read_journal(writer.path.read_bytes())
    assert not contents.complete and [record.body for record in contents.records] == [{"index": 0}]


async def test_evidence_verify_reads_back_every_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``daydream evidence verify`` reports each document's digest and fails on a damaged fork."""
    from daydream.cli import _command_handlers
    from daydream.trajectory import DaydreamPhase, run_document_path
    from tests.harness.trajectory import make_recorder, observe_text_and_result

    run_dir = tmp_path / "runs" / "run"
    recorder = make_recorder(tmp_path, path=run_document_path(run_dir), session_id="run", artifact_run_dir=run_dir)
    async with recorder:
        async with recorder.fork("reviewer") as child:
            async with child.invocation(phase=DaydreamPhase.REVIEW) as invocation:
                observe_text_and_result(invocation)
    verify = _command_handlers()["evidence"]
    assert verify(["verify", str(run_dir)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["complete"]
    assert [document["trajectory_id"] for document in report["documents"]] == ["run", "run:reviewer"]
    child.path.write_bytes(child.path.read_bytes()[:-5])
    assert verify(["verify", str(run_dir)]) == 1
    damaged = json.loads(capsys.readouterr().out)["documents"][1]
    assert not damaged["complete"] and "truncated" in damaged["problem"]
