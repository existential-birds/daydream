"""Supplement the runner's capture-loss proof with the retained receipt bound."""
from __future__ import annotations

from pathlib import Path

import pytest

from daydream.backends import ToolResultEvent, ToolStartEvent
from daydream.review_evidence import ReviewEvidence


@pytest.mark.parametrize('field', ['call-id', 'tool-name', 'native-status'])
def test_oversized_native_metadata_cannot_remain_in_retained_receipts(tmp_path: Path, field: str) -> None:
    source = 'VALUE = 1\n'
    (tmp_path / 'api.py').write_text(source)
    evidence = ReviewEvidence(None)
    evidence.configure_capture(tmp_path)
    too_large = 'x' * 2_049
    call_id = too_large if field == 'call-id' else 'read-source'
    name = too_large if field == 'tool-name' else 'Read'
    evidence.observe(ToolStartEvent(id=call_id, name=name, input={'file_path': 'api.py'}))
    evidence.observe(ToolResultEvent(id=call_id, output=source, is_error=False,
                                     status=too_large if field == 'native-status' else None))
    assert evidence.retention_overflow_results == 1
    assert evidence.capture_failure(['api.py']) is True
    assert all(len(value.encode()) <= 2_048 for receipt in evidence.receipts
               for value in (receipt.call.id, receipt.call.name, receipt.result.id, receipt.result.status or ''))
