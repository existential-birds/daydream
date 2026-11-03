"""Tests for multi-phase canonical synthesis with structured-output support.

These exercise the harness's per-phase Codex/Claude script synthesis. The
structured-output roundtrip drives the synthesized JSONL through the REAL
``CodexBackend`` parser and asserts the adapter forwards the requested JSON as assistant text
for host resolution — never that a builder was merely called.
"""




