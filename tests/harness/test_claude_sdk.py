"""Tests for the shared Claude-SDK mock/patch harness.

The harness stands in for the SDK types ``ClaudeBackend`` dispatches on, so its
own contract — the eight patched names, message replay, options capture, and the
``assistant_message`` / ``result_message`` override seam the metrics tests rely
on — is pinned here rather than inferred from its consumers.
"""

from __future__ import annotations
