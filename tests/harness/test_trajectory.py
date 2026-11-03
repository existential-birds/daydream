"""Self-test for the shared trajectory/manifest builders.

Drives the real ``TrajectoryRecorder`` through the harness builders and asserts
the observable outcome (a schema-valid trajectory on disk carrying the observed
text), plus the manifest and diff builders the archive/training tests rely on.
"""

from __future__ import annotations



