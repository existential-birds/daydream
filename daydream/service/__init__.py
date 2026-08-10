"""Durable neutral review-service core (Plan 008 Step 3, leaf-B).

This package contains the controller state machine, the storage + executor
ports, and the global admission/retry budget controller. It is deliberately
free of any execution-adapter, provider, or worker-asserted infrastructure
identity, and has no dependency on the daydream model-agent backends, executor
registry, or transactional store implementation (which live in consuming
leaves).
"""

from __future__ import annotations
