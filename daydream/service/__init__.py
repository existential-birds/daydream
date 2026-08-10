"""Durable review-service storage layer.

This leaf (Plan 008 leaf-C) contributes the transactional store: the neutral
``ServiceStore`` port, the in-memory conformance double, and the production
SQLite implementation. The controller state machine (leaf-B) and executor
registry (leaf-D) live beside these modules under ``daydream.service``; the
integrator reconciles their shared namespace.
"""
