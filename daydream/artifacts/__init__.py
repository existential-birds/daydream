"""Artifact storage: ownership, attested trees, publication, and crash recovery.

The public session boundary lives in :mod:`daydream.artifact_visibility`.
Leaf modules never import the session: filesystem and ownership feed durable
ledgers and transfers, external publication builds on those, and transactions
reconcile the resulting state.
"""
