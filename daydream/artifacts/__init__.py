"""Owned artifact storage, live routing, frozen publication, and crash recovery.

``artifact_visibility`` acquires leases and binds sessions to the current run.
``session`` owns routing, frozen evidence, publication, and rollback through
filesystem, ownership, ledger, and transaction capabilities.
"""
