"""Owned artifact storage, live routing, frozen publication, and crash recovery.

``artifact_visibility`` acquires leases and binds sessions to the current run.
``session`` routes writes; ``finalization`` freezes joined evidence and publishes
or restores it through filesystem, ownership, ledger, and transaction modules.
"""
