"""Daydream service-mode contracts (Plan 008, Step 2 leaf).

This package owns the immutable job model (``models``), the strictly passive
worker artifact envelope (``artifact``), and the fail-closed service review
worker (``worker``). It deliberately imports nothing at package level so
importing ``daydream.service`` never pulls the phase/agent stack in.
"""
