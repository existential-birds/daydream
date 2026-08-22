"""Daydream Harbor verifier core.

A stdlib-only source of truth for pure review scoring: strict gold/candidate/
reward models, deterministic candidate-ID derivation, maximum-cardinality
one-to-one matching over injected verdicts, per-task reward, and corpus micro
metrics. Imports only the Python standard library so issue #7 can copy this
module byte-for-byte into judge-free verifier images.
"""
