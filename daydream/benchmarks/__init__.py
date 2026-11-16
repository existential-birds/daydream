"""Optional benchmark boundary: Harbor-compatible ATIF export of local evidence.

Ordinary Daydream runs never import this package. Benchmark adapters call
:mod:`daydream.benchmarks.harbor` to export a finished run's local OpenTelemetry
evidence as ATIF trajectories.
"""
