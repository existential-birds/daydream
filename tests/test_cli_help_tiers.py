"""Tests for the two-tier help surface (``--help`` vs ``--help-all``)."""


from daydream.cli import _parse_args


def test_advanced_flags_still_parse() -> None:
    assert _parse_args(["--start-at", "fix", "/t"]).start_at == "fix"


def test_precision_flag_activates_precision_mode() -> None:
    """#232: ``--precision`` is the activation path into RunConfig.precision_mode.

    Absent the flag the field stays ``False`` (byte-identical default); with it,
    the field is ``True`` so the deep orchestrator's ``_precision_mode`` resolver
    runs the suppression pass.
    """
    assert _parse_args(["/t"]).precision_mode is False
    assert _parse_args(["--precision", "/t"]).precision_mode is True


def test_diagram_flags_parse_from_both_tiers() -> None:
    """#1113: both flags reach ``RunConfig`` through the production parser."""
    assert _parse_args(["--diagram", "off", "/t"]).diagram == "off"
    only = _parse_args(["--diagram-only", "both", "/t"])
    assert only.diagram == "both"
    assert only.output_mode == "diagram"
