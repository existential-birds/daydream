
from daydream import severity


def test_stronger_severity_never_fabricates_a_level() -> None:
    # Off-vocabulary and absent values lose to any canonical value and never
    # become one themselves (P6): two unusable inputs stay unusable.
    assert severity.stronger_severity(None, "low") == "low"
    assert severity.stronger_severity("CRITICAL", "medium") == "medium"
    assert severity.stronger_severity("high", None) == "high"
    assert severity.stronger_severity(None, None) is None
    assert severity.stronger_severity("bogus", 7) is None
