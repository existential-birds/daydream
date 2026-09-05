from daydream import severity


def test_canonical_levels_are_low_medium_high() -> None:
    assert severity.CANONICAL_LEVELS == ("low", "medium", "high")


def test_normalize_severity_maps_known_and_rejects_unknown() -> None:
    assert severity.normalize_severity("HIGH") == "high"
    assert severity.normalize_severity(" Medium ") == "medium"
    assert severity.normalize_severity("low") == "low"
    assert severity.normalize_severity("") is None
    assert severity.normalize_severity(None) is None
    assert severity.normalize_severity("CRITICAL") is None  # unknown: caller maps explicitly


def test_severity_rank_matches_current_ordering() -> None:
    # Pins the existing sort semantics: unknown/absent ranks as medium (1).
    assert severity.SEVERITY_RANK == {"low": 2, "medium": 1, "high": 0}
    assert severity.SEVERITY_RANK.get("bogus", 1) == 1


def test_stronger_severity_returns_the_more_severe_level() -> None:
    assert severity.stronger_severity("medium", "high") == "high"
    assert severity.stronger_severity("high", "medium") == "high"
    assert severity.stronger_severity("low", "medium") == "medium"
    assert severity.stronger_severity("low", "low") == "low"


def test_stronger_severity_never_fabricates_a_level() -> None:
    # Off-vocabulary and absent values lose to any canonical value and never
    # become one themselves (P6): two unusable inputs stay unusable.
    assert severity.stronger_severity(None, "low") == "low"
    assert severity.stronger_severity("CRITICAL", "medium") == "medium"
    assert severity.stronger_severity("high", None) == "high"
    assert severity.stronger_severity(None, None) is None
    assert severity.stronger_severity("bogus", 7) is None
