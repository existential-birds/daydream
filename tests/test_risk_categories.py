import pytest

from daydream.deep.risk_categories import (
    MANDATORY_RISK_CATEGORIES,
    UnknownRiskCategoryError,
    categories_in,
    validate_extra_categories,
)


@pytest.mark.parametrize(("category", "text"),
    [("security", "the handler must authenticate the caller's token"),
        ("concurrency", "this path can deadlock while holding the mutex"),
        ("persistence", "the change runs `alter table users`"), ("public-interface", "the grpc endpoint gains a field"),
        ("migration", "see migrations/0042_add_flag.py"),
    ],
)
def test_each_mandatory_category_matches_its_declared_triggers(category: str, text: str) -> None:
    assert category in categories_in(text)

def test_configured_extra_categories_are_validated_and_cannot_widen() -> None:
    """The declared vocabulary is the mandatory set, so a valid extra is a no-op."""
    validate_extra_categories(())
    validate_extra_categories(MANDATORY_RISK_CATEGORIES)
    validate_extra_categories(["security", "security"])

def test_unknown_extra_category_fails_loudly_naming_the_value() -> None:
    with pytest.raises(UnknownRiskCategoryError, match="public_interface"):
        validate_extra_categories(["public_interface"])
