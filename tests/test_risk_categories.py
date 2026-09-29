import pytest

from daydream.deep.risk_categories import (
    MANDATORY_RISK_CATEGORIES,
    UnknownRiskCategoryError,
    categories_in,
    resolve_mandatory_categories,
)


@pytest.mark.parametrize(
    ("category", "text"),
    [
        ("security", "the handler must authenticate the caller's token"),
        ("concurrency", "this path can deadlock while holding the mutex"),
        ("persistence", "the change runs `alter table users`"),
        ("public-interface", "the grpc endpoint gains a field"),
        ("migration", "see migrations/0042_add_flag.py"),
    ],
)
def test_each_mandatory_category_matches_its_declared_triggers(category: str, text: str) -> None:
    assert category in categories_in(text)


def test_builtin_categories_cannot_be_removed_and_extras_are_additive() -> None:
    assert resolve_mandatory_categories(()) == MANDATORY_RISK_CATEGORIES
    assert resolve_mandatory_categories([])[: len(MANDATORY_RISK_CATEGORIES)] == MANDATORY_RISK_CATEGORIES


def test_unknown_extra_category_fails_loudly_naming_the_value() -> None:
    with pytest.raises(UnknownRiskCategoryError, match="public_interface"):
        resolve_mandatory_categories(["public_interface"])
