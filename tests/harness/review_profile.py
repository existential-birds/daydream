"""Shared default-profile strategy lookup for prompt tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TypedDict

from daydream.pr_review import PRInfo
from daydream.review_profile import ResolvedProfile, build_default_profile


class PromptPaths(TypedDict):
    """The five on-disk path kwargs shared by the per-stack/fallback/arbiter builders.

    Declaring each key's type explicitly lets mypy reconcile ``**p`` unpacking
    with the builders' per-parameter signatures; a plain ``dict[str, Path]`` would
    spill ``Path`` onto unrelated kwargs like ``prior_commits``/``is_docs_only``.
    """

    diff_path: Path
    intent_path: Path
    alternatives_path: Path
    output_path: Path
    cwd: Path


def prompt_paths(tmp_path: Path, output_name: str = "stack-review.md") -> PromptPaths:
    """Canonical deep-review artifact path set; only the output basename varies."""
    return {
        "diff_path": tmp_path / ".daydream" / "diff.patch",
        "intent_path": tmp_path / ".daydream" / "deep" / "intent.md",
        "alternatives_path": tmp_path / ".daydream" / "deep" / "alternatives.json",
        "output_path": tmp_path / ".daydream" / "deep" / output_name,
        "cwd": tmp_path,
    }


def sample_pr() -> PRInfo:
    """The canonical test PRInfo used across submission/review/severity tests."""
    return PRInfo(
        number=42,
        head_sha="head123",
        base_sha="base456",
        base_ref="main",
        head_ref="feature",
        owner="acme",
        repo="widgets",
        url="https://github.com/acme/widgets/pull/42",
    )


def default_strategy(stage: str) -> str:
    return build_default_profile().strategies[stage].content


def exploration_strategies() -> dict[str, str]:
    """Return the default profile's exploration.* strategy content by key."""
    return {
        name: value.content
        for name, value in build_default_profile().strategies.items()
        if name.startswith("exploration.")
    }


def independent_alternatives_profile() -> ResolvedProfile:
    """Explicitly request the separate design pass in integration fixtures."""
    profile = build_default_profile()
    strategies = dict(profile.strategies)
    strategies["alternatives"] = replace(
        strategies["alternatives"], content=strategies["alternatives"].content + "\nCustom independent design policy.",
    )
    return ResolvedProfile(profile=replace(profile, strategies=strategies), source_kind="test")


def independent_exploration_profile(
    resolved: ResolvedProfile | None = None,
) -> ResolvedProfile:
    """Request model exploration when testing specialist dispatch and lifecycle."""
    profile = resolved.profile if resolved is not None else build_default_profile()
    strategies = dict(profile.strategies)
    key = "exploration.dependency_trace"
    strategies[key] = replace(
        strategies[key], content=strategies[key].content + "\nCustom dependency exploration policy.",
    )
    return ResolvedProfile(profile=replace(profile, strategies=strategies), source_kind="test")
