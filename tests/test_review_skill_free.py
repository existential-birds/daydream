"""Skill-token sweep across built-in Deep/Improve prompt + subprocess surfaces (M12)."""
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

FORBIDDEN = re.compile(r"/(beagle-|skill:)|\\$review-|review-verification-protocol|beagle-core")


def _all_builtin_sources() -> Iterator[Any]:
    import daydream

    root = Path(daydream.__file__).parent
    for p in sorted(root.rglob("*.py")):
        if "atif" in p.parts or "benchmark" in p.parts:
            continue
        yield p


def test_no_skill_tokens_in_builtin_prompt_sources() -> None:
    for p in _all_builtin_sources():
        if not any(name in p.name for name in ("prompts", "phases", "coverage", "detection", "sharding")):
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if FORBIDDEN.search(line):
                # historical archive fixtures / legacy-decode comments may keep them
                if "legacy" in line.lower() or "fixture" in line.lower():
                    continue
                m = FORBIDDEN.search(line)
                assert m is not None
                raise AssertionError(
                    f"{p}:{i}: skill token {m.group(0)!r} "
                    f"in {line.strip()!r}"
                )


def test_builtin_default_profile_has_no_skill_tokens() -> None:
    from daydream import review_profile as rp

    p = rp.build_default_profile()
    for key, strat in p.strategies.items():
        assert not FORBIDDEN.search(strat.content), f"{key} default contains a skill token"
