"""Completeness + invariant-immutability guards for the review profile (M14, M15)."""

import pytest

from daydream import review_profile as rp


def test_profile_cannot_alter_invariant_host_owned_keys() -> None:
    for field in ("findings_schema", "verifier", "judge", "scoring", "gold", "skill_name", "model"):
        with pytest.raises(rp.ProfileError):
            rp.parse_profile(f'schema_version = 1\nname = "p"\n{field} = "x"', source="s")
