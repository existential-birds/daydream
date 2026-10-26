"""Legacy manifests omit absent profile provenance without inventing values."""

from daydream.archive import manifest as m


def test_legacy_manifest_without_profile_fields_still_serializes() -> None:
    # Read legacy JSON without rewriting it; serialization omits absent profile fields.
    man = m.Manifest(  # profile_* all None (legacy shape)
        schema_version="1", session_id="s", archived_at="2026-01-01T00:00:00Z", status="complete",
    )
    d = man.to_dict()
    assert "profile_digest" not in d and "profile_name" not in d
    assert "profile_schema_version" not in d and "profile_source_kind" not in d
