"""Deterministic leak-resistant content compiler (issue #778)."""
from pathlib import Path
import hashlib

REPO = Path(__file__).resolve().parents[1]


def _seed_bare_bundle(tmp_path: Path) -> tuple[Path, bytes]:
    """Build a real base/head repo + bare mirror + build_bundle; return (mirror, bundle_bytes)."""
    import os
    import subprocess
    from daydream.benchmark import snapshot
    src = tmp_path / "src"; src.mkdir()
    env = {**os.environ, "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
           "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}

    def g(*a: str) -> str:
        return subprocess.run(["git", "-C", str(src), *a], check=True,
                              env=env, capture_output=True).stdout.decode().strip()

    g("init", "-q"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    (src / "f.py").write_text("x=1\n"); g("add", "."); g("commit", "-qm", "base"); base = g("rev-parse", "HEAD")
    (src / "f.py").write_text("x=2\n"); g("add", "."); g("commit", "-qm", "head"); head = g("rev-parse", "HEAD")
    m = snapshot.ensure_mirror(tmp_path, "o/r")
    # push the base/head commits (objects + refs) into the mirror so build_bundle can resolve trees
    subprocess.run(["git", "-C", str(src), "push", str(m),
                    f"{base}:refs/heads/base", f"{head}:refs/heads/head"],
                   check=True, env=env, capture_output=True)
    bundle = tmp_path / "b.bundle"
    snapshot.build_bundle(m, base, head, bundle)
    return m, bundle.read_bytes()


def test_spike_persisted_pull_request_field_set():
    """The import persists pull_request as a raw dict; the compiler must tolerate a missing body."""
    from daydream.benchmark.schema import ImportDocument
    assert ImportDocument.model_fields["pull_request"].annotation is dict
    # Field set is pinned by the constructor at github_import.py:1080-1090: it carries
    # number/url/title/state/base/head/created_at/updated_at/author and NO body.
    from daydream.benchmark import github_import as gi
    assert hasattr(gi, "fetch_and_normalize")  # module import guard (no network call here)


def test_spike_bundle_heads_is_exactly_base_head(tmp_path):
    from daydream.benchmark import snapshot
    m, bundle_bytes = _seed_bare_bundle(tmp_path)
    (tmp_path / "b.bundle").write_bytes(bundle_bytes)
    heads = snapshot.bundle_heads(tmp_path / "b.bundle")
    assert heads == {"refs/heads/base", "refs/heads/head"}


def test_derive_task_key_is_opaque_and_deterministic():
    from daydream.benchmark.harbor import build
    case_id = "pr-000101-1a2b3c4d5e6f"
    k = build.derive_task_key(case_id)
    assert k.startswith("case-") and len(k) == len("case-") + 12
    assert k == build.derive_task_key(case_id)          # deterministic
    assert k != build.derive_task_key("pr-000101-1a2b3c4d5e60")  # distinct case -> distinct key
    assert "pr-" not in k and case_id not in k          # reveals no authoring case id
    assert all(c in "0123456789abcdef" for c in k[len("case-"):])  # hex suffix