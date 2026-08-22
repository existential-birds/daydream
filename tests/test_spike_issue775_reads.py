"""Spike: confirm the read-enrichment primitives for issue-775 (throwaway)."""

from tests.test_benchmark_curation import _seed_ready_case


def test_spike_numstat_and_evidence_join(tmp_path, fake_gh):
    from daydream import git_ops
    from daydream.benchmark import curation as cu, snapshot
    from daydream.benchmark.storage import load_json_strict

    ws, case_id, head_sha = _seed_ready_case(tmp_path, fake_gh, lines=4, candidate=True)
    view = cu.get_case(ws, case_id)
    snap = view["snapshot"]
    # base2..head = base.py (1 add 1 del) + feature.py (4 adds)
    proc = git_ops._run_git(snapshot.mirror(ws),
        ["diff", "--numstat", snap["original_base_sha"], snap["original_head_sha"]], retries=0)
    rows = [ln.split("\t") for ln in proc.stdout.splitlines() if ln]
    assert [r[2] for r in rows] == ["base.py", "feature.py"]
    assert (len(rows), sum(int(r[0]) + int(r[1]) for r in rows)) == (2, 6)

    # evidence record for the candidate is reachable via source.import_file
    cand = next(c for c in view["candidates"] if c["exact_acceptable"])
    imp = load_json_strict(ws / view["source"]["import_file"])
    ev = next(e for e in imp["evidence"] if e["source_id"] == cand["source_id"])
    assert (ev["kind"], ev["author"]["login"], ev["author"]["type"]) == \
           ("inline_comment", "alice", "User")
    assert (ev["resolved"], ev["outdated"], ev["commit_id"]) == (False, False, head_sha)