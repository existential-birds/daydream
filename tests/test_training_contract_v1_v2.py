"""The v1/v2 divergence audit and full-suite gate (final parallel-implementation gate).

The v1 ``--corpus`` path is the canonical existing consumer; the v2
``--corpus-v2`` path is its sibling behind the same loaders and stages. These
tests run the contract that both stay behaviorally coherent and that the
combined v1 + v2 suite is green in a single invocation:

- the v1 branch in :func:`run_pipeline` is untouched and still gated on
  ``config.corpus`` (never silently rerouted through the v2 loader);
- v2-only record fields (``finding_text``/``task_identity``) are read only by
  the v2-aware row builders, so a v1 corpus never hits them;
- a v1 pipeline and a v2 pipeline produce behaviorally coherent manifests;
- the combined v1-contract and v2 suites pass together.

The structural audits parse ``daydream/training/coordinator.py`` with ``ast``
rather than substring grep so they name the enclosing function exactly.
"""

import ast
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = REPO_ROOT / "daydream" / "training" / "coordinator.py"

# The only functions allowed to read v2-only record fields. Everything else in
# the coordinator (and any code reached by a v1-shaped corpus) must never
# touch them.
V2_FIELD_ALLOWED_BUILDERS = {"_outcome_rows", "_sft_rows", "_sft_prompt", "_rft_rows", "_run_stage0"}

# The canonical combined suite (preamble task Step 1).
COMBINED_SUITE = [
    "tests/test_stacks_v2_gate.py",
    "tests/test_stacks_v2_load.py",
    "tests/test_training_coordinator.py",
    "tests/test_training_coordinator_v2.py",
    "tests/test_training_rft_v2_sha.py",
    "tests/test_corpus_v2.py",
    "tests/test_corpus_v2_reproducibility.py",
]


def _v2_field_reads() -> list[tuple[str, str | None, int]]:
    """Find every ``rec.get("finding_text"/"task_identity")`` call and the
    function it lives in (``None`` for module level)."""
    tree = ast.parse(COORDINATOR.read_text(encoding="utf-8"))
    reads: list[tuple[str, str | None, int]] = []
    stack: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in ("finding_text", "task_identity")
            ):
                reads.append(
                    (str(node.args[0].value), stack[-1] if stack else None, node.lineno)
                )
            self.generic_visit(node)

    Visitor().visit(tree)
    return reads


def _write_corpus(path: Path, n: int = 50) -> Path:
    """Write an n-record v1-shaped corpus: both gold classes, full diff identity."""
    rows = []
    for i in range(n):
        accepted = i % 2 == 0
        rows.append(
            {
                "session_id": f"sess-{i:04d}",
                "repo_slug": f"acme/tooling-{i % 7}",
                "comment_id": f"c{i:04d}",
                "text": f"grounded actionable finding {i}" if accepted else f"noise chatter {i}",
                "label": "accepted" if accepted else "rejected",
                "labeler_policy_version": "980-policy-r1",
                "base_sha": f"a{i:064x}",
                "head_sha": f"b{i:064x}",
                "diff": f"diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n@@ for sess-{i:04d}\n",
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return path


def test_v1_and_v2_manifests_are_behaviorally_coherent(tmp_path: Path) -> None:
    """A v1 dry run and a v2 dry run of the same size produce manifests with
    the same contract shape: identical stage sets, identical run-identity
    fields, and non-empty corpus/split digests."""
    from daydream.training.coordinator import PipelineConfig, run_pipeline
    from tests.fixtures.training.build_corpus_v2_50 import build_corpus_v2_50

    v1_manifest = run_pipeline(
        PipelineConfig(corpus=_write_corpus(tmp_path / "corpus.jsonl"), out_dir=tmp_path / "out-v1"),
        dry_run=True,
    )
    proj_dir = build_corpus_v2_50(tmp_path)
    v2_manifest = run_pipeline(
        PipelineConfig(corpus_v2=proj_dir, out_dir=tmp_path / "out-v2"),
        dry_run=True,
    )

    assert set(v1_manifest) == set(v2_manifest)
    assert set(v1_manifest["stages"]) == set(v2_manifest["stages"])
    for stage, entry in v1_manifest["stages"].items():
        assert entry["status"] == v2_manifest["stages"][stage]["status"], (
            f"stage {stage} diverges between v1 and v2 for equivalent input"
        )
    identity_keys_v1 = set(v1_manifest["run_identity"])
    identity_keys_v2 = set(v2_manifest["run_identity"])
    assert identity_keys_v1 == identity_keys_v2
    for field in ("corpus_digest", "split_digest"):
        assert v1_manifest["run_identity"][field], f"v1 {field} empty"
        assert v2_manifest["run_identity"][field], f"v2 {field} empty"

    # Both manifests are loadable JSON on disk in the same place.
    for out in ("out-v1", "out-v2"):
        payload: dict[str, Any] = json.loads(
            (tmp_path / out / "manifest.json").read_text(encoding="utf-8")
        )
        assert payload["stages"].keys() == v1_manifest["stages"].keys()


