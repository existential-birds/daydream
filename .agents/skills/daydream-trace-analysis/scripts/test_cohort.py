"""Cohort CLI contracts on real JSONL files, independent of runtime imports.

test_trace.py owns single-run retrieval. It cannot catch biased cohort selection,
lost malformed records, changed frozen evidence, or split sibling assignments.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("cohort.py")


def capture(run_id, captured_at, outcome="success", padding=0):
    return {"schema_version": "daydream.run.v1", "run_id": run_id, "captured_at": captured_at,
            "outcome": outcome, "completeness": {"trajectory": "complete"},
            "provenance": {"producer": {"version": "1", "commit": "producer-sha"},
                           "resolved_phases": {"fix": {"backend": "codex", "model": "model"}},
                           "effective_configuration": {"backend": "codex"},
                           "latency_profile": {"name": "balanced", "fix_budget_s": 100},
                           "file_configuration": {"review_max_turns": 10}},
            "original_task": {"status": "available", "value": {
                "repository": {"repo_slug": "owner/repo"},
                "analyzed_revision": {"head_sha": "reviewed-sha", "diff_key": "patch-key"},
                "diff_sha256": "d" * 64, "diff": "private diff", "changed_files": ["a.py"]}},
            "trajectories": {"status": "available", "value": {"root_trajectory_id": "root",
                "documents": [{"trajectory_id": "root", "steps": [{"message": "x" * padding}]},
                              {"trajectory_id": "sibling", "steps": []}]}}}


class CohortCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "runs.jsonl"

    def write(self, *records, path=None):
        (path or self.path).write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    def cli(self, *args, code=0):
        result = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
        self.assertEqual(result.returncode, code, result.stderr)
        return json.loads(result.stdout) if result.stdout else result.stderr

    def manifest(self):
        report = self.cli("select", self.path)
        path = self.root / "cohort.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return path, report

    def test_captured_time_selection_keeps_controls_and_true_producer_identity(self):
        # mtime sampling previously changes membership without a change to the runs.
        failed = capture("failed", "2026-10-16T23:30:00-04:00", "failed")
        unavailable = capture("missing", "2026-10-17T02:00:00Z", "interrupted")
        unavailable["trajectories"] = {"status": "failed", "value": None, "reason": "capture failed"}
        success = capture("success", "2026-10-17T01:00:00Z")
        self.path.write_text("\n" + json.dumps(failed) + "\n" + json.dumps(unavailable) + "\n")
        other = self.root / "other.jsonl"
        self.write(success, path=other)
        os.utime(other, (1, 1))
        first = self.cli("select", self.root)
        os.utime(other, (9999999, 9999999))
        second = self.cli("select", self.root)
        self.assertEqual(first, second)
        self.assertEqual([r["run_id"] for r in first["runs"]], ["failed", "missing", "success"])
        row = first["runs"][0]
        self.assertEqual(row["line"], 2)
        self.assertEqual(row["source"], str(self.path.resolve()))
        encoded = json.dumps(failed, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(row["sha256"], hashlib.sha256(encoded).hexdigest())
        self.assertEqual(row["bytes"], len(encoded))
        self.assertEqual(row["context"]["producer"]["commit"], "producer-sha")
        self.assertEqual(row["context"]["analyzed_revision"]["head_sha"], "reviewed-sha")
        self.assertEqual(row["context"]["latency_profile"]["fix_budget_s"], 100)
        self.assertEqual(row["context"]["file_configuration"]["review_max_turns"], 10)
        self.assertNotIn("private diff", json.dumps(first))
        self.assertIsNone(row["context"]["profile"])
        self.assertEqual(first["runs"][1]["availability"]["trajectories"]["reason"], "capture failed")
        limited = self.cli("select", self.root, "--limit", 1)
        self.assertEqual([r["run_id"] for r in limited["runs"]], ["failed"])
        self.assertEqual([r["reason"] for r in limited["exclusions"]], ["limit", "limit"])

    def test_filters_use_exact_json_values_and_inclusive_timestamp_bounds(self):
        records = [capture("yes", "2026-10-17T00:00:00Z"), capture("no", "2026-10-18T00:00:00Z"),
                   capture("unknown", None)]
        for row, value in zip(records, (True, 1, True), strict=True):
            row["provenance"]["escaped/~key"] = value
        self.write(*records)
        report = self.cli("select", self.path, "--where", "/provenance/escaped~1~0key=true",
                          "--since", "2026-10-17T00:00:00+00:00", "--until", "2026-10-17T00:00:00Z")
        self.assertEqual([r["run_id"] for r in report["runs"]], ["yes"])
        self.assertEqual({r["run_id"]: r["reason"] for r in report["exclusions"]},
                         {"no": "where", "unknown": "captured_at_unknown"})
        report = self.cli("select", self.path, "--where", '/missing="x"')
        self.assertEqual(report["runs"], [])
        self.assertEqual({r["reason"] for r in report["exclusions"]}, {"where_missing"})
        self.assertIn("timezone", self.cli("select", self.path, "--since", "2026-10-17T00:00:00", code=2))

    def test_malformed_lines_are_reported_without_losing_valid_neighbors(self):
        self.path.write_bytes((json.dumps(capture("one", "2026-10-17T00:00:00Z")) + "\n").encode()
                              + b'{broken\n\xff\n{"schema_version":"other","run_id":"bad"}\n'
                              + (json.dumps(capture("two", "2026-10-18T00:00:00Z")) + "\n").encode())
        report = self.cli("select", self.path, code=2)
        self.assertEqual({r["run_id"] for r in report["runs"]}, {"one", "two"})
        self.assertEqual([r["line"] for r in report["errors"]], [2, 3, 4])
        manifest = self.root / "partial.json"
        manifest.write_text(json.dumps(report))
        self.assertIn("errors", self.cli("batches", manifest, "--count", 2, "--question", "why", code=2))

    def test_identical_duplicates_are_once_and_conflicting_run_ids_are_rejected(self):
        record = capture("same", "2026-10-17T00:00:00Z")
        self.write(record, record, capture("other", "2026-10-17T00:00:00Z"))
        report = self.cli("select", self.path)
        self.assertEqual({r["run_id"] for r in report["runs"]}, {"same", "other"})
        self.assertEqual(report["exclusions"][0]["reason"], "duplicate")
        self.write(record, dict(record, outcome="failed"), capture("other", "2026-10-17T00:00:00Z"))
        report = self.cli("select", self.path, code=2)
        self.assertEqual([r["run_id"] for r in report["runs"]], ["other"])
        self.assertEqual({r["reason"] for r in report["exclusions"]}, {"conflicting_run_id"})
        self.assertIn("same", json.dumps(report["errors"]))

    def test_batch_plans_balance_whole_runs_and_reject_stale_or_tampered_evidence(self):
        records = [capture(name, "2026-10-17T00:00:00Z", padding=size)
                   for name, size in (("a", 9000), ("b", 6000), ("c", 5000), ("d", 1000))]
        self.write(*records)
        manifest, report = self.manifest()
        plan = self.cli("batches", manifest, "--count", 2, "--question", "Where was scope lost?",
                        "--chars", 120, "--max-observations", 3)
        self.assertEqual(plan["cohort_id"], report["cohort_id"])
        self.assertEqual(plan["evidence_chars"], 120)
        self.assertEqual(plan["max_observations"], 3)
        self.assertEqual([b["run_ids"] for b in plan["batches"]], [["a", "d"], ["b", "c"]])
        by_id = {r["run_id"]: r for r in report["runs"]}
        for batch in plan["batches"]:
            self.assertEqual(batch["bytes"], sum(by_id[r]["bytes"] for r in batch["run_ids"]))
        self.assertEqual(plan, self.cli("batches", manifest, "--count", 2, "--question", "Where was scope lost?",
                                        "--chars", 120, "--max-observations", 3))
        records[0]["trajectories"]["value"]["documents"][1]["steps"].append({"message": "changed"})
        self.write(*records)
        self.assertIn("changed", self.cli("batches", manifest, "--count", 2, "--question", "why", code=2))
        report["runs"][0]["bytes"] += 1
        manifest.write_text(json.dumps(report))
        self.assertIn("cohort_id", self.cli("batches", manifest, "--count", 2, "--question", "why", code=2))

    def test_reload_checks_physical_line_and_returns_all_sibling_documents(self):
        # Consumers must use the physical line, not a same-ID record elsewhere in the file.
        record = capture("one", "2026-10-17T00:00:00Z")
        self.path.write_text("\n" + json.dumps(record) + "\n")
        _, report = self.manifest()
        spec = importlib.util.spec_from_file_location("cohort_helper_test", SCRIPT)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        self.assertEqual([value for _, value in helper.iter_runs(report["runs"])], [record])
        self.path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "changed|line"):
            list(helper.iter_runs(report["runs"]))

    def test_grouped_reload_preserves_physical_records_and_rejects_late_source_changes(self):
        # A source stream must preserve skipped physical lines, siblings, and checks across files.
        first = capture("first", "2026-10-17T00:00:00Z")
        last = capture("last", "2026-10-18T00:00:00Z")
        other = capture("other", "2026-10-19T00:00:00Z")
        self.path.write_text("\n" + json.dumps(first) + "\n\n" + json.dumps(last) + "\n")
        other_path = self.root / "other.jsonl"
        self.write(other, path=other_path)
        manifest = self.cli("select", self.root)
        spec = importlib.util.spec_from_file_location("cohort_stream_test", SCRIPT)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        loaded = {row["run_id"]: (row, record) for row, record in helper.iter_runs(manifest["runs"])}
        expected = {record["run_id"]: record for record in (first, last, other)}
        self.assertEqual({run_id: record for run_id, (_, record) in loaded.items()}, expected)
        self.assertEqual(loaded["first"][0]["line"], 2)
        self.assertEqual(loaded["last"][0]["line"], 4)
        self.assertEqual(loaded["other"][0]["source"], str(other_path.resolve()))
        last["trajectories"]["value"]["documents"][1]["steps"].append({"message": "changed sibling"})
        self.path.write_text("\n" + json.dumps(first) + "\n\n" + json.dumps(last) + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            list(helper.iter_runs(manifest["runs"]))
        self.path.write_text("\n" + json.dumps(first) + "\n")
        with self.assertRaisesRegex(ValueError, "missing|line"):
            list(helper.iter_runs(manifest["runs"]))


if __name__ == "__main__":
    unittest.main()
