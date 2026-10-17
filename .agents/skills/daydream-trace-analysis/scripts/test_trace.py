"""CLI contracts for the standalone parser; no runtime code or private captures required.

Existing recorder tests do not exercise this offline consumer. These cases protect
evidence identity, accounting, and bounded retrieval against realistic ATIF traps.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("trace.py")


class TraceCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "capture.jsonl"
        steps = [
            {"step_id": 1, "source": "agent", "extra": {"daydream_phase": "deep"},
             "tool_calls": [
                 {"tool_call_id": "item_1", "function_name": "shell", "arguments": {"command": "pwd"}},
                 {"tool_call_id": "item_2", "function_name": "shell", "arguments": {"command": "bad"}}],
             "observation": {"results": [
                 {"source_call_id": "item_2", "content": "usage", "extra": {"exit_code": 2}},
                 {"source_call_id": "item_1", "content": "/repo", "extra": {"exit_code": 0}}]}},
            {"step_id": 2, "source": "agent", "extra": {"daydream_phase": "merge"},
             "tool_calls": [{"tool_call_id": "item_1", "function_name": "shell",
                             "arguments": {"command": "pwd"}}],
             "observation": {"results": [
                 {"source_call_id": "item_1", "content": "/other", "extra": {"exit_code": 0}}]}},
        ]
        root = {"schema_version": "ATIF-v1.7", "trajectory_id": "root", "session_id": "run",
                "steps": steps, "extra": {"subtrajectories": [
                    {"invocation_id": "one", "step_ids": [1]},
                    {"invocation_id": "two", "step_ids": [2]}]},
                "final_metrics": {"total_cost_usd": 5, "extra": {
                    "daydream_metric_scope": "whole_run_including_forks"}}}
        child = {"trajectory_id": "child", "steps": [steps[0]], "final_metrics": {"total_cost_usd": 3}}
        self.record = {"schema_version": "daydream.run.v1", "run_id": "run", "outcome": "success",
                       "trajectories": {"status": "available", "value": {
                           "root_trajectory_id": "root", "documents": [child, root]}},
                       "completeness": {"trajectory": "complete"}}
        self.path.write_text("\n" + json.dumps(self.record) + "\n")

    def cli(self, command, *args, ok=True):
        result = subprocess.run([sys.executable, str(SCRIPT), command, str(self.path), *args],
                                capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result.stderr

    def test_reordered_results_and_reused_ids_keep_exact_identity(self):
        report = self.cli("tools", "--doc", "1")
        rows = report["items"]
        self.assertEqual([r["invocation_id"] for r in rows], ["one", "one", "two"])
        self.assertEqual(rows[0]["results"][0]["content"], "/repo")
        self.assertFalse(rows[0]["failed"])
        self.assertTrue(rows[1]["failed"])
        self.assertEqual([r["result_count"] for r in rows], [1, 1, 1])
        self.assertEqual(rows[0]["results"][0]["pointer"],
                         "/trajectories/value/documents/1/steps/0/observation/results/1")
        self.assertEqual(rows[2]["results"][0]["content"], "/other")
        self.assertEqual(rows[2]["results"][0]["pointer"],
                         "/trajectories/value/documents/1/steps/1/observation/results/0")
        self.assertEqual(report["line"], 2)

    def test_summary_uses_declared_root_without_adding_fork_billing(self):
        report = self.cli("summary")
        self.assertEqual(report["root_trajectory_id"], "root")
        self.assertEqual(report["root_metrics"]["total_cost_usd"], 5)
        self.assertEqual(report["document_count"], 2)
        self.assertEqual(report["tool_call_count"], 5)

    def test_mining_returns_candidates_with_scope_and_error_evidence(self):
        report = self.cli("mine")
        duplicates = [r for r in report["items"] if r["kind"] == "identical_tool_call"]
        self.assertEqual(sorted(r["count"] for r in duplicates), [2, 3])
        pwd = next(r for r in duplicates if r["count"] == 3)
        self.assertEqual(pwd["scope"], "across_trajectories")
        self.assertEqual(len({r["pointer"] for r in pwd["evidence"]}), 3)
        self.assertTrue(any(r["kind"] == "tool_failure" for r in report["items"]))

    def test_search_and_show_bound_large_strings_and_support_slices(self):
        self.record["original_task"] = {"status": "available", "value": "a" * 10000 + "needle"}
        self.path.write_text(json.dumps(self.record) + "\n")
        report = self.cli("search", "--query", "needle", "--chars", "40", "--limit", "1")
        self.assertEqual(report["total"], 1)
        self.assertIn("needle", report["items"][0]["excerpt"])
        self.assertLessEqual(len(report["items"][0]["excerpt"]), 40)
        shown = self.cli("show", "--pointer", "/original_task/value", "--offset", "9998", "--chars", "8")
        self.assertEqual(shown["text"], "aaneedle")
        self.assertEqual(shown["total_chars"], 10006)
        # Case folding expands these characters; evidence offsets stay in the source text.
        for prefix in ("ß" * 100, "İ" * 100):
            with self.subTest(prefix=prefix[0]):
                value = prefix + "needle"
                self.record["original_task"]["value"] = value
                self.path.write_text(json.dumps(self.record) + "\n")
                report = self.cli("search", "--query", "NEEDLE", "--chars", "40")
                row = report["items"][0]
                self.assertEqual(row["match_offset"], len(prefix))
                self.assertEqual(row["excerpt_offset"], len(prefix) - 40 // 3)
                self.assertEqual(row["excerpt"], value[row["excerpt_offset"]:row["excerpt_offset"] + 40])
                self.assertIn("needle", row["excerpt"])

    def test_multiple_records_require_selection_and_bad_json_is_not_silently_skipped(self):
        second = dict(self.record, run_id="other")
        self.path.write_text(json.dumps(self.record) + "\n" + json.dumps(second) + "\n")
        self.assertIn("select", self.cli("summary", ok=False).lower())
        self.assertEqual(self.cli("summary", "--run-id", "other")["run_id"], "other")
        self.path.write_text(json.dumps(self.record) + "\n{" )
        self.assertIn(":2", self.cli("summary", ok=False))

    def test_raw_atif_and_unavailable_trajectories_are_distinguished(self):
        root = self.record["trajectories"]["value"]["documents"][1]
        self.path.write_text(json.dumps(root))
        self.assertEqual(self.cli("summary")["document_count"], 1)
        # Valid ATIF producers may serialize optional containers as null instead of omitting them.
        for nullable_path in (("extra",), ("steps", 0, "extra"), ("steps", 0, "tool_calls"),
                              ("steps", 0, "observation"),
                              ("steps", 0, "observation", "results", 0, "extra")):
            raw = json.loads(json.dumps(root))
            parent = raw
            for key in nullable_path[:-1]:
                parent = parent[key]
            parent[nullable_path[-1]] = None
            self.path.write_text(json.dumps(raw))
            for command in ("summary", "tools", "mine", "timeline"):
                with self.subTest(nullable_path=nullable_path, command=command):
                    self.assertEqual(self.cli(command)["run_id"], "run")
        self.record["trajectories"] = {"status": "unavailable", "reason": "capture failed", "value": None}
        self.path.write_text(json.dumps(self.record))
        report = self.cli("summary")
        self.assertEqual(report["document_count"], 0)
        self.assertEqual(report["availability"]["trajectories"]["reason"], "capture failed")

    def test_interrupted_phase_retains_a_retrievable_start_pointer(self):
        doc = self.record["trajectories"]["value"]["documents"][1]
        doc["extra"]["phase_events"] = [
            {"event": "phase_start", "scope_id": "scope", "session_id": "run", "phase": "deep",
             "timestamp": "2026-10-17T00:00:00Z", "metadata": {"reason": "x" * 1000}}]
        self.path.write_text(json.dumps(self.record))
        row = next(r for r in self.cli("timeline")["items"] if r["kind"] == "open_phase")
        self.assertEqual(row["trajectory_id"], "root")
        self.assertIsNone(row["duration_s"])
        value = self.cli("show", "--pointer", row["pointer"], "--chars", "2000")
        self.assertEqual(json.loads(value["text"])["metadata"]["reason"], "x" * 1000)

    def test_unmatched_interruption_is_visible_and_dispatch_links_are_not_errors(self):
        doc = self.record["trajectories"]["value"]["documents"][1]
        doc["steps"].append({"step_id": 3, "observation": {"results": [
            {"source_call_id": None, "content": "[interrupted]", "extra": {"status": "interrupted"}},
            {"subagent_trajectory_ref": [{"trajectory_id": "child"}]}]}})
        self.path.write_text(json.dumps(self.record))
        report = self.cli("tools", "--doc", "1")
        self.assertEqual(report["unmatched_results"]["total"], 1)
        row = report["unmatched_results"]["items"][0]
        self.assertTrue(row["failed"])
        report = self.cli("mine", "--doc", "1")
        self.assertTrue(any(r["kind"] == "unmatched_result_failure" for r in report["items"]))


if __name__ == "__main__":
    unittest.main()
