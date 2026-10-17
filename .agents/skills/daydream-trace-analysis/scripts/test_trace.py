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

    def test_window_keeps_decisions_results_and_invocation_boundaries_bounded(self):
        # Mine identifies calls but cannot show whether a later decision recovered.
        doc = self.record["trajectories"]["value"]["documents"][1]
        doc["steps"][0]["message"] = "Investigate the failure " + "x" * 10000
        doc["steps"][0]["observation"]["results"][0]["extra"]["native_payload"] = "z" * 10000
        doc["steps"][1]["message"] = "Changed approach after observing usage"
        doc["steps"][1]["reasoning_content"] = "Fresh verification context " + "y" * 10000
        # The selected range excludes later steps and independent sibling output.
        excluded = {"step_id": 3, "source": "agent", "tool_calls": [
            {"tool_call_id": "excluded", "function_name": "shell", "arguments": {"command": "other scope"}}],
            "observation": {"results": [{"source_call_id": "excluded", "content": "outside-window " * 10000}]}}
        doc["steps"].append(excluded)
        self.record["trajectories"]["value"]["documents"][0]["steps"] = [excluded]
        self.path.write_text(json.dumps(self.record))
        anchor = "/trajectories/value/documents/1/steps/0/tool_calls/1"
        report = self.cli("window", "--pointer", anchor, "--before", "2", "--after", "1",
                          "--chars", "50", "--evidence-limit", "1")
        self.assertEqual(report["anchor_pointer"], anchor)
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["document_step_count"], 3)
        self.assertEqual([row["invocation_id"] for row in report["items"]], ["one", "two"])
        self.assertEqual(report["items"][0]["tools"]["total"], 2)
        self.assertEqual(report["items"][0]["tools"]["remaining"], 1)
        selected = report["items"][0]["tools"]["items"][0]
        self.assertEqual(selected["pointer"], anchor)
        self.assertEqual(selected["results"][0]["content"], "usage")
        self.assertTrue(selected["failed"])
        self.assertLessEqual(len(selected["results"][0]["extra"]), 50)
        self.assertEqual(report["items"][1]["message"], "Changed approach after observing usage")
        self.assertEqual(report["items"][1]["phase"], "merge")
        self.assertLessEqual(len(report["items"][0]["message"]), 50)
        self.assertLessEqual(len(report["items"][1]["reasoning_content"]), 50)
        self.assertLess(len(json.dumps(report)), 4000)
        self.assertNotIn("outside-window", json.dumps(report))
        page = self.cli("window", "--pointer", anchor, "--after", "1", "--offset", "1", "--limit", "1")
        self.assertEqual(page["items"][0]["step_id"], 2)
        self.assertEqual(page["remaining"], 0)

    def test_window_keeps_anchored_matched_and_unmatched_results_in_recorded_order(self):
        # A later result of a later call needs priority at both bounded levels.
        step = self.record["trajectories"]["value"]["documents"][1]["steps"][0]
        step["observation"]["results"].extend([
            {"source_call_id": "item_2", "content": "late diagnostic", "extra": {"native": "x" * 10000}},
            {"source_call_id": "missing_1", "content": "earlier orphan"},
            {"source_call_id": "missing_2", "content": "anchored orphan", "extra": {"native": "y" * 10000}}])
        self.path.write_text(json.dumps(self.record))
        prefix = "/trajectories/value/documents/1/steps/0"
        anchor = prefix + "/observation/results/2/extra/native"
        report = self.cli("window", "--pointer", anchor, "--before", "0", "--after", "0",
                          "--chars", "40", "--evidence-limit", "1")
        selected = report["items"][0]["tools"]["items"][0]
        self.assertEqual(selected["tool_call_id"], "item_2")
        self.assertEqual(selected["result_count"], 2)
        self.assertEqual(selected["results_remaining"], 1)
        self.assertEqual(selected["results"][0]["content"], "late diagnostic")
        self.assertLessEqual(len(selected["results"][0]["extra"]), 40)
        ordered = self.cli("window", "--pointer", anchor, "--evidence-limit", "2")["items"][0]["tools"]["items"]
        self.assertEqual([row["tool_call_id"] for row in ordered], ["item_1", "item_2"])
        self.assertEqual([row["content"] for row in ordered[1]["results"]], ["usage", "late diagnostic"])
        anchor = prefix + "/observation/results/4/content"
        report = self.cli("window", "--pointer", anchor, "--evidence-limit", "1", "--chars", "40")
        unmatched = report["items"][0]["unmatched_results"]
        self.assertEqual(unmatched["total"], 2)
        self.assertEqual(unmatched["remaining"], 1)
        self.assertEqual(unmatched["items"][0]["content"], "anchored orphan")
        self.assertLessEqual(len(unmatched["items"][0]["extra"]), 40)

    def test_window_rejects_nonexistent_nonstep_and_invalid_pointers_and_negative_bounds(self):
        # A plausible-looking unresolved suffix must not masquerade as evidence.
        for pointer in ("/trajectories/value/documents/1/steps/0/missing", "/run_id"):
            with self.subTest(pointer=pointer):
                self.cli("window", "--pointer", pointer, ok=False)
        self.cli("window", "--pointer", "/trajectories/value/documents/1/steps/0", "--before", "-1", ok=False)
        for suffix in ("-1", "00", "+1"):
            pointer = f"/trajectories/value/documents/1/steps/0/observation/results/{suffix}/content"
            for command in ("window", "show"):
                with self.subTest(command=command, pointer=pointer):
                    self.cli(command, "--pointer", pointer, ok=False)

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

    def test_physical_line_selects_duplicate_ids_without_accepting_wrong_or_invalid_records(self):
        # Cohorts deduplicate identical IDs; run-id selection alone remains ambiguous.
        encoded = json.dumps(self.record)
        self.path.write_text("\n" + encoded + "\n" + encoded + "\n")
        self.assertIn("select", self.cli("summary", "--run-id", "run", ok=False).lower())
        selected = self.cli("summary", "--run-id", "run", "--line", "3")
        self.assertEqual((selected["run_id"], selected["line"]), ("run", 3))
        self.assertEqual(selected["document_count"], 2)
        self.assertEqual(self.cli("summary", "--line", "2")["line"], 2)
        self.assertIn("match", self.cli("summary", "--run-id", "other", "--line", "3", ok=False))
        self.path.write_text("{broken\n" + encoded + "\n\n")
        self.assertEqual(self.cli("summary", "--run-id", "run", "--line", "2")["line"], 2)
        self.assertIn(":1", self.cli("summary", "--line", "1", ok=False))
        for line in ("0", "-1", "3", "4"):
            with self.subTest(line=line):
                self.assertIn("line", self.cli("summary", "--line", line, ok=False))

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

    def test_timeline_bounds_generation_lifecycle_and_preserves_retrieval(self):
        # Completed invocations retain model output; phase-only coverage misses this nested payload.
        doc = self.record["trajectories"]["value"]["documents"][1]
        doc["extra"]["subtrajectories"][0]["generation_lifecycle"] = {
            "drafts": [{"generation_id": "generation-1", "choice_parts": [
                {"type": "text", "text": "x" * 100000}]}]}
        self.path.write_text(json.dumps(self.record))
        report = self.cli("timeline", "--doc", "1", "--chars", "40")
        row = next(r for r in report["items"] if r["invocation_id"] == "one")
        self.assertLess(len(json.dumps(report)), 3000)
        self.assertIsInstance(row["generation_lifecycle"], str)
        self.assertLessEqual(len(row["generation_lifecycle"]), 40)
        pointer = row["pointer"] + "/generation_lifecycle/drafts/0/choice_parts/0/text"
        shown = self.cli("show", "--pointer", pointer, "--offset", "99960", "--chars", "40")
        self.assertEqual(shown["total_chars"], 100000)
        self.assertEqual(shown["text"], "x" * 40)
        self.assertEqual(shown["remaining"], 0)

    def test_unmatched_interruption_is_visible_and_dispatch_links_are_not_errors(self):
        doc = self.record["trajectories"]["value"]["documents"][1]
        doc["steps"].append({"step_id": 3, "observation": {"results": [
            {"source_call_id": None, "content": "[interrupted]", "extra": {"status": "interrupted"}},
            {"subagent_trajectory_ref": [{"trajectory_id": "child"}]}]}})
        result = doc["steps"][-1]["observation"]["results"][0]
        for references in ({}, {"subagent_trajectory_ref": None}, {"subagent_trajectory_ref": []}):
            with self.subTest(references=references):
                result.update(references)
                self.path.write_text(json.dumps(self.record))
                report = self.cli("tools", "--doc", "1")
                self.assertEqual(report["unmatched_results"]["total"], 1)
                row = report["unmatched_results"]["items"][0]
                self.assertTrue(row["failed"])
                self.assertEqual(row["pointer"],
                                 "/trajectories/value/documents/1/steps/2/observation/results/0")
                report = self.cli("mine", "--doc", "1")
                self.assertTrue(any(r["kind"] == "unmatched_result_failure" for r in report["items"]))


if __name__ == "__main__":
    unittest.main()
