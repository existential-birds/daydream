"""Offline report CLI contracts absent from the single-run trace parser tests.

Real frozen JSONL inputs exercise recurrence denominators, assignment provenance,
and matched experiment conditions; no mocks can silently supply valid evidence.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ReportCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / "scripts" / "reports.py"
        shutil.copytree(Path(__file__).parent, self.script.parent, ignore=shutil.ignore_patterns("__pycache__"))
        self.source = self.root / "capture.jsonl"
        self.records = [{"schema_version": "daydream.run.v1", "run_id": f"run-{i}",
                         "outcome": "success", "completeness": {},
                         "provenance": {"effective_configuration": {"model": "other" if i == 4 else "same"}},
                         "original_task": {"status": "available", "value": "repeated search; later recovered"}}
                        for i in range(1, 6)]
        self.source.write_text("\n" + "\n".join(json.dumps(row) for row in self.records) + "\n")
        selected = subprocess.run([sys.executable, str(self.script.with_name("cohort.py")), "select", str(self.source)],
                                  capture_output=True, text=True)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        manifest = json.loads(selected.stdout)
        runs = manifest["runs"]
        self.manifest = self.write("manifest.json", manifest)
        self.plan_value = {"schema": "daydream.trace-batches.v1", "cohort_id": manifest["cohort_id"],
                           "question": "Which searches fail to change approach?", "evidence_chars": 100,
                           "max_observations": 2,
                           "batches": [{"batch_id": "batch-1", "bytes": sum(r["bytes"] for r in runs[:4]),
                                        "run_ids": [r["run_id"] for r in runs[:4]]},
                                       {"batch_id": "batch-2", "bytes": runs[4]["bytes"],
                                        "run_ids": ["run-5"]}]}
        self.plan_value["plan_id"] = self.plan_identity(self.plan_value)
        self.plan = self.write("plan.json", self.plan_value)
        self.report_value = {"schema": "daydream.trace-report.v1", "cohort_id": manifest["cohort_id"],
                             "plan_id": self.plan_value["plan_id"],
                             "batch_id": "batch-1", "runs": [
                                 {"run_id": "run-1", "status": "analyzed", "reason": "",
                                  "assessments": [self.assessment("present", 2)]},
                                 {"run_id": "run-2", "status": "analyzed", "reason": "",
                                  "assessments": [self.assessment("absent", 0)]},
                                 {"run_id": "run-3", "status": "analyzed", "reason": "",
                                  "assessments": [self.assessment("unknown", None)]},
                                 {"run_id": "run-4", "status": "skipped", "reason": "manual exclusion",
                                  "assessments": []}]}
        self.report = self.write("report.json", self.report_value)

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    @staticmethod
    def plan_identity(plan):
        content = {key: value for key, value in plan.items() if key != "plan_id"}
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def assessment(verdict, occurrences):
        evidence = {"pointer": "/original_task/value", "excerpt": "later recovered"}
        return {"pattern": "unchanged-search", "category": "investigation_loop", "verdict": verdict,
                "occurrences": occurrences, "observation": "Inspect recovery before judging repetition.",
                "confidence": "medium", "evidence": [evidence] if verdict == "present" else [],
                "counterevidence": [evidence] if verdict == "absent" else [],
                "intervention": "Require a changed hypothesis before another search."}

    def cli(self, *args, ok=True):
        result = subprocess.run([sys.executable, str(self.script), *map(str, args)], capture_output=True, text=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, "")
        return result.stderr

    def synthesize(self, ok=True):
        return self.cli("synthesize", self.manifest, self.plan, self.report, ok=ok)

    def test_recurrence_counts_runs_separately_from_occurrences_and_retains_unknowns(self):
        # Existing mine tests only count repeated calls within documents, not cohort recurrence.
        output = self.synthesize()
        pattern = output["patterns"][0]
        self.assertEqual(pattern["affected_runs"], ["run-1"])
        self.assertEqual(pattern["assessable_runs"], ["run-1", "run-2"])
        self.assertEqual(pattern["observed_occurrences"], 2)
        self.assertEqual(pattern["unknown_runs"], ["run-3"])
        self.assertEqual(pattern["counterexample_runs"], ["run-2"])
        self.assertEqual(pattern["skipped_runs"], ["run-4"])
        self.assertEqual(pattern["missing_runs"], ["run-5"])
        self.assertEqual(output["missing_batches"], ["batch-2"])
        self.assertFalse(output["complete"])
        self.assertEqual(len(output["strata"]), 2)
        self.assertEqual(len(pattern["strata"]), 2)
        self.assertEqual(output["runs"][0]["source"], str(self.source.resolve()))
        self.assertEqual(output["runs"][0]["line"], 2)

    def test_templates_cannot_be_mistaken_for_completed_analysis(self):
        template = self.cli("template", self.manifest, self.plan, "--batch", "batch-1")
        self.assertTrue(template["template"])
        self.assertEqual([r["run_id"] for r in template["runs"]], [f"run-{i}" for i in range(1, 5)])
        self.report = self.write("template.json", template)
        self.assertIn("template", self.synthesize(ok=False))
        assessment = self.cli("assessment-template", "--pattern", "unchanged-search",
                              "--category", "investigation_loop")
        self.assertIsNone(assessment["occurrences"])
        self.report_value["runs"][0]["assessments"] = [assessment]
        self.report = self.write("report.json", self.report_value)
        self.assertIn("observation", self.synthesize(ok=False))
        assessment["observation"] = "Capture lacks the next decision needed to assess recurrence."
        self.write("report.json", self.report_value)
        self.assertIn("run-1", self.synthesize()["patterns"][0]["unknown_runs"])

    def test_reports_remain_bound_to_the_question_limits_and_assignments(self):
        # Assignment membership alone cannot detect old reports reused for a new question.
        for field, value in (("question", "Did verification retain the fixes?"),
                             ("evidence_chars", 110), ("max_observations", 3),
                             ("batches", list(reversed(self.plan_value["batches"])))):
            with self.subTest(field=field):
                changed = dict(self.plan_value, **{field: value})
                self.write("plan.json", changed)
                self.assertIn("plan_id", self.synthesize(ok=False))
                changed["plan_id"] = self.plan_identity(changed)
                self.write("plan.json", changed)
                self.assertIn("plan_id", self.synthesize(ok=False))
        self.write("plan.json", self.plan_value)
        planned = subprocess.run([sys.executable, str(self.script.with_name("cohort.py")), "batches",
                                  str(self.manifest), "--count", "2", "--question", "Which searches repeat?"],
                                 capture_output=True, text=True)
        self.assertEqual(planned.returncode, 0, planned.stderr)
        produced = json.loads(planned.stdout)
        self.assertEqual(produced.get("plan_id"), self.plan_identity(produced))
        template = self.cli("template", self.manifest, self.plan, "--batch", "batch-1")
        self.assertEqual(template.get("plan_id"), self.plan_value["plan_id"])
        self.assertEqual(self.synthesize().get("plan_id"), self.plan_value["plan_id"])

    def test_report_rows_cannot_override_frozen_run_metadata(self):
        # Excerpt checks cannot protect source provenance if analyst keys overwrite it afterward.
        for key, value in (("source", "/invented"), ("line", 999), ("sha256", "bad"),
                           ("context", {"model": "invented"})):
            with self.subTest(key=key):
                changed = json.loads(json.dumps(self.report_value))
                changed["runs"][0][key] = value
                self.write("report.json", changed)
                self.assertIn("metadata", self.synthesize(ok=False))
        self.write("report.json", self.report_value)
        output = self.synthesize()
        frozen = json.loads(self.manifest.read_text())["runs"][0]
        for key in ("source", "line", "sha256", "context"):
            self.assertEqual(output["runs"][0][key], frozen[key])

    def test_synthesis_rejects_forged_evidence_and_unknown_as_zero(self):
        # A valid JSON pointer alone cannot prove an excerpt came from the captured record.
        for mutation, message in (({"pointer": "/missing"}, "pointer"),
                                  ({"pointer": "/original_task/value", "excerpt": "invented"}, "excerpt"),
                                  ({"pointer": "/original_task/value", "excerpt": "x" * 101}, "excerpt")):
            with self.subTest(mutation=mutation):
                self.report_value["runs"][0]["assessments"][0]["evidence"][0] = mutation
                self.write("report.json", self.report_value)
                self.assertIn(message, self.synthesize(ok=False))
        self.report_value["runs"][0]["assessments"] = [self.assessment("unknown", 0)]
        self.write("report.json", self.report_value)
        self.assertIn("null", self.synthesize(ok=False))
        self.report_value["runs"][0]["assessments"] = [self.assessment("absent", 0)]
        self.report_value["runs"][0]["assessments"][0]["counterevidence"] = []
        self.write("report.json", self.report_value)
        self.assertIn("evidence", self.synthesize(ok=False))

    def test_assignment_duplicates_and_source_mutations_fail_before_synthesis(self):
        self.report_value["runs"][0]["run_id"] = "run-5"
        self.write("report.json", self.report_value)
        self.assertIn("assignment", self.synthesize(ok=False))
        self.report_value["runs"][0]["run_id"] = "run-1"
        self.write("report.json", self.report_value)
        self.assertIn("duplicate", self.cli("synthesize", self.manifest, self.plan,
                                            self.report, self.report, ok=False))
        # Real frozen source changes must fail even if its run ID and evidence still match.
        self.records[0]["extra"] = "changed after selection"
        self.source.write_text("\n" + "\n".join(json.dumps(row) for row in self.records) + "\n")
        self.assertIn("changed", self.synthesize(ok=False).lower())

    def test_templates_and_synthesis_recheck_skipped_and_unreported_tail_records(self):
        # Streaming must still check sources whose runs have no assessable report entry.
        commands = [("template", self.manifest, self.plan, "--batch", "batch-1"),
                    ("synthesize", self.manifest, self.plan, self.report)]
        for command in commands:
            self.cli(*command)
        for index in (3, 4):
            changed = json.loads(json.dumps(self.records))
            changed[index]["extra"] = "late frozen record changed"
            self.source.write_text("\n" + "\n".join(json.dumps(row) for row in changed) + "\n")
            for command in commands:
                with self.subTest(run=index + 1, command=command[0]):
                    self.assertIn("changed", self.cli(*command, ok=False))
        unchanged_prefix = "\n" + "\n".join(json.dumps(row) for row in self.records[:-1]) + "\n"
        for ending, error in (("", "line"), ("{broken\n", "Expecting")):
            self.source.write_text(unchanged_prefix + ending)
            for command in commands:
                with self.subTest(ending=ending, command=command[0]):
                    self.assertIn(error, self.cli(*command, ok=False))

    def test_template_and_synthesis_memory_is_bounded_independently_of_corpus_payload(self):
        # Existing small captures cannot detect retaining every embedded trajectory in memory.
        padding = 512 * 1024
        with self.source.open("w", encoding="utf-8") as stream:
            for number in range(24):
                record = dict(self.records[0], run_id=f"bulk-{number:02d}", trajectories={
                    "status": "available", "value": {"documents": [
                        {"trajectory_id": "root", "steps": [{"message": "x" * padding}]},
                        {"trajectory_id": "sibling", "steps": [{"message": "retained sibling evidence"}]}]}})
                stream.write(json.dumps(record) + "\n")
        selected = subprocess.run([sys.executable, str(self.script.with_name("cohort.py")), "select", str(self.source)],
                                  capture_output=True, text=True)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        manifest = self.write("bulk-manifest.json", json.loads(selected.stdout))
        planned = subprocess.run([sys.executable, str(self.script.with_name("cohort.py")), "batches", str(manifest),
                                  "--count", "2", "--question", "Which captured searches recover?"],
                                 capture_output=True, text=True)
        self.assertEqual(planned.returncode, 0, planned.stderr)
        plan = self.write("bulk-plan.json", json.loads(planned.stdout))
        measure = """import json, runpy, sys, tracemalloc
from pathlib import Path
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).parent))
tracemalloc.start()
try:
    runpy.run_path(script, run_name="__main__")
finally:
    print(json.dumps({"peak_bytes": tracemalloc.get_traced_memory()[1]}), file=sys.stderr)
"""

        def measured(*args):
            result = subprocess.run([sys.executable, "-c", measure, str(self.script), *map(str, args)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout), json.loads(result.stderr)["peak_bytes"]

        report, template_peak = measured("template", manifest, plan, "--batch", "batch-1")
        report.pop("template")
        for row in report["runs"]:
            assessment = self.assessment("present", 1)
            assessment["evidence"] = [{"pointer": "/trajectories/value/documents/1/steps/0/message",
                                       "excerpt": "retained sibling evidence"}]
            row.update(status="analyzed", reason="", assessments=[assessment])
        report_path = self.write("bulk-report.json", report)
        output, synthesis_peak = measured("synthesize", manifest, plan, report_path)
        self.assertEqual(output["patterns"][0]["affected_count"], len(report["runs"]))
        self.assertEqual(len(output["missing_runs"]), 24 - len(report["runs"]))
        for command, peak in (("template", template_peak), ("synthesize", synthesis_peak)):
            with self.subTest(command=command):
                self.assertLess(peak, 12 * padding,
                                "Inspection must discard captures after checking them, rather than retain the corpus")

    def test_unassessed_patterns_and_per_run_limits_stay_explicit(self):
        self.report_value["runs"][1]["assessments"] = []
        self.write("report.json", self.report_value)
        pattern = self.synthesize()["patterns"][0]
        self.assertEqual(pattern["unassessed_runs"], ["run-2"])
        self.assertEqual(pattern["assessable_runs"], ["run-1"])
        first = self.report_value["runs"][0]["assessments"][0]
        self.report_value["runs"][0]["assessments"] = [dict(first, pattern=f"p{i}") for i in range(3)]
        self.write("report.json", self.report_value)
        self.assertIn("max_observations", self.synthesize(ok=False))

    def test_experiment_requires_matched_conditions_repeats_and_disjoint_holdout(self):
        synthesis = self.write("synthesis.json", self.synthesize())
        packet = self.cli("experiment-template", synthesis)
        path = self.write("experiment.json", packet)
        self.assertIn("template", self.cli("validate-experiment", path, ok=False))
        conditions = {"backend": "codex", "model": "same", "profile": "deep", "reasoning_effort": "high",
                      "budgets": {"tools": 50}, "temperature": 0.2,
                      "workloads": ["discovery-task", "heldout-task"], "producer_revision": "base-sha"}
        packet.update(template=False, change={"pattern": "unchanged-search", "description": "Change search guidance"},
                      predicted_effect="Reduce repeated calls while retaining correct findings.",
                      baseline=conditions, candidate=dict(conditions, producer_revision="candidate-sha"),
                      repeats=3, discovery_workloads=["discovery-task"], heldout_workloads=["heldout-task"],
                      quality_guardrails=["Retain independently adjudicated true findings and truthful coverage"],
                      rejection_criterion="Reject if any retained finding is lost or median latency increases")
        self.write("experiment.json", packet)
        self.assertTrue(self.cli("validate-experiment", path)["valid"])
        for field, value, message in (("model", "other", "matched"), ("reasoning_effort", "low", "matched"),
                                      ("temperature", 0.9, "matched"), ("seed", 7, "matched"),
                                      ("workloads", ["heldout-task"], "workloads")):
            with self.subTest(field=field):
                invalid = json.loads(json.dumps(packet))
                invalid["candidate"][field] = value
                self.write("experiment.json", invalid)
                self.assertIn(message, self.cli("validate-experiment", path, ok=False))
        invalid = json.loads(json.dumps(packet))
        del invalid["candidate"]["reasoning_effort"]
        self.write("experiment.json", invalid)
        self.assertIn("reasoning_effort", self.cli("validate-experiment", path, ok=False))
        template = self.cli("experiment-template", synthesis)
        self.assertIn("reasoning_effort", template["baseline"])
        self.assertIn("reasoning_effort", template["candidate"])
        packet["heldout_workloads"] = ["discovery-task"]
        self.write("experiment.json", packet)
        self.assertIn("disjoint", self.cli("validate-experiment", path, ok=False))


if __name__ == "__main__":
    unittest.main()
