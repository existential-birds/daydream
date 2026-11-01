import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "findings.py"


class FindingsTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("sweep_findings", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scan = self.root / "scan"
        self.apply = self.root / "apply"
        self.scan.mkdir()
        self.apply.mkdir()
        self.memory = self.root / "memory.json"

    def finding(self, path, net, **extra):
        (self.root / path).write_text("original\n")
        return dict(kind="simplify", path=path, symbol="function", reason="same behavior",
                    evidence="inspected callers", followups=[], est_net_lines=net,
                    confidence="high", **extra)

    def write_rows(self, name, rows):
        (self.scan / name).write_text("\n".join(json.dumps(row) for row in rows) + "\nSCAN_DONE-1\n")

    def plan(self, limit=1):
        self.module.plan(self.root, self.scan, self.apply, self.memory, 1, limit)
        return [json.loads(line) for line in (self.apply / "chunk-1.jsonl").read_text().splitlines()] if (self.apply / "chunk-1.jsonl").exists() else []

    def test_global_rank_before_truncation_and_carryover(self):
        self.write_rows("findings-1.jsonl", [self.finding(f"a{i}.py", -100) for i in range(8)])
        self.write_rows("findings-2.jsonl", [self.finding("winner.py", -600)])
        self.assertEqual(self.plan()[0]["path"], "winner.py")
        self.assertEqual(len(json.loads(self.memory.read_text())["candidates"]), 9)
        for file in self.scan.iterdir():
            file.unlink()
        self.assertEqual(self.plan()[0]["path"], "winner.py")

    def test_dedupe_investigations_and_invalid_rows(self):
        good = self.finding("ready.py", -50)
        low = self.finding("uncertain.py", -900)
        low["confidence"] = "low"
        malformed = self.finding("bad.py", -1)
        malformed["est_net_lines"] = True
        self.write_rows("findings-1.jsonl", [good, good, low, malformed])
        self.assertEqual([r["path"] for r in self.plan(5)], ["ready.py"])
        self.assertEqual(len(json.loads(self.memory.read_text())["candidates"]), 2)

    def test_outcomes_suppress_unchanged_and_revalidate_changed(self):
        self.write_rows("findings-1.jsonl", [self.finding("ready.py", -50)])
        selected = self.plan()[0]
        (self.root / "ready.py").write_text("smaller\n")
        (self.apply / "outcomes-1.jsonl").write_text(json.dumps({"id": selected["id"], "status": "applied", "reason": "consolidated"}) + "\n")
        self.module.finish(self.root, self.scan, self.apply, self.memory)
        self.assertEqual(self.plan(), [])
        (self.root / "ready.py").write_text("changed again\n")
        self.assertTrue(self.plan()[0]["needs_revalidation"])

    def test_overlapping_candidates_defer_and_path_escape_rejected(self):
        first = self.finding("shared.py", -80)
        second = dict(first, symbol="other", est_net_lines=-60)
        escape = dict(first, path="../outside.py", est_net_lines=-1000)
        self.write_rows("findings-1.jsonl", [first, second, escape])
        self.assertEqual(len(self.plan(5)), 1)

    def test_rejects_ambiguous_identity_and_path_aliases(self):
        first = self.finding("ready.py", -80)
        nameless = dict(first)
        del nameless["symbol"]
        alias = dict(first, path="./ready.py", symbol="alias")
        self.write_rows("findings-1.jsonl", [first, nameless, alias])
        self.assertEqual(len(json.loads(self.memory.read_text())["candidates"]) if self.memory.exists() else 0, 0)
        self.plan(5)
        self.assertEqual(len(json.loads(self.memory.read_text())["candidates"]), 1)

    def test_supporting_contract_change_invalidates_decision(self):
        (self.root / "contract.md").write_text("first contract\n")
        row = self.finding("ready.py", -50, evidence_paths=["contract.md"])
        self.write_rows("findings-1.jsonl", [row])
        selected = self.plan()[0]
        (self.apply / "outcomes-1.jsonl").write_text(json.dumps({"id": selected["id"], "status": "rejected", "reason": "contract requires it"}) + "\n")
        self.module.finish(self.root, self.scan, self.apply, self.memory)
        self.assertEqual(self.plan(), [])
        (self.root / "contract.md").write_text("new contract\n")
        self.assertTrue(self.plan()[0]["needs_revalidation"])


if __name__ == "__main__":
    unittest.main()
