"""Inventory/focus regressions against isolated Git repositories."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


TOOLKIT = Path(__file__).resolve().parents[1]


class PartitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return target

    def partition(self, k=2, **environment):
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        result = subprocess.run(
            ["python3", str(TOOLKIT / "partition.py"), str(self.root), str(k)],
            env={**os.environ, **environment}, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def test_tracks_resources_empty_and_binary_but_excludes_untracked_ignored(self):
        self.write(".gitignore", "cache/\n")
        self.write("cache/ignored.py", "pass\n")
        self.write(".github/workflows/check.yml", "name: check\n")
        self.write("pyproject.toml", "[project]\n")
        self.write("template.html", "<h1>Hello</h1>\n")
        self.write("empty.py", "")
        (self.root / "asset.png").write_bytes(b"\x89PNG\x00binary")
        (self.root / "outside.py").symlink_to("/missing/outside.py")
        data = self.partition()
        records = {item["path"]: item for item in data["files"]}
        self.assertEqual(set(records), {
            ".gitignore", ".github/workflows/check.yml", "pyproject.toml",
            "template.html", "empty.py", "asset.png", "outside.py",
        })
        self.assertEqual(records["asset.png"]["status"], "binary")
        self.assertEqual(records["empty.py"]["loc"], 0)
        self.assertEqual(records["outside.py"]["status"], "symlink")
        self.assertEqual(data["total_files"], len(records))
        self.assertCountEqual(
            [path for sl in data["slices"] for path in sl["paths"]], records,
        )

    def test_python_inventory_groups_production_and_test_and_bounds_focus(self):
        self.write("app/worker.py", "import json\n\ndef run():\n    return json.dumps({})\n")
        self.write("tests/test_worker.py", "def test_run():\n    assert True\n")
        for i in range(8):
            self.write(f"app/extra_{i}.py", "pass\n" * (i + 2))
        data = self.partition(k=3, SWEEP_FOCUS_MAX="2", SWEEP_FOCUS_LOC="10")
        worker = next(item for item in data["files"] if item["path"] == "app/worker.py")
        self.assertEqual(worker["imports"], ["json"])
        self.assertEqual(worker["definitions"][0]["name"], "run")
        self.assertIn("tests/test_worker.py", worker["related_paths"])
        owner = next(sl for sl in data["slices"] if "app/worker.py" in sl["paths"])
        self.assertIn("tests/test_worker.py", owner["paths"])
        for sl in data["slices"]:
            self.assertLessEqual(len(sl["focus_paths"]), 2)
            self.assertTrue(set(sl["focus_paths"]) <= set(sl["paths"]))
            self.assertIn("inventory_summary", sl)
            self.assertIn("role", sl)

    def test_completed_unchanged_focus_rotates_and_modified_files_are_revisited(self):
        for i in range(3):
            self.write(f"app/file_{i}.py", "pass\n" * (i + 1))
        first = self.partition(k=1, SWEEP_FOCUS_MAX="1")
        path = first["slices"][0]["focus_paths"][0]
        record = next(item for item in first["files"] if item["path"] == path)
        coverage = self.root / ".coverage-history.jsonl"
        coverage.write_text(json.dumps({"path": path, "sha256": record["sha256"], "status": "complete"}) + "\n")
        self.write(".gitignore", ".coverage-history.jsonl\n")
        options = {"SWEEP_COVERAGE_FILE": str(coverage), "SWEEP_FOCUS_MAX": "1"}
        second = self.partition(k=1, **options)
        self.assertNotEqual(second["slices"][0]["focus_paths"], [path])
        self.write(path, "pass\n" * 7)
        third = self.partition(k=1, **options)
        self.assertEqual(third["slices"][0]["focus_paths"], [path])


if __name__ == "__main__":
    unittest.main()
