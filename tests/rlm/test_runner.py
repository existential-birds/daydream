# tests/rlm/test_runner.py
"""Tests for RLM runner orchestration."""

import pytest

from daydream.rlm.runner import (
    RLMConfig,
    RLMRunner,
    load_codebase,
)


class TestRLMConfig:
    """Tests for RLMConfig dataclass."""

    def test_config_defaults(self):
        """RLMConfig should have sensible defaults."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        assert cfg.workspace_path == "/repo"
        assert cfg.languages == ["python"]
        assert cfg.model == "opus"
        assert cfg.sub_model == "haiku"
        assert cfg.use_container is True

    def test_config_pr_mode(self):
        """RLMConfig should support PR mode."""
        cfg = RLMConfig(
            workspace_path="/repo",
            languages=["python"],
            pr_number=123,
        )
        assert cfg.pr_number == 123


class TestLoadCodebase:
    """Tests for load_codebase function."""

    def test_load_codebase_python(self, tmp_path):
        """Should load Python files from directory."""
        # Create test files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "readme.md").write_text("# Readme")

        ctx = load_codebase(tmp_path, languages=["python"])

        assert ctx.file_count == 2
        assert "main.py" in ctx.files or str(tmp_path / "main.py") in ctx.files
        assert ctx.languages == ["python"]

    def test_load_codebase_excludes_hidden(self, tmp_path):
        """Should exclude hidden directories."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("gitconfig")
        (tmp_path / "main.py").write_text("x=1")

        ctx = load_codebase(tmp_path, languages=["python"])

        assert ctx.file_count == 1
        assert not any(".git" in f for f in ctx.files.keys())

    def test_load_codebase_excludes_node_modules(self, tmp_path):
        """Should exclude node_modules."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("module")
        (tmp_path / "app.ts").write_text("const x = 1")

        ctx = load_codebase(tmp_path, languages=["typescript"])

        assert ctx.file_count == 1


class TestRLMRunner:
    """Tests for RLMRunner class."""

    def test_runner_init(self):
        """RLMRunner should initialize with config."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)
        assert runner.config == cfg
