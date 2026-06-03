import pytest

from daydream.benchmark.score import model_results_dir, preflight_judge_env


def test_model_results_dir_sanitizes_slashes(tmp_path):
    assert model_results_dir(tmp_path, "anthropic/claude-opus-4.5").name == "anthropic_claude-opus-4.5"


def test_preflight_raises_when_key_unset(monkeypatch):
    monkeypatch.delenv("MARTIAN_API_KEY", raising=False)
    with pytest.raises(EnvironmentError) as e:
        preflight_judge_env()
    assert "MARTIAN_API_KEY" in str(e.value)


def test_preflight_passes_when_key_present(monkeypatch):
    monkeypatch.setenv("MARTIAN_API_KEY", "sk-or-x")
    preflight_judge_env()
