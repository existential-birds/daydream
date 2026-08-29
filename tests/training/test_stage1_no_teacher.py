"""AC6: Stage 1 uses dataset SFT and never routes through live-teacher
[orchestrator.algo] type = "sft" — failure mode is silent (a teacher endpoint
that does not exist), so this is asserted structurally, not at runtime."""
import tomllib
from pathlib import Path


def _all_rl_configs():
    return [p for p in (Path(__file__).parents[2] / "rl" / "train").glob("*.toml")]


def test_no_rl_config_declares_live_teacher_algo():
    for cfg in _all_rl_configs():
        c = tomllib.loads(cfg.read_text())
        algo = c.get("orchestrator", {}).get("algo")
        assert not (isinstance(algo, dict) and algo.get("type") == "sft"), (
            f"{cfg.name} declares live-teacher algo type='sft'; dataset SFT is the only permitted path (C4)"
        )


def test_sft_config_names_a_local_dataset_not_an_endpoint():
    c = tomllib.loads((Path(__file__).parents[2] / "rl" / "train" / "sft.toml").read_text())
    # prime-rl schema: the local dataset path lives in [data].name; [dataset]
    # is the daydream-side gating contract table (no path by design).
    data = c["data"]
    assert "path" in data or "corpus_dir" in data or ("name" in data and str(data["name"]).startswith("/")), (
        "sft.toml [data] must name a local dataset path, not a remote reference"
    )
    for ds in (c.get("dataset", {}), data):
        assert not any(k in ds for k in ("teacher", "endpoint", "api_base"))
