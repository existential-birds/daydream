import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUG = ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"


def test_bug_report_template_native() -> None:
    data = yaml.safe_load(BUG.read_text())  # must parse as valid YAML
    bodies = " ".join(
        (b.get("attributes", {}).get("label", "") + " "
         + b.get("attributes", {}).get("placeholder", "") + " "
         + b.get("attributes", {}).get("description", ""))
        for b in data["body"])
    for token in ("Daydream version", "backend", "provider", "model",
                  "reasoning effort", "digest", "source", "stacks",
                  "benchmark", "run ID"):
        assert token.lower() in bodies.lower(), f"missing field request {token!r}"
    for token in ("beagle", "plugin", "skill", "claude code version"):
        assert token.lower() not in bodies.lower(), f"stale field {token!r} still present"
    area = next(b for b in data["body"] if b.get("id") == "area")
    options = " ".join(area["attributes"]["options"]).lower()
    for cat in ("review", "improve", "profile", "benchmark"):
        assert cat in options, f"affected-area missing {cat!r}"
    assert "beagle" not in options