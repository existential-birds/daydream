"""AC #4: daydream.ui's public surface is declared via explicit PEP 484 export forms."""
from pathlib import Path

FACADE = Path("daydream/ui/__init__.py").read_text(encoding="utf-8")


def test_every_facade_reexport_uses_redundant_alias_or_all():
    assert "__all__" in FACADE or " as " in FACADE, (
        "ui facade must declare re-exports explicitly (PEP 484): "
        "`from x import Y as Y` aliases or an explicit __all__"
    )


def test_flows_engine_exports_loopgroup_explicitly():
    src = Path("daydream/flows/engine.py").read_text(encoding="utf-8")
    assert "LoopGroup as LoopGroup" in src or '"LoopGroup"' in src
