"""#1093: the naming grep-gate keeps the tree free of project-owned versioned names."""

import subprocess


def test_no_project_owned_versioned_names() -> None:
    """#1093 Should-Have: the naming decision cannot regress."""
    result = subprocess.run(
        ["bash", "scripts/check-naming.sh"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"versioned project names remain:\n{result.stdout}"
