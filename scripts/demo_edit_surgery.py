#!/usr/bin/env python3
"""Demo the Dream Surgery Edit visualization.

Usage:
    python scripts/demo_edit_surgery.py
"""

import sys
import time
from pathlib import Path

# Add parent to path so we can import daydream
sys.path.insert(0, str(Path(__file__).parent.parent))

from daydream.ui import (
    LiveToolPanelRegistry,
    create_console,
    print_phase_hero,
)


def main() -> None:
    """Demo the Edit Dream Surgery visualization."""
    console = create_console()

    # Hero banner
    print_phase_hero(console, "HEAL", "mending what was found")

    # Create registry
    registry = LiveToolPanelRegistry(console, quiet_mode=False)

    # Sample edit args
    edit_args = {
        "file_path": "/src/components/UserAuth.tsx",
        "old_string": 'const password = user.password  // plaintext',
        "new_string": 'const passwordHash = await bcrypt.hash(user.password, 12)',
        "replace_all": False,
    }

    # Create and start the panel
    panel = registry.create("demo-edit-1", "Edit", edit_args)

    # Let the animation run for a few seconds
    console.print()
    console.print("[dim]  (surgery in progress...)[/dim]")

    time.sleep(4)

    # Complete with result
    panel.set_result("Edit applied successfully")
    panel.finish()

    console.print()
    console.print("[bold green]  ~ the code has been healed ~[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
