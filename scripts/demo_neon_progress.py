#!/usr/bin/env python3
"""Demo script to visualize the neon_progress context manager.

Run with: python scripts/demo_neon_progress.py
"""

import time

from rich.console import Console

from daydream.ui import neon_progress


def main() -> None:
    console = Console()

    console.print("\n[bold cyan]Neon Progress Demo[/bold cyan]\n")

    # Demo 1: Short operation (2 seconds)
    console.print("[dim]Demo 1: Short operation (2 seconds)[/dim]")
    with neon_progress(console, "Loading configuration..."):
        time.sleep(2)
    console.print("[green]Done![/green]\n")

    # Demo 2: Medium operation (4 seconds)
    console.print("[dim]Demo 2: Medium operation (4 seconds)[/dim]")
    with neon_progress(console, "Analyzing codebase...", width=50):
        time.sleep(4)
    console.print("[green]Done![/green]\n")

    # Demo 3: Simulated work with checkpoints
    console.print("[dim]Demo 3: Multiple sequential operations[/dim]")

    tasks = [
        ("Fetching review skills...", 1.5),
        ("Parsing feedback...", 2.0),
        ("Applying fixes...", 2.5),
    ]

    for message, duration in tasks:
        with neon_progress(console, message):
            time.sleep(duration)
        console.print(f"  [green]\u2714[/green] {message.replace('...', ' complete')}")

    console.print("\n[bold green]All demos complete![/bold green]\n")


if __name__ == "__main__":
    main()
