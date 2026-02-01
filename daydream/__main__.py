"""Entry point for `python -m daydream`.

Enable execution of the daydream package as a module using the Python -m flag.
This module imports and invokes the main function from the CLI module when
the package is run directly.

Usage:
    python -m daydream [options]

Exports:
    None - This module is intended for direct execution only.
"""

from daydream.cli import main

if __name__ == "__main__":
    main()
