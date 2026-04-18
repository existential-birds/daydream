"""Deep-review mode package.

Exports run_deep once the orchestrator exists (plan 05-09).
"""

from daydream.deep.detection import StackAssignment, detect_stacks

__all__ = ["StackAssignment", "detect_stacks"]
