"""Configuration constants for daydream.

Provide centralized configuration values used throughout the daydream package.
This module contains constants for skill mappings, file paths, and regex patterns
used by the review and fix loop system.

Exports:
    ReviewSkillChoice: Enum for review skill menu choices.
    REVIEW_SKILLS: dict[ReviewSkillChoice, str] - Mapping of review type identifiers to skill names.
    REVIEW_OUTPUT_FILE: str - Default filename for storing review results.
    UNKNOWN_SKILL_PATTERN: str - Regex pattern for detecting unknown skill errors.
"""

from enum import Enum


class ReviewSkillChoice(Enum):
    """Enum for review skill menu choices."""

    PYTHON = "1"
    FRONTEND = "2"


# Skill mapping for review types
REVIEW_SKILLS: dict[ReviewSkillChoice, str] = {
    ReviewSkillChoice.PYTHON: "beagle-python:review-python",
    ReviewSkillChoice.FRONTEND: "beagle-react:review-frontend",
}

# CLI skill name to full skill path mapping
SKILL_MAP: dict[str, str] = {
    "python": "beagle-python:review-python",
    "frontend": "beagle-react:review-frontend",
}

# Output file for review results
REVIEW_OUTPUT_FILE = ".review-output.md"

# Pattern to detect unknown skill errors
UNKNOWN_SKILL_PATTERN = r"Unknown skill: ([\w:-]+)"
