"""Configuration constants for daydream.

Provide centralized configuration values used throughout the daydream package.
This module contains constants for skill mappings, file paths, and regex patterns
used by the review and fix loop system.

Exports:
    REVIEW_SKILLS: dict[str, str] - Mapping of review type identifiers to skill names.
    REVIEW_OUTPUT_FILE: str - Default filename for storing review results.
    UNKNOWN_SKILL_PATTERN: str - Regex pattern for detecting unknown skill errors.
"""

# Skill mapping for review types
REVIEW_SKILLS: dict[str, str] = {
    "1": "beagle:review-python",
    "2": "beagle:review-frontend",
}

# Output file for review results
REVIEW_OUTPUT_FILE = ".review-output.md"

# Pattern to detect unknown skill errors
UNKNOWN_SKILL_PATTERN = r"Unknown skill: ([\w:-]+)"
