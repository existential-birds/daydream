"""Configuration constants for daydream."""

# Skill mapping for review types
REVIEW_SKILLS: dict[str, str] = {
    "1": "beagle:review-python",
    "2": "beagle:review-frontend",
}

# Output file for review results
REVIEW_OUTPUT_FILE = ".review-output.md"

# Pattern to detect unknown skill errors
UNKNOWN_SKILL_PATTERN = r"Unknown skill: ([\w:-]+)"
