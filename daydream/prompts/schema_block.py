"""Shared JSON structured-output schema block for prompt builders."""

import json
from typing import Any


def schema_block(schema: dict[str, Any]) -> str:
    return "Return ONLY a JSON object matching this schema:\n```json\n" + json.dumps(schema, indent=2) + "\n```"
