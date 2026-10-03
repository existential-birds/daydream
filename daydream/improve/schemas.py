"""Strict model output contracts for the improve advisor."""

from typing import Any

from daydream.improve.command_contract import (
    _OPTIONAL_COMMAND_REF_SCHEMA,
    COMMAND_REF_SCHEMA as _COMMAND_REF_SCHEMA,
    DIRECTORY_SCOPE_SCHEMA as _DIRECTORY_SCOPE_SCHEMA,
    REPOSITORY_FILE_PATH_SCHEMA as _REPOSITORY_FILE_PATH_SCHEMA,
)
from daydream.output_schema import (
    array_schema,
    enum_schema,
    record_array_schema,
    result_array_schema,
    severity_enum_schema,
    strict_object,
    text_schema,
)

MAINTENANCE_SIGNALS: tuple[str, ...] = (
    "overengineered_structure",
    "reuse_existing",
    "hand_rolled_substitute",
    "duplicated_test_structure",
    "parameterizable_test_matrix",
    "excessive_comment",
    "self_evident_comment",
    "dead_code",
)

CHANGE_SHAPES: tuple[str, ...] = (
    "delete",
    "reuse",
    "consolidate",
    "neutral",
    "additive",
    "unknown",
)

_MAINTENANCE_FINDING_PROPERTIES: dict[str, Any] = {
    "maintenance_signals": array_schema(
        {"type": "string", "enum": list(MAINTENANCE_SIGNALS)},
        description="Stable classifications for codebase-growth pressure. Return an "
        "empty array when none apply; do not invent a signal merely to "
        "prefer a smaller patch.",
    ),
    "change_shape": enum_schema(
        list(CHANGE_SHAPES),
        description="The expected overall shape of the fix. This is a prioritization "
        "hint, not a promise about exact line counts.",
    ),
    "reuse_target": text_schema(
        nullable=True,
        max_length=500,
        description="For reuse_existing, identify the verified target as "
        "repo:<path>#<symbol>. For a standard-library or dependency "
        "substitute use stdlib:<qualified-name> or dep:<package>:<api>. "
        "The target must be cited in evidence. "
        "Return null when no concrete reuse target is proposed.",
    ),
}

AUDIT_FINDINGS_SCHEMA: dict[str, Any] = result_array_schema(
    "findings",
    {
        "title": {"type": "string"},
        "category": {"type": "string"},
        "path": {"type": "string"},
        "line": {"type": ["integer", "null"]},
        "body": {"type": "string"},
        "impact": {"enum": ["HIGH", "MED", "LOW"]},
        "effort": {"enum": ["S", "M", "L"]},
        "risk": {"enum": ["LOW", "MED", "HIGH"]},
        "confidence": {"enum": ["HIGH", "MED", "LOW"]},
        "evidence": array_schema({"type": "string"}),
        **_MAINTENANCE_FINDING_PROPERTIES,
    },
)

VET_SCHEMA: dict[str, Any] = result_array_schema(
    "verdicts",
    {
        "vet_id": {"type": "integer"},
        "keep": {"type": "boolean"},
        "reason": {"type": "string"},
        "severity": severity_enum_schema(nullable=True),
        "impact": enum_schema(["HIGH", "MED", "LOW", None], nullable=True),
        "effort": enum_schema(["S", "M", "L", None], nullable=True),
        "risk": enum_schema(["LOW", "MED", "HIGH", None], nullable=True),
        "confidence": enum_schema(["HIGH", "MED", "LOW", None], nullable=True),
        "path": {"type": ["string", "null"]},
        "line": {"type": ["integer", "null"]},
        **_MAINTENANCE_FINDING_PROPERTIES,
    },
)

_STEP_NUMBER_LIST_SCHEMA: dict[str, Any] = array_schema({"type": "integer", "minimum": 1})
_SYMBOL_NAME_SCHEMA: dict[str, Any] = text_schema(min_length=1, max_length=300)
_ROLE_STRING_SCHEMA: dict[str, Any] = text_schema(min_length=12, max_length=300)
_PATH_WITH_ROLE_SCHEMA: dict[str, Any] = strict_object(
    {
        "path": _REPOSITORY_FILE_PATH_SCHEMA,
        "role": _ROLE_STRING_SCHEMA,
    }
)
_STOP_CONDITION_BODY_PROPERTIES: dict[str, Any] = {
    "condition": {"type": "string", "minLength": 30, "maxLength": 800},
    "evidence_to_report": {"type": "string", "minLength": 20, "maxLength": 500},
    "related_paths": array_schema(_REPOSITORY_FILE_PATH_SCHEMA),
    "related_step_numbers": _STEP_NUMBER_LIST_SCHEMA,
}
# The model-facing authoring schema: judgment content only. The host derives
# numbering, command records, excerpt text, git policy, boilerplate stop
# conditions, and rendering (see daydream/improve/assemble.py).
PLAN_AUTHOR_SCHEMA: dict[str, Any] = strict_object(
    {
        "title": {"type": "string", "minLength": 12, "maxLength": 160},
        "covered_fingerprints": array_schema(
            {"type": "string", "minLength": 1, "maxLength": 128},
            minItems=1,
            description="Copy the selected finding's complete set of unique "
            "member_fingerprints exactly. Order is not significant. When "
            "it has no member_fingerprints, use a one-item array "
            "containing its fingerprint.",
        ),
        "why_this_matters": strict_object(
            {
                key: {"type": "string", "minLength": 30, "maxLength": 800}
                for key in ("problem", "concrete_cost", "intended_outcome")
            }
        ),
        "scope": strict_object(
            {
                "existing_paths": array_schema(_PATH_WITH_ROLE_SCHEMA),
                "new_paths": array_schema(_PATH_WITH_ROLE_SCHEMA),
                "out_of_scope_paths": record_array_schema(
                    {
                        "path": _DIRECTORY_SCOPE_SCHEMA,
                        "reason": text_schema(min_length=20, max_length=500),
                    },
                    minItems=1,
                ),
                "out_of_scope_behaviors": record_array_schema(
                    {key: text_schema(min_length=20, max_length=500) for key in ("behavior", "reason")}, minItems=1
                ),
            }
        ),
        "context_excerpts": record_array_schema(
            {
                "path": _REPOSITORY_FILE_PATH_SCHEMA,
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "file_role": _ROLE_STRING_SCHEMA,
            }
        ),
        "git_workflow": strict_object(
            {
                "commit_boundaries": text_schema(
                    min_length=20,
                    max_length=500,
                    description="How to split the work into commits, stated as a "
                    "decision the executor follows rather than a choice it "
                    "makes: say 'one commit' or list each commit and the "
                    "step numbers it covers. Never 'split as appropriate'.",
                ),
                "commit_message_example": text_schema(
                    min_length=5, max_length=200, description="The literal commit message to use, ready to paste."
                ),
            }
        ),
        "steps": record_array_schema(
            {
                "title": text_schema(
                    min_length=12,
                    max_length=200,
                    description="What this step accomplishes, in the imperative. "
                    "Steps are executed strictly in array order, so "
                    "order them by dependency.",
                ),
                "changes": record_array_schema(
                    {
                        "path": _REPOSITORY_FILE_PATH_SCHEMA,
                        "symbol": text_schema(
                            min_length=1,
                            max_length=300,
                            description="Exact name of the function, class, "
                            "constant, or block being changed, "
                            "copied verbatim from the file. Never "
                            "a description like 'the relevant "
                            "handler'.",
                        ),
                        "operation": enum_schema(
                            [
                                "create",
                                "modify",
                                "delete",
                                "move",
                                "rename",
                            ]
                        ),
                        "instruction": text_schema(
                            min_length=30,
                            max_length=4000,
                            description="Exactly what to do, written for an "
                            "executor that cannot infer anything "
                            "and will not look around the "
                            "repository. Name every identifier, "
                            "file, literal, header, key, and "
                            "import in full. State what must NOT "
                            "change. Banned: 'the relevant X', "
                            "'the appropriate Y', 'as "
                            "appropriate', 'as needed', 'if "
                            "necessary', 'where applicable', "
                            "'update accordingly', 'and similar', "
                            "'etc.', 'consider', 'you may want "
                            "to', 'try to' — each one is a "
                            "decision the executor cannot make. "
                            "If a change needs more than 4000 "
                            "characters to specify, split it into "
                            "several entries in this array or "
                            "into another step; it is never "
                            "truncated for you. For a delete "
                            "operation, name exactly what is "
                            "removed and do not invent a "
                            "replacement.",
                        ),
                        "target_state": text_schema(
                            min_length=30,
                            max_length=4000,
                            description="What is literally true of this file "
                            "once the instruction is done, phrased "
                            "so the executor can re-read the file "
                            "and check it sentence by sentence. "
                            "Describe observable content, not "
                            "intent or quality. For a delete "
                            "operation, state that the exact "
                            "symbol or block is absent; when the "
                            "whole file is deleted, state that the "
                            "path no longer exists.",
                        ),
                    },
                    minItems=1,
                ),
                "verification": _OPTIONAL_COMMAND_REF_SCHEMA,
            },
            minItems=1,
        ),
        "test_plan": strict_object(
            {
                "mode": enum_schema(
                    [
                        "new-or-updated-tests",
                        "existing-coverage",
                        "not-applicable",
                    ],
                    description="Use new-or-updated-tests when test code must change, "
                    "existing-coverage when named tests already prove the "
                    "change, and not-applicable only for documentation, "
                    "exact comment/docstring cleanup, or deletion of a "
                    "non-runtime artifact or redundant test that cannot "
                    "usefully be exercised by a test.",
                ),
                "rationale": text_schema(
                    min_length=20,
                    max_length=700,
                    description="Explain why this mode is sufficient for this exact "
                    "change. Deletion alone is not a reason to omit tests "
                    "when observable behavior changes.",
                ),
                "existing_coverage": record_array_schema(
                    {
                        "path": _REPOSITORY_FILE_PATH_SCHEMA,
                        "symbol": _SYMBOL_NAME_SCHEMA,
                        "behavior": text_schema(min_length=20, max_length=700),
                        "verification": _OPTIONAL_COMMAND_REF_SCHEMA,
                    },
                    description="Exact existing tests that already prove the planned "
                    "behavior. Populate only in existing-coverage mode. "
                    "These paths are evidence, not writable plan scope.",
                ),
                "exemplars": record_array_schema(
                    {
                        "path": _REPOSITORY_FILE_PATH_SCHEMA,
                        "symbol": _SYMBOL_NAME_SCHEMA,
                        "pattern_to_copy": text_schema(min_length=20, max_length=700),
                    },
                    description="Existing tests whose shape the new tests copy. Leave "
                    "this empty when the repository has no test to copy — "
                    "an invented exemplar is worse than none.",
                ),
                "cases": record_array_schema(
                    {
                        "name": text_schema(min_length=12, max_length=200),
                        "test_file": _REPOSITORY_FILE_PATH_SCHEMA,
                        "test_symbol": _SYMBOL_NAME_SCHEMA,
                        "kind": enum_schema(
                            [
                                "unit",
                                "integration",
                                "acceptance",
                                "static",
                            ]
                        ),
                        "setup": text_schema(
                            min_length=20,
                            max_length=1000,
                            description="The exact fixtures, helpers, and starting "
                            "values this test needs, named as they "
                            "appear in the repository. Prefer reusing "
                            "a named harness from an exemplar over "
                            "describing one.",
                        ),
                        "action": text_schema(
                            min_length=20,
                            max_length=1000,
                            description="The single call or interaction under test, with its literal arguments.",
                        ),
                        "assertions": array_schema(
                            text_schema(
                                min_length=15,
                                max_length=500,
                                description="One observable outcome and the exact "
                                "expected value. Assert on what the "
                                "user or caller sees, never that a "
                                "function was called.",
                            ),
                            minItems=1,
                        ),
                        "verification": _OPTIONAL_COMMAND_REF_SCHEMA,
                    }
                ),
            }
        ),
        "done_criteria": record_array_schema(
            {
                "kind": enum_schema(
                    [
                        "behavior",
                        "step-gate",
                        "test-gate",
                        "scope-integrity",
                        "static-invariant",
                    ]
                ),
                "description": text_schema(
                    min_length=20,
                    max_length=500,
                    description="A statement the executor can settle as true or "
                    "false without judgement, naming the exact test "
                    "symbol, file, or observable behaviour it turns "
                    "on. Not 'the code is clean' or 'performance "
                    "improves'.",
                ),
                "verification": _OPTIONAL_COMMAND_REF_SCHEMA,
            },
            minItems=1,
        ),
        "false_assumption": strict_object(_STOP_CONDITION_BODY_PROPERTIES),
        "additional_command_refs": array_schema(_COMMAND_REF_SCHEMA),
    }
)
