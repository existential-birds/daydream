# Design: Collapse Skill Tool Display

**Date:** 2026-02-01
**Status:** Approved

## Problem

The current Skill tool display is redundant and wastes vertical space:

1. **Skill panel** (cyan border): Shows `✨ Skill` header + skill name as pill badge
2. **Output panel** (purple border): Shows `Output` header + "Launching skill: X"

The skill name appears twice, and the output provides no additional information.

## Solution

Suppress the Output panel entirely for Skill tool calls. Show only the single Skill panel.

**Before:**
```
┌─ ✨ Skill ────────────────────────┐
│  beagle:review-python             │
└───────────────────────────────────┘

┌─ Output ──────────────────────────┐
│                                   │
│ Launching skill: beagle:review-   │
│ python                            │
└───────────────────────────────────┘
```

**After:**
```
┌─ ✨ Skill ────────────────────────┐
│  beagle:review-python             │
└───────────────────────────────────┘
```

## Implementation

In `daydream/ui.py`, modify `LiveToolPanel` to skip rendering the result section when the tool name is `"Skill"`:

1. In `set_result()`: Still store the result (for completeness), but flag that it should not be displayed
2. In `__rich__()`: Skip the Output section when `self._name == "Skill"`
3. In `finish()`: Skip printing the result panel when `self._name == "Skill"`

The logic is: **never render an Output panel for Skill tool calls**.

## Files to Modify

- `daydream/ui.py`: `LiveToolPanel` class methods (`__rich__`, `finish`)

## Testing

1. Run `daydream` with `--python` flag on a Python project
2. Verify only one panel appears for the Skill tool call
3. Verify other tool calls (Bash, Glob, etc.) still show Output panels normally
