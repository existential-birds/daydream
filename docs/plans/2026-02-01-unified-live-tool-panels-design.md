# Unified Live Tool Panels

## Problem

Currently, Grep and Read tool panels display as two separate bordered boxes:
1. A header panel with tool name and arguments
2. An output panel with results

Additionally:
- Normal mode uses `transient=True`, causing panels to disappear after completion
- Quiet mode prints two separate static panels instead of using a unified approach

## Solution

Combine header and output into a single persistent panel that:
1. Shows header + animated throbber while executing
2. Transitions in-place to header + result when complete
3. Stays visible after completion

Quiet mode shows the same panel but hides the result section.

## Changes

### 1. `LiveToolPanel.start()` - Remove quiet mode branching

```python
def start(self) -> None:
    """Start Live context and show tool call with animated throbber."""
    self._console.print()  # Add newline for separation

    self._live = Live(
        self,
        console=self._console,
        refresh_per_second=10,
        transient=False,  # Panel persists after stop()
    )
    self._live.start()
```

### 2. `LiveToolPanel.finish()` - Simplify to just stop Live

```python
def finish(self) -> None:
    """Stop Live context. Final panel state persists on screen."""
    if self._live is not None:
        self._live.stop()
        self._live = None
```

### 3. `LiveToolPanel._render_panel()` - Quiet mode hides result at render time

```python
def _render_panel(self) -> Panel:
    """Render current state as a Panel."""
    header = self._build_tool_header_content()

    # Border color based on tool type and error state
    if self._name == "Skill":
        border_color = NEON_COLORS["cyan"]
    elif self._is_error:
        border_color = NEON_COLORS["red"]
    else:
        border_color = NEON_COLORS["purple"]

    if self._result is None:
        # Still executing: header + throbber (both modes)
        content = Group(
            header,
            Text("\n"),
            self._throbber.render(width=40),
        )
    elif self._quiet_mode:
        # Quiet mode complete: header only
        content = Group(header)
    else:
        # Normal mode complete: header + result
        result_content = self._build_result_content_internal()

        if self._is_error:
            result_title = Text("\u274c Error", style=STYLE_BOLD_RED)
        else:
            result_title = Text("Output", style=STYLE_BOLD_CYAN)

        if isinstance(result_content, Text) and not result_content.plain.strip():
            content = Group(header)  # Empty result
        else:
            content = Group(
                header,
                Text("\n"),
                result_title,
                Text("\n"),
                result_content,
            )

    return Panel(
        content,
        box=box.ROUNDED,
        border_style=Style(color=border_color),
        style=STYLE_PANEL_BG,
        padding=(0, 1),
    )
```

## Files Changed

- `daydream/ui.py` - `LiveToolPanel` class only

## Files Unchanged

- `agent.py` - Already uses registry correctly
- `LiveToolPanelRegistry` - Just creates panels
- `print_tool_call()` / `print_tool_result()` - Keep for other use cases

## Testing

```bash
# Normal mode - persistent panels with results
daydream /path/to/project --python

# Quiet mode - persistent panels, headers only
daydream /path/to/project --python --quiet
```

Verify:
- Throbber animates during tool execution
- Panel transitions smoothly to result state
- Grep shows "Found N matches" formatting
- Read shows syntax-highlighted code
- Errors show red border and error title
