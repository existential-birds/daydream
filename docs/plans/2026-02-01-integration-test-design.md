# Integration Test Design

## Overview

Add one integration test that exercises the full review-fix-test flow, mocking at the Claude Agent SDK boundary.

## Test File

`tests/test_integration.py`

## Mock Strategy

Patch `claude_agent_sdk.ClaudeSDKClient` at the class level. The mock returns canned responses based on prompt content to simulate each phase:

| Phase | Prompt Detection | Mock Response |
|-------|------------------|---------------|
| Review | Contains `beagle:review-` | Markdown review with issues |
| Parse | Contains "parse" or "JSON" | JSON array of feedback items |
| Fix | Contains "fix" | Confirmation message |
| Test | Contains "test" | Passing test output |

## Mock Implementation

```python
class MockClaudeSDKClient:
    def __init__(self, options=None):
        self.options = options
        self._prompt = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def query(self, prompt: str):
        self._prompt = prompt

    async def receive_response(self):
        # Yield appropriate ResultMessage based on self._prompt
        response = self._get_response_for_prompt()
        yield ResultMessage(...)
```

## Test Case

```python
@pytest.mark.asyncio
async def test_full_fix_flow(tmp_path, monkeypatch):
    # Setup: create target dir with dummy Python file
    target = tmp_path / "project"
    target.mkdir()
    (target / "main.py").write_text("print('hello')")

    # Mock the SDK client
    monkeypatch.setattr("daydream.agent.ClaudeSDKClient", MockClaudeSDKClient)

    # Run full flow
    config = RunConfig(
        target=str(target),
        skill="python",
        quiet=True,
    )
    exit_code = await run(config)

    # Assert success
    assert exit_code == 0
    assert (target / ".review-output.md").exists()
```

## Dependencies

Add to `pyproject.toml` dev dependencies:
- `pytest-asyncio>=0.24`

## Tasks

1. Add `pytest-asyncio` to dev dependencies
2. Create `tests/test_integration.py` with mock and test
3. Run test to verify it passes
