# RLM Code Review Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Recursive Language Model (RLM) capabilities to Daydream for reviewing large codebases (1M+ tokens) using sandboxed Python REPL execution inside devcontainers.

**Architecture:** Three-process model with Host (Daydream CLI) orchestrating a devcontainer-sandboxed Python REPL. The REPL executes model-generated code and routes `llm_query()` calls back to the host via JSON-RPC over stdin/stdout.

**Tech Stack:** Python 3.12, asyncio, JSON-RPC 2.0, devcontainer CLI, tree-sitter (optional phase 3)

---

## Phase 1: Core Infrastructure

### Task 1: Create RLM module structure

**Files:**
- Create: `daydream/rlm/__init__.py`
- Create: `daydream/rlm/errors.py`

**Step 1: Write the failing test**

Create `tests/rlm/__init__.py` (empty) and `tests/rlm/test_errors.py`:

```python
# tests/rlm/__init__.py
# Empty - marks directory as test package
```

```python
# tests/rlm/test_errors.py
"""Tests for RLM error types."""

from daydream.rlm.errors import (
    RLMError,
    REPLCrashError,
    REPLTimeoutError,
    HeartbeatFailedError,
    ContainerError,
)


def test_rlm_error_is_exception():
    """RLMError should be a base Exception."""
    err = RLMError("test message")
    assert isinstance(err, Exception)
    assert str(err) == "test message"


def test_repl_crash_error_inherits_rlm_error():
    """REPLCrashError should inherit from RLMError."""
    err = REPLCrashError("process died")
    assert isinstance(err, RLMError)
    assert "process died" in str(err)


def test_repl_timeout_error_inherits_rlm_error():
    """REPLTimeoutError should inherit from RLMError."""
    err = REPLTimeoutError("exceeded 300s")
    assert isinstance(err, RLMError)


def test_heartbeat_failed_error_inherits_rlm_error():
    """HeartbeatFailedError should inherit from RLMError."""
    err = HeartbeatFailedError("no pong received")
    assert isinstance(err, RLMError)


def test_container_error_inherits_rlm_error():
    """ContainerError should inherit from RLMError."""
    err = ContainerError("failed to start")
    assert isinstance(err, RLMError)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rlm/test_errors.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'daydream.rlm'"

**Step 3: Write minimal implementation**

```python
# daydream/rlm/__init__.py
"""RLM (Recursive Language Model) code review module.

This module provides capabilities for reviewing large codebases (1M+ tokens)
using sandboxed Python REPL execution with sub-LLM orchestration.
"""

from daydream.rlm.errors import (
    ContainerError,
    HeartbeatFailedError,
    REPLCrashError,
    REPLTimeoutError,
    RLMError,
)

__all__ = [
    "RLMError",
    "REPLCrashError",
    "REPLTimeoutError",
    "HeartbeatFailedError",
    "ContainerError",
]
```

```python
# daydream/rlm/errors.py
"""Error types for the RLM module."""


class RLMError(Exception):
    """Base class for RLM errors."""


class REPLCrashError(RLMError):
    """REPL process exited unexpectedly."""


class REPLTimeoutError(RLMError):
    """Code execution exceeded timeout."""


class HeartbeatFailedError(RLMError):
    """REPL stopped responding to heartbeats."""


class ContainerError(RLMError):
    """Devcontainer failed to start or crashed."""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/rlm/test_errors.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add daydream/rlm/__init__.py daydream/rlm/errors.py tests/rlm/__init__.py tests/rlm/test_errors.py
git commit -m "feat(rlm): add module structure and error types"
```

---

### Task 2: Implement JSON-RPC protocol helpers

**Files:**
- Create: `daydream/rlm/ipc.py`
- Create: `tests/rlm/test_ipc.py`

**Step 1: Write the failing test**

```python
# tests/rlm/test_ipc.py
"""Tests for JSON-RPC IPC protocol helpers."""

import json

from daydream.rlm.ipc import (
    JsonRpcRequest,
    JsonRpcResponse,
    JsonRpcError,
    encode_request,
    encode_response,
    decode_message,
    generate_id,
)


class TestJsonRpcRequest:
    """Tests for JsonRpcRequest dataclass."""

    def test_request_with_params(self):
        """Request should include method, params, and id."""
        req = JsonRpcRequest(method="execute", params={"code": "print(1)"}, id="1")
        assert req.method == "execute"
        assert req.params == {"code": "print(1)"}
        assert req.id == "1"

    def test_request_without_params(self):
        """Request params should default to None."""
        req = JsonRpcRequest(method="ping", id="hb-1")
        assert req.params is None


class TestJsonRpcResponse:
    """Tests for JsonRpcResponse dataclass."""

    def test_response_with_result(self):
        """Response should include result and id."""
        resp = JsonRpcResponse(id="1", result={"output": "hello"})
        assert resp.id == "1"
        assert resp.result == {"output": "hello"}
        assert resp.error is None

    def test_response_with_error(self):
        """Response should include error when present."""
        err = JsonRpcError(code=-32600, message="Invalid Request")
        resp = JsonRpcResponse(id="1", error=err)
        assert resp.error.code == -32600
        assert resp.error.message == "Invalid Request"


class TestEncode:
    """Tests for encoding functions."""

    def test_encode_request(self):
        """encode_request should produce valid JSON-RPC 2.0."""
        req = JsonRpcRequest(method="execute", params={"code": "x=1"}, id="42")
        line = encode_request(req)
        assert line.endswith("\n")
        data = json.loads(line)
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "execute"
        assert data["params"] == {"code": "x=1"}
        assert data["id"] == "42"

    def test_encode_request_without_params(self):
        """encode_request should omit params if None."""
        req = JsonRpcRequest(method="ping", id="hb-1")
        line = encode_request(req)
        data = json.loads(line)
        assert "params" not in data

    def test_encode_response_with_result(self):
        """encode_response should produce valid JSON-RPC 2.0 result."""
        resp = JsonRpcResponse(id="1", result="pong")
        line = encode_response(resp)
        data = json.loads(line)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "1"
        assert data["result"] == "pong"
        assert "error" not in data

    def test_encode_response_with_error(self):
        """encode_response should produce valid JSON-RPC 2.0 error."""
        err = JsonRpcError(code=-32600, message="Invalid")
        resp = JsonRpcResponse(id="1", error=err)
        line = encode_response(resp)
        data = json.loads(line)
        assert data["error"]["code"] == -32600
        assert data["error"]["message"] == "Invalid"
        assert "result" not in data


class TestDecode:
    """Tests for decode_message function."""

    def test_decode_request(self):
        """decode_message should parse JSON-RPC request."""
        line = '{"jsonrpc":"2.0","method":"llm_query","params":{"prompt":"hi"},"id":"cb-1"}\n'
        msg = decode_message(line)
        assert isinstance(msg, JsonRpcRequest)
        assert msg.method == "llm_query"
        assert msg.params == {"prompt": "hi"}
        assert msg.id == "cb-1"

    def test_decode_response_with_result(self):
        """decode_message should parse JSON-RPC response with result."""
        line = '{"jsonrpc":"2.0","id":"1","result":"ok"}\n'
        msg = decode_message(line)
        assert isinstance(msg, JsonRpcResponse)
        assert msg.id == "1"
        assert msg.result == "ok"

    def test_decode_response_with_error(self):
        """decode_message should parse JSON-RPC response with error."""
        line = '{"jsonrpc":"2.0","id":"1","error":{"code":-32600,"message":"Bad"}}\n'
        msg = decode_message(line)
        assert isinstance(msg, JsonRpcResponse)
        assert msg.error.code == -32600

    def test_decode_invalid_json_raises(self):
        """decode_message should raise on invalid JSON."""
        import pytest
        with pytest.raises(json.JSONDecodeError):
            decode_message("not json\n")


class TestGenerateId:
    """Tests for generate_id function."""

    def test_generate_id_uniqueness(self):
        """generate_id should produce unique IDs."""
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_id_with_prefix(self):
        """generate_id should support prefix."""
        id1 = generate_id(prefix="hb")
        assert id1.startswith("hb-")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rlm/test_ipc.py -v`
Expected: FAIL with "cannot import name 'JsonRpcRequest'"

**Step 3: Write minimal implementation**

```python
# daydream/rlm/ipc.py
"""JSON-RPC 2.0 protocol helpers for REPL IPC.

This module provides encoding/decoding for newline-delimited JSON-RPC messages
used for communication between the host process and the sandboxed REPL.
"""

import json
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class JsonRpcError:
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Any = None


@dataclass
class JsonRpcRequest:
    """JSON-RPC 2.0 request message."""

    method: str
    id: str
    params: dict[str, Any] | None = None


@dataclass
class JsonRpcResponse:
    """JSON-RPC 2.0 response message."""

    id: str
    result: Any = None
    error: JsonRpcError | None = None


def generate_id(prefix: str = "") -> str:
    """Generate a unique message ID.

    Args:
        prefix: Optional prefix for the ID (e.g., "hb" for heartbeat).

    Returns:
        Unique string ID, optionally prefixed.
    """
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}-{uid}" if prefix else uid


def encode_request(request: JsonRpcRequest) -> str:
    """Encode a JSON-RPC request as a newline-delimited JSON string.

    Args:
        request: The request to encode.

    Returns:
        JSON string with trailing newline.
    """
    data: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": request.method,
        "id": request.id,
    }
    if request.params is not None:
        data["params"] = request.params
    return json.dumps(data) + "\n"


def encode_response(response: JsonRpcResponse) -> str:
    """Encode a JSON-RPC response as a newline-delimited JSON string.

    Args:
        response: The response to encode.

    Returns:
        JSON string with trailing newline.
    """
    data: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": response.id,
    }
    if response.error is not None:
        error_data: dict[str, Any] = {
            "code": response.error.code,
            "message": response.error.message,
        }
        if response.error.data is not None:
            error_data["data"] = response.error.data
        data["error"] = error_data
    else:
        data["result"] = response.result
    return json.dumps(data) + "\n"


def decode_message(line: str) -> JsonRpcRequest | JsonRpcResponse:
    """Decode a newline-delimited JSON-RPC message.

    Args:
        line: JSON string (may include trailing newline).

    Returns:
        JsonRpcRequest or JsonRpcResponse depending on message type.

    Raises:
        json.JSONDecodeError: If the line is not valid JSON.
        KeyError: If required fields are missing.
    """
    data = json.loads(line.strip())

    # Response has "result" or "error", request has "method"
    if "method" in data:
        return JsonRpcRequest(
            method=data["method"],
            id=data["id"],
            params=data.get("params"),
        )
    else:
        error = None
        if "error" in data:
            err_data = data["error"]
            error = JsonRpcError(
                code=err_data["code"],
                message=err_data["message"],
                data=err_data.get("data"),
            )
        return JsonRpcResponse(
            id=data["id"],
            result=data.get("result"),
            error=error,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/rlm/test_ipc.py -v`
Expected: PASS (14 tests)

**Step 5: Commit**

```bash
git add daydream/rlm/ipc.py tests/rlm/test_ipc.py
git commit -m "feat(rlm): add JSON-RPC 2.0 protocol helpers"
```

---

### Task 3: Implement REPL environment data structures

**Files:**
- Create: `daydream/rlm/environment.py`
- Create: `tests/rlm/test_environment.py`

**Step 1: Write the failing test**

```python
# tests/rlm/test_environment.py
"""Tests for REPL environment data structures."""

from daydream.rlm.environment import (
    FileInfo,
    Service,
    RepoContext,
)


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_file_info_creation(self):
        """FileInfo should store parsed file metadata."""
        info = FileInfo(
            language="python",
            functions=["main", "helper"],
            classes=["MyClass"],
            imports=["os", "sys"],
            exports=[],
        )
        assert info.language == "python"
        assert info.functions == ["main", "helper"]
        assert info.classes == ["MyClass"]
        assert info.imports == ["os", "sys"]
        assert info.exports == []


class TestService:
    """Tests for Service dataclass."""

    def test_service_creation(self):
        """Service should store service boundary metadata."""
        svc = Service(
            name="auth",
            root="services/auth",
            files=["services/auth/main.py", "services/auth/utils.py"],
            dependencies=["db", "cache"],
        )
        assert svc.name == "auth"
        assert svc.root == "services/auth"
        assert len(svc.files) == 2
        assert svc.dependencies == ["db", "cache"]


class TestRepoContext:
    """Tests for RepoContext dataclass."""

    def test_repo_context_creation(self):
        """RepoContext should hold all codebase data."""
        ctx = RepoContext(
            files={"main.py": "print('hello')"},
            structure={"main.py": FileInfo("python", ["main"], [], [], [])},
            services={},
            file_sizes={"main.py": 10},
            total_tokens=10,
            file_count=1,
            largest_files=[("main.py", 10)],
            languages=["python"],
            changed_files=None,
        )
        assert ctx.file_count == 1
        assert ctx.total_tokens == 10
        assert "main.py" in ctx.files

    def test_repo_context_with_changed_files(self):
        """RepoContext should support PR mode with changed_files."""
        ctx = RepoContext(
            files={},
            structure={},
            services={},
            file_sizes={},
            total_tokens=0,
            file_count=0,
            largest_files=[],
            languages=[],
            changed_files=["src/api.py", "tests/test_api.py"],
        )
        assert ctx.changed_files == ["src/api.py", "tests/test_api.py"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rlm/test_environment.py -v`
Expected: FAIL with "cannot import name 'FileInfo'"

**Step 3: Write minimal implementation**

```python
# daydream/rlm/environment.py
"""REPL environment data structures.

This module defines the data structures available to model-generated code
in the sandboxed REPL environment.
"""

from dataclasses import dataclass


@dataclass
class FileInfo:
    """Parsed metadata about a source file.

    Attributes:
        language: Programming language ("python", "typescript", "go").
        functions: List of function/method names defined in the file.
        classes: List of class/struct/interface names defined in the file.
        imports: List of import statements or imported modules.
        exports: List of exported symbols (primarily for TS/Go).
    """

    language: str
    functions: list[str]
    classes: list[str]
    imports: list[str]
    exports: list[str]


@dataclass
class Service:
    """Metadata about a detected service boundary.

    Attributes:
        name: Service identifier (e.g., "auth", "billing").
        root: Root directory path (e.g., "services/auth").
        files: List of all file paths belonging to this service.
        dependencies: List of other service names this service imports from.
    """

    name: str
    root: str
    files: list[str]
    dependencies: list[str]


@dataclass
class RepoContext:
    """Complete codebase context available to the REPL.

    This is the `repo` object exposed in the REPL namespace.

    Attributes:
        files: Mapping of file paths to their contents.
        structure: Mapping of file paths to parsed FileInfo.
        services: Mapping of service names to Service metadata.
        file_sizes: Mapping of file paths to token counts.
        total_tokens: Total token count across all files.
        file_count: Number of files in the repository.
        largest_files: Top files by token count as (path, tokens) tuples.
        languages: List of detected programming languages.
        changed_files: Files changed in PR (None for full repo review).
    """

    files: dict[str, str]
    structure: dict[str, FileInfo]
    services: dict[str, Service]
    file_sizes: dict[str, int]
    total_tokens: int
    file_count: int
    largest_files: list[tuple[str, int]]
    languages: list[str]
    changed_files: list[str] | None
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/rlm/test_environment.py -v`
Expected: PASS (4 tests)

**Step 5: Update module exports and commit**

Update `daydream/rlm/__init__.py` to export new types:

```python
# daydream/rlm/__init__.py
"""RLM (Recursive Language Model) code review module.

This module provides capabilities for reviewing large codebases (1M+ tokens)
using sandboxed Python REPL execution with sub-LLM orchestration.
"""

from daydream.rlm.environment import FileInfo, RepoContext, Service
from daydream.rlm.errors import (
    ContainerError,
    HeartbeatFailedError,
    REPLCrashError,
    REPLTimeoutError,
    RLMError,
)

__all__ = [
    # Errors
    "RLMError",
    "REPLCrashError",
    "REPLTimeoutError",
    "HeartbeatFailedError",
    "ContainerError",
    # Environment
    "FileInfo",
    "Service",
    "RepoContext",
]
```

```bash
git add daydream/rlm/environment.py daydream/rlm/__init__.py tests/rlm/test_environment.py
git commit -m "feat(rlm): add REPL environment data structures"
```

---

### Task 4: Implement REPL namespace builder

**Files:**
- Modify: `daydream/rlm/environment.py`
- Modify: `tests/rlm/test_environment.py`

**Step 1: Write the failing test**

Add to `tests/rlm/test_environment.py`:

```python
class TestBuildReplNamespace:
    """Tests for build_repl_namespace function."""

    def test_namespace_contains_repo(self):
        """Namespace should contain repo context."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={"a.py": "x=1"},
            structure={},
            services={},
            file_sizes={"a.py": 5},
            total_tokens=5,
            file_count=1,
            largest_files=[("a.py", 5)],
            languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "response")
        assert "repo" in ns
        assert ns["repo"].files == {"a.py": "x=1"}

    def test_namespace_contains_llm_query(self):
        """Namespace should contain llm_query function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        calls = []

        def mock_llm(prompt: str, model: str = "haiku") -> str:
            calls.append((prompt, model))
            return "mocked"

        ns = build_repl_namespace(ctx, llm_query_fn=mock_llm)
        result = ns["llm_query"]("test prompt")
        assert result == "mocked"
        assert calls == [("test prompt", "haiku")]

    def test_namespace_contains_llm_query_parallel(self):
        """Namespace should contain llm_query_parallel function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )

        def mock_llm(prompt: str, model: str = "haiku") -> str:
            return f"response to: {prompt[:10]}"

        def mock_parallel(prompts: list[str], model: str = "haiku") -> list[str]:
            return [f"parallel: {p[:5]}" for p in prompts]

        ns = build_repl_namespace(
            ctx,
            llm_query_fn=mock_llm,
            llm_query_parallel_fn=mock_parallel,
        )
        results = ns["llm_query_parallel"](["a", "b"])
        assert results == ["parallel: a", "parallel: b"]

    def test_namespace_contains_files_containing(self):
        """Namespace should contain files_containing search function."""
        from daydream.rlm.environment import build_repl_namespace
        import re

        ctx = RepoContext(
            files={
                "a.py": "def foo(): pass",
                "b.py": "def bar(): pass",
                "c.py": "x = 1",
            },
            structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=3, largest_files=[], languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")
        matches = ns["files_containing"](r"def \w+")
        assert set(matches) == {"a.py", "b.py"}

    def test_namespace_contains_files_importing(self):
        """Namespace should contain files_importing search function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={
                "a.py": "import os\nimport sys",
                "b.py": "from os import path",
                "c.py": "import json",
            },
            structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=3, largest_files=[], languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")
        matches = ns["files_importing"]("os")
        assert set(matches) == {"a.py", "b.py"}

    def test_namespace_contains_get_file_slice(self):
        """Namespace should contain get_file_slice function."""
        from daydream.rlm.environment import build_repl_namespace

        ctx = RepoContext(
            files={"a.py": "line1\nline2\nline3\nline4\nline5"},
            structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=1, largest_files=[], languages=["python"],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")
        # get_file_slice uses 1-based line numbers
        result = ns["get_file_slice"]("a.py", 2, 4)
        assert result == "line2\nline3\nline4"

    def test_namespace_contains_final_functions(self):
        """Namespace should contain FINAL and FINAL_VAR functions."""
        from daydream.rlm.environment import build_repl_namespace, FinalAnswer

        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        ns = build_repl_namespace(ctx, llm_query_fn=lambda p, m: "")

        # Test FINAL raises FinalAnswer with the answer
        try:
            ns["FINAL"]("my answer")
            assert False, "Should have raised FinalAnswer"
        except FinalAnswer as e:
            assert e.answer == "my answer"

        # Test FINAL_VAR raises FinalAnswer with variable value
        ns["my_var"] = "variable content"
        try:
            ns["FINAL_VAR"]("my_var")
            assert False, "Should have raised FinalAnswer"
        except FinalAnswer as e:
            assert e.answer == "variable content"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rlm/test_environment.py::TestBuildReplNamespace -v`
Expected: FAIL with "cannot import name 'build_repl_namespace'"

**Step 3: Write minimal implementation**

Add to `daydream/rlm/environment.py`:

```python
import re
from typing import Callable


class FinalAnswer(Exception):
    """Raised when FINAL() or FINAL_VAR() is called to signal completion.

    Attributes:
        answer: The final answer string to return to the user.
    """

    def __init__(self, answer: str):
        self.answer = answer
        super().__init__(answer)


def build_repl_namespace(
    ctx: RepoContext,
    llm_query_fn: Callable[[str, str], str],
    llm_query_parallel_fn: Callable[[list[str], str], list[str]] | None = None,
) -> dict[str, any]:
    """Build the namespace dict for the REPL environment.

    Creates a dictionary containing all objects and functions available
    to model-generated code in the REPL.

    Args:
        ctx: The RepoContext with codebase data.
        llm_query_fn: Function to call for llm_query(prompt, model).
        llm_query_parallel_fn: Optional function for parallel queries.
            If None, falls back to sequential calls.

    Returns:
        Dictionary to use as REPL globals/locals.
    """
    namespace: dict[str, any] = {}

    # Expose repo context
    namespace["repo"] = ctx

    # Wrap llm_query with default model parameter
    def llm_query(prompt: str, model: str = "haiku") -> str:
        """Fresh-context sub-LLM call. Returns response text."""
        return llm_query_fn(prompt, model)

    namespace["llm_query"] = llm_query

    # Parallel query function
    if llm_query_parallel_fn is not None:
        def llm_query_parallel(prompts: list[str], model: str = "haiku") -> list[str]:
            """Batch multiple independent queries for efficiency."""
            return llm_query_parallel_fn(prompts, model)
    else:
        def llm_query_parallel(prompts: list[str], model: str = "haiku") -> list[str]:
            """Fallback: execute queries sequentially."""
            return [llm_query_fn(p, model) for p in prompts]

    namespace["llm_query_parallel"] = llm_query_parallel

    # Search function: files_containing
    def files_containing(pattern: str) -> list[str]:
        """Grep-like regex search, returns matching file paths."""
        compiled = re.compile(pattern)
        return [path for path, content in ctx.files.items() if compiled.search(content)]

    namespace["files_containing"] = files_containing

    # Search function: files_importing
    def files_importing(module: str) -> list[str]:
        """Find files that import a given module."""
        # Match: import X, from X import, import X as Y
        pattern = rf"(?:^|\n)\s*(?:import\s+{re.escape(module)}|from\s+{re.escape(module)}\s+import)"
        compiled = re.compile(pattern)
        return [path for path, content in ctx.files.items() if compiled.search(content)]

    namespace["files_importing"] = files_importing

    # File slice function
    def get_file_slice(path: str, start_line: int, end_line: int) -> str:
        """Get specific line range from a file (1-based, inclusive)."""
        content = ctx.files.get(path, "")
        lines = content.split("\n")
        # Convert to 0-based index, end_line is inclusive
        selected = lines[start_line - 1 : end_line]
        return "\n".join(selected)

    namespace["get_file_slice"] = get_file_slice

    # FINAL function - signals completion with direct answer
    def FINAL(answer: str) -> None:
        """Signal task completion with direct answer."""
        raise FinalAnswer(answer)

    namespace["FINAL"] = FINAL

    # FINAL_VAR function - signals completion using a variable
    def FINAL_VAR(var_name: str) -> None:
        """Signal task completion, returning a REPL variable as output."""
        if var_name not in namespace:
            raise NameError(f"Variable '{var_name}' not found in namespace")
        raise FinalAnswer(str(namespace[var_name]))

    namespace["FINAL_VAR"] = FINAL_VAR

    return namespace
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/rlm/test_environment.py -v`
Expected: PASS (all tests)

**Step 5: Update exports and commit**

Update `daydream/rlm/__init__.py`:

```python
from daydream.rlm.environment import (
    FileInfo,
    FinalAnswer,
    RepoContext,
    Service,
    build_repl_namespace,
)
```

Add to `__all__`:
```python
    "FinalAnswer",
    "build_repl_namespace",
```

```bash
git add daydream/rlm/environment.py daydream/rlm/__init__.py tests/rlm/test_environment.py
git commit -m "feat(rlm): add REPL namespace builder with helper functions"
```

---

### Task 5: Add RLM constants to config

**Files:**
- Modify: `daydream/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
"""Tests for daydream configuration constants."""

from daydream.config import (
    RLM_REPL_INIT_TIMEOUT,
    RLM_CODE_EXEC_TIMEOUT,
    RLM_HEARTBEAT_TIMEOUT,
    RLM_LLM_QUERY_TIMEOUT,
    RLM_CONTAINER_STARTUP_TIMEOUT,
    RLM_HEARTBEAT_INTERVAL,
    RLM_OUTPUT_TRUNCATION_LIMIT,
)


def test_rlm_timeout_constants_exist():
    """RLM timeout constants should be defined."""
    assert RLM_REPL_INIT_TIMEOUT == 60
    assert RLM_CODE_EXEC_TIMEOUT == 300
    assert RLM_HEARTBEAT_TIMEOUT == 5
    assert RLM_LLM_QUERY_TIMEOUT == 60
    assert RLM_CONTAINER_STARTUP_TIMEOUT == 120


def test_rlm_heartbeat_interval():
    """RLM heartbeat interval should be defined."""
    assert RLM_HEARTBEAT_INTERVAL == 10


def test_rlm_output_truncation_limit():
    """RLM output truncation limit should be 50k chars."""
    assert RLM_OUTPUT_TRUNCATION_LIMIT == 50_000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "cannot import name 'RLM_REPL_INIT_TIMEOUT'"

**Step 3: Write minimal implementation**

Add to `daydream/config.py`:

```python
# RLM timeout configuration (in seconds)
RLM_REPL_INIT_TIMEOUT = 60  # REPL initialization
RLM_CODE_EXEC_TIMEOUT = 300  # Code execution (5 min)
RLM_HEARTBEAT_TIMEOUT = 5  # Heartbeat response
RLM_LLM_QUERY_TIMEOUT = 60  # Sub-LLM query
RLM_CONTAINER_STARTUP_TIMEOUT = 120  # Container startup

# RLM heartbeat interval (in seconds)
RLM_HEARTBEAT_INTERVAL = 10

# RLM output truncation limit (in characters)
RLM_OUTPUT_TRUNCATION_LIMIT = 50_000
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add daydream/config.py tests/test_config.py
git commit -m "feat(rlm): add RLM timeout and configuration constants"
```

---

## Phase 2: Container Integration

### Task 6: Implement devcontainer management

**Files:**
- Create: `daydream/rlm/container.py`
- Create: `tests/rlm/test_container.py`

**Step 1: Write the failing test**

```python
# tests/rlm/test_container.py
"""Tests for devcontainer management."""

import pytest

from daydream.rlm.container import (
    ContainerConfig,
    DevContainer,
    find_devcontainer_config,
)


class TestContainerConfig:
    """Tests for ContainerConfig dataclass."""

    def test_container_config_defaults(self):
        """ContainerConfig should have sensible defaults."""
        cfg = ContainerConfig(workspace_path="/repo")
        assert cfg.workspace_path == "/repo"
        assert cfg.container_id is None
        assert cfg.mount_readonly is True

    def test_container_config_custom(self):
        """ContainerConfig should accept custom values."""
        cfg = ContainerConfig(
            workspace_path="/repo",
            container_id="abc123",
            mount_readonly=False,
        )
        assert cfg.container_id == "abc123"
        assert cfg.mount_readonly is False


class TestFindDevcontainerConfig:
    """Tests for find_devcontainer_config function."""

    def test_find_devcontainer_config_not_found(self, tmp_path):
        """Should return None if no .devcontainer found."""
        result = find_devcontainer_config(tmp_path)
        assert result is None

    def test_find_devcontainer_config_found(self, tmp_path):
        """Should return path to devcontainer.json if found."""
        devcontainer_dir = tmp_path / ".devcontainer"
        devcontainer_dir.mkdir()
        config_file = devcontainer_dir / "devcontainer.json"
        config_file.write_text('{"name": "test"}')

        result = find_devcontainer_config(tmp_path)
        assert result == config_file


class TestDevContainer:
    """Tests for DevContainer class."""

    def test_devcontainer_init(self):
        """DevContainer should initialize with config."""
        cfg = ContainerConfig(workspace_path="/repo")
        dc = DevContainer(cfg)
        assert dc.config == cfg
        assert dc.process is None
        assert dc.container_id is None

    def test_devcontainer_not_running_initially(self):
        """DevContainer should not be running initially."""
        cfg = ContainerConfig(workspace_path="/repo")
        dc = DevContainer(cfg)
        assert dc.is_running is False

    @pytest.mark.asyncio
    async def test_devcontainer_exec_requires_running(self):
        """exec_command should raise if container not running."""
        cfg = ContainerConfig(workspace_path="/repo")
        dc = DevContainer(cfg)
        with pytest.raises(RuntimeError, match="not running"):
            await dc.exec_command(["echo", "test"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rlm/test_container.py -v`
Expected: FAIL with "cannot import name 'ContainerConfig'"

**Step 3: Write minimal implementation**

```python
# daydream/rlm/container.py
"""Devcontainer management for sandboxed REPL execution.

This module handles starting, stopping, and executing commands in
devcontainers for secure code execution.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.config import RLM_CONTAINER_STARTUP_TIMEOUT
from daydream.rlm.errors import ContainerError


@dataclass
class ContainerConfig:
    """Configuration for devcontainer setup.

    Attributes:
        workspace_path: Path to the workspace/repository to mount.
        container_id: Existing container ID to use (if already running).
        mount_readonly: Whether to mount workspace as read-only.
    """

    workspace_path: str
    container_id: str | None = None
    mount_readonly: bool = True


def find_devcontainer_config(workspace_path: Path) -> Path | None:
    """Find devcontainer.json in workspace.

    Args:
        workspace_path: Path to search for .devcontainer directory.

    Returns:
        Path to devcontainer.json if found, None otherwise.
    """
    devcontainer_dir = workspace_path / ".devcontainer"
    config_file = devcontainer_dir / "devcontainer.json"
    if config_file.exists():
        return config_file
    return None


class DevContainer:
    """Manages a devcontainer for sandboxed execution.

    Provides methods to start, stop, and execute commands in a
    devcontainer environment.
    """

    def __init__(self, config: ContainerConfig):
        """Initialize DevContainer with configuration.

        Args:
            config: Container configuration.
        """
        self.config = config
        self.process: asyncio.subprocess.Process | None = None
        self.container_id: str | None = config.container_id

    @property
    def is_running(self) -> bool:
        """Check if container is running."""
        return self.container_id is not None

    async def start(self, timeout: float = RLM_CONTAINER_STARTUP_TIMEOUT) -> None:
        """Start the devcontainer.

        Args:
            timeout: Maximum time to wait for container to start.

        Raises:
            ContainerError: If container fails to start.
        """
        if self.is_running:
            return

        workspace = Path(self.config.workspace_path)

        # Check for devcontainer config
        config_path = find_devcontainer_config(workspace)
        if config_path is None:
            raise ContainerError(
                f"No .devcontainer/devcontainer.json found in {workspace}"
            )

        try:
            # Start devcontainer using CLI
            proc = await asyncio.create_subprocess_exec(
                "devcontainer",
                "up",
                "--workspace-folder",
                str(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            if proc.returncode != 0:
                raise ContainerError(
                    f"devcontainer up failed: {stderr.decode()}"
                )

            # Parse container ID from output
            output = json.loads(stdout.decode())
            self.container_id = output.get("containerId")

            if not self.container_id:
                raise ContainerError("No container ID in devcontainer output")

        except asyncio.TimeoutError:
            raise ContainerError(f"Container startup timed out after {timeout}s")
        except json.JSONDecodeError as e:
            raise ContainerError(f"Failed to parse devcontainer output: {e}")

    async def stop(self) -> None:
        """Stop the devcontainer."""
        if not self.is_running:
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "devcontainer",
                "stop",
                "--container-id",
                self.container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
        finally:
            self.container_id = None

    async def exec_command(
        self,
        command: list[str],
        stdin: asyncio.StreamReader | None = None,
    ) -> tuple[asyncio.StreamWriter, asyncio.StreamReader, asyncio.StreamReader]:
        """Execute a command in the container.

        Args:
            command: Command and arguments to execute.
            stdin: Optional stdin stream to connect.

        Returns:
            Tuple of (stdin_writer, stdout_reader, stderr_reader).

        Raises:
            RuntimeError: If container is not running.
        """
        if not self.is_running:
            raise RuntimeError("Container is not running")

        # Build exec command
        exec_cmd = [
            "devcontainer",
            "exec",
            "--container-id",
            self.container_id,
            "-i",  # Keep stdin attached
            *command,
        ]

        self.process = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        return self.process.stdin, self.process.stdout, self.process.stderr

    async def __aenter__(self) -> "DevContainer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/rlm/test_container.py -v`
Expected: PASS (6 tests)

**Step 5: Update exports and commit**

Update `daydream/rlm/__init__.py`:

```python
from daydream.rlm.container import ContainerConfig, DevContainer, find_devcontainer_config
```

Add to `__all__`:
```python
    "ContainerConfig",
    "DevContainer",
    "find_devcontainer_config",
```

```bash
git add daydream/rlm/container.py daydream/rlm/__init__.py tests/rlm/test_container.py
git commit -m "feat(rlm): add devcontainer management"
```

---

### Task 7: Implement REPL process manager

**Files:**
- Create: `daydream/rlm/repl.py`
- Create: `tests/rlm/test_repl.py`

**Step 1: Write the failing test**

```python
# tests/rlm/test_repl.py
"""Tests for REPL process manager."""

import pytest

from daydream.rlm.repl import (
    REPLProcess,
    ExecuteResult,
)
from daydream.rlm.environment import RepoContext


class TestExecuteResult:
    """Tests for ExecuteResult dataclass."""

    def test_execute_result_success(self):
        """ExecuteResult should capture successful output."""
        result = ExecuteResult(
            output="hello\n",
            error=None,
            final_answer=None,
        )
        assert result.output == "hello\n"
        assert result.is_error is False
        assert result.is_final is False

    def test_execute_result_error(self):
        """ExecuteResult should capture errors."""
        result = ExecuteResult(
            output="",
            error="NameError: name 'x' is not defined",
            final_answer=None,
        )
        assert result.is_error is True
        assert "NameError" in result.error

    def test_execute_result_final(self):
        """ExecuteResult should capture final answer."""
        result = ExecuteResult(
            output="",
            error=None,
            final_answer="# Code Review Report\n...",
        )
        assert result.is_final is True
        assert result.final_answer.startswith("# Code Review")


class TestREPLProcess:
    """Tests for REPLProcess class."""

    def test_repl_process_init(self):
        """REPLProcess should initialize with context."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        repl = REPLProcess(ctx, llm_callback=lambda p, m: "")
        assert repl.context == ctx
        assert repl.is_running is False

    def test_repl_process_not_running_initially(self):
        """REPLProcess should not be running initially."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        repl = REPLProcess(ctx, llm_callback=lambda p, m: "")
        assert repl.is_running is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rlm/test_repl.py -v`
Expected: FAIL with "cannot import name 'REPLProcess'"

**Step 3: Write minimal implementation**

```python
# daydream/rlm/repl.py
"""REPL process manager for executing model-generated code.

This module manages the Python REPL process that executes code generated
by the model, handling IPC communication and output capture.
"""

import asyncio
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Callable

from daydream.config import RLM_OUTPUT_TRUNCATION_LIMIT
from daydream.rlm.environment import (
    FinalAnswer,
    RepoContext,
    build_repl_namespace,
)


@dataclass
class ExecuteResult:
    """Result of executing code in the REPL.

    Attributes:
        output: Captured stdout from execution.
        error: Error message if execution failed, None otherwise.
        final_answer: Final answer if FINAL/FINAL_VAR was called, None otherwise.
    """

    output: str
    error: str | None
    final_answer: str | None

    @property
    def is_error(self) -> bool:
        """Check if execution resulted in an error."""
        return self.error is not None

    @property
    def is_final(self) -> bool:
        """Check if execution produced a final answer."""
        return self.final_answer is not None


class REPLProcess:
    """Manages execution of Python code in a sandboxed namespace.

    This class provides the core REPL functionality, executing code
    in an isolated namespace with access to the codebase and LLM functions.
    """

    def __init__(
        self,
        context: RepoContext,
        llm_callback: Callable[[str, str], str],
        llm_parallel_callback: Callable[[list[str], str], list[str]] | None = None,
    ):
        """Initialize REPL process.

        Args:
            context: Repository context with codebase data.
            llm_callback: Function to handle llm_query calls.
            llm_parallel_callback: Function to handle parallel queries.
        """
        self.context = context
        self.llm_callback = llm_callback
        self.llm_parallel_callback = llm_parallel_callback
        self._namespace: dict | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Check if REPL is initialized and running."""
        return self._running

    def start(self) -> None:
        """Initialize the REPL namespace."""
        self._namespace = build_repl_namespace(
            self.context,
            llm_query_fn=self.llm_callback,
            llm_query_parallel_fn=self.llm_parallel_callback,
        )
        self._running = True

    def stop(self) -> None:
        """Clean up the REPL."""
        self._namespace = None
        self._running = False

    def execute(self, code: str) -> ExecuteResult:
        """Execute Python code in the REPL namespace.

        Args:
            code: Python code to execute.

        Returns:
            ExecuteResult with output, error, or final answer.
        """
        if not self._running or self._namespace is None:
            return ExecuteResult(
                output="",
                error="REPL is not running",
                final_answer=None,
            )

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, self._namespace, self._namespace)

            output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()

            # Combine stdout and stderr
            if stderr_output:
                output = output + stderr_output

            # Truncate if too long
            if len(output) > RLM_OUTPUT_TRUNCATION_LIMIT:
                output = (
                    output[:RLM_OUTPUT_TRUNCATION_LIMIT]
                    + "\n[truncated - use llm_query to analyze large outputs]"
                )

            return ExecuteResult(output=output, error=None, final_answer=None)

        except FinalAnswer as fa:
            # Model called FINAL() or FINAL_VAR()
            output = stdout_capture.getvalue()
            return ExecuteResult(
                output=output,
                error=None,
                final_answer=fa.answer,
            )

        except Exception:
            # Capture full traceback
            output = stdout_capture.getvalue()
            tb = traceback.format_exc()
            return ExecuteResult(
                output=output,
                error=tb,
                final_answer=None,
            )

    def __enter__(self) -> "REPLProcess":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/rlm/test_repl.py -v`
Expected: PASS (5 tests)

**Step 5: Add more comprehensive tests**

Add to `tests/rlm/test_repl.py`:

```python
class TestREPLProcessExecution:
    """Tests for REPLProcess code execution."""

    def test_execute_simple_code(self):
        """Should execute simple Python code."""
        ctx = RepoContext(
            files={"a.py": "x=1"},
            structure={}, services={}, file_sizes={"a.py": 3},
            total_tokens=3, file_count=1, largest_files=[("a.py", 3)],
            languages=["python"], changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute("print('hello world')")
            assert result.output.strip() == "hello world"
            assert result.is_error is False

    def test_execute_accesses_repo(self):
        """Should access repo context in execution."""
        ctx = RepoContext(
            files={"main.py": "print('hi')"},
            structure={}, services={}, file_sizes={"main.py": 10},
            total_tokens=10, file_count=1, largest_files=[("main.py", 10)],
            languages=["python"], changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute("print(len(repo.files))")
            assert "1" in result.output

    def test_execute_catches_errors(self):
        """Should capture execution errors."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute("undefined_variable")
            assert result.is_error is True
            assert "NameError" in result.error

    def test_execute_final_stops_execution(self):
        """FINAL() should produce final answer."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            result = repl.execute('FINAL("my report")')
            assert result.is_final is True
            assert result.final_answer == "my report"

    def test_execute_llm_query(self):
        """llm_query should route to callback."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        calls = []

        def mock_llm(prompt: str, model: str) -> str:
            calls.append((prompt, model))
            return "LLM response"

        with REPLProcess(ctx, llm_callback=mock_llm) as repl:
            result = repl.execute('result = llm_query("analyze this")\nprint(result)')
            assert "LLM response" in result.output
            assert len(calls) == 1
            assert calls[0][0] == "analyze this"
            assert calls[0][1] == "haiku"

    def test_execute_truncates_long_output(self):
        """Should truncate output exceeding limit."""
        ctx = RepoContext(
            files={}, structure={}, services={}, file_sizes={},
            total_tokens=0, file_count=0, largest_files=[], languages=[],
            changed_files=None,
        )
        with REPLProcess(ctx, llm_callback=lambda p, m: "") as repl:
            # Generate output longer than truncation limit
            result = repl.execute("print('x' * 100000)")
            assert "[truncated" in result.output
            assert len(result.output) < 60000  # Should be truncated
```

Run: `pytest tests/rlm/test_repl.py -v`
Expected: PASS (11 tests)

**Step 6: Update exports and commit**

Update `daydream/rlm/__init__.py`:

```python
from daydream.rlm.repl import ExecuteResult, REPLProcess
```

Add to `__all__`:
```python
    "ExecuteResult",
    "REPLProcess",
```

```bash
git add daydream/rlm/repl.py daydream/rlm/__init__.py tests/rlm/test_repl.py
git commit -m "feat(rlm): add REPL process manager with code execution"
```

---

## Phase 3: Orchestration

### Task 8: Implement RLM runner orchestration

**Files:**
- Create: `daydream/rlm/runner.py`
- Create: `tests/rlm/test_runner.py`

**Step 1: Write the failing test**

```python
# tests/rlm/test_runner.py
"""Tests for RLM runner orchestration."""

import pytest

from daydream.rlm.runner import (
    RLMConfig,
    RLMRunner,
    load_codebase,
)


class TestRLMConfig:
    """Tests for RLMConfig dataclass."""

    def test_config_defaults(self):
        """RLMConfig should have sensible defaults."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        assert cfg.workspace_path == "/repo"
        assert cfg.languages == ["python"]
        assert cfg.model == "opus"
        assert cfg.sub_model == "haiku"
        assert cfg.use_container is True

    def test_config_pr_mode(self):
        """RLMConfig should support PR mode."""
        cfg = RLMConfig(
            workspace_path="/repo",
            languages=["python"],
            pr_number=123,
        )
        assert cfg.pr_number == 123


class TestLoadCodebase:
    """Tests for load_codebase function."""

    def test_load_codebase_python(self, tmp_path):
        """Should load Python files from directory."""
        # Create test files
        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "readme.md").write_text("# Readme")

        ctx = load_codebase(tmp_path, languages=["python"])

        assert ctx.file_count == 2
        assert "main.py" in ctx.files or str(tmp_path / "main.py") in ctx.files
        assert ctx.languages == ["python"]

    def test_load_codebase_excludes_hidden(self, tmp_path):
        """Should exclude hidden directories."""
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("gitconfig")
        (tmp_path / "main.py").write_text("x=1")

        ctx = load_codebase(tmp_path, languages=["python"])

        assert ctx.file_count == 1
        assert not any(".git" in f for f in ctx.files.keys())

    def test_load_codebase_excludes_node_modules(self, tmp_path):
        """Should exclude node_modules."""
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "pkg.js").write_text("module")
        (tmp_path / "app.ts").write_text("const x = 1")

        ctx = load_codebase(tmp_path, languages=["typescript"])

        assert ctx.file_count == 1


class TestRLMRunner:
    """Tests for RLMRunner class."""

    def test_runner_init(self):
        """RLMRunner should initialize with config."""
        cfg = RLMConfig(workspace_path="/repo", languages=["python"])
        runner = RLMRunner(cfg)
        assert runner.config == cfg
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/rlm/test_runner.py -v`
Expected: FAIL with "cannot import name 'RLMConfig'"

**Step 3: Write minimal implementation**

```python
# daydream/rlm/runner.py
"""RLM runner orchestration.

This module provides the main orchestration for RLM code reviews,
coordinating the REPL, container, and LLM interactions.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from daydream.rlm.environment import FileInfo, RepoContext, Service


# File extensions by language
LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "python": [".py"],
    "typescript": [".ts", ".tsx"],
    "javascript": [".js", ".jsx"],
    "go": [".go"],
}

# Directories to exclude from codebase loading
EXCLUDED_DIRS = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".next",
    ".nuxt",
}


@dataclass
class RLMConfig:
    """Configuration for RLM code review.

    Attributes:
        workspace_path: Path to the repository to review.
        languages: List of languages to include in review.
        model: Model to use for root LLM (orchestration).
        sub_model: Model to use for sub-LLM calls (analysis).
        use_container: Whether to use devcontainer sandboxing.
        pr_number: PR number for focused review (optional).
        timeout: Maximum time for review in seconds.
    """

    workspace_path: str
    languages: list[str]
    model: str = "opus"
    sub_model: str = "haiku"
    use_container: bool = True
    pr_number: int | None = None
    timeout: float = 600.0


def _estimate_tokens(content: str) -> int:
    """Estimate token count for content.

    Uses simple heuristic of ~4 characters per token.

    Args:
        content: Text content to estimate.

    Returns:
        Estimated token count.
    """
    return len(content) // 4


def _should_exclude_dir(name: str) -> bool:
    """Check if directory should be excluded."""
    return name in EXCLUDED_DIRS or name.startswith(".")


def load_codebase(
    workspace_path: Path,
    languages: list[str],
    changed_files: list[str] | None = None,
) -> RepoContext:
    """Load codebase files into RepoContext.

    Args:
        workspace_path: Path to repository root.
        languages: List of languages to include.
        changed_files: Optional list of changed files for PR mode.

    Returns:
        RepoContext with loaded files and metadata.
    """
    # Build set of extensions to include
    extensions: set[str] = set()
    for lang in languages:
        extensions.update(LANGUAGE_EXTENSIONS.get(lang, []))

    files: dict[str, str] = {}
    file_sizes: dict[str, int] = {}
    structure: dict[str, FileInfo] = {}

    for root, dirs, filenames in os.walk(workspace_path):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not _should_exclude_dir(d)]

        for filename in filenames:
            # Check extension
            ext = os.path.splitext(filename)[1]
            if ext not in extensions:
                continue

            filepath = Path(root) / filename
            rel_path = str(filepath.relative_to(workspace_path))

            try:
                content = filepath.read_text(encoding="utf-8")
                files[rel_path] = content
                tokens = _estimate_tokens(content)
                file_sizes[rel_path] = tokens

                # Basic structure extraction (placeholder for tree-sitter)
                structure[rel_path] = FileInfo(
                    language=_detect_language(ext),
                    functions=[],  # TODO: tree-sitter parsing
                    classes=[],
                    imports=[],
                    exports=[],
                )
            except (UnicodeDecodeError, PermissionError):
                # Skip binary or unreadable files
                continue

    # Calculate totals
    total_tokens = sum(file_sizes.values())
    file_count = len(files)

    # Get largest files
    sorted_files = sorted(file_sizes.items(), key=lambda x: x[1], reverse=True)
    largest_files = sorted_files[:10]

    return RepoContext(
        files=files,
        structure=structure,
        services={},  # TODO: service detection
        file_sizes=file_sizes,
        total_tokens=total_tokens,
        file_count=file_count,
        largest_files=largest_files,
        languages=languages,
        changed_files=changed_files,
    )


def _detect_language(ext: str) -> str:
    """Detect language from file extension."""
    ext_to_lang = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".go": "go",
    }
    return ext_to_lang.get(ext, "unknown")


class RLMRunner:
    """Orchestrates RLM code review execution.

    Manages the iterative loop of:
    1. Sending code to execute
    2. Processing output and handling callbacks
    3. Continuing until FINAL is called or timeout
    """

    def __init__(self, config: RLMConfig):
        """Initialize RLM runner.

        Args:
            config: RLM configuration.
        """
        self.config = config
        self._context: RepoContext | None = None

    async def run(self) -> str:
        """Execute the RLM code review.

        Returns:
            Final review report.
        """
        # Load codebase
        workspace = Path(self.config.workspace_path)
        self._context = load_codebase(
            workspace,
            self.config.languages,
        )

        # TODO: Implement full orchestration loop
        # This will be expanded in subsequent tasks
        return "# Code Review\n\nReview not yet implemented."
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/rlm/test_runner.py -v`
Expected: PASS (6 tests)

**Step 5: Update exports and commit**

Update `daydream/rlm/__init__.py`:

```python
from daydream.rlm.runner import RLMConfig, RLMRunner, load_codebase
```

Add to `__all__`:
```python
    "RLMConfig",
    "RLMRunner",
    "load_codebase",
```

```bash
git add daydream/rlm/runner.py daydream/rlm/__init__.py tests/rlm/test_runner.py
git commit -m "feat(rlm): add RLM runner orchestration scaffold"
```

---

### Task 9: Add --rlm flag to CLI

**Files:**
- Modify: `daydream/cli.py`
- Modify: `daydream/runner.py`

**Step 1: Write the failing test**

Add to `tests/test_integration.py` or create `tests/test_cli.py`:

```python
# tests/test_cli.py
"""Tests for CLI argument parsing."""

import sys
from unittest.mock import patch

from daydream.cli import _parse_args
from daydream.runner import RunConfig


class TestParseArgsRLM:
    """Tests for --rlm flag parsing."""

    def test_rlm_flag_sets_mode(self):
        """--rlm flag should enable RLM mode."""
        with patch.object(sys, "argv", ["daydream", "/repo", "--rlm", "--python"]):
            config = _parse_args()
            assert config.rlm_mode is True

    def test_rlm_without_skill_allowed(self):
        """--rlm can be used with language flags."""
        with patch.object(sys, "argv", ["daydream", "/repo", "--rlm", "--python"]):
            config = _parse_args()
            assert config.rlm_mode is True
            assert config.skill == "python"

    def test_default_no_rlm(self):
        """RLM mode should be off by default."""
        with patch.object(sys, "argv", ["daydream", "/repo", "--python"]):
            config = _parse_args()
            assert config.rlm_mode is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with "AttributeError: 'RunConfig' object has no attribute 'rlm_mode'"

**Step 3: Write minimal implementation**

Update `daydream/runner.py` - add `rlm_mode` to `RunConfig`:

```python
@dataclass
class RunConfig:
    """Configuration for a daydream run.

    Attributes:
        target: Target directory path for the review. If None, prompts user.
        skill: Review skill to use ("python" or "frontend"). If None, prompts user.
        model: Claude model to use ("opus", "sonnet", or "haiku"). Default is "opus".
        debug: Enable debug logging to a timestamped file in the target directory.
        cleanup: Remove review output file after completion. If None, prompts user.
        quiet: Suppress verbose output from the agent.
        review_only: Run review phase only without applying fixes.
        start_at: Phase to start at ("review", "parse", "fix", or "test").
        rlm_mode: Use RLM mode for large codebase review.
    """

    target: str | None = None
    skill: str | None = None
    model: str = "opus"
    debug: bool = False
    cleanup: bool | None = None
    quiet: bool = True
    review_only: bool = False
    start_at: str = "review"
    rlm_mode: bool = False
```

Update `daydream/cli.py` - add `--rlm` argument:

```python
    parser.add_argument(
        "--rlm",
        action="store_true",
        default=False,
        help="Use RLM mode for large codebase review (1M+ tokens)",
    )
```

And in `_parse_args()` return:

```python
    return RunConfig(
        target=args.target,
        skill=args.skill,
        model=args.model,
        debug=args.debug,
        cleanup=args.cleanup,
        quiet=True,
        review_only=args.review_only,
        start_at=args.start_at,
        rlm_mode=args.rlm,
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add daydream/cli.py daydream/runner.py tests/test_cli.py
git commit -m "feat(rlm): add --rlm flag to CLI"
```

---

### Task 10: Integrate RLM mode into runner

**Files:**
- Modify: `daydream/runner.py`

**Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
import pytest


class TestRLMIntegration:
    """Tests for RLM integration in runner."""

    @pytest.mark.asyncio
    async def test_rlm_mode_prints_not_implemented(self, tmp_path, capsys):
        """RLM mode should indicate it's not yet complete."""
        from daydream.runner import run, RunConfig

        # Create a minimal Python file
        (tmp_path / "main.py").write_text("x = 1")

        config = RunConfig(
            target=str(tmp_path),
            skill="python",
            rlm_mode=True,
            review_only=True,
        )

        exit_code = await run(config)

        # Should complete without error for now
        # (Full implementation comes later)
        captured = capsys.readouterr()
        assert "RLM" in captured.out or exit_code == 0
```

**Step 2: Run test to verify current behavior**

Run: `pytest tests/test_cli.py::TestRLMIntegration -v`

**Step 3: Add RLM mode branch to runner**

Update `daydream/runner.py`:

```python
async def run(config: RunConfig | None = None) -> int:
    """Execute the review and fix loop.

    Args:
        config: Optional configuration. If provided, values skip prompts.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    if config is None:
        config = RunConfig()

    print_phase_hero(console, "DAYDREAM", phase_subtitle("DAYDREAM"))

    # Get target directory (from config or prompt)
    if config.target is not None:
        target_dir = Path(config.target).resolve()
    else:
        target_input = prompt_user(console, "Enter target directory", ".")
        target_dir = Path(target_input).resolve()

    if not target_dir.is_dir():
        print_error(console, "Invalid Path", f"'{target_dir}' is not a valid directory")
        return 1

    # RLM mode branch
    if config.rlm_mode:
        return await _run_rlm_mode(config, target_dir)

    # ... rest of existing code ...
```

Add the RLM mode function:

```python
async def _run_rlm_mode(config: RunConfig, target_dir: Path) -> int:
    """Execute RLM mode review.

    Args:
        config: Run configuration.
        target_dir: Target directory path.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from daydream.rlm import RLMConfig, RLMRunner, load_codebase
    from daydream.rlm.errors import RLMError

    # Map skill to languages
    skill_to_languages = {
        "python": ["python"],
        "frontend": ["typescript", "javascript"],
    }
    languages = skill_to_languages.get(config.skill or "python", ["python"])

    print_info(console, f"RLM mode: reviewing {target_dir}")
    print_info(console, f"Languages: {', '.join(languages)}")
    print_info(console, f"Model: {config.model}")
    console.print()

    # Load codebase to show stats
    ctx = load_codebase(target_dir, languages)
    print_info(console, f"Files: {ctx.file_count:,}")
    print_info(console, f"Estimated tokens: {ctx.total_tokens:,}")
    console.print()

    if ctx.file_count == 0:
        print_error(console, "No Files", "No matching files found in target directory")
        return 1

    # Show largest files
    if ctx.largest_files:
        print_dim(console, "Largest files:")
        for path, tokens in ctx.largest_files[:5]:
            print_dim(console, f"  {path}: {tokens:,} tokens")
        console.print()

    # TODO: Full RLM implementation
    print_info(console, "[RLM] Full orchestration not yet implemented")
    print_info(console, "[RLM] Codebase loaded successfully")

    return 0
```

Add import at top of `runner.py`:

```python
from daydream.ui import (
    # ... existing imports ...
    print_dim,
)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add daydream/runner.py
git commit -m "feat(rlm): integrate RLM mode branch into runner"
```

---

## Phase 4: Full Orchestration Loop

### Task 11: Implement LLM interaction loop

**Files:**
- Modify: `daydream/rlm/runner.py`
- Create: `tests/rlm/test_runner_integration.py`

This task implements the full orchestration loop that:
1. Generates system prompt with codebase metadata
2. Sends prompts to root LLM
3. Executes returned code in REPL
4. Handles llm_query callbacks
5. Loops until FINAL is called

**Implementation details in code comments of runner.py.**

---

### Task 12: Add graceful fallback

**Files:**
- Modify: `daydream/runner.py`

Implement the graceful degradation pattern from the design doc:

```python
async def run_rlm_review_with_fallback(cwd: Path, languages: list[str]) -> str:
    try:
        return await run_rlm_review(cwd, languages)
    except (REPLCrashError, HeartbeatFailedError, ContainerError) as e:
        console.print(f"[yellow]RLM mode failed: {e}[/yellow]")
        console.print("[yellow]Falling back to standard skill-based review...[/yellow]")
        return await run_standard_review(cwd, languages)
```

---

## Summary

**Total Tasks**: 12 tasks across 4 phases

**Phase 1 (Core Infrastructure)**: Tasks 1-5
- Module structure and errors
- JSON-RPC protocol
- Environment data structures
- REPL namespace builder
- Configuration constants

**Phase 2 (Container Integration)**: Tasks 6-7
- Devcontainer management
- REPL process manager

**Phase 3 (Orchestration)**: Tasks 8-10
- RLM runner scaffold
- CLI --rlm flag
- Runner integration

**Phase 4 (Full Loop)**: Tasks 11-12
- LLM interaction loop
- Graceful fallback

**Testing approach**: Each task follows TDD with unit tests first. Integration tests mock at the LLM boundary, allowing real container/REPL execution.
