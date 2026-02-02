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

    # Validate JSON-RPC version
    if data.get("jsonrpc") != "2.0":
        raise ValueError("Invalid or missing jsonrpc version")

    # Response has "result" or "error", request has "method"
    if "method" in data:
        return JsonRpcRequest(
            method=data["method"],
            id=data["id"],
            params=data.get("params"),
        )
    else:
        # Validate that exactly one of "result" or "error" is present
        if "result" not in data and "error" not in data:
            raise KeyError("result/error")
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
