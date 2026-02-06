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
