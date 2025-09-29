"""Unit tests for MCP server module."""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from mcp_forge.server import MCPServer, JSONRPCRequest, JSONRPCResponse, MCPError


class TestJSONRPCRequest:
    """Test cases for JSONRPCRequest class."""

    def test_request_creation(self):
        """Test creating JSON-RPC request."""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="test_method",
            params={"param1": "value1"},
            id="test-id"
        )
        
        assert request.jsonrpc == "2.0"
        assert request.method == "test_method"
        assert request.params == {"param1": "value1"}
        assert request.id == "test-id"

    def test_request_without_params(self):
        """Test creating JSON-RPC request without params."""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="test_method",
            id="test-id"
        )
        
        assert request.params is None

    def test_request_notification(self):
        """Test creating JSON-RPC notification (no id)."""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="notification_method"
        )
        
        assert request.id is None

    def test_request_serialization(self):
        """Test JSON-RPC request serialization."""
        request = JSONRPCRequest(
            jsonrpc="2.0",
            method="test_method",
            params={"key": "value"},
            id=123
        )
        
        data = request.dict()
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "test_method"
        assert data["params"] == {"key": "value"}
        assert data["id"] == 123


class TestJSONRPCResponse:
    """Test cases for JSONRPCResponse class."""

    def test_success_response(self):
        """Test creating successful JSON-RPC response."""
        response = JSONRPCResponse(
            jsonrpc="2.0",
            result={"success": True},
            id="test-id"
        )
        
        assert response.jsonrpc == "2.0"
        assert response.result == {"success": True}
        assert response.error is None
        assert response.id == "test-id"

    def test_error_response(self):
        """Test creating error JSON-RPC response."""
        error = {"code": -32602, "message": "Invalid params"}
        response = JSONRPCResponse(
            jsonrpc="2.0",
            error=error,
            id="test-id"
        )
        
        assert response.jsonrpc == "2.0"
        assert response.result is None
        assert response.error == error
        assert response.id == "test-id"

    def test_response_serialization(self):
        """Test JSON-RPC response serialization."""
        response = JSONRPCResponse(
            jsonrpc="2.0",
            result={"data": "test"},
            id=456
        )
        
        data = response.dict(exclude_none=True)
        assert data["jsonrpc"] == "2.0"
        assert data["result"] == {"data": "test"}
        assert data["id"] == 456
        assert "error" not in data


class TestMCPServer:
    """Test cases for MCPServer class."""

    @pytest.fixture
    def server(self):
        """Create MCPServer instance for testing."""
        return MCPServer()

    @pytest.fixture
    def client(self, server):
        """Create test client for MCPServer."""
        return TestClient(server.app)

    def test_server_initialization(self, server):
        """Test MCP server initialization."""
        assert server.app is not None
        assert len(server.tools) == 0
        assert len(server.resources) == 0

    def test_register_tool(self, server):
        """Test registering MCP tool."""
        def test_tool(params):
            return {"result": "test"}
        
        server.register_tool("test_tool", test_tool, "A test tool")
        
        assert "test_tool" in server.tools
        assert server.tools["test_tool"]["handler"] == test_tool
        assert server.tools["test_tool"]["description"] == "A test tool"

    def test_register_async_tool(self, server):
        """Test registering async MCP tool."""
        async def async_tool(params):
            return {"result": "async_test"}
        
        server.register_tool("async_tool", async_tool, "An async tool")
        
        assert "async_tool" in server.tools
        assert server.tools["async_tool"]["handler"] == async_tool

    def test_register_resource(self, server):
        """Test registering MCP resource."""
        def test_resource():
            return {"data": "resource_data"}
        
        server.register_resource("test_resource", test_resource, "A test resource")
        
        assert "test_resource" in server.resources
        assert server.resources["test_resource"]["handler"] == test_resource
        assert server.resources["test_resource"]["description"] == "A test resource"

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_mcp_capabilities_endpoint(self, client):
        """Test MCP capabilities endpoint."""
        response = client.get("/mcp/capabilities")
        
        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert "tools" in data["capabilities"]
        assert "resources" in data["capabilities"]

    def test_list_tools_endpoint(self, client, server):
        """Test list tools endpoint."""
        # Register a test tool
        server.register_tool("test_tool", lambda x: x, "Test tool")
        
        response = client.post("/mcp/tools/list", json={
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": "test-id"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "test-id"
        assert "result" in data
        assert "tools" in data["result"]
        assert len(data["result"]["tools"]) == 1

    def test_call_tool_endpoint(self, client, server):
        """Test call tool endpoint."""
        # Register a test tool
        def test_tool(params):
            return {"input": params.get("input", ""), "processed": True}
        
        server.register_tool("test_tool", test_tool, "Test tool")
        
        response = client.post("/mcp/tools/call", json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "test_tool",
                "arguments": {"input": "test_data"}
            },
            "id": "call-test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "call-test"
        assert "result" in data
        assert data["result"]["input"] == "test_data"
        assert data["result"]["processed"] is True

    def test_call_nonexistent_tool(self, client):
        """Test calling non-existent tool."""
        response = client.post("/mcp/tools/call", json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "nonexistent_tool",
                "arguments": {}
            },
            "id": "error-test"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "error-test"
        assert "error" in data
        assert data["error"]["code"] == -32602

    def test_list_resources_endpoint(self, client, server):
        """Test list resources endpoint."""
        # Register a test resource
        server.register_resource("test_resource", lambda: {"data": "test"}, "Test resource")
        
        response = client.post("/mcp/resources/list", json={
            "jsonrpc": "2.0",
            "method": "resources/list",
            "id": "resource-list"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "resource-list"
        assert "result" in data
        assert "resources" in data["result"]
        assert len(data["result"]["resources"]) == 1

    def test_read_resource_endpoint(self, client, server):
        """Test read resource endpoint."""
        # Register a test resource
        def test_resource():
            return {"content": "resource_content", "type": "text"}
        
        server.register_resource("test_resource", test_resource, "Test resource")
        
        response = client.post("/mcp/resources/read", json={
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {
                "uri": "test_resource"
            },
            "id": "resource-read"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "resource-read"
        assert "result" in data
        assert data["result"]["content"] == "resource_content"

    def test_read_nonexistent_resource(self, client):
        """Test reading non-existent resource."""
        response = client.post("/mcp/resources/read", json={
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {
                "uri": "nonexistent_resource"
            },
            "id": "resource-error"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "resource-error"
        assert "error" in data
        assert data["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, server):
        """Test async tool execution."""
        async def async_tool(params):
            await asyncio.sleep(0.1)  # Simulate async work
            return {"async": True, "input": params}
        
        server.register_tool("async_tool", async_tool, "Async tool")
        
        result = await server._execute_tool("async_tool", {"test": "data"})
        
        assert result["async"] is True
        assert result["input"]["test"] == "data"

    @pytest.mark.asyncio
    async def test_tool_execution_error(self, server):
        """Test tool execution with error."""
        def failing_tool(params):
            raise ValueError("Tool execution failed")
        
        server.register_tool("failing_tool", failing_tool, "Failing tool")
        
        with pytest.raises(MCPError):
            await server._execute_tool("failing_tool", {})

    @pytest.mark.asyncio
    async def test_resource_execution_error(self, server):
        """Test resource execution with error."""
        def failing_resource():
            raise RuntimeError("Resource access failed")
        
        server.register_resource("failing_resource", failing_resource, "Failing resource")
        
        with pytest.raises(MCPError):
            await server._execute_resource("failing_resource")

    def test_invalid_json_rpc_request(self, client):
        """Test invalid JSON-RPC request."""
        response = client.post("/mcp/tools/call", json={
            "invalid": "request"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == -32600  # Invalid Request

    def test_json_rpc_parse_error(self, client):
        """Test JSON-RPC parse error."""
        response = client.post("/mcp/tools/call", data="invalid json")
        
        assert response.status_code == 422  # FastAPI validation error

    def test_method_not_found(self, client):
        """Test JSON-RPC method not found."""
        response = client.post("/mcp/unknown", json={
            "jsonrpc": "2.0",
            "method": "unknown/method",
            "id": "unknown"
        })
        
        assert response.status_code == 404

    def test_server_metrics(self, server):
        """Test server metrics collection."""
        metrics = server.get_metrics()
        
        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "tools" in metrics
        assert "resources" in metrics

    def test_server_with_authentication(self):
        """Test server with authentication enabled."""
        server = MCPServer(require_auth=True, api_key="test-key")
        client = TestClient(server.app)
        
        # Request without auth should fail
        response = client.get("/health")
        assert response.status_code == 401
        
        # Request with correct auth should succeed
        response = client.get("/health", headers={"Authorization": "Bearer test-key"})
        assert response.status_code == 200

    def test_server_cors_configuration(self):
        """Test server CORS configuration."""
        server = MCPServer(enable_cors=True, cors_origins=["http://localhost:3000"])
        client = TestClient(server.app)
        
        response = client.options("/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET"
        })
        
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers

    def test_server_rate_limiting(self):
        """Test server rate limiting."""
        server = MCPServer(enable_rate_limiting=True, rate_limit="10/minute")
        client = TestClient(server.app)
        
        # First request should succeed
        response = client.get("/health")
        assert response.status_code == 200

    def test_tool_with_schema_validation(self, server):
        """Test tool with parameter schema validation."""
        tool_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        def validated_tool(params):
            return {"greeting": f"Hello {params['name']}, age {params.get('age', 'unknown')}"}
        
        server.register_tool("validated_tool", validated_tool, "Tool with validation", schema=tool_schema)
        
        client = TestClient(server.app)
        
        # Valid request
        response = client.post("/mcp/tools/call", json={
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "validated_tool",
                "arguments": {"name": "Alice", "age": 30}
            },
            "id": "valid"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "Hello Alice" in data["result"]["greeting"]

    def test_resource_with_caching(self, server):
        """Test resource with caching enabled."""
        call_count = 0
        
        def cached_resource():
            nonlocal call_count
            call_count += 1
            return {"data": f"call_{call_count}", "timestamp": call_count}
        
        server.register_resource("cached_resource", cached_resource, "Cached resource", cache_ttl=60)
        
        client = TestClient(server.app)
        
        # First call
        response1 = client.post("/mcp/resources/read", json={
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": "cached_resource"},
            "id": "cache1"
        })
        
        # Second call should return cached result
        response2 = client.post("/mcp/resources/read", json={
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": "cached_resource"},
            "id": "cache2"
        })
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Should be same result due to caching
        data1 = response1.json()["result"]
        data2 = response2.json()["result"]
        assert data1["timestamp"] == data2["timestamp"]

    def test_server_shutdown_gracefully(self, server):
        """Test server graceful shutdown."""
        # This would test shutdown hooks and cleanup
        server.shutdown()
        # Verify cleanup was performed
        assert len(server.tools) == 0
        assert len(server.resources) == 0

    def test_websocket_support(self):
        """Test WebSocket support for real-time communication."""
        server = MCPServer(enable_websocket=True)
        client = TestClient(server.app)
        
        with client.websocket_connect("/ws") as websocket:
            # Send JSON-RPC request over WebSocket
            websocket.send_json({
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": "ws-test"
            })
            
            response = websocket.receive_json()
            assert response["jsonrpc"] == "2.0"
            assert response["id"] == "ws-test"

    def test_server_logging_configuration(self, server):
        """Test server logging configuration."""
        # Verify logging is properly configured
        assert server.logger is not None
        
        # Test log message
        server.logger.info("Test log message")
        
        # In a real test, you would verify the log output


class TestMCPError:
    """Test cases for MCPError exception class."""

    def test_error_creation(self):
        """Test creating MCP error."""
        error = MCPError("Test error", code=-32603)
        assert str(error) == "Test error"
        assert error.code == -32603

    def test_error_default_code(self):
        """Test MCP error with default code."""
        error = MCPError("Default error")
        assert error.code == -32603  # Internal error

    def test_error_with_data(self):
        """Test MCP error with additional data."""
        error = MCPError("Error with data", code=-32602, data={"field": "value"})
        assert error.data == {"field": "value"}

    def test_error_to_dict(self):
        """Test MCP error serialization."""
        error = MCPError("Serialization test", code=-32601, data={"extra": "info"})
        error_dict = error.to_dict()
        
        assert error_dict["code"] == -32601
        assert error_dict["message"] == "Serialization test"
        assert error_dict["data"] == {"extra": "info"}


if __name__ == "__main__":
    pytest.main([__file__])
