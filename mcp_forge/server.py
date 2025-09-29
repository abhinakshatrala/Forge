"""
MCP Server implementation with JSON-RPC protocol support
"""

import asyncio
import json
import logging
import sys
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 request"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    id: Optional[Union[str, int]] = None


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 response"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[str, int]] = None


@dataclass
class JSONRPCError:
    """JSON-RPC 2.0 error"""
    code: int
    message: str
    data: Optional[Any] = None


class MCPCapabilities(BaseModel):
    """MCP server capabilities"""
    tools: Dict[str, Any] = {}
    resources: Dict[str, Any] = {}
    prompts: Dict[str, Any] = {}
    logging: Dict[str, Any] = {}


class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPResource(BaseModel):
    """MCP resource definition"""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None


class MCPPrompt(BaseModel):
    """MCP prompt definition"""
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


class MCPServer:
    """
    Model Context Protocol server implementation
    Supports JSON-RPC 2.0 over stdio, unix socket, and HTTPS
    """
    
    def __init__(self, config):
        self.config = config
        self.capabilities = MCPCapabilities()
        self.tools: Dict[str, Callable] = {}
        self.resources: Dict[str, Callable] = {}
        self.prompts: Dict[str, Callable] = {}
        self.session_id = str(uuid.uuid4())
        self.client_info = {}
        
        # Register built-in tools
        self._register_builtin_tools()
        
    def _register_builtin_tools(self):
        """Register built-in MCP tools"""
        
        # Schema validation tool
        self.register_tool(
            name="validate_schema",
            description="Validate JSON data against a schema",
            input_schema={
                "type": "object",
                "properties": {
                    "data": {"type": "object", "description": "Data to validate"},
                    "schema_id": {"type": "string", "description": "Schema identifier"}
                },
                "required": ["data", "schema_id"]
            },
            handler=self._validate_schema_tool
        )
        
        # Artifact generation tool
        self.register_tool(
            name="generate_artifact",
            description="Generate artifact for a specific profile",
            input_schema={
                "type": "object",
                "properties": {
                    "profile": {"type": "string", "enum": ["pm", "tpm", "dev"]},
                    "data": {"type": "object", "description": "Artifact data"},
                    "version": {"type": "string", "description": "Semantic version"}
                },
                "required": ["profile", "data"]
            },
            handler=self._generate_artifact_tool
        )
        
        # LLM routing tool
        self.register_tool(
            name="route_llm_request",
            description="Route LLM request based on task and complexity",
            input_schema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task type"},
                    "complexity": {"type": "integer", "minimum": 1, "maximum": 10},
                    "profile": {"type": "string", "enum": ["pm", "tpm", "dev"]},
                    "prompt": {"type": "string", "description": "LLM prompt"}
                },
                "required": ["task", "prompt"]
            },
            handler=self._route_llm_request_tool
        )
        
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any], handler: Callable):
        """Register a new tool"""
        tool = MCPTool(
            name=name,
            description=description,
            inputSchema=input_schema
        )
        self.capabilities.tools[name] = tool.dict()
        self.tools[name] = handler
        
    def register_resource(self, uri: str, name: str, description: str, handler: Callable, mime_type: str = None):
        """Register a new resource"""
        resource = MCPResource(
            uri=uri,
            name=name,
            description=description,
            mimeType=mime_type
        )
        self.capabilities.resources[uri] = resource.dict()
        self.resources[uri] = handler
        
    def register_prompt(self, name: str, description: str, handler: Callable, arguments: List[Dict[str, Any]] = None):
        """Register a new prompt"""
        prompt = MCPPrompt(
            name=name,
            description=description,
            arguments=arguments
        )
        self.capabilities.prompts[name] = prompt.dict()
        self.prompts[name] = handler
        
    async def _validate_schema_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in schema validation tool"""
        from .schema_registry import SchemaRegistry
        
        registry = SchemaRegistry(self.config)
        result = await registry.validate(params["data"], params["schema_id"])
        
        return {
            "valid": result.valid,
            "errors": result.errors,
            "repaired_data": result.repaired_data if hasattr(result, 'repaired_data') else None
        }
        
    async def _generate_artifact_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in artifact generation tool"""
        from .artifact_manager import ArtifactManager
        
        manager = ArtifactManager(self.config)
        artifact_path = await manager.save_artifact(
            profile=params["profile"],
            data=params["data"],
            version=params.get("version", "1.0.0")
        )
        
        return {
            "artifact_path": str(artifact_path),
            "profile": params["profile"],
            "version": params.get("version", "1.0.0")
        }
        
    async def _route_llm_request_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Built-in LLM routing tool"""
        from .llm_router import LLMRouter
        
        router = LLMRouter(self.config)
        route_result = await router.route_request(
            task=params["task"],
            complexity=params.get("complexity", 5),
            profile=params.get("profile", "pm"),
            prompt=params["prompt"]
        )
        
        return {
            "model": route_result.model,
            "endpoint": route_result.endpoint,
            "response": route_result.response
        }
        
    async def handle_request(self, request_data: str) -> str:
        """Handle incoming JSON-RPC request"""
        try:
            # Parse JSON-RPC request
            request_dict = json.loads(request_data)
            request = JSONRPCRequest(**request_dict)
            
            # Handle different MCP methods
            if request.method == "initialize":
                result = await self._handle_initialize(request.params or {})
            elif request.method == "tools/list":
                result = await self._handle_tools_list()
            elif request.method == "tools/call":
                result = await self._handle_tools_call(request.params or {})
            elif request.method == "resources/list":
                result = await self._handle_resources_list()
            elif request.method == "resources/read":
                result = await self._handle_resources_read(request.params or {})
            elif request.method == "prompts/list":
                result = await self._handle_prompts_list()
            elif request.method == "prompts/get":
                result = await self._handle_prompts_get(request.params or {})
            elif request.method == "logging/setLevel":
                result = await self._handle_logging_set_level(request.params or {})
            else:
                raise ValueError(f"Unknown method: {request.method}")
                
            response = JSONRPCResponse(result=result, id=request.id)
            
        except json.JSONDecodeError as e:
            error = JSONRPCError(code=-32700, message="Parse error", data=str(e))
            response = JSONRPCResponse(error=asdict(error), id=None)
        except ValidationError as e:
            error = JSONRPCError(code=-32602, message="Invalid params", data=str(e))
            response = JSONRPCResponse(error=asdict(error), id=request.id if 'request' in locals() else None)
        except ValueError as e:
            error = JSONRPCError(code=-32601, message="Method not found", data=str(e))
            response = JSONRPCResponse(error=asdict(error), id=request.id if 'request' in locals() else None)
        except Exception as e:
            logger.error(f"Internal error: {e}\n{traceback.format_exc()}")
            error = JSONRPCError(code=-32603, message="Internal error", data=str(e))
            response = JSONRPCResponse(error=asdict(error), id=request.id if 'request' in locals() else None)
            
        return json.dumps(asdict(response), default=str)
        
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        self.client_info = params.get("clientInfo", {})
        
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": self.capabilities.dict(),
            "serverInfo": {
                "name": "mcp-forge",
                "version": "1.0.0"
            },
            "instructions": "MCP Forge server ready. Use tools for schema validation, artifact generation, and LLM routing."
        }
        
    async def _handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools/list request"""
        return {
            "tools": list(self.capabilities.tools.values())
        }
        
    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get("name")
        tool_params = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise ValueError(f"Tool not found: {tool_name}")
            
        handler = self.tools[tool_name]
        result = await handler(tool_params)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        }
        
    async def _handle_resources_list(self) -> Dict[str, Any]:
        """Handle resources/list request"""
        return {
            "resources": list(self.capabilities.resources.values())
        }
        
    async def _handle_resources_read(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resources/read request"""
        uri = params.get("uri")
        
        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")
            
        handler = self.resources[uri]
        content = await handler(params)
        
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "application/json",
                    "text": json.dumps(content, indent=2)
                }
            ]
        }
        
    async def _handle_prompts_list(self) -> Dict[str, Any]:
        """Handle prompts/list request"""
        return {
            "prompts": list(self.capabilities.prompts.values())
        }
        
    async def _handle_prompts_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts/get request"""
        prompt_name = params.get("name")
        prompt_params = params.get("arguments", {})
        
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_name}")
            
        handler = self.prompts[prompt_name]
        result = await handler(prompt_params)
        
        return {
            "description": result.get("description", ""),
            "messages": result.get("messages", [])
        }
        
    async def _handle_logging_set_level(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle logging/setLevel request"""
        level = params.get("level", "INFO")
        logging.getLogger().setLevel(getattr(logging, level.upper()))
        
        return {"success": True}
        
    async def run_stdio(self):
        """Run server over stdio transport"""
        logger.info("Starting MCP server on stdio")
        
        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                    
                # Handle request
                response = await self.handle_request(line.strip())
                
                # Write response to stdout
                print(response, flush=True)
                
            except KeyboardInterrupt:
                logger.info("Shutting down MCP server")
                break
            except Exception as e:
                logger.error(f"Error in stdio loop: {e}")
                
    async def run_unix_socket(self, socket_path: str):
        """Run server over Unix socket transport"""
        import socket
        import os
        
        # Remove existing socket file
        if os.path.exists(socket_path):
            os.unlink(socket_path)
            
        server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_socket.bind(socket_path)
        server_socket.listen(1)
        
        logger.info(f"Starting MCP server on Unix socket: {socket_path}")
        
        try:
            while True:
                client_socket, _ = server_socket.accept()
                asyncio.create_task(self._handle_unix_client(client_socket))
        except KeyboardInterrupt:
            logger.info("Shutting down MCP server")
        finally:
            server_socket.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
                
    async def _handle_unix_client(self, client_socket):
        """Handle Unix socket client connection"""
        try:
            while True:
                data = client_socket.recv(4096)
                if not data:
                    break
                    
                response = await self.handle_request(data.decode('utf-8'))
                client_socket.send(response.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error handling Unix socket client: {e}")
        finally:
            client_socket.close()
            
    async def run_https(self, host: str = "127.0.0.1", port: int = 8443):
        """Run server over HTTPS transport with SSE support"""
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import StreamingResponse
        import uvicorn
        import ssl
        
        app = FastAPI(title="MCP Forge Server", version="1.0.0")
        
        @app.post("/mcp")
        async def handle_mcp_request(request: Request):
            """Handle MCP JSON-RPC requests over HTTPS"""
            try:
                body = await request.body()
                response = await self.handle_request(body.decode('utf-8'))
                return json.loads(response)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        @app.get("/events")
        async def sse_events():
            """Server-Sent Events endpoint for real-time updates"""
            async def event_generator():
                while True:
                    # Send keepalive every 15 seconds
                    yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.now().isoformat()})}\n\n"
                    await asyncio.sleep(15)
                    
            return StreamingResponse(event_generator(), media_type="text/plain")
            
        @app.get("/healthz")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "server": "mcp-forge", "version": "1.0.0"}
            
        # Configure SSL context
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        if self.config.transports.https.get("tls", {}).get("local_ca_path"):
            ssl_context.load_cert_chain(
                self.config.transports.https["tls"]["local_ca_path"]
            )
            
        logger.info(f"Starting MCP server on HTTPS: {host}:{port}")
        
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            ssl_context=ssl_context if ssl_context else None,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
