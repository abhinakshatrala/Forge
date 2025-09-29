"""
Business User Conversation UI for capturing requirements
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ConversationMessage(BaseModel):
    """Conversation message structure"""
    id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}


class ConversationSession(BaseModel):
    """Conversation session"""
    id: str
    user_id: str
    project_id: Optional[str] = None
    status: str = "active"  # active, completed, archived
    created_at: datetime
    updated_at: datetime
    messages: List[ConversationMessage] = []
    requirements_draft: Dict[str, Any] = {}


class BusinessUserUI:
    """
    Business User Conversation Interface
    Provides web UI for capturing requirements conversationally
    """
    
    def __init__(self, config):
        self.config = config
        self.app = FastAPI(title="MCP Forge - Business User Interface")
        self.sessions: Dict[str, ConversationSession] = {}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Setup templates and static files
        self.templates_dir = Path(__file__).parent / "templates"
        self.static_dir = Path(__file__).parent / "static"
        self.templates_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)
        
        self.templates = Jinja2Templates(directory=str(self.templates_dir))
        
        # Create default templates if they don't exist
        self._create_default_templates()
        
        # Setup routes
        self._setup_routes()
        
    def _create_default_templates(self):
        """Create default HTML templates"""
        
        # Main conversation template
        conversation_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Forge - Requirements Capture</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .chat-container { display: flex; gap: 20px; height: 70vh; }
        .chat-panel { flex: 2; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; flex-direction: column; }
        .requirements-panel { flex: 1; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; }
        .messages { flex: 1; padding: 20px; overflow-y: auto; }
        .message { margin-bottom: 15px; }
        .message.user { text-align: right; }
        .message.assistant { text-align: left; }
        .message-content { display: inline-block; padding: 10px 15px; border-radius: 18px; max-width: 70%; }
        .message.user .message-content { background: #007AFF; color: white; }
        .message.assistant .message-content { background: #E5E5EA; color: black; }
        .input-area { padding: 20px; border-top: 1px solid #eee; }
        .input-group { display: flex; gap: 10px; }
        .input-group input { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 20px; outline: none; }
        .input-group button { padding: 12px 24px; background: #007AFF; color: white; border: none; border-radius: 20px; cursor: pointer; }
        .input-group button:hover { background: #0056CC; }
        .requirements-list { margin-top: 20px; }
        .requirement-item { background: #f8f9fa; padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid #007AFF; }
        .requirement-title { font-weight: bold; margin-bottom: 5px; }
        .requirement-description { color: #666; font-size: 14px; }
        .status-indicator { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
        .status-active { background: #34C759; }
        .status-draft { background: #FF9500; }
        .actions { margin-top: 20px; }
        .btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; margin-right: 10px; }
        .btn-primary { background: #007AFF; color: white; }
        .btn-secondary { background: #8E8E93; color: white; }
        .btn:hover { opacity: 0.8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MCP Forge - Requirements Capture</h1>
            <p>Describe your project requirements in natural language. I'll help structure them into a formal specification.</p>
            <div style="margin-top: 10px;">
                <span class="status-indicator status-active"></span>
                <span>Session: {{ session_id }}</span>
                <span style="margin-left: 20px;">Project: {{ project_id or 'New Project' }}</span>
            </div>
        </div>
        
        <div class="chat-container">
            <div class="chat-panel">
                <div class="messages" id="messages">
                    <!-- Messages will be populated here -->
                </div>
                <div class="input-area">
                    <div class="input-group">
                        <input type="text" id="messageInput" placeholder="Describe your requirements..." onkeypress="handleKeyPress(event)">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
            
            <div class="requirements-panel">
                <h3>Captured Requirements</h3>
                <div class="requirements-list" id="requirements">
                    <p style="color: #666; font-style: italic;">Requirements will appear here as we discuss your project...</p>
                </div>
                
                <div class="actions">
                    <button class="btn btn-primary" onclick="generatePMJson()">Generate PM JSON</button>
                    <button class="btn btn-secondary" onclick="saveSession()">Save Session</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        const sessionId = '{{ session_id }}';
        let ws = null;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://localhost:8788/ws/${sessionId}`);
            
            ws.onopen = function(event) {
                console.log('WebSocket connected');
                loadMessages();
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'message') {
                    addMessage(data.role, data.content);
                } else if (data.type === 'requirements_update') {
                    updateRequirements(data.requirements);
                } else if (data.type === 'validation_result') {
                    showValidationResult(data.result);
                }
            };
            
            ws.onclose = function(event) {
                console.log('WebSocket disconnected');
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            if (message && ws) {
                ws.send(JSON.stringify({
                    type: 'user_message',
                    content: message
                }));
                input.value = '';
            }
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function addMessage(role, content) {
            const messages = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            messageDiv.innerHTML = `<div class="message-content">${content}</div>`;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function updateRequirements(requirements) {
            const requirementsDiv = document.getElementById('requirements');
            if (requirements.length === 0) {
                requirementsDiv.innerHTML = '<p style="color: #666; font-style: italic;">Requirements will appear here as we discuss your project...</p>';
                return;
            }
            
            let html = '';
            requirements.forEach(req => {
                html += `
                    <div class="requirement-item">
                        <div class="requirement-title">${req.title}</div>
                        <div class="requirement-description">${req.description}</div>
                        <div style="margin-top: 8px; font-size: 12px; color: #666;">
                            Priority: ${req.priority} | Complexity: ${req.complexity}
                        </div>
                    </div>
                `;
            });
            requirementsDiv.innerHTML = html;
        }
        
        function loadMessages() {
            if (ws) {
                ws.send(JSON.stringify({
                    type: 'load_session'
                }));
            }
        }
        
        function generatePMJson() {
            if (ws) {
                ws.send(JSON.stringify({
                    type: 'generate_pm_json'
                }));
            }
        }
        
        function saveSession() {
            if (ws) {
                ws.send(JSON.stringify({
                    type: 'save_session'
                }));
                alert('Session saved successfully!');
            }
        }
        
        function showValidationResult(result) {
            if (result.valid) {
                alert('Requirements JSON generated successfully!');
            } else {
                alert('Validation issues found. Auto-repair attempted.');
            }
        }
        
        // Initialize WebSocket connection
        connectWebSocket();
    </script>
</body>
</html>
        """
        
        conversation_template = self.templates_dir / "conversation.html"
        if not conversation_template.exists():
            with open(conversation_template, 'w') as f:
                f.write(conversation_html)
                
        # Index template
        index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Forge - Welcome</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .welcome-container { background: white; padding: 40px; border-radius: 12px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); max-width: 500px; text-align: center; }
        .logo { font-size: 48px; font-weight: bold; color: #667eea; margin-bottom: 20px; }
        .subtitle { color: #666; margin-bottom: 30px; font-size: 18px; }
        .form-group { margin-bottom: 20px; text-align: left; }
        .form-group label { display: block; margin-bottom: 5px; font-weight: 500; }
        .form-group input { width: 100%; padding: 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
        .btn { width: 100%; padding: 15px; background: #667eea; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-bottom: 15px; }
        .btn:hover { background: #5a6fd8; }
        .btn-secondary { background: #8E8E93; }
        .btn-secondary:hover { background: #7d7d7d; }
        .sessions-list { margin-top: 30px; text-align: left; }
        .session-item { background: #f8f9fa; padding: 15px; margin-bottom: 10px; border-radius: 6px; cursor: pointer; }
        .session-item:hover { background: #e9ecef; }
        .session-title { font-weight: bold; }
        .session-meta { color: #666; font-size: 14px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="welcome-container">
        <div class="logo">MCP Forge</div>
        <div class="subtitle">Local-first Model Context Protocol Server</div>
        
        <form method="post" action="/start-session">
            <div class="form-group">
                <label for="project_name">Project Name</label>
                <input type="text" id="project_name" name="project_name" placeholder="Enter your project name" required>
            </div>
            <div class="form-group">
                <label for="user_name">Your Name</label>
                <input type="text" id="user_name" name="user_name" placeholder="Enter your name" required>
            </div>
            <button type="submit" class="btn">Start New Requirements Session</button>
        </form>
        
        <button class="btn btn-secondary" onclick="showSessions()">View Previous Sessions</button>
        
        <div class="sessions-list" id="sessions" style="display: none;">
            <h3>Previous Sessions</h3>
            <!-- Sessions will be loaded here -->
        </div>
    </div>

    <script>
        function showSessions() {
            const sessionsDiv = document.getElementById('sessions');
            if (sessionsDiv.style.display === 'none') {
                loadSessions();
                sessionsDiv.style.display = 'block';
            } else {
                sessionsDiv.style.display = 'none';
            }
        }
        
        function loadSessions() {
            fetch('/api/sessions')
                .then(response => response.json())
                .then(sessions => {
                    const sessionsDiv = document.getElementById('sessions');
                    if (sessions.length === 0) {
                        sessionsDiv.innerHTML = '<h3>Previous Sessions</h3><p>No previous sessions found.</p>';
                        return;
                    }
                    
                    let html = '<h3>Previous Sessions</h3>';
                    sessions.forEach(session => {
                        html += `
                            <div class="session-item" onclick="resumeSession('${session.id}')">
                                <div class="session-title">${session.project_id || 'Untitled Project'}</div>
                                <div class="session-meta">
                                    Created: ${new Date(session.created_at).toLocaleDateString()} | 
                                    Status: ${session.status} | 
                                    Messages: ${session.message_count}
                                </div>
                            </div>
                        `;
                    });
                    sessionsDiv.innerHTML = html;
                });
        }
        
        function resumeSession(sessionId) {
            window.location.href = `/conversation/${sessionId}`;
        }
    </script>
</body>
</html>
        """
        
        index_template = self.templates_dir / "index.html"
        if not index_template.exists():
            with open(index_template, 'w') as f:
                f.write(index_html)
                
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
            
        @self.app.post("/start-session")
        async def start_session(project_name: str = Form(...), user_name: str = Form(...)):
            session_id = str(uuid.uuid4())
            session = ConversationSession(
                id=session_id,
                user_id=user_name,
                project_id=project_name,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.sessions[session_id] = session
            return RedirectResponse(url=f"/conversation/{session_id}", status_code=303)
            
        @self.app.get("/conversation/{session_id}", response_class=HTMLResponse)
        async def conversation(request: Request, session_id: str):
            if session_id not in self.sessions:
                return RedirectResponse(url="/", status_code=404)
                
            session = self.sessions[session_id]
            return self.templates.TemplateResponse("conversation.html", {
                "request": request,
                "session_id": session_id,
                "project_id": session.project_id
            })
            
        @self.app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await self._handle_websocket(websocket, session_id)
            
        @self.app.get("/api/sessions")
        async def list_sessions():
            return [
                {
                    "id": session.id,
                    "project_id": session.project_id,
                    "status": session.status,
                    "created_at": session.created_at.isoformat(),
                    "message_count": len(session.messages)
                }
                for session in self.sessions.values()
            ]
            
    async def _handle_websocket(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection for real-time conversation"""
        await websocket.accept()
        
        if session_id not in self.sessions:
            await websocket.close(code=4004, reason="Session not found")
            return
            
        self.active_connections[session_id] = websocket
        session = self.sessions[session_id]
        
        try:
            while True:
                data = await websocket.receive_json()
                await self._process_websocket_message(session, data, websocket)
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error for session {session_id}: {e}")
        finally:
            if session_id in self.active_connections:
                del self.active_connections[session_id]
                
    async def _process_websocket_message(self, session: ConversationSession, data: Dict[str, Any], websocket: WebSocket):
        """Process incoming WebSocket message"""
        message_type = data.get("type")
        
        if message_type == "user_message":
            await self._handle_user_message(session, data["content"], websocket)
        elif message_type == "load_session":
            await self._load_session_messages(session, websocket)
        elif message_type == "generate_pm_json":
            await self._generate_pm_json(session, websocket)
        elif message_type == "save_session":
            await self._save_session(session)
            
    async def _handle_user_message(self, session: ConversationSession, content: str, websocket: WebSocket):
        """Handle user message and generate assistant response"""
        # Add user message
        user_message = ConversationMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=content,
            timestamp=datetime.now()
        )
        session.messages.append(user_message)
        
        # Send user message to client
        await websocket.send_json({
            "type": "message",
            "role": "user",
            "content": content
        })
        
        # Generate assistant response
        assistant_response = await self._generate_assistant_response(session, content)
        
        assistant_message = ConversationMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=assistant_response,
            timestamp=datetime.now()
        )
        session.messages.append(assistant_message)
        
        # Send assistant response to client
        await websocket.send_json({
            "type": "message",
            "role": "assistant",
            "content": assistant_response
        })
        
        # Update requirements based on conversation
        await self._update_requirements(session, websocket)
        
        session.updated_at = datetime.now()
        
    async def _generate_assistant_response(self, session: ConversationSession, user_input: str) -> str:
        """Generate assistant response using LLM router"""
        try:
            from .llm_router import LLMRouter
            
            router = LLMRouter(self.config)
            
            # Build conversation context
            context = self._build_conversation_context(session)
            
            prompt = f"""You are a product manager assistant helping to capture requirements. 
            
Context: {context}

User input: {user_input}

Please respond helpfully to gather more details about the requirements. Ask clarifying questions when needed.
Focus on understanding:
- What the user wants to build
- Who will use it
- Key features and functionality
- Success criteria
- Constraints and limitations

Keep responses conversational and engaging."""

            result = await router.route_request(
                task="requirements-gathering",
                complexity=3,
                profile="pm",
                prompt=prompt
            )
            
            return result.response or "I understand. Could you tell me more about that?"
            
        except Exception as e:
            logger.error(f"Failed to generate assistant response: {e}")
            return "I'm here to help capture your requirements. Could you tell me more about your project?"
            
    def _build_conversation_context(self, session: ConversationSession) -> str:
        """Build conversation context for LLM"""
        context = f"Project: {session.project_id}\n"
        context += f"User: {session.user_id}\n"
        
        if session.requirements_draft:
            context += f"Current requirements draft: {json.dumps(session.requirements_draft, indent=2)}\n"
            
        # Include recent messages
        recent_messages = session.messages[-6:]  # Last 6 messages
        context += "Recent conversation:\n"
        for msg in recent_messages:
            context += f"{msg.role}: {msg.content}\n"
            
        return context
        
    async def _update_requirements(self, session: ConversationSession, websocket: WebSocket):
        """Extract and update requirements from conversation"""
        try:
            # Simple requirement extraction (in a real implementation, use more sophisticated NLP)
            requirements = []
            
            for message in session.messages:
                if message.role == "user":
                    # Look for requirement indicators
                    content = message.content.lower()
                    if any(keyword in content for keyword in ["need", "want", "should", "must", "require"]):
                        # Extract potential requirement
                        requirement = {
                            "id": f"REQ-{len(requirements) + 1:03d}",
                            "title": message.content[:50] + "..." if len(message.content) > 50 else message.content,
                            "description": message.content,
                            "priority": "medium",
                            "complexity": 5,
                            "source_message_id": message.id
                        }
                        requirements.append(requirement)
                        
            session.requirements_draft = {"requirements": requirements}
            
            # Send requirements update to client
            await websocket.send_json({
                "type": "requirements_update",
                "requirements": requirements
            })
            
        except Exception as e:
            logger.error(f"Failed to update requirements: {e}")
            
    async def _load_session_messages(self, session: ConversationSession, websocket: WebSocket):
        """Load and send existing session messages"""
        for message in session.messages:
            await websocket.send_json({
                "type": "message",
                "role": message.role,
                "content": message.content
            })
            
        # Send current requirements
        if session.requirements_draft.get("requirements"):
            await websocket.send_json({
                "type": "requirements_update",
                "requirements": session.requirements_draft["requirements"]
            })
            
    async def _generate_pm_json(self, session: ConversationSession, websocket: WebSocket):
        """Generate formal PM requirements JSON"""
        try:
            from .schema_registry import SchemaRegistry
            from .artifact_manager import ArtifactManager
            
            # Build formal requirements structure
            requirements_data = {
                "metadata": {
                    "version": "1.0.0",
                    "created_at": datetime.now().isoformat(),
                    "author": session.user_id,
                    "project_id": session.project_id or f"proj-{session.id[:8]}",
                    "provenance": f"conversation-session-{session.id}",
                    "data_classification": "internal"
                },
                "requirements": session.requirements_draft.get("requirements", []),
                "acceptance_criteria": [],
                "constraints": {
                    "budget": {},
                    "technical": [],
                    "regulatory": []
                },
                "stakeholders": [
                    {
                        "name": session.user_id,
                        "role": "Product Owner",
                        "involvement": "owner",
                        "contact": ""
                    }
                ]
            }
            
            # Validate against schema
            registry = SchemaRegistry(self.config)
            validation_result = await registry.validate(requirements_data, "https://forge.dev/schemas/requirements-1.0.0.json")
            
            if validation_result.valid or validation_result.repair_applied:
                # Save artifact
                manager = ArtifactManager(self.config)
                final_data = validation_result.repaired_data if validation_result.repaired_data else requirements_data
                artifact_path = await manager.save_artifact("pm", final_data)
                
                await websocket.send_json({
                    "type": "validation_result",
                    "result": {
                        "valid": True,
                        "artifact_path": str(artifact_path),
                        "repaired": validation_result.repair_applied
                    }
                })
            else:
                await websocket.send_json({
                    "type": "validation_result",
                    "result": {
                        "valid": False,
                        "errors": validation_result.errors
                    }
                })
                
        except Exception as e:
            logger.error(f"Failed to generate PM JSON: {e}")
            await websocket.send_json({
                "type": "validation_result",
                "result": {
                    "valid": False,
                    "error": str(e)
                }
            })
            
    async def _save_session(self, session: ConversationSession):
        """Save session to persistent storage"""
        try:
            session_file = Path(self.config.storage.root) / "sessions" / f"{session.id}.json"
            session_file.parent.mkdir(parents=True, exist_ok=True)
            
            session_data = {
                "id": session.id,
                "user_id": session.user_id,
                "project_id": session.project_id,
                "status": session.status,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "messages": [
                    {
                        "id": msg.id,
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "metadata": msg.metadata
                    }
                    for msg in session.messages
                ],
                "requirements_draft": session.requirements_draft
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            logger.info(f"Saved session: {session.id}")
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            
    def get_app(self) -> FastAPI:
        """Get FastAPI application"""
        return self.app
