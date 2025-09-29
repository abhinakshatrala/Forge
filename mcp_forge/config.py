"""
Configuration management for MCP Forge
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class TransportType(str, Enum):
    STDIO = "stdio"
    UNIX_SOCKET = "unix_socket"
    HTTPS = "https"


class AuthMode(str, Enum):
    API_KEY = "api_key"
    MTLS = "mtls"
    API_KEY_OR_MTLS = "api_key_or_mtls"


class LLMProvider(BaseModel):
    type: str
    endpoint: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class LLMRoute(BaseModel):
    match: Dict[str, Any]
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    structured_output: Optional[bool] = None


class LLMRouter(BaseModel):
    policy: str = "balanced"
    default_model: str = "local-small"
    routes: List[LLMRoute] = []
    fallbacks: List[str] = []
    retry_on_nonconforming: bool = True


class Profile(BaseModel):
    artifact_schema_id: str
    artifact_output_dir: str
    readme_output_dir: str
    readme_template: str
    auto_transform_from: Optional[str] = None
    llm_router: LLMRouter
    ide_targets: Optional[Dict[str, Any]] = None


class TransportConfig(BaseModel):
    stdio: Dict[str, bool] = {"enabled": True}
    unix_socket: Dict[str, Any] = {"enabled": True, "path": "/run/mcp.sock", "permissions": "0600"}
    https: Dict[str, Any] = {
        "enabled": False,
        "bind": "127.0.0.1:8443",
        "tls": {
            "min_version": "TLS1.2",
            "prefer": "TLS1.3",
            "local_ca_path": "/etc/mcp/ca.pem",
            "require_client_certificate": True
        },
        "sse": {"enabled": True, "path": "/events", "idle_keepalive_ms": 15000}
    }


class AuthConfig(BaseModel):
    mode: AuthMode = AuthMode.API_KEY_OR_MTLS
    api_key_header: str = "X-API-Key"
    key_rotation_days: int = 30
    hash_at_rest: bool = True
    reject_query_param_credentials: bool = True
    disable_unused_keys: bool = True
    key_inactivity_disable_days: int = 30


class SchemaRegistryConfig(BaseModel):
    path: str = "/var/lib/mcp/schemas"
    versioning: str = "semver"
    dialect: str = "json-schema-2020-12"
    strict_validation: bool = True
    deterministic_errors: bool = True
    auto_repair_prompts: bool = True
    compatibility_mode: str = "backward"
    diffs: Dict[str, Any] = {
        "enabled": True,
        "strategy": "semantic",
        "store_path": "/var/lib/mcp/schemas/diffs"
    }


class StorageConfig(BaseModel):
    driver: str = "filesystem"
    root: str = "/var/lib/mcp/data"
    version_history: bool = True
    diffs: bool = True


class BackupConfig(BaseModel):
    enabled: bool = True
    schedule_cron: str = "0 2 * * *"
    target: str = "/mnt/removable/mcp-backup"
    offline_restore_drills: bool = True
    rpo_minutes: int = 60
    rto_minutes: int = 30


class ObservabilityConfig(BaseModel):
    dashboard: Dict[str, Any] = {"enabled": True, "bind": "127.0.0.1:8787"}
    metrics: Dict[str, int] = {"latency_p50_target_ms": 50, "latency_p99_target_ms": 250}
    error_taxonomy: List[str] = ["validation", "auth", "transport", "tool", "workflow"]


class MCPConfig(BaseModel):
    """Main configuration class for MCP Forge"""
    
    # Core server settings
    protocol: str = "jsonrpc2"
    transports: TransportConfig = Field(default_factory=TransportConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    
    # Schema and validation
    schema_registry: SchemaRegistryConfig = Field(default_factory=SchemaRegistryConfig)
    
    # Profiles for different personas
    profiles: Dict[str, Profile] = {}
    
    # LLM providers
    llm_providers: Dict[str, LLMProvider] = {
        "local-small": LLMProvider(type="llama.cpp", endpoint="http://127.0.0.1:9001"),
        "local-medium": LLMProvider(type="vllm", endpoint="http://127.0.0.1:9002"),
        "local-large": LLMProvider(type="vllm", endpoint="http://127.0.0.1:9003"),
        "local-structured": LLMProvider(type="json-mode", endpoint="http://127.0.0.1:9004")
    }
    
    # Storage and persistence
    storage: StorageConfig = Field(default_factory=StorageConfig)
    backups: BackupConfig = Field(default_factory=BackupConfig)
    
    # Observability
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    # Paths
    paths: Dict[str, str] = {
        "schemas": "/var/lib/mcp/schemas",
        "artifacts": "/var/lib/mcp/handoffs",
        "data_root": "/var/lib/mcp/data",
        "audit_log": "/var/log/mcp/audit.log",
        "index_root": "/var/lib/mcp/index",
        "backup_target": "/mnt/removable/mcp-backup",
        "ide_windsurf": "/var/lib/mcp/ide/windsurf",
        "ide_claude_code": "/var/lib/mcp/ide/claude-code"
    }
    
    # Business user UI
    ui_enabled: bool = True
    ui_bind: str = "127.0.0.1:8788"
    
    # Security settings
    default_deny_egress: bool = True
    max_workers: str = "auto"
    queue_bound: int = 512
    
    @classmethod
    def load_from_env(cls) -> "MCPConfig":
        """Load configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        if os.getenv("MCP_DATA_ROOT"):
            config.storage.root = os.getenv("MCP_DATA_ROOT")
        
        if os.getenv("MCP_SCHEMA_PATH"):
            config.schema_registry.path = os.getenv("MCP_SCHEMA_PATH")
            
        if os.getenv("MCP_UI_BIND"):
            config.ui_bind = os.getenv("MCP_UI_BIND")
            
        return config
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        dirs_to_create = [
            self.storage.root,
            self.schema_registry.path,
            self.paths["artifacts"],
            self.paths["index_root"],
            Path(self.paths["audit_log"]).parent,
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = MCPConfig.load_from_env()
