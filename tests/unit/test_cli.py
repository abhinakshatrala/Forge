"""Unit tests for CLI module."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from mcp_forge.cli import app, init_project, serve_command, ui_command


class TestCLI:
    """Test cases for CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "MCP Forge" in result.stdout
        assert "init" in result.stdout
        assert "serve" in result.stdout
        assert "ui" in result.stdout

    def test_init_command_success(self, runner, temp_dir):
        """Test successful project initialization."""
        with patch('mcp_forge.cli.Path.cwd', return_value=temp_dir):
            result = runner.invoke(app, ["init", str(temp_dir)])
            
            assert result.exit_code == 0
            assert "initialized successfully" in result.stdout.lower()
            
            # Check that config file was created
            config_file = temp_dir / "config.env"
            assert config_file.exists()

    def test_init_command_existing_project(self, runner, temp_dir):
        """Test initialization of existing project."""
        # Create existing config file
        config_file = temp_dir / "config.env"
        config_file.write_text("MCP_DATA_ROOT=/existing/path")
        
        with patch('mcp_forge.cli.Path.cwd', return_value=temp_dir):
            result = runner.invoke(app, ["init", str(temp_dir)])
            
            # Should warn about existing project
            assert result.exit_code == 0
            assert "already initialized" in result.stdout.lower() or "existing" in result.stdout.lower()

    def test_init_command_force_overwrite(self, runner, temp_dir):
        """Test force initialization over existing project."""
        # Create existing config file
        config_file = temp_dir / "config.env"
        config_file.write_text("MCP_DATA_ROOT=/old/path")
        
        with patch('mcp_forge.cli.Path.cwd', return_value=temp_dir):
            result = runner.invoke(app, ["init", str(temp_dir), "--force"])
            
            assert result.exit_code == 0
            
            # Config should be updated
            new_content = config_file.read_text()
            assert "/old/path" not in new_content

    def test_init_command_custom_config(self, runner, temp_dir):
        """Test initialization with custom configuration."""
        with patch('mcp_forge.cli.Path.cwd', return_value=temp_dir):
            result = runner.invoke(app, [
                "init", str(temp_dir),
                "--data-root", "/custom/data",
                "--ui-bind", "0.0.0.0:9000"
            ])
            
            assert result.exit_code == 0
            
            config_file = temp_dir / "config.env"
            config_content = config_file.read_text()
            assert "/custom/data" in config_content
            assert "0.0.0.0:9000" in config_content

    @patch('mcp_forge.cli.MCPServer')
    @patch('uvicorn.run')
    def test_serve_command_stdio(self, mock_uvicorn, mock_server, runner):
        """Test serve command with stdio transport."""
        result = runner.invoke(app, ["serve", "--transport", "stdio"])
        
        assert result.exit_code == 0
        # Should not call uvicorn for stdio transport
        mock_uvicorn.assert_not_called()

    @patch('mcp_forge.cli.MCPServer')
    @patch('uvicorn.run')
    def test_serve_command_http(self, mock_uvicorn, mock_server, runner):
        """Test serve command with HTTP transport."""
        result = runner.invoke(app, ["serve", "--transport", "http", "--port", "8080"])
        
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()
        
        # Check uvicorn was called with correct parameters
        call_args = mock_uvicorn.call_args
        assert call_args[1]["port"] == 8080

    @patch('mcp_forge.cli.MCPServer')
    @patch('uvicorn.run')
    def test_serve_command_unix_socket(self, mock_uvicorn, mock_server, runner, temp_dir):
        """Test serve command with Unix socket transport."""
        socket_path = temp_dir / "mcp.sock"
        
        result = runner.invoke(app, [
            "serve", 
            "--transport", "unix",
            "--socket-path", str(socket_path)
        ])
        
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once()
        
        # Check uvicorn was called with Unix socket
        call_args = mock_uvicorn.call_args
        assert str(socket_path) in str(call_args)

    @patch('mcp_forge.cli.MCPServer')
    @patch('uvicorn.run')
    def test_serve_command_with_auth(self, mock_uvicorn, mock_server, runner):
        """Test serve command with authentication."""
        result = runner.invoke(app, [
            "serve",
            "--transport", "http",
            "--require-auth",
            "--api-key", "test-key-123"
        ])
        
        assert result.exit_code == 0
        
        # Check server was created with auth
        mock_server.assert_called_once()
        call_args = mock_server.call_args
        assert call_args[1]["require_auth"] is True
        assert call_args[1]["api_key"] == "test-key-123"

    @patch('mcp_forge.cli.MCPServer')
    @patch('uvicorn.run')
    def test_serve_command_with_tls(self, mock_uvicorn, mock_server, runner, temp_dir):
        """Test serve command with TLS."""
        cert_file = temp_dir / "cert.pem"
        key_file = temp_dir / "key.pem"
        cert_file.write_text("fake cert")
        key_file.write_text("fake key")
        
        result = runner.invoke(app, [
            "serve",
            "--transport", "https",
            "--tls-cert", str(cert_file),
            "--tls-key", str(key_file)
        ])
        
        assert result.exit_code == 0
        
        # Check uvicorn was called with TLS settings
        call_args = mock_uvicorn.call_args
        assert call_args[1]["ssl_certfile"] == str(cert_file)
        assert call_args[1]["ssl_keyfile"] == str(key_file)

    @patch('mcp_forge.ui.start_ui_server')
    def test_ui_command_default(self, mock_ui_server, runner):
        """Test UI command with default settings."""
        result = runner.invoke(app, ["ui"])
        
        assert result.exit_code == 0
        mock_ui_server.assert_called_once()
        
        # Check default parameters
        call_args = mock_ui_server.call_args
        assert call_args[1]["host"] == "127.0.0.1"
        assert call_args[1]["port"] == 8788

    @patch('mcp_forge.ui.start_ui_server')
    def test_ui_command_custom_bind(self, mock_ui_server, runner):
        """Test UI command with custom bind address."""
        result = runner.invoke(app, [
            "ui",
            "--host", "0.0.0.0",
            "--port", "9000"
        ])
        
        assert result.exit_code == 0
        
        call_args = mock_ui_server.call_args
        assert call_args[1]["host"] == "0.0.0.0"
        assert call_args[1]["port"] == 9000

    @patch('mcp_forge.ui.start_ui_server')
    def test_ui_command_with_data_root(self, mock_ui_server, runner, temp_dir):
        """Test UI command with custom data root."""
        result = runner.invoke(app, [
            "ui",
            "--data-root", str(temp_dir)
        ])
        
        assert result.exit_code == 0
        
        call_args = mock_ui_server.call_args
        assert call_args[1]["data_root"] == str(temp_dir)

    def test_artifact_create_command(self, runner, temp_dir):
        """Test artifact create command."""
        data_file = temp_dir / "requirements.json"
        data_file.write_text(json.dumps({
            "requirements": [{"id": "REQ-001", "title": "Test requirement"}]
        }))
        
        with patch('mcp_forge.cli.ArtifactManager') as mock_manager:
            mock_instance = mock_manager.return_value
            mock_instance.create_artifact.return_value = "artifact-id-123"
            
            result = runner.invoke(app, [
                "artifact", "create",
                "--profile", "pm",
                "--data-file", str(data_file)
            ])
            
            assert result.exit_code == 0
            assert "artifact-id-123" in result.stdout
            mock_instance.create_artifact.assert_called_once()

    def test_artifact_list_command(self, runner):
        """Test artifact list command."""
        with patch('mcp_forge.cli.ArtifactManager') as mock_manager:
            mock_instance = mock_manager.return_value
            mock_instance.list_artifacts.return_value = [
                {"id": "pm-1.0.0", "profile": "pm", "version": "1.0.0"},
                {"id": "tpm-1.0.0", "profile": "tpm", "version": "1.0.0"}
            ]
            
            result = runner.invoke(app, ["artifact", "list"])
            
            assert result.exit_code == 0
            assert "pm-1.0.0" in result.stdout
            assert "tpm-1.0.0" in result.stdout

    def test_artifact_transform_command(self, runner):
        """Test artifact transform command."""
        with patch('mcp_forge.cli.ArtifactManager') as mock_manager:
            mock_instance = mock_manager.return_value
            mock_instance.transform_artifact.return_value = "tpm-1.0.0"
            
            result = runner.invoke(app, [
                "artifact", "transform",
                "--from-profile", "pm",
                "--to-profile", "tpm",
                "--from-version", "1.0.0"
            ])
            
            assert result.exit_code == 0
            assert "tpm-1.0.0" in result.stdout
            mock_instance.transform_artifact.assert_called_once()

    def test_workflow_create_command(self, runner, temp_dir):
        """Test workflow create command."""
        workflow_file = temp_dir / "workflow.json"
        workflow_file.write_text(json.dumps([
            {
                "id": "step1",
                "name": "First Step",
                "handler": "handler1"
            }
        ]))
        
        with patch('mcp_forge.cli.WorkflowEngine') as mock_engine:
            mock_instance = mock_engine.return_value
            mock_instance.create_workflow.return_value = "workflow-123"
            
            result = runner.invoke(app, [
                "workflow", "create",
                "--name", "Test Workflow",
                "--steps-file", str(workflow_file)
            ])
            
            assert result.exit_code == 0
            assert "workflow-123" in result.stdout

    def test_workflow_execute_command(self, runner):
        """Test workflow execute command."""
        with patch('mcp_forge.cli.WorkflowEngine') as mock_engine:
            mock_instance = mock_engine.return_value
            mock_instance.execute_workflow = Mock()
            mock_instance.execute_workflow.return_value = {"success": True}
            
            result = runner.invoke(app, [
                "workflow", "execute", "workflow-123"
            ])
            
            assert result.exit_code == 0
            mock_instance.execute_workflow.assert_called_once_with(
                "workflow-123", resume_from_checkpoint=False
            )

    def test_workflow_status_command(self, runner):
        """Test workflow status command."""
        with patch('mcp_forge.cli.WorkflowEngine') as mock_engine:
            mock_instance = mock_engine.return_value
            mock_instance.get_workflow_status.return_value = {
                "id": "workflow-123",
                "status": "running",
                "progress": 0.5
            }
            
            result = runner.invoke(app, [
                "workflow", "status", "workflow-123"
            ])
            
            assert result.exit_code == 0
            assert "running" in result.stdout
            assert "50%" in result.stdout or "0.5" in result.stdout

    def test_validate_command(self, runner, temp_dir):
        """Test validate command."""
        data_file = temp_dir / "data.json"
        data_file.write_text(json.dumps({"name": "test", "age": 25}))
        
        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }))
        
        with patch('mcp_forge.cli.SchemaRegistry') as mock_registry:
            mock_instance = mock_registry.return_value
            mock_instance.validate_data.return_value = Mock(is_valid=True, errors=[])
            
            result = runner.invoke(app, [
                "validate",
                str(data_file),
                str(schema_file)
            ])
            
            assert result.exit_code == 0
            assert "valid" in result.stdout.lower()

    def test_validate_command_with_auto_repair(self, runner, temp_dir):
        """Test validate command with auto-repair."""
        data_file = temp_dir / "invalid_data.json"
        data_file.write_text(json.dumps({"age": "twenty"}))  # Invalid type
        
        schema_file = temp_dir / "schema.json"
        schema_file.write_text(json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }))
        
        with patch('mcp_forge.cli.SchemaRegistry') as mock_registry:
            mock_instance = mock_registry.return_value
            mock_instance.validate_data.return_value = Mock(is_valid=False, errors=["Invalid type"])
            mock_instance.auto_repair_data.return_value = {"name": "default", "age": 0}
            
            result = runner.invoke(app, [
                "validate",
                str(data_file),
                str(schema_file),
                "--auto-repair"
            ])
            
            assert result.exit_code == 0
            mock_instance.auto_repair_data.assert_called_once()

    def test_health_command(self, runner):
        """Test health command."""
        with patch('mcp_forge.cli.MCPServer') as mock_server:
            mock_instance = mock_server.return_value
            mock_instance.health_check.return_value = {
                "status": "healthy",
                "components": {"server": "ok", "database": "ok"}
            }
            
            result = runner.invoke(app, ["health"])
            
            assert result.exit_code == 0
            assert "healthy" in result.stdout.lower()

    def test_health_command_llm_only(self, runner):
        """Test health command with LLM-only check."""
        with patch('mcp_forge.cli.LLMRouter') as mock_router:
            mock_instance = mock_router.return_value
            mock_instance.health_check.return_value = {
                "overall_status": "healthy",
                "providers": {"small": {"is_healthy": True}}
            }
            
            result = runner.invoke(app, ["health", "--llm-only"])
            
            assert result.exit_code == 0
            assert "healthy" in result.stdout.lower()

    def test_schema_compatibility_command(self, runner, temp_dir):
        """Test schema compatibility command."""
        old_schema = temp_dir / "old_schema.json"
        old_schema.write_text(json.dumps({
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }))
        
        new_schema = temp_dir / "new_schema.json"
        new_schema.write_text(json.dumps({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }))
        
        with patch('mcp_forge.cli.SchemaRegistry') as mock_registry:
            mock_instance = mock_registry.return_value
            mock_instance.check_compatibility.return_value = {
                "compatible": True,
                "breaking_changes": []
            }
            
            result = runner.invoke(app, [
                "schema", "compatibility",
                "--old-schema", str(old_schema),
                "--new-schema", str(new_schema)
            ])
            
            assert result.exit_code == 0
            assert "compatible" in result.stdout.lower()

    def test_cli_error_handling(self, runner):
        """Test CLI error handling."""
        # Test with invalid command
        result = runner.invoke(app, ["invalid-command"])
        assert result.exit_code != 0

    def test_cli_verbose_mode(self, runner):
        """Test CLI verbose mode."""
        with patch('mcp_forge.cli.setup_logging') as mock_logging:
            result = runner.invoke(app, ["--verbose", "health"])
            
            mock_logging.assert_called_once()
            # Should set up debug logging in verbose mode

    def test_cli_config_file_loading(self, runner, temp_dir):
        """Test CLI configuration file loading."""
        config_file = temp_dir / "config.env"
        config_file.write_text("""
MCP_DATA_ROOT=/custom/data
MCP_UI_BIND=127.0.0.1:9000
MCP_LLM_SMALL=http://localhost:9001
        """)
        
        with patch('mcp_forge.cli.load_config') as mock_load_config:
            mock_load_config.return_value = {
                "MCP_DATA_ROOT": "/custom/data",
                "MCP_UI_BIND": "127.0.0.1:9000"
            }
            
            result = runner.invoke(app, [
                "--config", str(config_file),
                "health"
            ])
            
            mock_load_config.assert_called_once_with(str(config_file))

    def test_cli_environment_variable_override(self, runner):
        """Test CLI environment variable override."""
        with patch.dict('os.environ', {'MCP_DATA_ROOT': '/env/data'}):
            with patch('mcp_forge.cli.get_config_value') as mock_get_config:
                mock_get_config.return_value = '/env/data'
                
                result = runner.invoke(app, ["health"])
                
                # Environment variable should be used
                mock_get_config.assert_called()


class TestCLIUtilities:
    """Test CLI utility functions."""

    def test_init_project_function(self, temp_dir):
        """Test init_project utility function."""
        result = init_project(
            project_path=str(temp_dir),
            data_root="/custom/data",
            ui_bind="0.0.0.0:8000",
            force=False
        )
        
        assert result is True
        
        config_file = temp_dir / "config.env"
        assert config_file.exists()
        
        config_content = config_file.read_text()
        assert "/custom/data" in config_content
        assert "0.0.0.0:8000" in config_content

    def test_init_project_existing_no_force(self, temp_dir):
        """Test init_project with existing config and no force."""
        config_file = temp_dir / "config.env"
        config_file.write_text("MCP_DATA_ROOT=/existing")
        
        result = init_project(
            project_path=str(temp_dir),
            force=False
        )
        
        # Should not overwrite existing config
        assert result is False
        assert "/existing" in config_file.read_text()

    def test_init_project_existing_with_force(self, temp_dir):
        """Test init_project with existing config and force flag."""
        config_file = temp_dir / "config.env"
        config_file.write_text("MCP_DATA_ROOT=/existing")
        
        result = init_project(
            project_path=str(temp_dir),
            data_root="/new/data",
            force=True
        )
        
        # Should overwrite existing config
        assert result is True
        config_content = config_file.read_text()
        assert "/new/data" in config_content
        assert "/existing" not in config_content

    @patch('uvicorn.run')
    def test_serve_command_function(self, mock_uvicorn):
        """Test serve_command utility function."""
        serve_command(
            transport="http",
            host="0.0.0.0",
            port=8080,
            require_auth=True,
            api_key="test-key"
        )
        
        mock_uvicorn.assert_called_once()
        call_args = mock_uvicorn.call_args
        assert call_args[1]["host"] == "0.0.0.0"
        assert call_args[1]["port"] == 8080

    @patch('mcp_forge.ui.start_ui_server')
    def test_ui_command_function(self, mock_ui_server):
        """Test ui_command utility function."""
        ui_command(
            host="127.0.0.1",
            port=8788,
            data_root="/custom/data"
        )
        
        mock_ui_server.assert_called_once()
        call_args = mock_ui_server.call_args
        assert call_args[1]["host"] == "127.0.0.1"
        assert call_args[1]["port"] == 8788
        assert call_args[1]["data_root"] == "/custom/data"


if __name__ == "__main__":
    pytest.main([__file__])
