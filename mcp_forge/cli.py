"""
Command Line Interface for MCP Forge
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .config import MCPConfig
from .server import MCPServer
from .ui import BusinessUserUI
from .workflow_engine import WorkflowEngine
from .schema_registry import SchemaRegistry
from .artifact_manager import ArtifactManager
from .llm_router import LLMRouter


app = typer.Typer(name="mcp-forge", help="Local-first Model Context Protocol Server")
console = Console()


@app.command()
def serve(
    transport: str = typer.Option("stdio", help="Transport type: stdio, unix, https"),
    host: str = typer.Option("127.0.0.1", help="Host for HTTPS transport"),
    port: int = typer.Option(8443, help="Port for HTTPS transport"),
    socket_path: str = typer.Option("/run/mcp.sock", help="Unix socket path"),
    config_file: Optional[str] = typer.Option(None, help="Configuration file path"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Start the MCP server"""
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = MCPConfig.load_from_env()
    if config_file:
        # In a full implementation, load from config file
        pass
        
    config.ensure_directories()
    
    # Create and start server
    server = MCPServer(config)
    
    rprint(Panel.fit(
        f"[bold blue]MCP Forge Server[/bold blue]\n"
        f"Transport: {transport}\n"
        f"Version: 1.0.0\n"
        f"Protocol: JSON-RPC 2.0",
        title="Starting Server"
    ))
    
    try:
        if transport == "stdio":
            asyncio.run(server.run_stdio())
        elif transport == "unix":
            asyncio.run(server.run_unix_socket(socket_path))
        elif transport == "https":
            asyncio.run(server.run_https(host, port))
        else:
            rprint(f"[red]Unknown transport: {transport}[/red]")
            sys.exit(1)
    except KeyboardInterrupt:
        rprint("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        rprint(f"[red]Server error: {e}[/red]")
        sys.exit(1)


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="UI host"),
    port: int = typer.Option(8788, help="UI port"),
    log_level: str = typer.Option("INFO", help="Log level"),
):
    """Start the business user conversation UI"""
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = MCPConfig.load_from_env()
    config.ensure_directories()
    
    ui_app = BusinessUserUI(config)
    
    rprint(Panel.fit(
        f"[bold green]MCP Forge Business UI[/bold green]\n"
        f"URL: http://{host}:{port}\n"
        f"Version: 1.0.0",
        title="Starting UI Server"
    ))
    
    try:
        uvicorn.run(
            ui_app.get_app(),
            host=host,
            port=port,
            log_level=log_level.lower()
        )
    except KeyboardInterrupt:
        rprint("\n[yellow]UI server stopped by user[/yellow]")


@app.command()
def validate(
    file_path: str = typer.Argument(..., help="Path to JSON file to validate"),
    schema_id: str = typer.Argument(..., help="Schema ID to validate against"),
    auto_repair: bool = typer.Option(True, help="Attempt auto-repair on validation errors"),
):
    """Validate a JSON file against a schema"""
    
    async def run_validation():
        config = MCPConfig.load_from_env()
        registry = SchemaRegistry(config)
        
        # Load file
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            rprint(f"[red]File not found: {file_path}[/red]")
            return False
            
        import json
        with open(file_path_obj, 'r') as f:
            data = json.load(f)
            
        # Validate
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Validating...", total=None)
            result = await registry.validate(data, schema_id)
            progress.remove_task(task)
            
        # Display results
        if result.valid:
            rprint("[green]✓ Validation successful[/green]")
            return True
        else:
            rprint("[red]✗ Validation failed[/red]")
            
            # Show errors
            table = Table(title="Validation Errors")
            table.add_column("Path", style="cyan")
            table.add_column("Error", style="red")
            
            for error in result.errors:
                path = ".".join(str(p) for p in error.get("instancePath", []))
                table.add_row(path or "root", error.get("message", "Unknown error"))
                
            console.print(table)
            
            if result.repair_applied:
                rprint("[yellow]Auto-repair was applied[/yellow]")
                
                # Save repaired data
                repaired_path = file_path_obj.with_suffix('.repaired.json')
                with open(repaired_path, 'w') as f:
                    json.dump(result.repaired_data, f, indent=2)
                rprint(f"[green]Repaired data saved to: {repaired_path}[/green]")
                
            return False
            
    success = asyncio.run(run_validation())
    if not success:
        sys.exit(1)


@app.command()
def workflow(
    action: str = typer.Argument(..., help="Action: create, execute, status, list"),
    workflow_id: Optional[str] = typer.Option(None, help="Workflow ID"),
    name: Optional[str] = typer.Option(None, help="Workflow name for create"),
    description: Optional[str] = typer.Option(None, help="Workflow description for create"),
    steps_file: Optional[str] = typer.Option(None, help="JSON file with workflow steps"),
):
    """Manage workflows"""
    
    async def run_workflow_command():
        config = MCPConfig.load_from_env()
        engine = WorkflowEngine(config)
        
        if action == "create":
            if not name or not steps_file:
                rprint("[red]Name and steps file required for create action[/red]")
                return False
                
            # Load steps from file
            import json
            with open(steps_file, 'r') as f:
                steps = json.load(f)
                
            workflow_id = await engine.create_workflow(
                name=name,
                description=description or f"Workflow: {name}",
                steps=steps
            )
            
            rprint(f"[green]Created workflow: {workflow_id}[/green]")
            
        elif action == "execute":
            if not workflow_id:
                rprint("[red]Workflow ID required for execute action[/red]")
                return False
                
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Executing workflow...", total=None)
                result = await engine.execute_workflow(workflow_id)
                progress.remove_task(task)
                
            rprint(f"[green]Workflow execution completed[/green]")
            rprint(f"Status: {result['status']}")
            rprint(f"Steps completed: {result['completed_steps']}/{result['total_steps']}")
            
            if result.get('error'):
                rprint(f"[red]Error: {result['error']}[/red]")
                
        elif action == "status":
            if not workflow_id:
                rprint("[red]Workflow ID required for status action[/red]")
                return False
                
            status = engine.get_workflow_status(workflow_id)
            if not status:
                rprint(f"[red]Workflow not found: {workflow_id}[/red]")
                return False
                
            # Display status table
            table = Table(title=f"Workflow Status: {status['name']}")
            table.add_column("Step", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Attempts", style="yellow")
            
            for step in status['steps']:
                table.add_row(
                    step['name'],
                    step['status'],
                    str(step['attempt'])
                )
                
            console.print(table)
            
            rprint(f"Overall Status: {status['status']}")
            rprint(f"Progress: {status['progress']['completed']}/{status['progress']['total']}")
            
        elif action == "list":
            workflows = engine.list_workflows()
            
            if not workflows:
                rprint("[yellow]No workflows found[/yellow]")
                return True
                
            table = Table(title="Workflows")
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Status", style="yellow")
            table.add_column("Created", style="blue")
            table.add_column("Steps", style="magenta")
            
            for wf in workflows:
                table.add_row(
                    wf['id'][:8] + "...",
                    wf['name'],
                    wf['status'],
                    wf['created_at'][:10] if wf['created_at'] else "Unknown",
                    str(wf['step_count'])
                )
                
            console.print(table)
            
        else:
            rprint(f"[red]Unknown action: {action}[/red]")
            return False
            
        return True
        
    success = asyncio.run(run_workflow_command())
    if not success:
        sys.exit(1)


@app.command()
def artifact(
    action: str = typer.Argument(..., help="Action: create, list, transform"),
    profile: Optional[str] = typer.Option(None, help="Profile: pm, tpm, dev"),
    data_file: Optional[str] = typer.Option(None, help="JSON data file"),
    version: Optional[str] = typer.Option("1.0.0", help="Artifact version"),
    from_profile: Optional[str] = typer.Option(None, help="Source profile for transform"),
    to_profile: Optional[str] = typer.Option(None, help="Target profile for transform"),
    from_version: Optional[str] = typer.Option(None, help="Source version for transform"),
):
    """Manage artifacts"""
    
    async def run_artifact_command():
        config = MCPConfig.load_from_env()
        manager = ArtifactManager(config)
        
        if action == "create":
            if not profile or not data_file:
                rprint("[red]Profile and data file required for create action[/red]")
                return False
                
            # Load data from file
            import json
            with open(data_file, 'r') as f:
                data = json.load(f)
                
            artifact_path = await manager.save_artifact(profile, data, version)
            rprint(f"[green]Created artifact: {artifact_path}[/green]")
            
        elif action == "list":
            artifacts = await manager.list_artifacts(profile)
            
            if not artifacts:
                rprint("[yellow]No artifacts found[/yellow]")
                return True
                
            table = Table(title="Artifacts")
            table.add_column("Profile", style="cyan")
            table.add_column("Version", style="green")
            table.add_column("Created", style="blue")
            table.add_column("Path", style="yellow")
            
            for artifact in artifacts:
                table.add_row(
                    artifact.profile,
                    artifact.version,
                    artifact.created_at.strftime("%Y-%m-%d %H:%M"),
                    str(Path(artifact.file_path).name)
                )
                
            console.print(table)
            
        elif action == "transform":
            if not from_profile or not to_profile or not from_version:
                rprint("[red]from_profile, to_profile, and from_version required for transform[/red]")
                return False
                
            artifact_path = await manager.auto_transform(from_profile, to_profile, from_version)
            rprint(f"[green]Transformed artifact: {artifact_path}[/green]")
            
        else:
            rprint(f"[red]Unknown action: {action}[/red]")
            return False
            
        return True
        
    success = asyncio.run(run_artifact_command())
    if not success:
        sys.exit(1)


@app.command()
def health():
    """Check system health"""
    
    async def run_health_check():
        config = MCPConfig.load_from_env()
        
        rprint("[bold blue]MCP Forge Health Check[/bold blue]\n")
        
        # Check directories
        rprint("[cyan]Checking directories...[/cyan]")
        for name, path in config.paths.items():
            path_obj = Path(path)
            if path_obj.exists():
                rprint(f"  ✓ {name}: {path}")
            else:
                rprint(f"  ✗ {name}: {path} [red](missing)[/red]")
                
        # Check schemas
        rprint("\n[cyan]Checking schemas...[/cyan]")
        registry = SchemaRegistry(config)
        schemas = registry.list_schemas()
        for schema in schemas:
            rprint(f"  ✓ {schema.id} v{schema.version}")
            
        # Check LLM providers
        rprint("\n[cyan]Checking LLM providers...[/cyan]")
        router = LLMRouter(config)
        health_results = await router.health_check()
        
        for model, result in health_results.items():
            if result['status'] == 'healthy':
                rprint(f"  ✓ {model}: {result['endpoint']} ({result['latency_ms']}ms)")
            else:
                rprint(f"  ✗ {model}: {result['endpoint']} [red]({result['error']})[/red]")
                
        rprint("\n[green]Health check completed[/green]")
        
    asyncio.run(run_health_check())


@app.command()
def init(
    directory: str = typer.Option(".", help="Directory to initialize"),
    force: bool = typer.Option(False, help="Force initialization even if directory exists"),
):
    """Initialize a new MCP Forge project"""
    
    project_dir = Path(directory)
    
    if project_dir.exists() and not force and any(project_dir.iterdir()):
        rprint(f"[red]Directory {directory} is not empty. Use --force to override.[/red]")
        sys.exit(1)
        
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create basic project structure
    directories = [
        "schemas",
        "templates",
        "workflows",
        "var/lib/mcp/data",
        "var/lib/mcp/handoffs",
        "var/log/mcp"
    ]
    
    for dir_name in directories:
        (project_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
    # Copy default schemas if not already present
    import shutil
    schema_source = Path(__file__).parent.parent / "schemas"
    schema_dest = project_dir / "schemas"
    
    if schema_source.exists() and schema_source != schema_dest:
        for schema_file in schema_source.glob("*.json"):
            dest_file = schema_dest / schema_file.name
            if not dest_file.exists() or force:
                try:
                    shutil.copy2(schema_file, dest_file)
                except shutil.SameFileError:
                    pass  # Skip if same file
            
    # Create basic config file
    config_content = """# MCP Forge Configuration
# Set environment variables to override defaults

# Data directories
export MCP_DATA_ROOT="./var/lib/mcp/data"
export MCP_SCHEMA_PATH="./schemas"

# UI settings
export MCP_UI_BIND="127.0.0.1:8788"

# LLM endpoints (update with your local endpoints)
export MCP_LLM_SMALL="http://127.0.0.1:9001"
export MCP_LLM_MEDIUM="http://127.0.0.1:9002"
export MCP_LLM_LARGE="http://127.0.0.1:9003"
export MCP_LLM_STRUCTURED="http://127.0.0.1:9004"
"""
    
    with open(project_dir / "config.env", 'w') as f:
        f.write(config_content)
        
    # Create README
    readme_content = """# MCP Forge Project

This is a local-first Model Context Protocol server project.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the business user UI:
   ```bash
   mcp-forge ui
   ```

3. Start the MCP server:
   ```bash
   mcp-forge serve --transport stdio
   ```

## Directory Structure

- `schemas/` - JSON Schema definitions
- `templates/` - README templates for artifacts
- `workflows/` - Workflow definitions
- `var/lib/mcp/` - Data storage
- `var/log/mcp/` - Log files

## Configuration

Edit `config.env` to customize settings, then source it:
```bash
source config.env
```

## Usage

- **Business User UI**: Capture requirements conversationally
- **MCP Server**: Integrate with IDEs and other tools
- **CLI Tools**: Manage artifacts, workflows, and validation

See the documentation for more details.
"""
    
    with open(project_dir / "README.md", 'w') as f:
        f.write(readme_content)
        
    rprint(f"[green]✓ Initialized MCP Forge project in {directory}[/green]")
    rprint("\nNext steps:")
    rprint("1. cd " + str(project_dir))
    rprint("2. source config.env")
    rprint("3. mcp-forge ui")


def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
