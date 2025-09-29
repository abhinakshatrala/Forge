# MCP Forge Project

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
