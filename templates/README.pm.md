# PM README - {title}

## Overview
This document contains the product requirements generated for project **{project_id}**.

**Generated Artifacts:**
- Requirements JSON: `{artifact_path}`
- Version: {version}
- Created: {created_at}
- Schema: {schema_id}

## Local Setup (Python 3.11, venv, CLI)

### Prerequisites
- Python 3.11 or higher
- Git (for version control)

### Installation Steps

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

2. **Activate Virtual Environment**
   ```bash
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize MCP Forge**
   ```bash
   mcp-forge init .
   source config.env
   ```

## Transports (stdio/unix socket by default)

MCP Forge supports multiple transport mechanisms:

### STDIO Transport (Default)
```bash
mcp-forge serve --transport stdio
```
- Best for IDE integration
- Direct stdin/stdout communication
- No network dependencies

### Unix Socket Transport
```bash
mcp-forge serve --transport unix --socket-path /run/mcp.sock
```
- Local inter-process communication
- Better performance than stdio
- Supports multiple concurrent clients

### HTTPS Transport (Optional)
```bash
mcp-forge serve --transport https --host 127.0.0.1 --port 8443
```
- Web-based access
- Server-Sent Events (SSE) support
- Requires TLS configuration

## Optional HTTPS+SSE with mTLS (local CA)

For secure HTTPS communication with mutual TLS:

### Setup Local CA
```bash
# Generate CA certificate
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 365 -key ca.key -out ca.pem

# Generate server certificate
openssl genrsa -out server.key 4096
openssl req -new -key server.key -out server.csr
openssl x509 -req -days 365 -in server.csr -CA ca.pem -CAkey ca.key -out server.pem
```

### Configure mTLS
Set environment variables:
```bash
export MCP_TLS_CA_PATH="/path/to/ca.pem"
export MCP_TLS_CERT_PATH="/path/to/server.pem"
export MCP_TLS_KEY_PATH="/path/to/server.key"
export MCP_REQUIRE_CLIENT_CERT=true
```

### Server-Sent Events
Access real-time updates at: `https://localhost:8443/events`

## Schema Validation & Auto-Repair

### Validation Process
The requirements are validated against schema: `{schema_id}`

### Auto-Repair Features
- **Missing Required Fields**: Automatically adds with default values
- **Type Coercion**: Converts compatible types (string ↔ number)
- **Format Correction**: Fixes datetime formats to ISO 8601
- **Structure Repair**: Adds missing nested objects/arrays

### Manual Validation
```bash
mcp-forge validate requirements.json {schema_id}
```

### Validation Results
- ✅ **Valid**: No issues found
- 🔧 **Repaired**: Auto-repair applied successfully  
- ❌ **Failed**: Manual intervention required

## LLM Router Configuration (PM stage)

Current LLM routing policy: **{llm_policy}**

### Available Models
- `local-small`: Fast responses, simple tasks
- `local-medium`: Balanced performance/quality
- `local-large`: Complex reasoning, high quality
- `local-structured`: JSON/structured output

### PM Stage Routing Rules
```json
{{
  "policy": "{llm_policy}",
  "routes": [
    {{
      "match": {{"task": "ideation", "complexity_max": 3}},
      "model": "local-small",
      "temperature": 0.7
    }},
    {{
      "match": {{"task": "requirements-refinement", "complexity_min": 3}},
      "model": "local-medium", 
      "temperature": 0.3
    }},
    {{
      "match": {{"task": "schema-binding"}},
      "model": "local-structured",
      "temperature": 0.0,
      "structured_output": true
    }}
  ]
}}
```

### Model Health Check
```bash
mcp-forge health
```

## Saving Artifacts and Versioning

### Artifact Structure
```
/var/lib/mcp/handoffs/pm/
├── 1.0.0/
│   ├── pm-1.0.0.json          # Requirements JSON
│   ├── pm-1.0.0.md            # This README
│   └── pm-1.0.0.metadata.json # Artifact metadata
└── 1.0.1/
    └── ...
```

### Version Management
- **Semantic Versioning**: MAJOR.MINOR.PATCH
- **Auto-increment**: Patch version increments automatically
- **Manual Versioning**: Specify version explicitly

### Creating Artifacts
```bash
# Via CLI
mcp-forge artifact create --profile pm --data-file requirements.json --version 1.0.0

# Via API
curl -X POST http://localhost:8788/api/artifacts \
  -H "Content-Type: application/json" \
  -d '{{"profile": "pm", "data": {{...}}, "version": "1.0.0"}}'
```

### Listing Artifacts
```bash
mcp-forge artifact list --profile pm
```

## Troubleshooting

### Common Issues

#### 1. Schema Validation Errors
**Problem**: Requirements don't match schema
**Solution**: 
```bash
mcp-forge validate requirements.json {schema_id} --auto-repair
```

#### 2. LLM Provider Unreachable
**Problem**: Cannot connect to local LLM endpoints
**Solution**:
- Check endpoints in `config.env`
- Verify LLM servers are running
- Run health check: `mcp-forge health`

#### 3. Permission Errors
**Problem**: Cannot write to `/var/lib/mcp/`
**Solution**:
```bash
# Use local directory
export MCP_DATA_ROOT="./data"
mkdir -p data
```

#### 4. UI Not Loading
**Problem**: Business UI not accessible
**Solution**:
- Check port availability: `netstat -an | grep 8788`
- Try different port: `mcp-forge ui --port 8789`
- Check firewall settings

### Debug Mode
```bash
mcp-forge serve --log-level DEBUG
```

### Log Files
- Application logs: `mcp-forge.log`
- Audit logs: `/var/log/mcp/audit.log`
- Error logs: Check systemd journal if using service

### Getting Help
1. Check logs for error details
2. Verify configuration with `mcp-forge health`
3. Test with minimal example
4. Review schema documentation

## Next Steps

### Handoff to TPM
1. **Review Requirements**: Ensure completeness and clarity
2. **Generate Technical Implementation**: 
   ```bash
   mcp-forge artifact transform --from-profile pm --to-profile tpm --from-version {version}
   ```
3. **Validate Handoff**: Check generated technical specification
4. **Update Stakeholders**: Notify TPM of completed requirements

### Quality Gates
- ✅ All requirements have acceptance criteria
- ✅ Business value is clearly defined
- ✅ Constraints and dependencies identified
- ✅ Stakeholders reviewed and approved
- ✅ Schema validation passes

---

*Generated by MCP Forge v1.0.0 - Local-first Model Context Protocol Server*
