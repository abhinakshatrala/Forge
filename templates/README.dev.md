# Dev README - {title}

## Overview
This document contains the development tasks and implementation details for project **{project_id}**.

**Generated Artifacts:**
- Development Tasks JSON: `{artifact_path}`
- Version: {version}
- Created: {created_at}
- Based on Technical Implementation: {technical_impl_version}

## IDE Integration (Windsurf / Claude Code)

### Windsurf IDE Integration
MCP Forge provides native Windsurf integration for seamless development:

```bash
# Setup Windsurf project
mcp-forge init --windsurf-config
```

**Windsurf Configuration:**
- **Project Config**: `.windsurf/project.json` with MCP settings
- **Context Files**: Automatic context loading from artifacts
- **Task Integration**: Development tasks available in IDE
- **Schema Validation**: Real-time validation against MCP schemas

**Recommended Extensions:**
- MCP Protocol Support
- JSON Schema Validator
- Python/FastAPI Support
- Docker Integration

### Claude Code Integration
Enhanced development experience with Claude Code:

```bash
# Generate Claude Code context
cat {artifact_path} | jq '.workflow_integration.ide_handoff.claude_code'
```

**Context Strategy:**
- **File Priorities**: Core implementation files loaded first
- **Architecture Context**: System design and component relationships
- **Requirements Traceability**: Link code to original requirements
- **Testing Context**: Test files and validation criteria

**Usage Patterns:**
```bash
# Load development context
claude-code --context-files="src/main.py,{artifact_path},schemas/*.json"

# Generate code from tasks
claude-code --task="TASK-001" --template="fastapi-endpoint"
```

## Local-Only Execution and Egress Policy

### Default-Deny Egress
All development execution follows strict local-only policy:

**Network Restrictions:**
- ❌ **Outbound Internet**: Blocked by default
- ✅ **Local Services**: LLM endpoints, databases
- ✅ **Development Tools**: Local package managers, build tools
- ⚠️ **Exceptions**: Explicitly allowlisted endpoints only

### Local Development Environment
```bash
# Setup local environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify local-only setup
mcp-forge health --check-egress
```

### Allowlisted Endpoints
Configure necessary external access in `config.env`:
```bash
# Package repositories (if needed)
export MCP_ALLOW_PYPI="https://pypi.org/simple/"

# Local LLM endpoints
export MCP_LLM_ENDPOINTS="http://127.0.0.1:9001,http://127.0.0.1:9002"

# Development tools
export MCP_ALLOW_LOCALHOST="127.0.0.1,localhost"
```

### Offline Development
- **Documentation**: Bundled offline docs in `/docs`
- **Dependencies**: Pre-downloaded in local cache
- **Testing**: All tests run without network access
- **Build Process**: Fully offline build pipeline

## Testing Harness (fixtures & mocks)

### Test Structure
```
tests/
├── unit/           # Component unit tests
├── integration/    # System integration tests  
├── fixtures/       # Test data and scenarios
├── mocks/         # Mock services and dependencies
└── performance/   # Load and performance tests
```

### Unit Testing
**Command**: `{unit_test_command}`

```bash
# Run all unit tests
{unit_test_command}

# Run specific test module
python -m pytest tests/unit/test_server.py -v

# Run with coverage
python -m pytest tests/unit/ --cov=mcp_forge --cov-report=html
```

**Test Categories:**
- **Schema Validation**: JSON schema compliance
- **LLM Router**: Model selection and routing logic
- **Artifact Manager**: CRUD operations and versioning
- **Workflow Engine**: Step execution and error handling

### Integration Testing
**Command**: `{integration_test_command}`

```bash
# Run integration tests
{integration_test_command}

# Test specific workflow
python -m pytest tests/integration/test_pm_to_tpm_workflow.py

# Test with real LLM endpoints
python -m pytest tests/integration/ --llm-endpoints
```

**Integration Scenarios:**
- **End-to-End Workflows**: PM → TPM → Dev transformation
- **UI Integration**: Business user conversation flows
- **MCP Protocol**: Client-server communication
- **Multi-Transport**: stdio, unix socket, HTTPS

### Test Fixtures
Located in `/var/lib/mcp/test/fixtures/`:

```bash
# Load test fixtures
ls /var/lib/mcp/test/fixtures/
├── pm-requirements-sample.json
├── tpm-implementation-sample.json
├── dev-tasks-sample.json
├── conversation-sessions/
└── workflow-definitions/
```

### Mock Services
Located in `/var/lib/mcp/test/mocks/`:

```bash
# Start mock LLM server
python tests/mocks/llm_server.py --port 9001

# Mock external APIs
python tests/mocks/api_server.py --config tests/mocks/api-config.json
```

**Available Mocks:**
- **LLM Providers**: Simulated local-small, local-medium, local-large
- **File System**: In-memory storage for testing
- **Network Services**: Mock HTTP endpoints
- **Time/Clock**: Controllable time for testing workflows

### Performance Testing
```bash
# Load testing
python -m pytest tests/performance/test_load.py

# Memory profiling  
python -m pytest tests/performance/test_memory.py --profile

# Benchmark critical paths
python -m pytest tests/performance/test_benchmarks.py --benchmark-only
```

## LLM Router Configuration (Dev stage)

Current LLM routing policy: **latency_then_cost**

### Dev Stage Routing Rules
```json
{{
  "policy": "latency_then_cost",
  "default_model": "local-small",
  "routes": [
    {{
      "match": {{"task": "code-diff", "complexity_min": 3}},
      "model": "local-medium",
      "temperature": 0.2,
      "max_tokens": 6000
    }},
    {{
      "match": {{"task": "unit-tests"}},
      "model": "local-structured",
      "temperature": 0.0,
      "structured_output": true
    }}
  ],
  "fallbacks": ["local-small"]
}}
```

### Development-Specific Tasks
- **Code Generation**: Fast iteration with local-small
- **Code Review**: Quality analysis with local-medium  
- **Test Generation**: Structured output with local-structured
- **Documentation**: Balanced approach with local-medium

### Model Performance Optimization
```bash
# Monitor LLM usage during development
mcp-forge llm-stats --profile dev

# Optimize model selection
mcp-forge llm-benchmark --tasks="code-diff,unit-tests"
```

## Deployment Steps

{deployment_steps}

### Environment Setup
**Python Version**: {python_version}

```bash
# Create development environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Build Process
```bash
# Install build dependencies
pip install build wheel

# Build package
python -m build

# Verify build
pip install dist/mcp_forge-*.whl
```

### Local Development Server
```bash
# Start all services
mcp-forge serve --transport stdio &
mcp-forge ui --host 127.0.0.1 --port 8788 &

# Or start individual components
python -m mcp_forge.main
```

### Production Deployment
```bash
# Build Docker image
docker build -t mcp-forge:latest .

# Run with Docker Compose
docker-compose up -d

# Deploy to Kubernetes (optional)
kubectl apply -f k8s/
```

### Health Checks
```bash
# Application health
curl http://localhost:8787/healthz

# MCP server health
echo '{{"jsonrpc": "2.0", "method": "ping", "id": 1}}' | mcp-forge serve --transport stdio

# UI health
curl http://localhost:8788/api/health
```

## Development Workflow

### Task-Driven Development
```bash
# List development tasks
cat {artifact_path} | jq '.tasks[] | {{id, title, priority, estimated_hours}}'

# Start working on a task
mcp-forge task start TASK-001

# Track progress
mcp-forge task update TASK-001 --status in_progress --notes "Implementing API endpoints"

# Complete task
mcp-forge task complete TASK-001 --result "API endpoints implemented and tested"
```

### Code Generation Workflow
```bash
# Generate code from task specification
mcp-forge generate-code \
  --task TASK-001 \
  --template fastapi-endpoint \
  --output src/api/endpoints.py

# Validate generated code
python -m py_compile src/api/endpoints.py
mcp-forge validate-code src/api/endpoints.py
```

### Testing Workflow
```bash
# Test-driven development
mcp-forge generate-tests --task TASK-001 --output tests/unit/test_endpoints.py

# Run tests for specific task
python -m pytest tests/unit/test_endpoints.py -v

# Update tests based on implementation
mcp-forge update-tests --task TASK-001 --code-changes src/api/endpoints.py
```

## Applying Diffs and Running Workflows

### Diff Management
```bash
# View changes between versions
mcp-forge artifact diff \
  --profile dev \
  --from-version 1.0.0 \
  --to-version 1.0.1

# Apply diff to codebase
mcp-forge apply-diff \
  --diff-file changes.json \
  --target-dir src/

# Validate diff application
mcp-forge validate-diff --diff-file changes.json --verify
```

### Workflow Execution
```bash
# Create development workflow
cat > dev-workflow.json << EOF
[
  {{
    "id": "setup",
    "name": "Environment Setup", 
    "handler": "setup_environment",
    "params": {{"python_version": "{python_version}"}}
  }},
  {{
    "id": "generate",
    "name": "Code Generation",
    "handler": "generate_code", 
    "params": {{"tasks": ["TASK-001", "TASK-002"]}},
    "depends_on": ["setup"]
  }},
  {{
    "id": "test",
    "name": "Run Tests",
    "handler": "run_tests",
    "params": {{"test_suite": "unit"}},
    "depends_on": ["generate"]
  }}
]
EOF

# Execute workflow
mcp-forge workflow create \
  --name "Development Pipeline" \
  --steps-file dev-workflow.json

mcp-forge workflow execute <workflow-id>
```

### Continuous Integration
```bash
# Setup CI workflow
cat > .github/workflows/ci.yml << EOF
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '{python_version}'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: {unit_test_command}
      - name: Run integration tests  
        run: {integration_test_command}
EOF
```

## Troubleshooting

### Common Development Issues

#### 1. LLM Endpoint Unreachable
**Problem**: Cannot connect to local LLM services
**Solution**:
```bash
# Check LLM service status
mcp-forge health --llm-only

# Start mock LLM for development
python tests/mocks/llm_server.py --port 9001 &

# Update configuration
export MCP_LLM_SMALL="http://127.0.0.1:9001"
```

#### 2. Schema Validation Failures
**Problem**: Generated artifacts don't validate
**Solution**:
```bash
# Debug validation errors
mcp-forge validate {artifact_path} schemas/dev-tasks-1.0.0.json --verbose

# Auto-repair common issues
mcp-forge validate {artifact_path} schemas/dev-tasks-1.0.0.json --auto-repair

# Update schema if needed
mcp-forge schema update schemas/dev-tasks-1.0.0.json --version 1.0.1
```

#### 3. Test Failures
**Problem**: Tests failing in local environment
**Solution**:
```bash
# Run tests with detailed output
{unit_test_command} -v --tb=long

# Check test environment
python -m pytest --collect-only tests/

# Reset test database/fixtures
mcp-forge test-reset --fixtures --mocks
```

#### 4. Performance Issues
**Problem**: Slow code generation or validation
**Solution**:
```bash
# Profile performance bottlenecks
python -m cProfile -o profile.stats -m mcp_forge.main

# Optimize LLM routing
mcp-forge llm-optimize --profile dev --task-types="code-diff,unit-tests"

# Use faster models for development
export MCP_DEV_FAST_MODE=true
```

### Debug Mode
```bash
# Enable debug logging
export MCP_LOG_LEVEL=DEBUG
mcp-forge serve --log-level DEBUG

# Debug specific components
export MCP_DEBUG_COMPONENTS="llm_router,workflow_engine"
```

### Development Tools
```bash
# Interactive development shell
mcp-forge shell

# Code quality checks
pre-commit run --all-files

# Security scanning
bandit -r src/

# Dependency vulnerability check
safety check
```

## IDE-Specific Setup

### Windsurf Configuration
```json
// .windsurf/project.json
{{
  "mcp": {{
    "enabled": true,
    "artifacts_path": "/var/lib/mcp/handoffs/dev",
    "schemas_path": "schemas/",
    "auto_context": ["requirements", "architecture", "tasks"]
  }},
  "python": {{
    "interpreter": "./venv/bin/python",
    "linting": ["flake8", "mypy"],
    "formatting": ["black", "isort"]
  }}
}}
```

### VS Code Configuration
```json
// .vscode/settings.json
{{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "files.associations": {{
    "*.json": "jsonc"
  }},
  "json.schemas": [
    {{
      "fileMatch": ["**/dev-tasks-*.json"],
      "url": "./schemas/dev-tasks-1.0.0.json"
    }}
  ]
}}
```

## Next Steps

### Implementation Checklist
- ✅ **Environment Setup**: Development environment configured
- ⏳ **Task Implementation**: Core development tasks in progress
- ⏳ **Testing**: Unit and integration tests being written
- ⏳ **Documentation**: Code documentation and examples
- ⏳ **Performance**: Optimization and benchmarking
- ⏳ **Deployment**: Production deployment preparation

### Quality Gates
- **Code Coverage**: Minimum 80% unit test coverage
- **Performance**: Meet NFR latency requirements
- **Security**: Pass security scanning and review
- **Documentation**: Complete API and user documentation
- **Integration**: Successful end-to-end testing

### Deployment Readiness
```bash
# Pre-deployment checklist
mcp-forge deployment-check \
  --coverage-threshold 80 \
  --performance-benchmark \
  --security-scan \
  --integration-tests
```

---

*Generated by MCP Forge v1.0.0 - Development Implementation*
*Ready for local-first development with IDE integration*
