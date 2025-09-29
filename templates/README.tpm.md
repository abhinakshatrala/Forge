# TPM README - {title}

## Overview
This document contains the technical implementation plan generated from PM requirements for project **{project_id}**.

**Generated Artifacts:**
- Technical Implementation JSON: `{artifact_path}`
- Version: {version}
- Created: {created_at}
- Based on Requirements: {requirements_version}

## Consuming PM Requirements

### Source Requirements
- **Requirements Version**: {requirements_version}
- **Project ID**: {project_id}
- **Requirements Path**: `/var/lib/mcp/handoffs/pm/{requirements_version}/`

### Requirements Analysis
The technical implementation addresses all requirements from the PM specification:

```bash
# View requirements summary
mcp-forge artifact list --profile pm

# Load specific requirements version
cat /var/lib/mcp/handoffs/pm/{requirements_version}/pm-{requirements_version}.json | jq '.requirements[] | {{id, title, priority}}'
```

### Traceability Matrix
Each technical component maps back to specific requirements:
- Requirements coverage analysis
- Gap identification
- Impact assessment for changes

## Auto-Generate Technical Implementation JSON

### Transformation Process
The technical implementation is auto-generated from PM requirements using:

```bash
mcp-forge artifact transform \
  --from-profile pm \
  --to-profile tpm \
  --from-version {requirements_version}
```

### Transformation Rules
1. **Requirements → Components**: High-level requirements become system components
2. **Acceptance Criteria → Interfaces**: Testable criteria define component interfaces  
3. **Constraints → NFRs**: Business constraints become non-functional requirements
4. **Dependencies → Architecture**: Requirement dependencies inform system architecture

### Manual Refinement
After auto-generation, review and refine:
- Component responsibilities and boundaries
- Interface definitions and contracts
- Technology stack selections
- Implementation phases and milestones

## Architecture Overview

{architecture_summary}

### System Components
```bash
# View component details
cat {artifact_path} | jq '.architecture.components[] | {{id, name, type, responsibilities}}'
```

### Data Flow Analysis
- **Real-time flows**: User interactions, API calls
- **Batch flows**: Data processing, reporting
- **Event flows**: System notifications, state changes

### Technology Stack
- **Languages**: Python 3.11+, JavaScript/TypeScript
- **Frameworks**: FastAPI, React/Vue.js
- **Databases**: PostgreSQL, Redis
- **Infrastructure**: Docker, Kubernetes

## NFRs and Dependencies

### Non-Functional Requirements

{nfrs_summary}

#### Performance Requirements
- **Response Time**: P50 < 50ms, P99 < 250ms
- **Throughput**: 1000+ requests/second
- **Concurrent Users**: 500+ simultaneous users

#### Scalability Requirements
- **Horizontal Scaling**: Auto-scale based on load
- **Data Volume**: Handle 10GB+ datasets
- **Geographic Distribution**: Multi-region deployment

#### Reliability Requirements
- **Availability**: 99.9% uptime (8.76 hours downtime/year)
- **Recovery Time**: RTO < 30 minutes
- **Data Loss**: RPO < 60 minutes
- **Backup Strategy**: Daily automated backups

#### Security Requirements
- **Authentication**: Multi-factor authentication
- **Authorization**: Role-based access control (RBAC)
- **Encryption**: TLS 1.3 in transit, AES-256 at rest
- **Compliance**: GDPR, SOC 2 Type II

### Dependencies
```bash
# View dependency graph
cat {artifact_path} | jq '.architecture.components[] | {{id, name, dependencies}}'
```

#### External Dependencies
- Third-party APIs and services
- Cloud provider services
- Open source libraries and frameworks

#### Internal Dependencies
- Shared libraries and components
- Database schemas and migrations
- Configuration and secrets management

## LLM Router Configuration (TPM stage)

Current LLM routing policy: **quality_then_latency**

### TPM Stage Routing Rules
```json
{{
  "policy": "quality_then_latency",
  "routes": [
    {{
      "match": {{"task": "design-elaboration", "complexity_min": 4}},
      "model": "local-large",
      "temperature": 0.2,
      "max_tokens": 12000
    }},
    {{
      "match": {{"task": "acceptance-criteria"}},
      "model": "local-structured", 
      "temperature": 0.0,
      "structured_output": true
    }}
  ],
  "fallbacks": ["local-medium", "local-small"]
}}
```

### Model Selection Strategy
- **Complex Design Tasks**: Use `local-large` for architectural decisions
- **Structured Output**: Use `local-structured` for JSON/schema generation
- **Quick Analysis**: Use `local-medium` for routine technical analysis

## Implementation Phases

{phases_summary}

### Phase Management
```bash
# View phase details
cat {artifact_path} | jq '.implementation_plan.phases[] | {{id, name, duration_weeks, deliverables}}'
```

### Phase Dependencies
- **Sequential Phases**: Must complete in order
- **Parallel Phases**: Can execute concurrently
- **Critical Path**: Phases that determine overall timeline

### Milestone Tracking
```bash
# Create workflow for phase tracking
mcp-forge workflow create \
  --name "Implementation Tracking" \
  --steps-file phase-workflow.json
```

### Risk Mitigation
- **Technical Risks**: Proof of concepts, prototypes
- **Resource Risks**: Team scaling, skill gaps
- **Timeline Risks**: Buffer time, parallel execution
- **Quality Risks**: Code reviews, automated testing

## Validation and Diffs

### Schema Validation
Technical implementation validated against: `schemas/technical-impl-1.0.0.json`

```bash
# Validate implementation
mcp-forge validate {artifact_path} schemas/technical-impl-1.0.0.json
```

### Change Management
```bash
# Generate diff from previous version
mcp-forge artifact diff \
  --profile tpm \
  --from-version 1.0.0 \
  --to-version 1.0.1
```

### Version Control Integration
- **Semantic Versioning**: Track breaking vs. compatible changes
- **Change Documentation**: Auto-generate change logs
- **Impact Analysis**: Identify affected components and phases

### Approval Workflow
1. **Technical Review**: Architecture and design validation
2. **Stakeholder Review**: Business alignment confirmation  
3. **Security Review**: Security and compliance verification
4. **Final Approval**: Sign-off for development handoff

## Handoff to Dev

### Development Handoff Package
```bash
# Generate dev tasks from technical implementation
mcp-forge artifact transform \
  --from-profile tpm \
  --to-profile dev \
  --from-version {version}
```

### Handoff Checklist
- ✅ **Architecture Documented**: All components and interfaces defined
- ✅ **NFRs Specified**: Performance, security, reliability requirements clear
- ✅ **Dependencies Identified**: All external and internal dependencies mapped
- ✅ **Phases Planned**: Implementation roadmap with milestones
- ✅ **Risks Assessed**: Mitigation strategies documented
- ✅ **Acceptance Criteria**: Clear definition of done for each component

### Development Support
- **Architecture Reviews**: Regular check-ins during implementation
- **Technical Guidance**: Support for complex technical decisions
- **Change Management**: Process for handling requirement changes
- **Quality Assurance**: Validation of implementation against design

### Monitoring and Metrics
```bash
# Setup monitoring workflow
mcp-forge workflow create \
  --name "Implementation Monitoring" \
  --steps-file monitoring-workflow.json
```

#### Key Metrics
- **Progress Tracking**: Phase completion, milestone achievement
- **Quality Metrics**: Code coverage, defect rates, performance benchmarks
- **Risk Indicators**: Schedule variance, resource utilization, technical debt

## Tools and Integration

### IDE Integration
- **Windsurf**: Native MCP support for technical specifications
- **Claude Code**: Context-aware development assistance
- **VS Code**: Extensions for MCP protocol integration

### Development Tools
```bash
# Generate development environment setup
cat {artifact_path} | jq '.code_structure.dependencies'

# Create development workflow
mcp-forge workflow create \
  --name "Development Setup" \
  --description "Automated development environment setup"
```

### Testing Strategy
- **Unit Testing**: Component-level validation
- **Integration Testing**: Interface and data flow validation  
- **Performance Testing**: NFR validation and benchmarking
- **Security Testing**: Vulnerability assessment and penetration testing

## Troubleshooting

### Common Issues

#### 1. Architecture Complexity
**Problem**: Design too complex for implementation timeline
**Solution**: 
- Break down into smaller components
- Identify MVP scope for first phase
- Defer non-critical features to later phases

#### 2. NFR Conflicts
**Problem**: Performance vs. security trade-offs
**Solution**:
- Prioritize based on business requirements
- Design configurable solutions
- Plan performance optimization phases

#### 3. Dependency Issues
**Problem**: External dependencies not available
**Solution**:
- Identify alternative solutions
- Plan abstraction layers
- Create fallback implementations

#### 4. Resource Constraints
**Problem**: Implementation requires more resources than available
**Solution**:
- Re-scope phases and deliverables
- Identify automation opportunities
- Plan team scaling strategy

### Debug and Analysis
```bash
# Analyze implementation complexity
cat {artifact_path} | jq '.implementation_plan.phases[].duration_weeks' | awk '{{sum+=$1}} END {{print "Total weeks:", sum}}'

# Check NFR completeness
mcp-forge validate {artifact_path} schemas/technical-impl-1.0.0.json --verbose
```

## Next Steps

### Immediate Actions
1. **Review and Approve**: Technical implementation review cycle
2. **Generate Dev Tasks**: Transform to development specifications
3. **Setup Monitoring**: Implementation tracking and metrics
4. **Resource Planning**: Team allocation and timeline confirmation

### Long-term Planning
- **Phase 2+ Planning**: Extended roadmap beyond MVP
- **Maintenance Strategy**: Post-launch support and evolution
- **Scaling Preparation**: Growth and expansion planning

---

*Generated by MCP Forge v1.0.0 - Technical Implementation Planning*
*Next: Hand off to Development Team for implementation*
