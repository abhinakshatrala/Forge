"""
Artifact Manager for handling profile-specific handoffs and versioning.

This module manages artifacts for different profiles (PM, TPM, Dev),
handles versioning, README generation, and handoffs between stages.
"""

import json
import logging
import hashlib
import shutil
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Union, List
from dataclasses import dataclass, asdict
from pydantic import BaseModel


logger = logging.getLogger(__name__)


@dataclass
class ArtifactMetadata:
    """Artifact metadata"""
    profile: str
    version: str
    created_at: datetime
    checksum: str
    file_path: str
    readme_path: Optional[str] = None


class ArtifactManager:
    """
    Manages artifacts for different profiles (PM, TPM, Dev)
    Handles versioning, README generation, and handoffs
    """
    
    def __init__(self, config):
        self.config = config
        self.base_dir = Path(config.paths["artifacts"])
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load README templates
        self.readme_templates = self._load_readme_templates()
        
    def _load_readme_templates(self) -> Dict[str, str]:
        """Load README templates for each profile"""
        templates = {}
        
        for profile_name, profile_config in self.config.profiles.items():
            template_path = Path(profile_config.readme_template)
            if template_path.exists():
                with open(template_path, 'r') as f:
                    templates[profile_name] = f.read()
            else:
                # Use default template
                templates[profile_name] = self._get_default_readme_template(profile_name)
                
        return templates
        
    def _get_default_readme_template(self, profile: str) -> str:
        """Get default README template for profile"""
        templates = {
            "pm": """# PM README - {title}

## Overview
This document contains the product requirements generated for {project_id}.

## Generated Artifacts
- Requirements JSON: `{artifact_path}`
- Version: {version}
- Created: {created_at}

## Local Setup
1. Install Python 3.11+
2. Create virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Schema Validation
The requirements are validated against the schema: `{schema_id}`

Auto-repair is enabled for common validation issues.

## LLM Router Configuration
This profile uses the following LLM routing policy: {llm_policy}

## Next Steps
Hand off to TPM for technical implementation planning.
""",
            "tpm": """# TPM README - {title}

## Overview
This document contains the technical implementation plan generated from PM requirements.

## Generated Artifacts
- Technical Implementation JSON: `{artifact_path}`
- Version: {version}
- Created: {created_at}
- Based on Requirements: {requirements_version}

## Architecture Overview
{architecture_summary}

## Implementation Phases
{phases_summary}

## Non-Functional Requirements
{nfrs_summary}

## Next Steps
Hand off to Dev team for implementation.
""",
            "dev": """# Dev README - {title}

## Overview
This document contains the development tasks and implementation details.

## Generated Artifacts
- Development Tasks JSON: `{artifact_path}`
- Version: {version}
- Created: {created_at}

## IDE Integration
This project supports:
- Windsurf IDE
- Claude Code

## Local Execution
All execution is local-only with default-deny egress policy.

## Testing
- Unit tests: `{unit_test_command}`
- Integration tests: `{integration_test_command}`
- Coverage: `{coverage_command}`

## Deployment
{deployment_steps}
"""
        }
        
        return templates.get(profile, "# {title}\n\nGenerated artifact for {profile} profile.")
        
    async def save_artifact(self, profile: str, data: Dict[str, Any], version: str = "1.0.0") -> Path:
        """
        Save artifact for specified profile with versioning
        """
        try:
            # Validate profile
            if profile not in self.config.profiles:
                raise ValueError(f"Unknown profile: {profile}")
                
            profile_config = self.config.profiles[profile]
            
            # Create profile directory
            profile_dir = self.base_dir / profile / version
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate artifact filename
            artifact_filename = f"{profile}-{version}.json"
            artifact_path = profile_dir / artifact_filename
            
            # Add metadata to data
            enhanced_data = {
                **data,
                "metadata": {
                    **data.get("metadata", {}),
                    "version": version,
                    "created_at": datetime.now().isoformat(),
                    "profile": profile,
                    "checksum": self._calculate_checksum(data)
                }
            }
            
            # Save artifact
            with open(artifact_path, 'w') as f:
                json.dump(enhanced_data, f, indent=2, default=str)
                
            # Generate and save README
            readme_path = await self._generate_readme(profile, enhanced_data, artifact_path)
            
            # Create artifact metadata
            metadata = ArtifactMetadata(
                profile=profile,
                version=version,
                created_at=datetime.now(),
                checksum=self._calculate_checksum(enhanced_data),
                file_path=str(artifact_path),
                readme_path=str(readme_path) if readme_path else None
            )
            
            # Save metadata
            metadata_path = profile_dir / f"{profile}-{version}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.dict(), f, indent=2, default=str)
                
            logger.info(f"Saved artifact for {profile} v{version}: {artifact_path}")
            
            return artifact_path
            
        except Exception as e:
            logger.error(f"Failed to save artifact: {e}")
            raise
            
    async def _generate_readme(self, profile: str, data: Dict[str, Any], artifact_path: Path) -> Optional[Path]:
        """Generate README for artifact"""
        try:
            template = self.readme_templates.get(profile, "")
            if not template:
                return None
                
            # Extract data for template variables
            metadata = data.get("metadata", {})
            
            # Prepare template variables
            template_vars = {
                "title": data.get("title", f"{profile.upper()} Artifact"),
                "project_id": metadata.get("project_id", "unknown"),
                "version": metadata.get("version", "1.0.0"),
                "created_at": metadata.get("created_at", datetime.now().isoformat()),
                "artifact_path": artifact_path.name,
                "schema_id": self.config.profiles[profile].artifact_schema_id,
                "llm_policy": self.config.profiles[profile].llm_router.policy,
                "profile": profile
            }
            
            # Profile-specific variables
            if profile == "tpm":
                template_vars.update({
                    "requirements_version": data.get("metadata", {}).get("requirements_version", "unknown"),
                    "architecture_summary": self._summarize_architecture(data),
                    "phases_summary": self._summarize_phases(data),
                    "nfrs_summary": self._summarize_nfrs(data)
                })
            elif profile == "dev":
                template_vars.update({
                    "unit_test_command": self._get_test_command(data, "unit"),
                    "integration_test_command": self._get_test_command(data, "integration"),
                    "coverage_command": self._get_test_command(data, "coverage"),
                    "deployment_steps": self._summarize_deployment(data)
                })
                
            # Format template
            readme_content = template.format(**template_vars)
            
            # Save README
            readme_path = artifact_path.parent / f"{profile}-{metadata.get('version', '1.0.0')}.md"
            with open(readme_path, 'w') as f:
                f.write(readme_content)
                
            logger.info(f"Generated README: {readme_path}")
            return readme_path
            
        except Exception as e:
            logger.error(f"Failed to generate README: {e}")
            return None
            
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 checksum of data"""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
        
    def _summarize_architecture(self, data: Dict[str, Any]) -> str:
        """Summarize architecture for TPM README"""
        architecture = data.get("architecture", {})
        components = architecture.get("components", [])
        
        if not components:
            return "No architecture components defined."
            
        summary = f"System consists of {len(components)} components:\n"
        for comp in components[:5]:  # Limit to first 5
            summary += f"- {comp.get('name', 'Unknown')}: {comp.get('type', 'unknown type')}\n"
            
        if len(components) > 5:
            summary += f"... and {len(components) - 5} more components.\n"
            
        return summary
        
    def _summarize_phases(self, data: Dict[str, Any]) -> str:
        """Summarize implementation phases for TPM README"""
        phases = data.get("implementation_plan", {}).get("phases", [])
        
        if not phases:
            return "No implementation phases defined."
            
        summary = f"Implementation planned in {len(phases)} phases:\n"
        for phase in phases:
            summary += f"- {phase.get('name', 'Unknown')}: {phase.get('duration_weeks', 'unknown')} weeks\n"
            
        return summary
        
    def _summarize_nfrs(self, data: Dict[str, Any]) -> str:
        """Summarize NFRs for TPM README"""
        nfrs = data.get("nfrs", {})
        
        if not nfrs:
            return "No non-functional requirements defined."
            
        summary = "Key NFRs:\n"
        
        performance = nfrs.get("performance", {})
        if performance:
            if "response_time_ms" in performance:
                summary += f"- Response time: {performance['response_time_ms']} ms\n"
            if "throughput_rps" in performance:
                summary += f"- Throughput: {performance['throughput_rps']} RPS\n"
                
        reliability = nfrs.get("reliability", {})
        if reliability and "availability_percent" in reliability:
            summary += f"- Availability: {reliability['availability_percent']}%\n"
            
        return summary
        
    def _get_test_command(self, data: Dict[str, Any], test_type: str) -> str:
        """Get test command for dev README"""
        testing = data.get("testing", {})
        commands = testing.get("test_commands", {})
        return commands.get(test_type, f"# {test_type} test command not defined")
        
    def _summarize_deployment(self, data: Dict[str, Any]) -> str:
        """Summarize deployment steps for dev README"""
        deployment = data.get("deployment", {})
        build_steps = deployment.get("build_steps", [])
        run_steps = deployment.get("run_steps", [])
        
        if not build_steps and not run_steps:
            return "No deployment steps defined."
            
        summary = "Deployment steps:\n"
        
        if build_steps:
            summary += "\nBuild:\n"
            for step in build_steps[:3]:  # Limit to first 3
                summary += f"- {step.get('name', 'Unknown')}: `{step.get('command', 'unknown')}`\n"
                
        if run_steps:
            summary += "\nRun:\n"
            for step in run_steps[:3]:  # Limit to first 3
                summary += f"- {step.get('name', 'Unknown')}: `{step.get('command', 'unknown')}`\n"
                
        return summary
        
    async def load_artifact(self, profile: str, version: str) -> Optional[Dict[str, Any]]:
        """Load artifact by profile and version"""
        try:
            artifact_path = self.base_dir / profile / version / f"{profile}-{version}.json"
            
            if not artifact_path.exists():
                return None
                
            with open(artifact_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Failed to load artifact {profile} v{version}: {e}")
            return None
            
    async def list_artifacts(self, profile: Optional[str] = None) -> List[ArtifactMetadata]:
        """List all artifacts, optionally filtered by profile"""
        artifacts = []
        
        try:
            search_dirs = [self.base_dir / profile] if profile else list(self.base_dir.iterdir())
            
            for profile_dir in search_dirs:
                if not profile_dir.is_dir():
                    continue
                    
                for version_dir in profile_dir.iterdir():
                    if not version_dir.is_dir():
                        continue
                        
                    metadata_file = version_dir / f"{profile_dir.name}-{version_dir.name}.metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata_data = json.load(f)
                            artifacts.append(ArtifactMetadata(**metadata_data))
                            
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            
        return sorted(artifacts, key=lambda x: x.created_at, reverse=True)
        
    async def auto_transform(self, from_profile: str, to_profile: str, from_version: str) -> Optional[Path]:
        """
        Auto-transform artifact from one profile to another
        Used for PM -> TPM -> Dev pipeline
        """
        try:
            # Check if auto-transform is configured
            to_profile_config = self.config.profiles.get(to_profile, {})
            if to_profile_config.get("auto_transform_from") != from_profile:
                raise ValueError(f"Auto-transform not configured from {from_profile} to {to_profile}")
                
            # Load source artifact
            source_data = await self.load_artifact(from_profile, from_version)
            if not source_data:
                raise ValueError(f"Source artifact not found: {from_profile} v{from_version}")
                
            # Transform data based on profiles
            transformed_data = await self._transform_data(source_data, from_profile, to_profile)
            
            # Generate new version
            new_version = self._generate_next_version(to_profile)
            
            # Save transformed artifact
            return await self.save_artifact(to_profile, transformed_data, new_version)
            
        except Exception as e:
            logger.error(f"Auto-transform failed: {e}")
            raise
            
    async def _transform_data(self, source_data: Dict[str, Any], from_profile: str, to_profile: str) -> Dict[str, Any]:
        """Transform data between profiles"""
        if from_profile == "pm" and to_profile == "tpm":
            return await self._transform_pm_to_tpm(source_data)
        elif from_profile == "tpm" and to_profile == "dev":
            return await self._transform_tpm_to_dev(source_data)
        else:
            raise ValueError(f"Unsupported transformation: {from_profile} -> {to_profile}")
            
    async def _transform_pm_to_tpm(self, pm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform PM requirements to TPM technical implementation"""
        requirements = pm_data.get("requirements", [])
        
        # Generate basic technical implementation structure
        tpm_data = {
            "metadata": {
                "project_id": pm_data.get("metadata", {}).get("project_id"),
                "requirements_version": pm_data.get("metadata", {}).get("version"),
                "author": "auto-transform"
            },
            "architecture": {
                "components": self._generate_components_from_requirements(requirements),
                "data_flow": [],
                "technology_stack": {
                    "languages": [{"name": "Python", "version": ">=3.11", "purpose": "Main implementation"}],
                    "frameworks": [{"name": "FastAPI", "version": ">=0.104.1", "component": "API"}],
                    "databases": [{"name": "SQLite", "type": "relational", "purpose": "Local storage"}],
                    "infrastructure": [{"name": "Docker", "type": "container", "purpose": "Deployment"}]
                }
            },
            "implementation_plan": {
                "phases": self._generate_phases_from_requirements(requirements),
                "milestones": [],
                "risk_mitigation": []
            },
            "nfrs": {
                "performance": {"response_time_ms": {"p50": 50, "p95": 200, "p99": 500}},
                "scalability": {"concurrent_users": 100},
                "reliability": {"availability_percent": 99.9},
                "security": {"authentication": ["api-key"], "authorization": "rbac"}
            }
        }
        
        return tpm_data
        
    async def _transform_tpm_to_dev(self, tpm_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform TPM technical implementation to Dev tasks"""
        components = tpm_data.get("architecture", {}).get("components", [])
        phases = tpm_data.get("implementation_plan", {}).get("phases", [])
        
        # Generate development tasks
        dev_data = {
            "metadata": {
                "project_id": tpm_data.get("metadata", {}).get("project_id"),
                "technical_impl_version": tmp_data.get("metadata", {}).get("version"),
                "author": "auto-transform"
            },
            "tasks": self._generate_tasks_from_components(components, phases),
            "code_structure": {
                "directories": self._generate_directory_structure(components),
                "key_files": self._generate_key_files(components),
                "dependencies": tpm_data.get("architecture", {}).get("technology_stack", {})
            },
            "testing": {
                "unit_tests": [],
                "integration_tests": [],
                "test_commands": {
                    "unit": "python -m pytest tests/unit/",
                    "integration": "python -m pytest tests/integration/",
                    "coverage": "python -m pytest --cov=src tests/"
                }
            },
            "deployment": {
                "environment": {"python_version": ">=3.11"},
                "build_steps": [
                    {"name": "install_deps", "command": "pip install -r requirements.txt"},
                    {"name": "run_tests", "command": "python -m pytest"}
                ],
                "run_steps": [
                    {"name": "start_server", "command": "python -m src.main"}
                ]
            }
        }
        
        return dev_data
        
    def _generate_components_from_requirements(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate architecture components from requirements"""
        components = [
            {
                "id": "COMP-001",
                "name": "API Gateway",
                "type": "gateway",
                "responsibilities": ["Request routing", "Authentication", "Rate limiting"]
            },
            {
                "id": "COMP-002", 
                "name": "Business Logic Service",
                "type": "service",
                "responsibilities": ["Core business logic", "Data processing"]
            },
            {
                "id": "COMP-003",
                "name": "Data Storage",
                "type": "database", 
                "responsibilities": ["Data persistence", "Query processing"]
            }
        ]
        
        return components
        
    def _generate_phases_from_requirements(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate implementation phases from requirements"""
        high_priority_reqs = [req for req in requirements if req.get("priority") in ["critical", "high"]]
        
        phases = [
            {
                "id": "PHASE-001",
                "name": "Foundation",
                "duration_weeks": 2,
                "deliverables": ["Basic API structure", "Authentication", "Database setup"],
                "requirements_covered": [req.get("id") for req in high_priority_reqs[:3]]
            },
            {
                "id": "PHASE-002", 
                "name": "Core Features",
                "duration_weeks": 4,
                "deliverables": ["Main business logic", "API endpoints", "Testing"],
                "requirements_covered": [req.get("id") for req in requirements[3:]]
            }
        ]
        
        return phases
        
    def _generate_tasks_from_components(self, components: List[Dict[str, Any]], phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate development tasks from components and phases"""
        tasks = []
        task_id = 1
        
        for component in components:
            tasks.append({
                "id": f"TASK-{task_id:03d}",
                "title": f"Implement {component.get('name')}",
                "description": f"Develop {component.get('name')} component with responsibilities: {', '.join(component.get('responsibilities', []))}",
                "type": "feature",
                "priority": "high",
                "estimated_hours": 16,
                "component_id": component.get("id")
            })
            task_id += 1
            
        return tasks
        
    def _generate_directory_structure(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate directory structure from components"""
        return [
            {"path": "src/", "purpose": "Source code"},
            {"path": "tests/", "purpose": "Test files"},
            {"path": "docs/", "purpose": "Documentation"},
            {"path": "config/", "purpose": "Configuration files"}
        ]
        
    def _generate_key_files(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate key files from components"""
        return [
            {"path": "src/main.py", "purpose": "Application entry point", "template": "FastAPI app", "language": "python"},
            {"path": "requirements.txt", "purpose": "Python dependencies", "template": "pip requirements", "language": "text"},
            {"path": "README.md", "purpose": "Project documentation", "template": "markdown", "language": "markdown"}
        ]
        
    def _generate_next_version(self, profile: str) -> str:
        """Generate next semantic version for profile"""
        try:
            existing_artifacts = [a for a in self.list_artifacts(profile) if a.profile == profile]
            if not existing_artifacts:
                return "1.0.0"
                
            # Get latest version and increment patch
            latest = max(existing_artifacts, key=lambda x: x.version)
            version_parts = latest.version.split('.')
            patch = int(version_parts[2]) + 1
            
            return f"{version_parts[0]}.{version_parts[1]}.{patch}"
            
        except Exception:
            return "1.0.0"
