"""
Workflow Engine with idempotency, checkpoints, and retry logic
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import traceback

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Individual workflow step"""
    id: str
    name: str
    description: str
    handler: str  # Function name to call
    params: Dict[str, Any]
    depends_on: List[str] = None
    retry_config: Optional[Dict[str, Any]] = None
    timeout_seconds: int = 300
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    attempt: int = 0


@dataclass
class Workflow:
    """Workflow definition and state"""
    id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    checkpoint_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.checkpoint_data is None:
            self.checkpoint_data = {}


class RetryConfig(BaseModel):
    """Retry configuration"""
    strategy: str = "exponential"  # exponential, linear, fixed
    base_ms: int = 200
    max_retries: int = 5
    max_delay_ms: int = 30000
    backoff_multiplier: float = 2.0


class WorkflowEngine:
    """
    Workflow engine with idempotency, checkpoints, and retry logic
    """
    
    def __init__(self, config):
        self.config = config
        self.workflows: Dict[str, Workflow] = {}
        self.step_handlers: Dict[str, Callable] = {}
        self.storage_path = Path(config.storage.root) / "workflows"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Register built-in step handlers
        self._register_builtin_handlers()
        
        # Load persisted workflows
        self._load_workflows()
        
    def _register_builtin_handlers(self):
        """Register built-in step handlers"""
        self.register_step_handler("validate_schema", self._validate_schema_step)
        self.register_step_handler("generate_artifact", self._generate_artifact_step)
        self.register_step_handler("transform_artifact", self._transform_artifact_step)
        self.register_step_handler("route_llm", self._route_llm_step)
        self.register_step_handler("save_checkpoint", self._save_checkpoint_step)
        self.register_step_handler("notify", self._notify_step)
        
    def register_step_handler(self, name: str, handler: Callable):
        """Register a step handler function"""
        self.step_handlers[name] = handler
        
    def _load_workflows(self):
        """Load persisted workflows from storage"""
        try:
            for workflow_file in self.storage_path.glob("*.json"):
                with open(workflow_file, 'r') as f:
                    workflow_data = json.load(f)
                    
                # Convert to Workflow object
                steps = [WorkflowStep(**step_data) for step_data in workflow_data.pop("steps", [])]
                workflow = Workflow(steps=steps, **workflow_data)
                
                # Convert datetime strings back to datetime objects
                if workflow.created_at and isinstance(workflow.created_at, str):
                    workflow.created_at = datetime.fromisoformat(workflow.created_at)
                if workflow.started_at and isinstance(workflow.started_at, str):
                    workflow.started_at = datetime.fromisoformat(workflow.started_at)
                if workflow.completed_at and isinstance(workflow.completed_at, str):
                    workflow.completed_at = datetime.fromisoformat(workflow.completed_at)
                    
                for step in workflow.steps:
                    if step.started_at and isinstance(step.started_at, str):
                        step.started_at = datetime.fromisoformat(step.started_at)
                    if step.completed_at and isinstance(step.completed_at, str):
                        step.completed_at = datetime.fromisoformat(step.completed_at)
                        
                self.workflows[workflow.id] = workflow
                logger.info(f"Loaded workflow: {workflow.id} ({workflow.status})")
                
        except Exception as e:
            logger.error(f"Failed to load workflows: {e}")
            
    def _save_workflow(self, workflow: Workflow):
        """Persist workflow to storage"""
        try:
            workflow_file = self.storage_path / f"{workflow.id}.json"
            with open(workflow_file, 'w') as f:
                json.dump(asdict(workflow), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save workflow {workflow.id}: {e}")
            
    async def create_workflow(self, name: str, description: str, steps: List[Dict[str, Any]]) -> str:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        # Convert step dictionaries to WorkflowStep objects
        workflow_steps = []
        for step_data in steps:
            step = WorkflowStep(
                id=step_data.get("id", str(uuid.uuid4())),
                name=step_data["name"],
                description=step_data["description"],
                handler=step_data["handler"],
                params=step_data.get("params", {}),
                depends_on=step_data.get("depends_on", []),
                retry_config=step_data.get("retry_config"),
                timeout_seconds=step_data.get("timeout_seconds", 300)
            )
            workflow_steps.append(step)
            
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            steps=workflow_steps
        )
        
        self.workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        
        logger.info(f"Created workflow: {workflow_id} - {name}")
        return workflow_id
        
    async def execute_workflow(self, workflow_id: str, resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """Execute a workflow with checkpoint resume capability"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        # Check if workflow can be resumed
        if resume_from_checkpoint and workflow.status in [WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            logger.info(f"Resuming workflow from checkpoint: {workflow_id}")
        else:
            # Reset workflow state for fresh execution
            workflow.status = WorkflowStatus.RUNNING
            workflow.started_at = datetime.now()
            workflow.completed_at = None
            workflow.error = None
            
            # Reset step states
            for step in workflow.steps:
                if not resume_from_checkpoint or step.status not in [StepStatus.COMPLETED]:
                    step.status = StepStatus.PENDING
                    step.started_at = None
                    step.completed_at = None
                    step.error = None
                    step.result = None
                    step.attempt = 0
                    
        try:
            # Execute steps in dependency order
            await self._execute_steps(workflow)
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            logger.info(f"Workflow completed: {workflow_id}")
            
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(e)
            logger.error(f"Workflow failed: {workflow_id} - {e}")
            
        finally:
            self._save_workflow(workflow)
            
        return {
            "workflow_id": workflow_id,
            "status": workflow.status,
            "completed_steps": len([s for s in workflow.steps if s.status == StepStatus.COMPLETED]),
            "total_steps": len(workflow.steps),
            "duration_seconds": (workflow.completed_at - workflow.started_at).total_seconds() if workflow.completed_at and workflow.started_at else None,
            "error": workflow.error
        }
        
    async def _execute_steps(self, workflow: Workflow):
        """Execute workflow steps in dependency order"""
        remaining_steps = [s for s in workflow.steps if s.status not in [StepStatus.COMPLETED, StepStatus.SKIPPED]]
        max_concurrent = self.config.concurrency.get("max_workers", 4)
        
        if max_concurrent == "auto":
            max_concurrent = min(4, len(remaining_steps))
        elif isinstance(max_concurrent, str):
            max_concurrent = 4
            
        semaphore = asyncio.Semaphore(max_concurrent)
        
        while remaining_steps:
            # Find steps that can be executed (dependencies satisfied)
            ready_steps = []
            for step in remaining_steps:
                if self._are_dependencies_satisfied(step, workflow.steps):
                    ready_steps.append(step)
                    
            if not ready_steps:
                # Check for circular dependencies or unsatisfied dependencies
                pending_steps = [s for s in remaining_steps if s.status == StepStatus.PENDING]
                if pending_steps:
                    raise Exception(f"Circular dependency or unsatisfied dependencies detected: {[s.id for s in pending_steps]}")
                break
                
            # Execute ready steps concurrently
            tasks = []
            for step in ready_steps:
                task = asyncio.create_task(self._execute_step_with_semaphore(semaphore, workflow, step))
                tasks.append(task)
                
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update remaining steps
            remaining_steps = [s for s in workflow.steps if s.status not in [StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED]]
            
            # Save checkpoint after each batch
            self._save_workflow(workflow)
            
    def _are_dependencies_satisfied(self, step: WorkflowStep, all_steps: List[WorkflowStep]) -> bool:
        """Check if step dependencies are satisfied"""
        if not step.depends_on:
            return True
            
        for dep_id in step.depends_on:
            dep_step = next((s for s in all_steps if s.id == dep_id), None)
            if not dep_step or dep_step.status != StepStatus.COMPLETED:
                return False
                
        return True
        
    async def _execute_step_with_semaphore(self, semaphore: asyncio.Semaphore, workflow: Workflow, step: WorkflowStep):
        """Execute step with concurrency control"""
        async with semaphore:
            await self._execute_step(workflow, step)
            
    async def _execute_step(self, workflow: Workflow, step: WorkflowStep):
        """Execute a single workflow step with retry logic"""
        if step.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]:
            return
            
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        retry_config = RetryConfig(**(step.retry_config or {}))
        
        for attempt in range(retry_config.max_retries + 1):
            step.attempt = attempt + 1
            
            try:
                # Get step handler
                handler = self.step_handlers.get(step.handler)
                if not handler:
                    raise ValueError(f"Handler not found: {step.handler}")
                    
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(workflow, step),
                    timeout=step.timeout_seconds
                )
                
                step.result = result
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.now()
                
                logger.info(f"Step completed: {step.id} - {step.name}")
                return
                
            except asyncio.TimeoutError:
                error_msg = f"Step timed out after {step.timeout_seconds} seconds"
                step.error = error_msg
                logger.error(f"Step timeout: {step.id} - {error_msg}")
                
            except Exception as e:
                error_msg = f"Step failed: {str(e)}"
                step.error = error_msg
                logger.error(f"Step error: {step.id} - {error_msg}\n{traceback.format_exc()}")
                
            # Check if we should retry
            if attempt < retry_config.max_retries:
                delay_ms = self._calculate_retry_delay(retry_config, attempt)
                logger.info(f"Retrying step {step.id} in {delay_ms}ms (attempt {attempt + 2}/{retry_config.max_retries + 1})")
                await asyncio.sleep(delay_ms / 1000)
            else:
                step.status = StepStatus.FAILED
                step.completed_at = datetime.now()
                raise Exception(f"Step failed after {retry_config.max_retries + 1} attempts: {step.error}")
                
    def _calculate_retry_delay(self, retry_config: RetryConfig, attempt: int) -> int:
        """Calculate retry delay based on strategy"""
        if retry_config.strategy == "exponential":
            delay = retry_config.base_ms * (retry_config.backoff_multiplier ** attempt)
        elif retry_config.strategy == "linear":
            delay = retry_config.base_ms * (attempt + 1)
        else:  # fixed
            delay = retry_config.base_ms
            
        return min(delay, retry_config.max_delay_ms)
        
    # Built-in step handlers
    async def _validate_schema_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Built-in schema validation step"""
        from .schema_registry import SchemaRegistry
        
        registry = SchemaRegistry(self.config)
        data = step.params.get("data")
        schema_id = step.params.get("schema_id")
        
        if not data or not schema_id:
            raise ValueError("Missing required parameters: data, schema_id")
            
        result = await registry.validate(data, schema_id)
        
        if not result.valid and not result.repair_applied:
            raise ValueError(f"Validation failed: {result.errors}")
            
        return {
            "valid": result.valid,
            "repaired": result.repair_applied,
            "data": result.repaired_data if result.repaired_data else data
        }
        
    async def _generate_artifact_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Built-in artifact generation step"""
        from .artifact_manager import ArtifactManager
        
        manager = ArtifactManager(self.config)
        profile = step.params.get("profile")
        data = step.params.get("data")
        version = step.params.get("version", "1.0.0")
        
        if not profile or not data:
            raise ValueError("Missing required parameters: profile, data")
            
        artifact_path = await manager.save_artifact(profile, data, version)
        
        return {
            "artifact_path": str(artifact_path),
            "profile": profile,
            "version": version
        }
        
    async def _transform_artifact_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Built-in artifact transformation step"""
        from .artifact_manager import ArtifactManager
        
        manager = ArtifactManager(self.config)
        from_profile = step.params.get("from_profile")
        to_profile = step.params.get("to_profile")
        from_version = step.params.get("from_version")
        
        if not from_profile or not to_profile or not from_version:
            raise ValueError("Missing required parameters: from_profile, to_profile, from_version")
            
        artifact_path = await manager.auto_transform(from_profile, to_profile, from_version)
        
        return {
            "artifact_path": str(artifact_path),
            "from_profile": from_profile,
            "to_profile": to_profile
        }
        
    async def _route_llm_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Built-in LLM routing step"""
        from .llm_router import LLMRouter
        
        router = LLMRouter(self.config)
        task = step.params.get("task")
        complexity = step.params.get("complexity", 5)
        profile = step.params.get("profile", "pm")
        prompt = step.params.get("prompt")
        
        if not task or not prompt:
            raise ValueError("Missing required parameters: task, prompt")
            
        result = await router.route_request(task, complexity, profile, prompt)
        
        return {
            "model": result.model,
            "response": result.response,
            "latency_ms": result.latency_ms
        }
        
    async def _save_checkpoint_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Built-in checkpoint save step"""
        checkpoint_data = step.params.get("data", {})
        workflow.checkpoint_data.update(checkpoint_data)
        
        self._save_workflow(workflow)
        
        return {"checkpoint_saved": True, "data": checkpoint_data}
        
    async def _notify_step(self, workflow: Workflow, step: WorkflowStep) -> Dict[str, Any]:
        """Built-in notification step"""
        message = step.params.get("message", "Workflow notification")
        recipients = step.params.get("recipients", [])
        
        # In a real implementation, this would send actual notifications
        logger.info(f"Notification: {message} (recipients: {recipients})")
        
        return {"message": message, "recipients": recipients, "sent": True}
        
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status and progress"""
        if workflow_id not in self.workflows:
            return None
            
        workflow = self.workflows[workflow_id]
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status,
            "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "error": workflow.error,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "status": step.status,
                    "attempt": step.attempt,
                    "error": step.error
                }
                for step in workflow.steps
            ],
            "progress": {
                "completed": len([s for s in workflow.steps if s.status == StepStatus.COMPLETED]),
                "failed": len([s for s in workflow.steps if s.status == StepStatus.FAILED]),
                "total": len(workflow.steps)
            }
        }
        
    def list_workflows(self, status_filter: Optional[WorkflowStatus] = None) -> List[Dict[str, Any]]:
        """List all workflows with optional status filter"""
        workflows = list(self.workflows.values())
        
        if status_filter:
            workflows = [w for w in workflows if w.status == status_filter]
            
        return [
            {
                "id": w.id,
                "name": w.name,
                "status": w.status,
                "created_at": w.created_at.isoformat() if w.created_at else None,
                "step_count": len(w.steps)
            }
            for w in sorted(workflows, key=lambda x: x.created_at or datetime.min, reverse=True)
        ]
        
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        
        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            self._save_workflow(workflow)
            logger.info(f"Cancelled workflow: {workflow_id}")
            return True
            
        return False
