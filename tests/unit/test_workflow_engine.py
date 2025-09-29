"""Unit tests for workflow engine module."""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from mcp_forge.workflow_engine import (
    WorkflowEngine, WorkflowStep, Workflow, WorkflowError, 
    WorkflowStatus, StepStatus, RetryConfig
)


class TestRetryConfig:
    """Test cases for RetryConfig class."""

    def test_retry_config_creation(self):
        """Test creating retry configuration."""
        config = RetryConfig(
            strategy="exponential",
            base_ms=1000,
            max_retries=3,
            max_delay_ms=30000
        )
        
        assert config.strategy == "exponential"
        assert config.base_ms == 1000
        assert config.max_retries == 3
        assert config.max_delay_ms == 30000

    def test_retry_config_defaults(self):
        """Test retry configuration with defaults."""
        config = RetryConfig()
        
        assert config.strategy == "exponential"
        assert config.base_ms == 1000
        assert config.max_retries == 3
        assert config.max_delay_ms == 60000

    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(strategy="exponential", base_ms=1000)
        
        assert config.calculate_delay(0) == 1000
        assert config.calculate_delay(1) == 2000
        assert config.calculate_delay(2) == 4000
        assert config.calculate_delay(3) == 8000

    def test_calculate_delay_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(strategy="linear", base_ms=1000)
        
        assert config.calculate_delay(0) == 1000
        assert config.calculate_delay(1) == 2000
        assert config.calculate_delay(2) == 3000
        assert config.calculate_delay(3) == 4000

    def test_calculate_delay_fixed(self):
        """Test fixed delay calculation."""
        config = RetryConfig(strategy="fixed", base_ms=1000)
        
        assert config.calculate_delay(0) == 1000
        assert config.calculate_delay(1) == 1000
        assert config.calculate_delay(2) == 1000

    def test_calculate_delay_max_limit(self):
        """Test delay calculation with max limit."""
        config = RetryConfig(strategy="exponential", base_ms=1000, max_delay_ms=5000)
        
        # Should cap at max_delay_ms
        assert config.calculate_delay(10) == 5000

    def test_retry_config_serialization(self):
        """Test retry config serialization."""
        config = RetryConfig(strategy="exponential", base_ms=2000, max_retries=5)
        data = config.to_dict()
        
        assert data["strategy"] == "exponential"
        assert data["base_ms"] == 2000
        assert data["max_retries"] == 5


class TestWorkflowStep:
    """Test cases for WorkflowStep class."""

    def test_step_creation(self):
        """Test creating workflow step."""
        step = WorkflowStep(
            id="test-step",
            name="Test Step",
            description="A test step",
            handler="test_handler",
            params={"param1": "value1"},
            depends_on=["step1", "step2"],
            timeout_seconds=300
        )
        
        assert step.id == "test-step"
        assert step.name == "Test Step"
        assert step.description == "A test step"
        assert step.handler == "test_handler"
        assert step.params == {"param1": "value1"}
        assert step.depends_on == ["step1", "step2"]
        assert step.timeout_seconds == 300
        assert step.status == StepStatus.PENDING

    def test_step_with_retry_config(self):
        """Test step with retry configuration."""
        retry_config = RetryConfig(max_retries=5)
        step = WorkflowStep(
            id="retry-step",
            name="Retry Step",
            handler="handler",
            retry_config=retry_config
        )
        
        assert step.retry_config == retry_config
        assert step.retry_config.max_retries == 5

    def test_step_status_transitions(self):
        """Test step status transitions."""
        step = WorkflowStep(id="test", name="Test", handler="handler")
        
        # Initial state
        assert step.status == StepStatus.PENDING
        
        # Start execution
        step.start_execution()
        assert step.status == StepStatus.RUNNING
        assert step.started_at is not None
        
        # Complete successfully
        step.complete_successfully({"result": "success"})
        assert step.status == StepStatus.COMPLETED
        assert step.completed_at is not None
        assert step.result == {"result": "success"}
        
        # Reset for failure test
        step.status = StepStatus.RUNNING
        step.completed_at = None
        step.result = None
        
        # Fail execution
        step.fail_execution("Test error")
        assert step.status == StepStatus.FAILED
        assert step.completed_at is not None
        assert step.error == "Test error"

    def test_step_execution_time(self):
        """Test step execution time calculation."""
        step = WorkflowStep(id="test", name="Test", handler="handler")
        
        # No execution time when not started
        assert step.execution_time is None
        
        # Start execution
        step.start_execution()
        
        # Should have execution time when running
        assert step.execution_time is not None
        assert step.execution_time >= 0
        
        # Complete execution
        step.complete_successfully({})
        
        # Should still have execution time when completed
        assert step.execution_time is not None

    def test_step_serialization(self):
        """Test step serialization."""
        step = WorkflowStep(
            id="serialize-test",
            name="Serialize Test",
            handler="handler",
            params={"key": "value"},
            depends_on=["dep1"]
        )
        
        data = step.to_dict()
        
        assert data["id"] == "serialize-test"
        assert data["name"] == "Serialize Test"
        assert data["handler"] == "handler"
        assert data["params"] == {"key": "value"}
        assert data["depends_on"] == ["dep1"]
        assert data["status"] == "pending"


class TestWorkflow:
    """Test cases for Workflow class."""

    @pytest.fixture
    def sample_steps(self):
        """Create sample workflow steps."""
        return [
            WorkflowStep(
                id="step1",
                name="First Step",
                handler="handler1",
                params={"input": "data1"}
            ),
            WorkflowStep(
                id="step2",
                name="Second Step",
                handler="handler2",
                depends_on=["step1"]
            ),
            WorkflowStep(
                id="step3",
                name="Third Step",
                handler="handler3",
                depends_on=["step1", "step2"]
            )
        ]

    def test_workflow_creation(self, sample_steps):
        """Test creating workflow."""
        workflow = Workflow(
            id="test-workflow",
            name="Test Workflow",
            description="A test workflow",
            steps=sample_steps
        )
        
        assert workflow.id == "test-workflow"
        assert workflow.name == "Test Workflow"
        assert workflow.description == "A test workflow"
        assert len(workflow.steps) == 3
        assert workflow.status == WorkflowStatus.PENDING

    def test_workflow_step_lookup(self, sample_steps):
        """Test workflow step lookup by ID."""
        workflow = Workflow(id="test", name="Test", steps=sample_steps)
        
        step = workflow.get_step("step2")
        assert step is not None
        assert step.id == "step2"
        assert step.name == "Second Step"
        
        # Non-existent step
        assert workflow.get_step("non-existent") is None

    def test_workflow_dependency_validation(self):
        """Test workflow dependency validation."""
        # Valid dependencies
        valid_steps = [
            WorkflowStep(id="step1", name="Step 1", handler="handler1"),
            WorkflowStep(id="step2", name="Step 2", handler="handler2", depends_on=["step1"])
        ]
        workflow = Workflow(id="valid", name="Valid", steps=valid_steps)
        assert workflow.validate_dependencies() is True
        
        # Invalid dependencies (circular)
        invalid_steps = [
            WorkflowStep(id="step1", name="Step 1", handler="handler1", depends_on=["step2"]),
            WorkflowStep(id="step2", name="Step 2", handler="handler2", depends_on=["step1"])
        ]
        workflow = Workflow(id="invalid", name="Invalid", steps=invalid_steps)
        assert workflow.validate_dependencies() is False

    def test_workflow_execution_order(self, sample_steps):
        """Test workflow execution order calculation."""
        workflow = Workflow(id="test", name="Test", steps=sample_steps)
        
        execution_order = workflow.get_execution_order()
        
        # step1 should be first (no dependencies)
        assert execution_order[0].id == "step1"
        
        # step2 should be second (depends on step1)
        assert execution_order[1].id == "step2"
        
        # step3 should be last (depends on step1 and step2)
        assert execution_order[2].id == "step3"

    def test_workflow_ready_steps(self, sample_steps):
        """Test getting ready-to-execute steps."""
        workflow = Workflow(id="test", name="Test", steps=sample_steps)
        
        # Initially, only step1 should be ready (no dependencies)
        ready_steps = workflow.get_ready_steps()
        assert len(ready_steps) == 1
        assert ready_steps[0].id == "step1"
        
        # Complete step1
        workflow.get_step("step1").complete_successfully({})
        
        # Now step2 should be ready
        ready_steps = workflow.get_ready_steps()
        assert len(ready_steps) == 1
        assert ready_steps[0].id == "step2"
        
        # Complete step2
        workflow.get_step("step2").complete_successfully({})
        
        # Now step3 should be ready
        ready_steps = workflow.get_ready_steps()
        assert len(ready_steps) == 1
        assert ready_steps[0].id == "step3"

    def test_workflow_progress_calculation(self, sample_steps):
        """Test workflow progress calculation."""
        workflow = Workflow(id="test", name="Test", steps=sample_steps)
        
        # No steps completed
        assert workflow.progress == 0.0
        
        # Complete one step
        workflow.get_step("step1").complete_successfully({})
        assert workflow.progress == pytest.approx(1/3, rel=1e-2)
        
        # Complete second step
        workflow.get_step("step2").complete_successfully({})
        assert workflow.progress == pytest.approx(2/3, rel=1e-2)
        
        # Complete all steps
        workflow.get_step("step3").complete_successfully({})
        assert workflow.progress == 1.0

    def test_workflow_status_updates(self, sample_steps):
        """Test workflow status updates based on step status."""
        workflow = Workflow(id="test", name="Test", steps=sample_steps)
        
        # Initially pending
        assert workflow.status == WorkflowStatus.PENDING
        
        # Start first step
        workflow.get_step("step1").start_execution()
        workflow.update_status()
        assert workflow.status == WorkflowStatus.RUNNING
        
        # Fail a step
        workflow.get_step("step1").fail_execution("Test error")
        workflow.update_status()
        assert workflow.status == WorkflowStatus.FAILED
        
        # Reset and complete all steps
        for step in workflow.steps:
            step.status = StepStatus.COMPLETED
        workflow.update_status()
        assert workflow.status == WorkflowStatus.COMPLETED

    def test_workflow_serialization(self, sample_steps):
        """Test workflow serialization."""
        workflow = Workflow(
            id="serialize-test",
            name="Serialize Test",
            description="Test serialization",
            steps=sample_steps
        )
        
        data = workflow.to_dict()
        
        assert data["id"] == "serialize-test"
        assert data["name"] == "Serialize Test"
        assert data["description"] == "Test serialization"
        assert len(data["steps"]) == 3
        assert data["status"] == "pending"
        assert "progress" in data


class TestWorkflowEngine:
    """Test cases for WorkflowEngine class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for workflow data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_workflow(self):
        """Create sample workflow for testing."""
        steps = [
            WorkflowStep(
                id="validate",
                name="Validate Input",
                handler="validate_handler",
                params={"schema": "test-schema"}
            ),
            WorkflowStep(
                id="process",
                name="Process Data",
                handler="process_handler",
                depends_on=["validate"]
            ),
            WorkflowStep(
                id="save",
                name="Save Result",
                handler="save_handler",
                depends_on=["process"]
            )
        ]
        
        return Workflow(
            id="test-workflow",
            name="Test Workflow",
            description="A test workflow",
            steps=steps
        )

    @pytest.fixture
    def engine(self, temp_data_dir):
        """Create WorkflowEngine instance."""
        return WorkflowEngine(data_root=str(temp_data_dir))

    def test_engine_initialization(self, engine, temp_data_dir):
        """Test workflow engine initialization."""
        assert engine.data_root == str(temp_data_dir)
        assert len(engine.workflows) == 0
        assert len(engine.handlers) == 0

    def test_register_handler(self, engine):
        """Test registering workflow step handler."""
        def test_handler(params):
            return {"result": "test"}
        
        engine.register_handler("test_handler", test_handler)
        
        assert "test_handler" in engine.handlers
        assert engine.handlers["test_handler"] == test_handler

    def test_register_async_handler(self, engine):
        """Test registering async workflow step handler."""
        async def async_handler(params):
            return {"result": "async_test"}
        
        engine.register_handler("async_handler", async_handler)
        
        assert "async_handler" in engine.handlers
        assert engine.handlers["async_handler"] == async_handler

    def test_create_workflow(self, engine, sample_workflow):
        """Test creating workflow."""
        workflow_id = engine.create_workflow(sample_workflow)
        
        assert workflow_id == sample_workflow.id
        assert workflow_id in engine.workflows
        assert engine.workflows[workflow_id] == sample_workflow

    def test_create_workflow_duplicate_id(self, engine, sample_workflow):
        """Test creating workflow with duplicate ID."""
        engine.create_workflow(sample_workflow)
        
        with pytest.raises(WorkflowError, match="already exists"):
            engine.create_workflow(sample_workflow)

    def test_get_workflow(self, engine, sample_workflow):
        """Test getting workflow by ID."""
        engine.create_workflow(sample_workflow)
        
        retrieved = engine.get_workflow(sample_workflow.id)
        assert retrieved == sample_workflow
        
        # Non-existent workflow
        assert engine.get_workflow("non-existent") is None

    def test_list_workflows(self, engine, sample_workflow):
        """Test listing workflows."""
        # Empty initially
        workflows = engine.list_workflows()
        assert len(workflows) == 0
        
        # Add workflow
        engine.create_workflow(sample_workflow)
        workflows = engine.list_workflows()
        assert len(workflows) == 1
        assert workflows[0].id == sample_workflow.id

    def test_list_workflows_by_status(self, engine, sample_workflow):
        """Test listing workflows by status."""
        engine.create_workflow(sample_workflow)
        
        # Filter by pending status
        pending = engine.list_workflows(status=WorkflowStatus.PENDING)
        assert len(pending) == 1
        
        # Filter by running status (should be empty)
        running = engine.list_workflows(status=WorkflowStatus.RUNNING)
        assert len(running) == 0

    @pytest.mark.asyncio
    async def test_execute_workflow_success(self, engine, sample_workflow):
        """Test successful workflow execution."""
        # Register handlers
        def validate_handler(params):
            return {"validated": True}
        
        def process_handler(params):
            return {"processed": True}
        
        def save_handler(params):
            return {"saved": True}
        
        engine.register_handler("validate_handler", validate_handler)
        engine.register_handler("process_handler", process_handler)
        engine.register_handler("save_handler", save_handler)
        
        # Create and execute workflow
        engine.create_workflow(sample_workflow)
        result = await engine.execute_workflow(sample_workflow.id)
        
        assert result["success"] is True
        assert result["workflow_id"] == sample_workflow.id
        
        # Check workflow status
        workflow = engine.get_workflow(sample_workflow.id)
        assert workflow.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_workflow_step_failure(self, engine, sample_workflow):
        """Test workflow execution with step failure."""
        # Register handlers with one that fails
        def validate_handler(params):
            return {"validated": True}
        
        def failing_handler(params):
            raise Exception("Handler failed")
        
        engine.register_handler("validate_handler", validate_handler)
        engine.register_handler("process_handler", failing_handler)
        
        # Create and execute workflow
        engine.create_workflow(sample_workflow)
        result = await engine.execute_workflow(sample_workflow.id)
        
        assert result["success"] is False
        assert "error" in result
        
        # Check workflow status
        workflow = engine.get_workflow(sample_workflow.id)
        assert workflow.status == WorkflowStatus.FAILED

    @pytest.mark.asyncio
    async def test_execute_workflow_with_retry(self, engine):
        """Test workflow execution with retry logic."""
        # Create workflow with retry configuration
        retry_config = RetryConfig(max_retries=2, base_ms=100)
        step = WorkflowStep(
            id="retry-step",
            name="Retry Step",
            handler="retry_handler",
            retry_config=retry_config
        )
        workflow = Workflow(id="retry-workflow", name="Retry Test", steps=[step])
        
        # Handler that fails first time, succeeds second time
        call_count = 0
        def retry_handler(params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt fails")
            return {"success": True, "attempt": call_count}
        
        engine.register_handler("retry_handler", retry_handler)
        engine.create_workflow(workflow)
        
        result = await engine.execute_workflow(workflow.id)
        
        assert result["success"] is True
        assert call_count == 2  # Should have retried once

    @pytest.mark.asyncio
    async def test_execute_workflow_timeout(self, engine):
        """Test workflow execution with timeout."""
        # Create workflow with short timeout
        step = WorkflowStep(
            id="timeout-step",
            name="Timeout Step",
            handler="slow_handler",
            timeout_seconds=1
        )
        workflow = Workflow(id="timeout-workflow", name="Timeout Test", steps=[step])
        
        # Slow handler that exceeds timeout
        async def slow_handler(params):
            await asyncio.sleep(2)  # Longer than timeout
            return {"result": "too_slow"}
        
        engine.register_handler("slow_handler", slow_handler)
        engine.create_workflow(workflow)
        
        result = await engine.execute_workflow(workflow.id)
        
        assert result["success"] is False
        assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_resume_workflow_from_checkpoint(self, engine, sample_workflow):
        """Test resuming workflow from checkpoint."""
        # Register handlers
        def validate_handler(params):
            return {"validated": True}
        
        def process_handler(params):
            return {"processed": True}
        
        def save_handler(params):
            return {"saved": True}
        
        engine.register_handler("validate_handler", validate_handler)
        engine.register_handler("process_handler", process_handler)
        engine.register_handler("save_handler", save_handler)
        
        # Create workflow and complete first step manually
        engine.create_workflow(sample_workflow)
        workflow = engine.get_workflow(sample_workflow.id)
        workflow.get_step("validate").complete_successfully({"validated": True})
        
        # Resume execution
        result = await engine.execute_workflow(sample_workflow.id, resume_from_checkpoint=True)
        
        assert result["success"] is True
        
        # Validate step should not have been executed again
        # (this would be verified by checking handler call counts in a real implementation)

    def test_save_checkpoint(self, engine, sample_workflow, temp_data_dir):
        """Test saving workflow checkpoint."""
        engine.create_workflow(sample_workflow)
        workflow = engine.get_workflow(sample_workflow.id)
        
        # Complete first step
        workflow.get_step("validate").complete_successfully({"validated": True})
        
        # Save checkpoint
        engine.save_checkpoint(sample_workflow.id)
        
        # Check that checkpoint file exists
        checkpoint_file = temp_data_dir / "checkpoints" / f"{sample_workflow.id}.json"
        assert checkpoint_file.exists()

    def test_load_checkpoint(self, engine, sample_workflow, temp_data_dir):
        """Test loading workflow checkpoint."""
        engine.create_workflow(sample_workflow)
        workflow = engine.get_workflow(sample_workflow.id)
        
        # Complete first step and save checkpoint
        workflow.get_step("validate").complete_successfully({"validated": True})
        engine.save_checkpoint(sample_workflow.id)
        
        # Reset workflow state
        workflow.get_step("validate").status = StepStatus.PENDING
        workflow.get_step("validate").result = None
        
        # Load checkpoint
        engine.load_checkpoint(sample_workflow.id)
        
        # Verify state was restored
        assert workflow.get_step("validate").status == StepStatus.COMPLETED
        assert workflow.get_step("validate").result == {"validated": True}

    def test_delete_workflow(self, engine, sample_workflow):
        """Test deleting workflow."""
        engine.create_workflow(sample_workflow)
        
        # Verify workflow exists
        assert engine.get_workflow(sample_workflow.id) is not None
        
        # Delete workflow
        engine.delete_workflow(sample_workflow.id)
        
        # Verify workflow is gone
        assert engine.get_workflow(sample_workflow.id) is None

    def test_get_workflow_status(self, engine, sample_workflow):
        """Test getting workflow status."""
        engine.create_workflow(sample_workflow)
        
        status = engine.get_workflow_status(sample_workflow.id)
        
        assert status["id"] == sample_workflow.id
        assert status["status"] == "pending"
        assert status["progress"] == 0.0
        assert "steps" in status

    def test_get_workflow_metrics(self, engine, sample_workflow):
        """Test getting workflow metrics."""
        engine.create_workflow(sample_workflow)
        
        metrics = engine.get_workflow_metrics(sample_workflow.id)
        
        assert "total_steps" in metrics
        assert "completed_steps" in metrics
        assert "failed_steps" in metrics
        assert "execution_time" in metrics

    def test_engine_health_check(self, engine):
        """Test engine health check."""
        health = engine.health_check()
        
        assert "status" in health
        assert "workflows" in health
        assert "handlers" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]

    def test_engine_serialization(self, engine, sample_workflow):
        """Test engine serialization."""
        engine.create_workflow(sample_workflow)
        
        data = engine.to_dict()
        
        assert "workflows" in data
        assert "handlers" in data
        assert "metrics" in data
        assert len(data["workflows"]) == 1


class TestWorkflowError:
    """Test WorkflowError exception class."""

    def test_error_creation(self):
        """Test creating WorkflowError."""
        error = WorkflowError("Test workflow error")
        assert str(error) == "Test workflow error"

    def test_error_with_workflow_id(self):
        """Test creating WorkflowError with workflow ID."""
        error = WorkflowError("Workflow failed", workflow_id="test-workflow")
        assert error.workflow_id == "test-workflow"
        assert "Workflow failed" in str(error)


if __name__ == "__main__":
    pytest.main([__file__])
