"""Unit tests for schema registry module."""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from mcp_forge.schema_registry import SchemaRegistry, SchemaValidationError


class TestSchemaRegistry:
    """Test cases for SchemaRegistry class."""

    @pytest.fixture
    def temp_schema_dir(self):
        """Create temporary directory for schema files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_schema(self):
        """Sample JSON schema for testing."""
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://forge.dev/schemas/test-1.0.0.json",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }

    @pytest.fixture
    def registry(self, temp_schema_dir):
        """Create SchemaRegistry instance with temp directory."""
        return SchemaRegistry(schema_path=str(temp_schema_dir))

    def test_init_creates_schema_directory(self, temp_schema_dir):
        """Test that SchemaRegistry creates schema directory if it doesn't exist."""
        schema_path = temp_schema_dir / "schemas"
        registry = SchemaRegistry(schema_path=str(schema_path))
        assert schema_path.exists()
        assert schema_path.is_dir()

    def test_load_schema_success(self, registry, temp_schema_dir, sample_schema):
        """Test successful schema loading."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        loaded_schema = registry.load_schema("test-schema.json")
        assert loaded_schema == sample_schema

    def test_load_schema_file_not_found(self, registry):
        """Test schema loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            registry.load_schema("non-existent.json")

    def test_load_schema_invalid_json(self, registry, temp_schema_dir):
        """Test schema loading with invalid JSON."""
        schema_file = temp_schema_dir / "invalid.json"
        with open(schema_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            registry.load_schema("invalid.json")

    def test_validate_data_success(self, registry, temp_schema_dir, sample_schema):
        """Test successful data validation."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        valid_data = {"name": "John", "age": 30}
        result = registry.validate_data(valid_data, "test-schema.json")
        assert result.is_valid is True
        assert result.errors == []

    def test_validate_data_failure(self, registry, temp_schema_dir, sample_schema):
        """Test data validation failure."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        invalid_data = {"age": -5}  # Missing required 'name', invalid age
        result = registry.validate_data(invalid_data, "test-schema.json")
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_auto_repair_missing_required_field(self, registry, temp_schema_dir, sample_schema):
        """Test auto-repair of missing required fields."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        data_missing_name = {"age": 25}
        repaired = registry.auto_repair_data(data_missing_name, "test-schema.json")
        
        assert "name" in repaired
        assert repaired["age"] == 25

    def test_auto_repair_invalid_type(self, registry, temp_schema_dir, sample_schema):
        """Test auto-repair of invalid data types."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        data_wrong_type = {"name": "John", "age": "thirty"}
        repaired = registry.auto_repair_data(data_wrong_type, "test-schema.json")
        
        assert repaired["name"] == "John"
        assert isinstance(repaired["age"], int) or repaired["age"] is None

    def test_auto_repair_constraint_violation(self, registry, temp_schema_dir, sample_schema):
        """Test auto-repair of constraint violations."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        data_negative_age = {"name": "John", "age": -5}
        repaired = registry.auto_repair_data(data_negative_age, "test-schema.json")
        
        assert repaired["name"] == "John"
        assert repaired["age"] >= 0

    def test_get_schema_info(self, registry, temp_schema_dir, sample_schema):
        """Test getting schema information."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        info = registry.get_schema_info("test-schema.json")
        assert info["id"] == sample_schema["$id"]
        assert info["version"] == "1.0.0"
        assert "properties" in info

    def test_list_schemas(self, registry, temp_schema_dir, sample_schema):
        """Test listing available schemas."""
        # Create multiple schema files
        for i in range(3):
            schema_file = temp_schema_dir / f"schema-{i}.json"
            with open(schema_file, 'w') as f:
                json.dump(sample_schema, f)
        
        schemas = registry.list_schemas()
        assert len(schemas) == 3
        assert all(schema.endswith('.json') for schema in schemas)

    def test_semantic_diff_no_changes(self, registry, temp_schema_dir, sample_schema):
        """Test semantic diff with no changes."""
        schema_file = temp_schema_dir / "test-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        diff = registry.semantic_diff(sample_schema, sample_schema)
        assert diff["has_changes"] is False
        assert diff["breaking_changes"] == []
        assert diff["additions"] == []
        assert diff["modifications"] == []

    def test_semantic_diff_with_changes(self, registry):
        """Test semantic diff with actual changes."""
        old_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        
        new_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}  # Addition
            },
            "required": ["name", "email"]  # Breaking change
        }
        
        diff = registry.semantic_diff(old_schema, new_schema)
        assert diff["has_changes"] is True
        assert len(diff["additions"]) > 0
        assert len(diff["breaking_changes"]) > 0

    def test_check_compatibility_compatible(self, registry):
        """Test compatibility check for compatible schemas."""
        old_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        
        new_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}  # Optional addition
            },
            "required": ["name"]
        }
        
        result = registry.check_compatibility(old_schema, new_schema)
        assert result["compatible"] is True
        assert result["breaking_changes"] == []

    def test_check_compatibility_incompatible(self, registry):
        """Test compatibility check for incompatible schemas."""
        old_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        
        new_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name", "email"]  # Breaking change
        }
        
        result = registry.check_compatibility(old_schema, new_schema)
        assert result["compatible"] is False
        assert len(result["breaking_changes"]) > 0

    @patch('builtins.open', new_callable=mock_open)
    @patch('os.path.exists')
    def test_save_schema(self, mock_exists, mock_file, registry):
        """Test saving schema to file."""
        mock_exists.return_value = True
        schema = {"type": "object", "properties": {}}
        
        registry.save_schema(schema, "new-schema.json")
        
        mock_file.assert_called_once()
        handle = mock_file()
        handle.write.assert_called()

    def test_validate_schema_format_valid(self, registry, sample_schema):
        """Test validation of schema format - valid schema."""
        result = registry.validate_schema_format(sample_schema)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_schema_format_invalid(self, registry):
        """Test validation of schema format - invalid schema."""
        invalid_schema = {
            "type": "invalid_type",  # Invalid type
            "properties": "not_an_object"  # Should be object
        }
        
        result = registry.validate_schema_format(invalid_schema)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_error_handling_with_exception(self, registry, temp_schema_dir):
        """Test error handling when file operations fail."""
        # Create a file that will cause permission error
        schema_file = temp_schema_dir / "readonly.json"
        schema_file.write_text('{"type": "object"}')
        schema_file.chmod(0o000)  # Remove all permissions
        
        try:
            with pytest.raises(PermissionError):
                registry.load_schema("readonly.json")
        finally:
            schema_file.chmod(0o644)  # Restore permissions for cleanup

    def test_caching_behavior(self, registry, temp_schema_dir, sample_schema):
        """Test that schemas are cached after first load."""
        schema_file = temp_schema_dir / "cached-schema.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        # Load schema twice
        schema1 = registry.load_schema("cached-schema.json")
        schema2 = registry.load_schema("cached-schema.json")
        
        # Should be the same object (cached)
        assert schema1 is schema2

    def test_clear_cache(self, registry, temp_schema_dir, sample_schema):
        """Test cache clearing functionality."""
        schema_file = temp_schema_dir / "cache-test.json"
        with open(schema_file, 'w') as f:
            json.dump(sample_schema, f)
        
        # Load and cache schema
        registry.load_schema("cache-test.json")
        assert len(registry._schema_cache) > 0
        
        # Clear cache
        registry.clear_cache()
        assert len(registry._schema_cache) == 0


@pytest.mark.asyncio
class TestSchemaRegistryAsync:
    """Test async functionality of SchemaRegistry."""

    @pytest.fixture
    def registry(self):
        """Create SchemaRegistry instance."""
        return SchemaRegistry()

    async def test_async_validate_data(self, registry):
        """Test async data validation."""
        # This would test async validation if implemented
        pass

    async def test_async_auto_repair(self, registry):
        """Test async auto-repair functionality."""
        # This would test async auto-repair if implemented
        pass


class TestSchemaValidationError:
    """Test SchemaValidationError exception class."""

    def test_exception_creation(self):
        """Test creating SchemaValidationError with message."""
        error = SchemaValidationError("Test error message")
        assert str(error) == "Test error message"

    def test_exception_with_errors_list(self):
        """Test creating SchemaValidationError with errors list."""
        errors = ["Error 1", "Error 2"]
        error = SchemaValidationError("Validation failed", errors=errors)
        assert error.errors == errors
        assert "Validation failed" in str(error)


if __name__ == "__main__":
    pytest.main([__file__])
