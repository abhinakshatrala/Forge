"""
Schema Registry with validation, auto-repair, and versioning
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import re

import jsonschema
from jsonschema import Draft202012Validator, ValidationError
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Exception raised for schema validation errors"""
    
    def __init__(self, message: str, errors: Optional[List[str]] = None):
        super().__init__(message)
        self.errors = errors or []


@dataclass
class ValidationResult:
    """Result of schema validation"""
    valid: bool
    errors: List[Dict[str, Any]]
    repaired_data: Optional[Dict[str, Any]] = None
    repair_applied: bool = False


class SchemaInfo(BaseModel):
    """Schema metadata"""
    id: str
    version: str
    title: str
    description: str
    file_path: str
    checksum: str
    created_at: datetime
    updated_at: datetime


class SchemaRegistry:
    """
    JSON Schema registry with validation, auto-repair, and versioning
    Supports JSON Schema 2020-12 dialect
    """
    
    def __init__(self, config):
        self.config = config
        self.schema_path = Path(config.schema_registry.path)
        self.schema_path.mkdir(parents=True, exist_ok=True)
        
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.schema_info: Dict[str, SchemaInfo] = {}
        
        # Load all schemas
        self._load_schemas()
        
    def _load_schemas(self):
        """Load all schemas from the registry directory"""
        logger.info(f"Loading schemas from {self.schema_path}")
        
        for schema_file in self.schema_path.glob("*.json"):
            try:
                with open(schema_file, 'r') as f:
                    schema_data = json.load(f)
                    
                schema_id = schema_data.get("$id", schema_file.stem)
                self.schemas[schema_id] = schema_data
                
                # Extract version from filename or schema
                version_match = re.search(r'-(\d+\.\d+\.\d+)\.json$', schema_file.name)
                version = version_match.group(1) if version_match else "1.0.0"
                
                # Calculate checksum
                checksum = hashlib.sha256(json.dumps(schema_data, sort_keys=True).encode()).hexdigest()
                
                # Store schema info
                self.schema_info[schema_id] = SchemaInfo(
                    id=schema_id,
                    version=version,
                    title=schema_data.get("title", schema_file.stem),
                    description=schema_data.get("description", ""),
                    file_path=str(schema_file),
                    checksum=checksum,
                    created_at=datetime.fromtimestamp(schema_file.stat().st_ctime),
                    updated_at=datetime.fromtimestamp(schema_file.stat().st_mtime)
                )
                
                logger.info(f"Loaded schema: {schema_id} v{version}")
                
            except Exception as e:
                logger.error(f"Failed to load schema {schema_file}: {e}")
                
    def get_schema(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Get schema by ID"""
        return self.schemas.get(schema_id)
        
    def list_schemas(self) -> List[SchemaInfo]:
        """List all available schemas"""
        return list(self.schema_info.values())
        
    async def validate(self, data: Dict[str, Any], schema_id: str) -> ValidationResult:
        """
        Validate data against schema with optional auto-repair
        """
        schema = self.get_schema(schema_id)
        if not schema:
            return ValidationResult(
                valid=False,
                errors=[{"message": f"Schema not found: {schema_id}"}]
            )
            
        try:
            # Create validator
            validator = Draft202012Validator(schema)
            
            # Validate data
            errors = []
            for error in validator.iter_errors(data):
                errors.append({
                    "instancePath": list(error.absolute_path),
                    "schemaPath": list(error.schema_path),
                    "keyword": error.validator,
                    "message": error.message,
                    "failingValue": error.instance if len(str(error.instance)) < 100 else str(error.instance)[:100] + "..."
                })
                
            if not errors:
                return ValidationResult(valid=True, errors=[])
                
            # Attempt auto-repair if enabled
            if self.config.schema_registry.auto_repair_prompts:
                repaired_data = await self._attempt_repair(data, schema, errors)
                if repaired_data:
                    # Validate repaired data
                    repair_errors = []
                    for error in validator.iter_errors(repaired_data):
                        repair_errors.append({
                            "instancePath": list(error.absolute_path),
                            "schemaPath": list(error.schema_path),
                            "keyword": error.validator,
                            "message": error.message
                        })
                        
                    return ValidationResult(
                        valid=len(repair_errors) == 0,
                        errors=repair_errors,
                        repaired_data=repaired_data,
                        repair_applied=True
                    )
                    
            return ValidationResult(valid=False, errors=errors)
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                valid=False,
                errors=[{"message": f"Validation failed: {str(e)}"}]
            )
            
    async def _attempt_repair(self, data: Dict[str, Any], schema: Dict[str, Any], errors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Attempt to automatically repair data based on validation errors
        """
        try:
            repaired_data = json.loads(json.dumps(data))  # Deep copy
            
            for error in errors:
                path = error["instancePath"]
                keyword = error["keyword"]
                
                if keyword == "required":
                    # Add missing required fields with default values
                    missing_props = error["message"].split("'")[1::2]  # Extract property names
                    target = repaired_data
                    
                    # Navigate to the target object
                    for key in path:
                        if isinstance(target, dict) and key in target:
                            target = target[key]
                        else:
                            break
                    else:
                        # Add missing properties with appropriate defaults
                        if isinstance(target, dict):
                            for prop in missing_props:
                                if prop not in target:
                                    target[prop] = self._get_default_value_for_property(schema, path + [prop])
                                    
                elif keyword == "type":
                    # Attempt type coercion
                    target = repaired_data
                    for key in path[:-1]:
                        if isinstance(target, dict) and key in target:
                            target = target[key]
                        else:
                            break
                    else:
                        if isinstance(target, dict) and path[-1] in target:
                            current_value = target[path[-1]]
                            expected_type = self._extract_expected_type(error["message"])
                            coerced_value = self._coerce_type(current_value, expected_type)
                            if coerced_value is not None:
                                target[path[-1]] = coerced_value
                                
                elif keyword == "format":
                    # Attempt format correction
                    if "date-time" in error["message"]:
                        target = repaired_data
                        for key in path[:-1]:
                            if isinstance(target, dict) and key in target:
                                target = target[key]
                            else:
                                break
                        else:
                            if isinstance(target, dict) and path[-1] in target:
                                # Try to fix datetime format
                                current_value = target[path[-1]]
                                if isinstance(current_value, str):
                                    fixed_datetime = self._fix_datetime_format(current_value)
                                    if fixed_datetime:
                                        target[path[-1]] = fixed_datetime
                                        
            return repaired_data
            
        except Exception as e:
            logger.error(f"Auto-repair failed: {e}")
            return None
            
    def _get_default_value_for_property(self, schema: Dict[str, Any], path: List[str]) -> Any:
        """Get appropriate default value for a property based on schema"""
        try:
            # Navigate schema to find property definition
            current_schema = schema
            for key in path:
                if "properties" in current_schema and key in current_schema["properties"]:
                    current_schema = current_schema["properties"][key]
                elif "items" in current_schema:
                    current_schema = current_schema["items"]
                else:
                    break
                    
            # Return default based on type
            prop_type = current_schema.get("type", "string")
            if "default" in current_schema:
                return current_schema["default"]
            elif prop_type == "string":
                return ""
            elif prop_type == "integer":
                return 0
            elif prop_type == "number":
                return 0.0
            elif prop_type == "boolean":
                return False
            elif prop_type == "array":
                return []
            elif prop_type == "object":
                return {}
            else:
                return None
                
        except Exception:
            return ""
            
    def _extract_expected_type(self, error_message: str) -> str:
        """Extract expected type from validation error message"""
        if "is not of type" in error_message:
            # Extract type from message like "... is not of type 'string'"
            type_match = re.search(r"is not of type '(\w+)'", error_message)
            if type_match:
                return type_match.group(1)
        return "string"
        
    def _coerce_type(self, value: Any, expected_type: str) -> Any:
        """Attempt to coerce value to expected type"""
        try:
            if expected_type == "string":
                return str(value)
            elif expected_type == "integer":
                if isinstance(value, str) and value.isdigit():
                    return int(value)
                elif isinstance(value, (int, float)):
                    return int(value)
            elif expected_type == "number":
                if isinstance(value, str):
                    return float(value)
                elif isinstance(value, (int, float)):
                    return float(value)
            elif expected_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                else:
                    return bool(value)
            elif expected_type == "array":
                if not isinstance(value, list):
                    return [value]
                return value
        except (ValueError, TypeError):
            pass
        return None
        
    def _fix_datetime_format(self, value: str) -> Optional[str]:
        """Attempt to fix datetime format to ISO 8601"""
        try:
            # Try parsing common datetime formats
            from datetime import datetime
            
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y/%m/%d %H:%M:%S",
                "%d/%m/%Y %H:%M:%S",
                "%Y-%m-%d",
                "%d-%m-%Y"
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
                    
            # If all else fails, try to add current time to date-only strings
            if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
                return f"{value}T00:00:00"
                
        except Exception:
            pass
        return None
        
    async def validate_and_save_diff(self, old_data: Dict[str, Any], new_data: Dict[str, Any], schema_id: str) -> Dict[str, Any]:
        """
        Validate new data and save semantic diff if enabled
        """
        validation_result = await self.validate(new_data, schema_id)
        
        if self.config.schema_registry.diffs["enabled"]:
            diff = self._generate_semantic_diff(old_data, new_data)
            await self._save_diff(schema_id, diff)
            
        return {
            "validation": validation_result,
            "diff_saved": self.config.schema_registry.diffs["enabled"]
        }
        
    def _generate_semantic_diff(self, old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate semantic diff between two data objects"""
        def deep_diff(obj1, obj2, path=""):
            changes = []
            
            if type(obj1) != type(obj2):
                changes.append({
                    "path": path,
                    "type": "type_change",
                    "old_type": type(obj1).__name__,
                    "new_type": type(obj2).__name__,
                    "old_value": obj1,
                    "new_value": obj2
                })
                return changes
                
            if isinstance(obj1, dict):
                all_keys = set(obj1.keys()) | set(obj2.keys())
                for key in all_keys:
                    key_path = f"{path}.{key}" if path else key
                    
                    if key not in obj1:
                        changes.append({
                            "path": key_path,
                            "type": "addition",
                            "value": obj2[key]
                        })
                    elif key not in obj2:
                        changes.append({
                            "path": key_path,
                            "type": "deletion",
                            "value": obj1[key]
                        })
                    else:
                        changes.extend(deep_diff(obj1[key], obj2[key], key_path))
                        
            elif isinstance(obj1, list):
                if len(obj1) != len(obj2):
                    changes.append({
                        "path": path,
                        "type": "length_change",
                        "old_length": len(obj1),
                        "new_length": len(obj2)
                    })
                    
                for i, (item1, item2) in enumerate(zip(obj1, obj2)):
                    changes.extend(deep_diff(item1, item2, f"{path}[{i}]"))
                    
            else:
                if obj1 != obj2:
                    changes.append({
                        "path": path,
                        "type": "value_change",
                        "old_value": obj1,
                        "new_value": obj2
                    })
                    
            return changes
            
        return {
            "timestamp": datetime.now().isoformat(),
            "changes": deep_diff(old_data, new_data)
        }
        
    async def _save_diff(self, schema_id: str, diff: Dict[str, Any]):
        """Save diff to storage"""
        try:
            diff_dir = Path(self.config.schema_registry.diffs["store_path"])
            diff_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            diff_file = diff_dir / f"{schema_id}_{timestamp}.json"
            
            with open(diff_file, 'w') as f:
                json.dump(diff, f, indent=2)
                
            logger.info(f"Saved diff for {schema_id} to {diff_file}")
            
        except Exception as e:
            logger.error(f"Failed to save diff: {e}")
            
    def get_schema_compatibility(self, schema_id: str, old_version: str, new_version: str) -> Dict[str, Any]:
        """Check compatibility between schema versions"""
        # This is a simplified compatibility check
        # In a full implementation, you'd compare actual schema structures
        
        def version_tuple(v):
            return tuple(map(int, v.split('.')))
            
        old_ver = version_tuple(old_version)
        new_ver = version_tuple(new_version)
        
        if new_ver[0] > old_ver[0]:
            compatibility = "breaking"
        elif new_ver[1] > old_ver[1]:
            compatibility = "compatible"
        else:
            compatibility = "patch"
            
        return {
            "schema_id": schema_id,
            "old_version": old_version,
            "new_version": new_version,
            "compatibility": compatibility,
            "backward_compatible": compatibility in ["compatible", "patch"]
        }
