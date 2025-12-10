"""Configuration validation."""

from typing import Dict, Any, List, Optional


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure and values.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ConfigValidationError: If validation fails
    """
    errors = []
    
    # Validate API section
    if "api" not in config:
        errors.append("Missing 'api' section in configuration")
    else:
        api_config = config["api"]
        if "provider" not in api_config:
            errors.append("Missing 'api.provider' in configuration")
        if "model" not in api_config:
            errors.append("Missing 'api.model' in configuration")
        if "keys" not in api_config:
            errors.append("Missing 'api.keys' in configuration")
        elif isinstance(api_config["keys"], str):
            # If it's a string, it should be comma-separated
            if not api_config["keys"]:
                errors.append("'api.keys' cannot be empty")
    
    # Validate processing section
    if "processing" not in config:
        errors.append("Missing 'processing' section in configuration")
    else:
        proc_config = config["processing"]
        if "batch_size" not in proc_config:
            errors.append("Missing 'processing.batch_size' in configuration")
        elif not isinstance(proc_config["batch_size"], int) or proc_config["batch_size"] <= 0:
            errors.append("'processing.batch_size' must be a positive integer")
    
    # Validate tasks section
    if "tasks" not in config:
        errors.append("Missing 'tasks' section in configuration")
    elif not isinstance(config["tasks"], list):
        errors.append("'tasks' must be a list")
    elif len(config["tasks"]) == 0:
        errors.append("'tasks' list cannot be empty")
    else:
        for i, task in enumerate(config["tasks"]):
            if not isinstance(task, dict):
                errors.append(f"Task {i+1} must be a dictionary")
                continue
            
            required_fields = ["name", "input_column", "output_column", "prompt_template"]
            for field in required_fields:
                if field not in task:
                    errors.append(f"Task {i+1} missing required field: '{field}'")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_msg)

