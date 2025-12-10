"""Configuration file loader with environment variable substitution."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file. If None, looks for .env in project root.
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent.parent.parent / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try to load from current directory
        load_dotenv()


def substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in config values.
    Supports ${VAR_NAME} and ${VAR_NAME:default} syntax.
    
    Args:
        value: Config value (can be dict, list, or string)
    
    Returns:
        Value with environment variables substituted
    """
    if isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        # Extract variable name and optional default
        var_expr = value[2:-1]
        if ":" in var_expr:
            var_name, default = var_expr.split(":", 1)
            return os.getenv(var_name.strip(), default.strip())
        else:
            env_value = os.getenv(var_expr.strip())
            if env_value is None:
                raise ValueError(f"Environment variable '{var_expr.strip()}' not found")
            return env_value
    return value


def load_config(config_path: Path, env_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and parse YAML configuration file with environment variable substitution.
    
    Args:
        config_path: Path to YAML config file
        env_path: Path to .env file (optional)
    
    Returns:
        Parsed configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    # Load environment variables first
    load_env_file(env_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        config = {}
    
    # Substitute environment variables
    config = substitute_env_vars(config)
    
    return config

