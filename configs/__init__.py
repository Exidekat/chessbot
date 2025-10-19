"""
Configuration Management

Provides utilities for loading, validating, and accessing configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
from .config_schema import DEFAULT_CONFIG, merge_configs, validate_config


DEFAULT_CONFIG_PATH = Path(__file__).parent / "current.yaml"


def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file.

    Falls back to defaults if file doesn't exist.
    Merges loaded config with defaults.

    Args:
        config_path: Path to config YAML file (default: configs/current.yaml)

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    config_file = Path(config_path)

    # Start with defaults
    config = DEFAULT_CONFIG.copy()

    # Load and merge if file exists
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}

            # Merge with defaults (loaded config takes precedence)
            config = merge_configs(config, loaded_config)

            print(f"[Config] Loaded configuration from {config_file}")

        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config: {e}")
    else:
        print(f"[Config] Config file not found at {config_file}, using defaults")

    # Validate
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return config


def save_config(config: Dict, config_path: Optional[str] = None):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config (default: configs/current.yaml)

    Raises:
        ValueError: If configuration is invalid
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # Validate before saving
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Cannot save invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"[Config] Configuration saved to {config_file}")

    except Exception as e:
        raise ValueError(f"Error saving config: {e}")


def get_camera_config() -> Dict:
    """Get camera configuration from current config."""
    config = load_config()
    return config.get("cameras", {})


def get_model_config() -> Dict:
    """Get model paths from current config."""
    config = load_config()
    return config.get("models", {})


def get_path_config() -> Dict:
    """Get file paths from current config."""
    config = load_config()
    return config.get("paths", {})


def get_viz_config() -> Dict:
    """Get visualization server config."""
    config = load_config()
    return config.get("visualization", {})


def get_guidance_config() -> Dict:
    """Get guidance system config."""
    config = load_config()
    return config.get("guidance", {})


__all__ = [
    'load_config',
    'save_config',
    'get_camera_config',
    'get_model_config',
    'get_path_config',
    'get_viz_config',
    'get_guidance_config',
    'DEFAULT_CONFIG',
    'validate_config',
    'merge_configs',
]
